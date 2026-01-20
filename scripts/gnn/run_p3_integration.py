#!/usr/bin/env python3
"""Run P3 integration and HPO retuning.

This script:
1. Loads the rebuilt graph cache
2. Runs P3 inference to get refined scores
3. Evaluates pipeline variants with/without P3
4. Runs HPO to find optimal P3 integration parameters

Usage:
    python scripts/gnn/run_p3_integration.py --graph_dir data/cache/gnn/<timestamp>
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_p3_model(checkpoint_path: Path, input_dim: int, device: str = "cuda"):
    """Load P3 model from checkpoint."""
    from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if checkpoint is wrapped or raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Wrapped format
        config = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
    else:
        # Raw state_dict format (used by train_p3_graph_reranker.py)
        state_dict = checkpoint
        config = {}

    # Infer dimensions from state_dict or use provided
    hidden_dim = config.get("hidden_dim", 128)
    num_layers = config.get("num_layers", 2)

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha_init=0.7,
        learn_alpha=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def run_p3_inference(
    model,
    graphs: List,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Run P3 inference to get refined scores."""
    refined_scores_list = []

    with torch.no_grad():
        for g in tqdm(graphs, desc="P3 inference"):
            g = g.to(device)

            # Get reranker scores as input
            original_scores = g.reranker_scores.to(device)

            # Run P3 model (returns only refined_scores)
            refined_scores = model(g.x, g.edge_index, original_scores)

            refined_scores_list.append(refined_scores.cpu().numpy())

    return refined_scores_list


def compute_ranking_metrics(
    gold_mask: np.ndarray,
    scores: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute ranking metrics."""
    metrics = {}

    # Sort by scores (descending)
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    # Recall@K
    for k in k_values:
        n_gold = gold_mask.sum()
        if n_gold > 0:
            recall_k = sorted_gold[:k].sum() / n_gold
        else:
            recall_k = 0.0
        metrics[f"recall@{k}"] = recall_k

    # MRR
    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) > 0:
        mrr = 1.0 / (gold_positions[0] + 1)
    else:
        mrr = 0.0
    metrics["mrr"] = mrr

    # nDCG@K
    for k in k_values:
        dcg = 0.0
        for i in range(min(k, len(sorted_gold))):
            if sorted_gold[i]:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        n_gold = int(gold_mask.sum())
        idcg = 0.0
        for i in range(min(k, n_gold)):
            idcg += 1.0 / np.log2(i + 2)

        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics


def evaluate_with_p3(
    graphs: List,
    refined_scores_list: Optional[List[np.ndarray]] = None,
    use_p3: bool = False,
) -> pd.DataFrame:
    """Evaluate ranking with or without P3."""
    results = []

    for i, g in enumerate(graphs):
        # Get gold mask (ensure CPU)
        node_labels = g.node_labels.cpu() if g.node_labels.is_cuda else g.node_labels
        gold_mask = node_labels.numpy() > 0
        if not gold_mask.any():
            continue

        # Get scores
        if use_p3 and refined_scores_list is not None:
            scores = refined_scores_list[i]
        else:
            reranker_scores = g.reranker_scores.cpu() if g.reranker_scores.is_cuda else g.reranker_scores
            scores = reranker_scores.numpy()

        # Compute metrics
        metrics = compute_ranking_metrics(gold_mask, scores)
        metrics["query_id"] = g.query_id
        metrics["post_id"] = g.post_id
        metrics["criterion_id"] = g.criterion_id
        metrics["has_evidence"] = int(g.y.item())
        metrics["n_gold"] = int(gold_mask.sum())
        metrics["n_candidates"] = len(gold_mask)
        metrics["fold_id"] = g.fold_id
        metrics["use_p3"] = use_p3

        results.append(metrics)

    return pd.DataFrame(results)


def load_graph_dataset(graph_dir: Path) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        fold_graphs[fold_id] = data["graphs"]
        logger.info(f"Loaded fold {fold_id}: {len(data['graphs'])} graphs")

    return fold_graphs, metadata


def run_cv_evaluation(
    fold_graphs: Dict[int, List],
    p3_checkpoint_dir: Path,
    device: str = "cuda",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run cross-validation evaluation with/without P3."""
    all_results_no_p3 = []
    all_results_with_p3 = []

    n_folds = len(fold_graphs)

    # Get input dimension from first graph
    sample_graph = fold_graphs[0][0]
    input_dim = sample_graph.x.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    for fold_id in range(n_folds):
        logger.info(f"\n=== Fold {fold_id} ===")

        graphs = fold_graphs[fold_id]
        logger.info(f"  {len(graphs)} graphs")

        # Filter to positive examples only (has evidence)
        pos_graphs = [g for g in graphs if g.y.item() > 0]
        logger.info(f"  {len(pos_graphs)} positive graphs")

        if len(pos_graphs) == 0:
            continue

        # Load P3 model for this fold
        p3_checkpoint = p3_checkpoint_dir / f"fold_{fold_id}_best.pt"
        if not p3_checkpoint.exists():
            logger.warning(f"  P3 checkpoint not found: {p3_checkpoint}")
            # Try without fold suffix
            p3_checkpoint = p3_checkpoint_dir / "best.pt"
            if not p3_checkpoint.exists():
                logger.warning(f"  Skipping P3 for fold {fold_id}")
                continue

        logger.info(f"  Loading P3 model from {p3_checkpoint}")
        p3_model = load_p3_model(p3_checkpoint, input_dim, device)

        # Run P3 inference
        logger.info("  Running P3 inference...")
        refined_scores = run_p3_inference(p3_model, pos_graphs, device)

        # Evaluate without P3
        logger.info("  Evaluating without P3...")
        results_no_p3 = evaluate_with_p3(pos_graphs, use_p3=False)
        all_results_no_p3.append(results_no_p3)

        # Evaluate with P3
        logger.info("  Evaluating with P3...")
        results_with_p3 = evaluate_with_p3(pos_graphs, refined_scores, use_p3=True)
        all_results_with_p3.append(results_with_p3)

        # Log fold metrics
        for name, df in [("Without P3", results_no_p3), ("With P3", results_with_p3)]:
            mrr = df["mrr"].mean()
            ndcg10 = df["ndcg@10"].mean()
            recall5 = df["recall@5"].mean()
            logger.info(f"  {name}: MRR={mrr:.4f}, nDCG@10={ndcg10:.4f}, Recall@5={recall5:.4f}")

    # Combine results
    df_no_p3 = pd.concat(all_results_no_p3, ignore_index=True) if all_results_no_p3 else pd.DataFrame()
    df_with_p3 = pd.concat(all_results_with_p3, ignore_index=True) if all_results_with_p3 else pd.DataFrame()

    return df_no_p3, df_with_p3


def main():
    parser = argparse.ArgumentParser(description="P3 Integration and Evaluation")
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Graph dataset directory",
    )
    parser.add_argument(
        "--p3_checkpoint_dir",
        type=str,
        default="outputs/gnn_research/20260117_p3_final/20260117_030023/p3_graph_reranker",
        help="P3 checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"outputs/p3_integration/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Load graph dataset
    graph_dir = Path(args.graph_dir)
    fold_graphs, metadata = load_graph_dataset(graph_dir)

    # Run CV evaluation
    p3_checkpoint_dir = Path(args.p3_checkpoint_dir)
    df_no_p3, df_with_p3 = run_cv_evaluation(
        fold_graphs,
        p3_checkpoint_dir,
        args.device,
    )

    if df_no_p3.empty or df_with_p3.empty:
        logger.error("No results generated. Check graph data and P3 checkpoints.")
        return

    # Save detailed results
    df_no_p3.to_csv(output_dir / "results_no_p3.csv", index=False)
    df_with_p3.to_csv(output_dir / "results_with_p3.csv", index=False)

    # Compute summary metrics
    metrics_no_p3 = {
        "mrr": {"mean": df_no_p3["mrr"].mean(), "std": df_no_p3["mrr"].std()},
        "ndcg@5": {"mean": df_no_p3["ndcg@5"].mean(), "std": df_no_p3["ndcg@5"].std()},
        "ndcg@10": {"mean": df_no_p3["ndcg@10"].mean(), "std": df_no_p3["ndcg@10"].std()},
        "recall@5": {"mean": df_no_p3["recall@5"].mean(), "std": df_no_p3["recall@5"].std()},
        "recall@10": {"mean": df_no_p3["recall@10"].mean(), "std": df_no_p3["recall@10"].std()},
    }

    metrics_with_p3 = {
        "mrr": {"mean": df_with_p3["mrr"].mean(), "std": df_with_p3["mrr"].std()},
        "ndcg@5": {"mean": df_with_p3["ndcg@5"].mean(), "std": df_with_p3["ndcg@5"].std()},
        "ndcg@10": {"mean": df_with_p3["ndcg@10"].mean(), "std": df_with_p3["ndcg@10"].std()},
        "recall@5": {"mean": df_with_p3["recall@5"].mean(), "std": df_with_p3["recall@5"].std()},
        "recall@10": {"mean": df_with_p3["recall@10"].mean(), "std": df_with_p3["recall@10"].std()},
    }

    # Compute improvements
    improvements = {}
    for metric in metrics_no_p3:
        base = metrics_no_p3[metric]["mean"]
        p3 = metrics_with_p3[metric]["mean"]
        improvements[metric] = {
            "absolute": p3 - base,
            "relative_pct": 100 * (p3 - base) / base if base > 0 else 0,
        }

    # Save summary
    summary = {
        "timestamp": timestamp,
        "graph_dir": str(graph_dir),
        "p3_checkpoint_dir": str(p3_checkpoint_dir),
        "n_queries_evaluated": len(df_no_p3),
        "without_p3": metrics_no_p3,
        "with_p3": metrics_with_p3,
        "improvements": improvements,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: P3 Graph Reranker Integration")
    logger.info("=" * 60)
    logger.info(f"Total queries evaluated: {len(df_no_p3)}")
    logger.info("")
    logger.info("Metric Comparison:")
    logger.info(f"{'Metric':<12} {'Without P3':>15} {'With P3':>15} {'Improvement':>15}")
    logger.info("-" * 60)

    for metric in ["mrr", "ndcg@5", "ndcg@10", "recall@5", "recall@10"]:
        base = metrics_no_p3[metric]["mean"]
        p3 = metrics_with_p3[metric]["mean"]
        imp = improvements[metric]["absolute"]
        imp_pct = improvements[metric]["relative_pct"]
        logger.info(f"{metric:<12} {base:>15.4f} {p3:>15.4f} {imp:>+10.4f} ({imp_pct:>+.1f}%)")

    logger.info("")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
