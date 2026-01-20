#!/usr/bin/env python3
"""Evaluate pipeline with P3 Graph Reranker enabled.

This script:
1. Loads pre-trained P3 models for each fold
2. Runs P3 inference to get refined scores
3. Evaluates pipeline with and without P3
4. Compares ranking metrics
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
from tqdm import tqdm

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN
from final_sc_review.metrics.ranking import recall_at_k, mrr_at_k, ndcg_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_p3_model(checkpoint_path: Path, device: str = "cuda") -> GraphRerankerGNN:
    """Load P3 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    config = checkpoint.get("config", {})
    input_dim = config.get("input_dim", 1032)
    hidden_dim = config.get("hidden_dim", 128)
    num_layers = config.get("num_layers", 2)

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha_init=0.7,
        learn_alpha=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def run_p3_inference(
    model: GraphRerankerGNN,
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

            # Run P3 model
            refined_scores, alpha = model(g.x, g.edge_index, original_scores)

            refined_scores_list.append(refined_scores.cpu().numpy())

    return refined_scores_list


def compute_ranking_metrics(
    gold_ids: List[str],
    ranked_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute ranking metrics."""
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(gold_ids, ranked_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(gold_ids, ranked_ids, k)

    metrics["mrr"] = mrr_at_k(gold_ids, ranked_ids, max(k_values))

    return metrics


def evaluate_ranking(
    graphs: List,
    refined_scores_list: Optional[List[np.ndarray]] = None,
    use_p3: bool = False,
) -> pd.DataFrame:
    """Evaluate ranking with or without P3 refined scores."""
    results = []

    for i, g in enumerate(graphs):
        # Skip if no gold evidence
        gold_mask = g.node_labels.numpy() > 0 if hasattr(g, 'node_labels') else np.zeros(g.x.size(0), dtype=bool)
        if not gold_mask.any():
            continue

        candidate_ids = g.candidate_uids
        gold_ids = [candidate_ids[j] for j in np.where(gold_mask)[0]]

        # Get scores
        if use_p3 and refined_scores_list is not None:
            scores = refined_scores_list[i]
        else:
            scores = g.reranker_scores.numpy()

        # Rank by scores
        sorted_idx = np.argsort(-scores)
        ranked_ids = [candidate_ids[j] for j in sorted_idx]

        # Compute metrics
        metrics = compute_ranking_metrics(gold_ids, ranked_ids)
        metrics["query_id"] = g.query_id
        metrics["post_id"] = g.post_id
        metrics["criterion_id"] = g.criterion_id
        metrics["n_gold"] = len(gold_ids)
        metrics["n_candidates"] = len(candidate_ids)
        metrics["use_p3"] = use_p3

        results.append(metrics)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate with P3 Graph Reranker")
    parser.add_argument(
        "--graph_dir",
        type=str,
        default="data/cache/gnn/20260117_003135",
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
        output_dir = Path(f"outputs/p3_eval/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Load graph dataset
    graph_dir = Path(args.graph_dir)
    metadata_path = graph_dir / "metadata.json"

    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    logger.info(f"Loading {n_folds}-fold graph dataset from {graph_dir}")

    # Results storage
    all_results_no_p3 = []
    all_results_with_p3 = []

    for fold_id in range(n_folds):
        logger.info(f"\n=== Fold {fold_id} ===")

        # Load graphs for this fold
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]
        logger.info(f"  Loaded {len(graphs)} graphs")

        # Load P3 model for this fold
        p3_checkpoint = Path(args.p3_checkpoint_dir) / f"fold_{fold_id}_best.pt"
        if not p3_checkpoint.exists():
            logger.warning(f"  P3 checkpoint not found: {p3_checkpoint}")
            continue

        logger.info(f"  Loading P3 model from {p3_checkpoint}")
        p3_model = load_p3_model(p3_checkpoint, args.device)

        # Run P3 inference
        logger.info("  Running P3 inference...")
        refined_scores = run_p3_inference(p3_model, graphs, args.device)

        # Evaluate without P3
        logger.info("  Evaluating without P3...")
        results_no_p3 = evaluate_ranking(graphs, use_p3=False)
        results_no_p3["fold_id"] = fold_id
        all_results_no_p3.append(results_no_p3)

        # Evaluate with P3
        logger.info("  Evaluating with P3...")
        results_with_p3 = evaluate_ranking(graphs, refined_scores, use_p3=True)
        results_with_p3["fold_id"] = fold_id
        all_results_with_p3.append(results_with_p3)

        # Log fold metrics
        for name, df in [("Without P3", results_no_p3), ("With P3", results_with_p3)]:
            mrr = df["mrr"].mean()
            ndcg10 = df["ndcg@10"].mean()
            recall5 = df["recall@5"].mean()
            logger.info(f"  {name}: MRR={mrr:.4f}, nDCG@10={ndcg10:.4f}, Recall@5={recall5:.4f}")

    # Combine results
    df_no_p3 = pd.concat(all_results_no_p3, ignore_index=True)
    df_with_p3 = pd.concat(all_results_with_p3, ignore_index=True)

    # Save detailed results
    df_no_p3.to_csv(output_dir / "results_no_p3.csv", index=False)
    df_with_p3.to_csv(output_dir / "results_with_p3.csv", index=False)

    # Compute aggregated metrics
    metrics_no_p3 = {
        "mrr": {"mean": df_no_p3["mrr"].mean(), "std": df_no_p3["mrr"].std()},
        "ndcg@1": {"mean": df_no_p3["ndcg@1"].mean(), "std": df_no_p3["ndcg@1"].std()},
        "ndcg@3": {"mean": df_no_p3["ndcg@3"].mean(), "std": df_no_p3["ndcg@3"].std()},
        "ndcg@5": {"mean": df_no_p3["ndcg@5"].mean(), "std": df_no_p3["ndcg@5"].std()},
        "ndcg@10": {"mean": df_no_p3["ndcg@10"].mean(), "std": df_no_p3["ndcg@10"].std()},
        "recall@1": {"mean": df_no_p3["recall@1"].mean(), "std": df_no_p3["recall@1"].std()},
        "recall@3": {"mean": df_no_p3["recall@3"].mean(), "std": df_no_p3["recall@3"].std()},
        "recall@5": {"mean": df_no_p3["recall@5"].mean(), "std": df_no_p3["recall@5"].std()},
        "recall@10": {"mean": df_no_p3["recall@10"].mean(), "std": df_no_p3["recall@10"].std()},
    }

    metrics_with_p3 = {
        "mrr": {"mean": df_with_p3["mrr"].mean(), "std": df_with_p3["mrr"].std()},
        "ndcg@1": {"mean": df_with_p3["ndcg@1"].mean(), "std": df_with_p3["ndcg@1"].std()},
        "ndcg@3": {"mean": df_with_p3["ndcg@3"].mean(), "std": df_with_p3["ndcg@3"].std()},
        "ndcg@5": {"mean": df_with_p3["ndcg@5"].mean(), "std": df_with_p3["ndcg@5"].std()},
        "ndcg@10": {"mean": df_with_p3["ndcg@10"].mean(), "std": df_with_p3["ndcg@10"].std()},
        "recall@1": {"mean": df_with_p3["recall@1"].mean(), "std": df_with_p3["recall@1"].std()},
        "recall@3": {"mean": df_with_p3["recall@3"].mean(), "std": df_with_p3["recall@3"].std()},
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
        "n_queries_evaluated": len(df_no_p3),
        "n_folds": n_folds,
        "without_p3": metrics_no_p3,
        "with_p3": metrics_with_p3,
        "improvements": improvements,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: P3 Graph Reranker Evaluation")
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
        logger.info(f"{metric:<12} {base:>15.4f} {p3:>15.4f} {imp:>+15.4f}")

    logger.info("")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
