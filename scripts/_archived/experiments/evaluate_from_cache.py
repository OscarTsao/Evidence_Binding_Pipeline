#!/usr/bin/env python3
"""Evaluate baseline and GNN from existing graph cache.

Uses the pre-built graph cache (with NV-Embed-v2 + Jina-v3 scores) for
consistent 5-fold evaluation comparing baseline vs GNN.

Usage:
    python scripts/experiments/evaluate_from_cache.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/unified_evaluation

Author: Evidence Binding Pipeline
Date: 2026-01-21
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss
from final_sc_review.constants import EXCLUDED_CRITERIA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metric Computation (Single Source of Truth)
# ============================================================================

def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute nDCG@K with binary relevance."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    # DCG
    dcg = 0.0
    for i in range(min(k, len(sorted_gold))):
        if sorted_gold[i]:
            dcg += 1.0 / math.log2(i + 2)

    # IDCG - based on gold items in candidate pool
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))

    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Recall@K."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    n_gold = gold_mask.sum()
    if n_gold == 0:
        return 0.0

    return sorted_gold[:k].sum() / n_gold


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) == 0:
        return 0.0

    return 1.0 / (gold_positions[0] + 1)


def compute_precision_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Precision@K."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    return sorted_gold[:k].sum() / k


def compute_hit_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Hit@K (1 if any relevant in top-k, else 0)."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    return 1.0 if sorted_gold[:k].sum() > 0 else 0.0


def compute_map_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Mean Average Precision@K."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx][:k]
    n_gold = gold_mask.sum()
    if n_gold == 0:
        return 0.0
    precisions = []
    hits = 0
    for i, rel in enumerate(sorted_gold):
        if rel:
            hits += 1
            precisions.append(hits / (i + 1))
    if not precisions:
        return 0.0
    return sum(precisions) / min(k, n_gold)


def compute_all_ranking_metrics(
    gold_mask: np.ndarray,
    scores: np.ndarray,
    ks: List[int] = [1, 3, 5, 10, 20],
) -> Dict[str, float]:
    """Compute all ranking metrics for a single query."""
    metrics = {}

    for k in ks:
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(gold_mask, scores, k)
        metrics[f"recall@{k}"] = compute_recall_at_k(gold_mask, scores, k)
        metrics[f"precision@{k}"] = compute_precision_at_k(gold_mask, scores, k)
        metrics[f"hit@{k}"] = compute_hit_at_k(gold_mask, scores, k)
        metrics[f"map@{k}"] = compute_map_at_k(gold_mask, scores, k)

    metrics["mrr"] = compute_mrr(gold_mask, scores)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across queries."""
    if not all_metrics:
        return {}

    agg = {}
    keys = all_metrics[0].keys()

    for key in keys:
        values = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    return agg


# ============================================================================
# Data Loading
# ============================================================================

def load_graph_dataset(
    graph_dir: Path,
    exclude_criteria: Optional[List[str]] = None,
) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        # Filter out excluded criteria
        if exclude_criteria:
            original_count = len(graphs)
            graphs = [g for g in graphs if getattr(g, 'criterion_id', None) not in exclude_criteria]
            logger.info(f"Loaded fold {fold_id}: {original_count} -> {len(graphs)} graphs (excluded {exclude_criteria})")
        else:
            logger.info(f"Loaded fold {fold_id}: {len(graphs)} graphs")

        fold_graphs[fold_id] = graphs

    return fold_graphs, metadata


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_baseline(graphs: List) -> Tuple[Dict, List[Dict]]:
    """Evaluate baseline (reranker scores only)."""
    all_metrics = []

    for g in graphs:
        gold_mask = g.node_labels.cpu().numpy()

        if not gold_mask.any():
            continue

        scores = g.reranker_scores.cpu().numpy()
        metrics = compute_all_ranking_metrics(gold_mask, scores)
        all_metrics.append(metrics)

    return aggregate_metrics(all_metrics), all_metrics


def evaluate_gnn(
    model: nn.Module,
    graphs: List,
    device: str = "cuda",
) -> Tuple[Dict, List[Dict]]:
    """Evaluate GNN (refined scores)."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for g in graphs:
            # Ensure graph is on CPU first, then move to device
            g = ensure_cpu(g)
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy()

            if not gold_mask.any():
                continue

            refined = model(g.x, g.edge_index, g.reranker_scores)
            scores = refined.cpu().numpy()

            metrics = compute_all_ranking_metrics(gold_mask, scores)
            all_metrics.append(metrics)

    return aggregate_metrics(all_metrics), all_metrics


# ============================================================================
# GNN Training
# ============================================================================

def ensure_cpu(graph):
    """Ensure all tensors in graph are on CPU."""
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph


def train_gnn_fold(
    train_graphs: List,
    val_graphs: List,
    config: Dict,
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    """Train GNN on a single fold."""
    # Ensure all graphs are on CPU before DataLoader (it will handle batching)
    train_graphs = [ensure_cpu(g) for g in train_graphs]
    val_graphs = [ensure_cpu(g) for g in val_graphs]

    input_dim = train_graphs[0].x.shape[1]

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        alpha_init=config["alpha_init"],
        learn_alpha=config["learn_alpha"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    loss_fn = GraphRerankerLoss(
        alpha_rank=config["alpha_rank"],
        alpha_align=config["alpha_align"],
        alpha_reg=config["alpha_reg"],
        margin=config["margin"],
    )

    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)

    best_val_ndcg = -1
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(config["max_epochs"]):
        # Training
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss, _ = loss_fn(refined, batch.reranker_scores, batch.node_labels, batch.batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_agg, _ = evaluate_gnn(model, val_graphs, device)
        val_ndcg = val_agg.get("ndcg@10", {}).get("mean", 0)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    return model, {"best_val_ndcg": best_val_ndcg, "best_epoch": best_epoch}


# ============================================================================
# Main Evaluation
# ============================================================================

def run_5fold_evaluation(
    graph_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    exclude_a10: bool = True,
    hpo_config_override: Optional[Dict] = None,
) -> Dict:
    """Run 5-fold CV evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load graphs
    exclude_criteria = EXCLUDED_CRITERIA if exclude_a10 else None
    fold_graphs, metadata = load_graph_dataset(graph_dir, exclude_criteria)
    n_folds = metadata["n_folds"]

    # GNN config (default values)
    gnn_config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "max_epochs": 30,
        "patience": 10,
        "alpha_init": 0.7,
        "learn_alpha": True,
        "alpha_rank": 1.0,
        "alpha_align": 0.5,
        "alpha_reg": 0.1,
        "margin": 0.1,
    }

    # Override with HPO-optimized params if provided
    if hpo_config_override:
        for key, value in hpo_config_override.items():
            if key in gnn_config:
                gnn_config[key] = value
            elif key == "n_epochs":
                gnn_config["max_epochs"] = value  # Map n_epochs to max_epochs
        logger.info(f"Using HPO-optimized config: {gnn_config}")

    fold_results = []

    for fold_id in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_id}")
        logger.info(f"{'='*60}")

        # Get train and val graphs
        train_graphs = []
        for fid in range(n_folds):
            if fid != fold_id:
                train_graphs.extend(fold_graphs[fid])
        val_graphs = fold_graphs[fold_id]

        # Filter to only queries with evidence (positives_only protocol)
        train_graphs_pos = [g for g in train_graphs if g.node_labels.sum() > 0]
        val_graphs_pos = [g for g in val_graphs if g.node_labels.sum() > 0]

        logger.info(f"Train: {len(train_graphs_pos)} graphs with evidence")
        logger.info(f"Val: {len(val_graphs_pos)} graphs with evidence")

        # Evaluate baseline
        logger.info("Evaluating baseline (Jina-v3 only)...")
        baseline_agg, baseline_per_query = evaluate_baseline(val_graphs_pos)

        # Train and evaluate GNN
        logger.info("Training GNN...")
        model, train_info = train_gnn_fold(
            train_graphs=train_graphs_pos,
            val_graphs=val_graphs_pos,
            config=gnn_config,
            device=device,
        )

        logger.info("Evaluating GNN...")
        gnn_agg, gnn_per_query = evaluate_gnn(model, val_graphs_pos, device)

        # Compute improvement
        improvement = {}
        for key in baseline_agg:
            if key in gnn_agg:
                base_val = baseline_agg[key]["mean"]
                gnn_val = gnn_agg[key]["mean"]
                improvement[key] = {
                    "absolute": gnn_val - base_val,
                    "relative_pct": ((gnn_val - base_val) / base_val * 100) if base_val > 0 else 0,
                }

        # Get final alpha
        final_alpha = float(model.alpha.item())

        fold_result = {
            "fold_id": fold_id,
            "n_train": len(train_graphs_pos),
            "n_val": len(val_graphs_pos),
            "best_epoch": train_info["best_epoch"],
            "final_alpha": final_alpha,
            "baseline": baseline_agg,
            "gnn": gnn_agg,
            "improvement": improvement,
        }
        fold_results.append(fold_result)

        # Log key metrics
        logger.info(f"Baseline nDCG@10: {baseline_agg['ndcg@10']['mean']:.4f} ± {baseline_agg['ndcg@10']['std']:.4f}")
        logger.info(f"GNN nDCG@10:      {gnn_agg['ndcg@10']['mean']:.4f} ± {gnn_agg['ndcg@10']['std']:.4f}")
        logger.info(f"Improvement:      {improvement['ndcg@10']['absolute']:+.4f} ({improvement['ndcg@10']['relative_pct']:+.2f}%)")
        logger.info(f"Final alpha:      {final_alpha:.4f}")

    # Aggregate across folds
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATED RESULTS (5-Fold CV)")
    logger.info(f"{'='*60}")

    aggregated = {
        "baseline": {},
        "gnn": {},
        "improvement": {},
    }

    metric_keys = fold_results[0]["baseline"].keys()

    for key in metric_keys:
        baseline_means = [fr["baseline"][key]["mean"] for fr in fold_results]
        gnn_means = [fr["gnn"][key]["mean"] for fr in fold_results]

        aggregated["baseline"][key] = f"{np.mean(baseline_means):.4f} ± {np.std(baseline_means):.4f}"
        aggregated["gnn"][key] = f"{np.mean(gnn_means):.4f} ± {np.std(gnn_means):.4f}"

        delta = np.mean(gnn_means) - np.mean(baseline_means)
        delta_std = np.std([g - b for g, b in zip(gnn_means, baseline_means)])
        aggregated["improvement"][key] = f"{delta:+.4f} ± {delta_std:.4f}"

    # Print summary table
    print("\n" + "="*80)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    print(f"\nConfig: top_k_final=10 (graphs built with top_k_rerank=20, evaluated at k=10)")
    print(f"Excluded A.10: {exclude_a10}")
    print(f"\n{'Metric':<15} {'Baseline':<25} {'GNN':<25} {'Δ':<20}")
    print("-"*80)

    for key in ["ndcg@10", "ndcg@5", "ndcg@3", "ndcg@1", "mrr", "recall@10", "recall@5"]:
        if key in aggregated["baseline"]:
            print(f"{key:<15} {aggregated['baseline'][key]:<25} {aggregated['gnn'][key]:<25} {aggregated['improvement'][key]:<20}")

    # Build results dict
    results = {
        "config": {
            "graph_dir": str(graph_dir),
            "exclude_a10": exclude_a10,
            "gnn_config": gnn_config,
            "n_folds": n_folds,
            "evaluation_protocol": "positives_only",
        },
        "fold_results": fold_results,
        "aggregated": aggregated,
        "timestamp": timestamp,
    }

    # Save results
    results_path = output_dir / f"5fold_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logger.info(f"\nSaved results to {results_path}")

    # Save as latest
    latest_path = output_dir / "latest_5fold_results.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate from Graph Cache")
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default="outputs/unified_evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include_a10", action="store_true")
    parser.add_argument("--hpo_params", type=str, help="Path to HPO best_params.json to use optimized config")

    args = parser.parse_args()

    # Load HPO params if provided
    hpo_config_override = None
    if args.hpo_params:
        with open(args.hpo_params) as f:
            hpo_data = json.load(f)
            hpo_config_override = hpo_data.get("best_params", {})
            logger.info(f"Loaded HPO params from {args.hpo_params}: {hpo_config_override}")

    results = run_5fold_evaluation(
        graph_dir=Path(args.graph_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        exclude_a10=not args.include_a10,
        hpo_config_override=hpo_config_override,
    )


if __name__ == "__main__":
    main()
