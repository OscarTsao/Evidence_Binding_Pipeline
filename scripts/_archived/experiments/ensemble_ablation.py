#!/usr/bin/env python3
"""Ensemble Methods Ablation Experiments.

Tests different ensemble strategies:
1. Multi-seed ensemble (same architecture, different seeds)
2. Multi-architecture ensemble (SAGE + GCN)
3. Score fusion methods (mean, weighted, learned)

Usage:
    python scripts/experiments/ensemble_ablation.py
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss
from final_sc_review.gnn.config import GNNType, GNNModelConfig
from final_sc_review.constants import EXCLUDED_CRITERIA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(
    graph_dir: Path,
    exclude_criteria: List[str],
) -> Tuple[Dict[int, List], Dict]:
    """Load graphs."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        if exclude_criteria:
            graphs = [g for g in graphs
                     if getattr(g, 'criterion_id', None) not in exclude_criteria]

        fold_graphs[fold_id] = graphs

    return fold_graphs, metadata


def compute_ranking_metrics(gold_mask: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute ranking metrics."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    metrics = {}

    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0
    metrics["mrr"] = mrr

    for k in [5, 10]:
        dcg = 0.0
        for i in range(min(k, len(sorted_gold))):
            if sorted_gold[i]:
                dcg += 1.0 / np.log2(i + 2)
        n_gold = int(gold_mask.sum())
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_gold)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def train_single_model(
    train_graphs: List[Data],
    gnn_type: str,
    seed: int,
    config: Dict,
    device: str = "cuda",
    max_epochs: int = 15,
) -> nn.Module:
    """Train a single model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_graphs[0].x.shape[1]
    train_pos = [g for g in train_graphs if g.y.item() > 0]

    if len(train_pos) < 5:
        return None

    gnn_config = GNNModelConfig(
        gnn_type=GNNType(gnn_type),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        layer_norm=False,
        residual=True,
    )
    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        alpha_init=config["alpha_init"],
        learn_alpha=True,
        config=gnn_config,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = GraphRerankerLoss(margin=config["margin"])

    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss, _ = loss_fn(refined, batch.reranker_scores, batch.node_labels, batch.batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model


def get_ensemble_predictions(
    models: List[nn.Module],
    graph: Data,
    device: str,
    fusion: str = "mean",
) -> np.ndarray:
    """Get ensemble predictions using specified fusion method."""
    all_preds = []

    for model in models:
        model.eval()
        g = graph.to(device)
        with torch.no_grad():
            pred = model(g.x, g.edge_index, g.reranker_scores)
            all_preds.append(pred.cpu().numpy())

    all_preds = np.array(all_preds)  # [n_models, n_nodes]

    if fusion == "mean":
        return np.mean(all_preds, axis=0)
    elif fusion == "max":
        return np.max(all_preds, axis=0)
    elif fusion == "weighted":
        # Weight by inverse variance
        var = np.var(all_preds, axis=0) + 1e-8
        weights = 1.0 / var
        weights = weights / weights.sum()
        return np.sum(all_preds * weights[None, :], axis=0)
    else:
        return np.mean(all_preds, axis=0)


def evaluate_ensemble(
    models: List[nn.Module],
    graphs: List[Data],
    device: str,
    fusion: str = "mean",
) -> Dict[str, float]:
    """Evaluate ensemble on graphs."""
    all_metrics = []

    for g in graphs:
        gold_mask = g.node_labels.cpu().numpy() > 0.5

        if not gold_mask.any():
            continue

        scores = get_ensemble_predictions(models, g, device, fusion)
        metrics = compute_ranking_metrics(gold_mask, scores)
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[key] = np.mean(values)

    return agg


def run_ablation(
    fold_graphs: Dict[int, List[Data]],
    output_dir: Path,
    device: str = "cuda",
):
    """Run ensemble ablation experiments."""
    train_config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 0.0001,
        "weight_decay": 1e-5,
        "alpha_init": 0.7,
        "margin": 0.1,
    }

    # Use fold 0 for validation
    val_fold = 0
    train_graphs = []
    for fid, graphs in fold_graphs.items():
        if fid != val_fold:
            train_graphs.extend(graphs)
    val_graphs = [g for g in fold_graphs[val_fold] if g.y.item() > 0]

    results = []

    # Experiment 1: Single model baseline
    logger.info("\n=== Experiment 1: Single Model Baselines ===")

    for gnn_type in ["sage", "gcn"]:
        logger.info(f"Training {gnn_type} single model...")
        model = train_single_model(train_graphs, gnn_type, seed=42, config=train_config, device=device)

        if model:
            metrics = evaluate_ensemble([model], val_graphs, device, "mean")
            results.append({
                "method": f"single_{gnn_type}",
                "n_models": 1,
                "fusion": "none",
                "ndcg@10": metrics.get("ndcg@10", 0.0),
                "mrr": metrics.get("mrr", 0.0),
            })
            logger.info(f"  {gnn_type}: nDCG@10={metrics.get('ndcg@10', 0.0):.4f}")

    # Experiment 2: Multi-seed ensemble (SAGE)
    logger.info("\n=== Experiment 2: Multi-Seed Ensemble (SAGE) ===")

    seeds = [42, 123, 456, 789, 1000]
    sage_models = []

    for seed in seeds:
        logger.info(f"Training SAGE with seed {seed}...")
        model = train_single_model(train_graphs, "sage", seed=seed, config=train_config, device=device)
        if model:
            sage_models.append(model)

    # Test different ensemble sizes
    for n_models in [2, 3, 5]:
        if len(sage_models) >= n_models:
            for fusion in ["mean", "max"]:
                metrics = evaluate_ensemble(sage_models[:n_models], val_graphs, device, fusion)
                results.append({
                    "method": f"multi_seed_sage_{n_models}",
                    "n_models": n_models,
                    "fusion": fusion,
                    "ndcg@10": metrics.get("ndcg@10", 0.0),
                    "mrr": metrics.get("mrr", 0.0),
                })
                logger.info(f"  {n_models} models ({fusion}): nDCG@10={metrics.get('ndcg@10', 0.0):.4f}")

    # Experiment 3: Multi-architecture ensemble
    logger.info("\n=== Experiment 3: Multi-Architecture Ensemble ===")

    gcn_model = train_single_model(train_graphs, "gcn", seed=42, config=train_config, device=device)

    if sage_models and gcn_model:
        arch_models = [sage_models[0], gcn_model]

        for fusion in ["mean", "max"]:
            metrics = evaluate_ensemble(arch_models, val_graphs, device, fusion)
            results.append({
                "method": "multi_arch_sage_gcn",
                "n_models": 2,
                "fusion": fusion,
                "ndcg@10": metrics.get("ndcg@10", 0.0),
                "mrr": metrics.get("mrr", 0.0),
            })
            logger.info(f"  SAGE+GCN ({fusion}): nDCG@10={metrics.get('ndcg@10', 0.0):.4f}")

    # Sort by nDCG@10
    results.sort(key=lambda x: -x["ndcg@10"])

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ensemble_ablation.csv", index=False)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ENSEMBLE ABLATION RESULTS")
    logger.info("=" * 70)

    for i, r in enumerate(results):
        logger.info(
            f"{i+1}. {r['method']} ({r['fusion']}): nDCG@10={r['ndcg@10']:.4f}, "
            f"n_models={r['n_models']}"
        )

    best = results[0]
    logger.info(f"\nBest: {best['method']} with {best['fusion']} fusion")
    logger.info(f"  nDCG@10: {best['ndcg@10']:.4f}")

    # Compare to single model
    single_sage = next((r for r in results if r["method"] == "single_sage"), None)
    if single_sage:
        improvement = (best["ndcg@10"] - single_sage["ndcg@10"]) / single_sage["ndcg@10"] * 100
        logger.info(f"  Improvement over single SAGE: {improvement:+.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"outputs/experiments/ensemble/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    graph_dir = Path(args.graph_dir)
    fold_graphs, _ = load_data(graph_dir, exclude_criteria=EXCLUDED_CRITERIA)

    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    results = run_ablation(fold_graphs, output_dir, args.device)

    summary = {
        "timestamp": timestamp,
        "n_experiments": len(results),
        "best": results[0] if results else None,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
