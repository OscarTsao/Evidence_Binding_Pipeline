#!/usr/bin/env python3
"""Hyperparameter Ablation Experiments.

Systematic testing of training hyperparameters:
- Loss function margins
- Loss weights (alpha_rank, alpha_align, alpha_reg)
- Learning rates
- Alpha initialization
- Batch sizes

Usage:
    python scripts/experiments/hyperparameter_ablation.py --experiment margin
    python scripts/experiments/hyperparameter_ablation.py --experiment lr
    python scripts/experiments/hyperparameter_ablation.py --experiment alpha
    python scripts/experiments/hyperparameter_ablation.py --experiment batch
    python scripts/experiments/hyperparameter_ablation.py --experiment all
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def load_data(graph_dir: Path, exclude_criteria: List[str]) -> Dict[int, List]:
    """Load graphs from cache."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    fold_graphs = {}
    for fold_id in range(metadata["n_folds"]):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        if exclude_criteria:
            graphs = [g for g in graphs
                     if getattr(g, 'criterion_id', None) not in exclude_criteria]
        fold_graphs[fold_id] = graphs

    return fold_graphs


def compute_metrics(gold_mask: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute ranking metrics."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    # MRR
    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0

    # nDCG@10
    dcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, n_gold)))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    return {"mrr": mrr, "ndcg@10": ndcg10}


def evaluate(model: nn.Module, graphs: List[Data], device: str) -> Dict[str, float]:
    """Evaluate model on graphs."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy() > 0.5
            if not gold_mask.any():
                continue

            refined = model(g.x, g.edge_index, g.reranker_scores)
            metrics = compute_metrics(gold_mask, refined.cpu().numpy())
            all_metrics.append(metrics)

    if not all_metrics:
        return {"mrr": 0.0, "ndcg@10": 0.0}

    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def train_and_evaluate(
    train_graphs: List[Data],
    val_graphs: List[Data],
    config: Dict,
    device: str = "cuda",
    max_epochs: int = 25,
    patience: int = 8,
) -> Tuple[float, float]:
    """Train model and return best nDCG@10 and final alpha."""
    input_dim = train_graphs[0].x.shape[1]
    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0, 0.0

    gnn_config = GNNModelConfig(
        gnn_type=GNNType.SAGE,
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.05),
        layer_norm=False,
        residual=True,
    )

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.05),
        alpha_init=config.get("alpha_init", 0.65),
        learn_alpha=config.get("learn_alpha", True),
        config=gnn_config,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.get("lr", 3.69e-5),
        weight_decay=config.get("weight_decay", 9.06e-6),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    loss_fn = GraphRerankerLoss(
        alpha_rank=config.get("alpha_rank", 1.0),
        alpha_align=config.get("alpha_align", 0.5),
        alpha_reg=config.get("alpha_reg", 0.1),
        margin=config.get("margin", 0.1),
    )

    train_loader = DataLoader(train_pos, batch_size=config.get("batch_size", 32), shuffle=True)

    best_ndcg = 0.0
    patience_counter = 0

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
        scheduler.step()

        metrics = evaluate(model, val_pos, device)
        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_ndcg, model.alpha.item()


def run_margin_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    """Test different margin values."""
    margins = [0.05, 0.1, 0.15, 0.2, 0.25]
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    base_config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "lr": 3.69e-5, "weight_decay": 9.06e-6, "alpha_init": 0.65,
        "batch_size": 32, "alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.1,
    }

    for margin in margins:
        logger.info(f"Testing margin={margin}")
        config = {**base_config, "margin": margin}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"margin": margin, "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def run_loss_weight_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    """Test different loss weight combinations."""
    configs = [
        {"alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.1},  # baseline
        {"alpha_rank": 1.0, "alpha_align": 0.3, "alpha_reg": 0.1},
        {"alpha_rank": 1.0, "alpha_align": 0.7, "alpha_reg": 0.1},
        {"alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.05},
        {"alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.2},
        {"alpha_rank": 1.5, "alpha_align": 0.5, "alpha_reg": 0.1},
        {"alpha_rank": 0.5, "alpha_align": 0.5, "alpha_reg": 0.1},
    ]
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    base_config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "lr": 3.69e-5, "weight_decay": 9.06e-6, "alpha_init": 0.65,
        "batch_size": 32, "margin": 0.1,
    }

    for loss_config in configs:
        name = f"rank{loss_config['alpha_rank']}_align{loss_config['alpha_align']}_reg{loss_config['alpha_reg']}"
        logger.info(f"Testing {name}")
        config = {**base_config, **loss_config}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({**loss_config, "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def run_lr_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    """Test different learning rates."""
    lrs = [1e-5, 2e-5, 3.69e-5, 5e-5, 1e-4, 2e-4]
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    base_config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "weight_decay": 9.06e-6, "alpha_init": 0.65,
        "batch_size": 32, "margin": 0.1,
        "alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.1,
    }

    for lr in lrs:
        logger.info(f"Testing lr={lr}")
        config = {**base_config, "lr": lr}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"lr": lr, "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def run_alpha_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    """Test different alpha_init values."""
    alphas = [0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9]
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    base_config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "lr": 3.69e-5, "weight_decay": 9.06e-6,
        "batch_size": 32, "margin": 0.1,
        "alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.1,
    }

    for alpha_init in alphas:
        logger.info(f"Testing alpha_init={alpha_init}")
        config = {**base_config, "alpha_init": alpha_init}
        ndcg, final_alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"alpha_init": alpha_init, "ndcg@10": ndcg, "final_alpha": final_alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}, final_alpha: {final_alpha:.3f}")

    return results


def run_batch_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    """Test different batch sizes."""
    batch_sizes = [8, 16, 32, 64, 128]
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    base_config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "lr": 3.69e-5, "weight_decay": 9.06e-6, "alpha_init": 0.65,
        "margin": 0.1, "alpha_rank": 1.0, "alpha_align": 0.5, "alpha_reg": 0.1,
    }

    for bs in batch_sizes:
        logger.info(f"Testing batch_size={bs}")
        config = {**base_config, "batch_size": bs}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"batch_size": bs, "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/optimized_20260130")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["margin", "loss_weights", "lr", "alpha", "batch", "all"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/experiments/hyperparameter/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    fold_graphs = load_data(Path(args.graph_dir), EXCLUDED_CRITERIA)
    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    all_results = {}

    experiments = [args.experiment] if args.experiment != "all" else ["margin", "loss_weights", "lr", "alpha", "batch"]

    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {exp} ablation")
        logger.info(f"{'='*60}")

        if exp == "margin":
            results = run_margin_ablation(fold_graphs, args.device)
        elif exp == "loss_weights":
            results = run_loss_weight_ablation(fold_graphs, args.device)
        elif exp == "lr":
            results = run_lr_ablation(fold_graphs, args.device)
        elif exp == "alpha":
            results = run_alpha_ablation(fold_graphs, args.device)
        elif exp == "batch":
            results = run_batch_ablation(fold_graphs, args.device)
        else:
            continue

        all_results[exp] = results

        # Sort and print
        results.sort(key=lambda x: -x["ndcg@10"])
        logger.info(f"\n{exp.upper()} RESULTS (sorted by nDCG@10):")
        for i, r in enumerate(results):
            logger.info(f"  {i+1}. {r}")

        # Save
        pd.DataFrame(results).to_csv(output_dir / f"{exp}_ablation.csv", index=False)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
