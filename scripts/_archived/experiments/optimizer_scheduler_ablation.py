#!/usr/bin/env python3
"""Optimizer and Learning Rate Scheduler Ablation.

Tests:
1. Optimizers: Adam, AdamW, SGD with momentum
2. Schedulers: CosineAnnealing, StepLR, ReduceLROnPlateau, Linear warmup

Usage:
    python scripts/experiments/optimizer_scheduler_ablation.py
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, LambdaLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN
from final_sc_review.gnn.config import GNNType, GNNModelConfig
from final_sc_review.constants import EXCLUDED_CRITERIA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(graph_dir: Path, exclude_criteria: List[str]) -> Dict[int, List]:
    """Load graphs from cache."""
    with open(graph_dir / "metadata.json") as f:
        metadata = json.load(f)

    fold_graphs = {}
    for fold_id in range(metadata["n_folds"]):
        data = torch.load(graph_dir / f"fold_{fold_id}.pt", weights_only=False)
        graphs = [g for g in data["graphs"]
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


class PairwiseMarginLoss(nn.Module):
    """Simple pairwise margin loss."""
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels, batch=None):
        if batch is None:
            return self._single(scores, labels)
        total, n = 0.0, 0
        for b in batch.unique():
            mask = batch == b
            loss = self._single(scores[mask], labels[mask])
            if loss > 0:
                total += loss
                n += 1
        return total / max(n, 1)

    def _single(self, scores, labels):
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device)
        losses = torch.relu(self.margin - scores[pos_mask].unsqueeze(1) + scores[neg_mask].unsqueeze(0))
        return losses.mean()


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def train_and_evaluate(
    train_graphs: List[Data],
    val_graphs: List[Data],
    optimizer_name: str,
    scheduler_name: str,
    device: str = "cuda",
    max_epochs: int = 25,
    patience: int = 8,
) -> Tuple[float, float]:
    """Train model with specific optimizer and scheduler."""
    input_dim = train_graphs[0].x.shape[1]
    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0, 0.0

    # Best configuration from previous experiments
    gnn_config = GNNModelConfig(
        gnn_type=GNNType.SAGE,
        hidden_dim=128,
        num_layers=2,
        dropout=0.05,
        layer_norm=False,
        residual=True,
    )

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.05,
        alpha_init=0.65,
        learn_alpha=True,
        config=gnn_config,
    ).to(device)

    # Select optimizer
    lr = 3.69e-5
    weight_decay = 9.06e-6

    if optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=lr * 10, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Select scheduler
    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)
    total_steps = max_epochs * len(train_loader)

    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    elif scheduler_name == "warmup":
        warmup_steps = len(train_loader) * 2  # 2 epochs warmup
        scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    elif scheduler_name == "none":
        scheduler = None
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    loss_fn = PairwiseMarginLoss(margin=0.1)

    best_ndcg = 0.0
    patience_counter = 0
    step = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss = loss_fn(refined, batch.node_labels, batch.batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Step-level scheduler (for warmup)
            if scheduler_name == "warmup":
                scheduler.step()
            step += 1

        # Epoch-level scheduler
        if scheduler_name in ["cosine", "step"]:
            scheduler.step()

        metrics = evaluate(model, val_pos, device)

        # Plateau scheduler needs metric
        if scheduler_name == "plateau":
            scheduler.step(metrics["ndcg@10"])

        if metrics["ndcg@10"] > best_ndcg:
            best_ndcg = metrics["ndcg@10"]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_ndcg, model.alpha.item()


def run_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    """Run optimizer and scheduler ablation."""
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    # Test combinations
    optimizers = ["adamw", "adam", "sgd"]
    schedulers = ["cosine", "step", "plateau", "warmup", "none"]

    for opt in optimizers:
        for sched in schedulers:
            name = f"{opt}_{sched}"
            logger.info(f"Testing {name}")
            try:
                ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, opt, sched, device)
                results.append({
                    "optimizer": opt,
                    "scheduler": sched,
                    "ndcg@10": ndcg,
                    "alpha": alpha
                })
                logger.info(f"  nDCG@10: {ndcg:.4f}")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                results.append({
                    "optimizer": opt,
                    "scheduler": sched,
                    "ndcg@10": 0.0,
                    "alpha": 0.0,
                    "error": str(e)
                })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/experiments/optimizer_scheduler/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    fold_graphs = load_data(Path(args.graph_dir), EXCLUDED_CRITERIA)
    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    results = run_ablation(fold_graphs, args.device)

    # Sort by nDCG@10
    results.sort(key=lambda x: -x.get("ndcg@10", 0))

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZER/SCHEDULER ABLATION RESULTS")
    logger.info("=" * 60)
    for i, r in enumerate(results):
        logger.info(f"{i+1}. {r['optimizer']}+{r['scheduler']}: nDCG@10={r.get('ndcg@10', 0):.4f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
