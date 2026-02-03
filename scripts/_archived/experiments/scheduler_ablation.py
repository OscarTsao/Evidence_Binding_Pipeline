#!/usr/bin/env python3
"""Ablation study on learning rate scheduler with GELU."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

from final_sc_review.constants import EXCLUDED_CRITERIA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(graph_dir: Path, exclude_criteria: List[str]) -> Dict[int, List]:
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
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0

    dcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, n_gold)))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    return {"mrr": mrr, "ndcg@10": ndcg10}


class SAGERerankerGELU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.05, alpha_init: float = 0.65):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout
        self._alpha_logit = nn.Parameter(torch.tensor(self._logit(alpha_init)))

    @staticmethod
    def _logit(p):
        import math
        p = max(min(p, 0.999), 0.001)
        return math.log(p / (1 - p))

    @property
    def alpha(self):
        return torch.sigmoid(self._alpha_logit)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        residual = self.input_proj(x)
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = self.activation(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h_new = h_new + (residual if i == 0 else h)
            h = h_new
        gnn_scores = self.score_head(h).squeeze(-1)
        return self.alpha * reranker_scores + (1 - self.alpha) * gnn_scores


class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels, batch=None):
        if batch is None:
            return self._single(scores, labels)
        losses = []
        for b in batch.unique():
            mask = batch == b
            loss = self._single(scores[mask], labels[mask])
            if loss.requires_grad:
                losses.append(loss)
        if not losses:
            return (scores * 0).sum()
        return torch.stack(losses).mean()

    def _single(self, scores, labels):
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        losses = torch.relu(self.margin - scores[pos_mask].unsqueeze(1) + scores[neg_mask].unsqueeze(0))
        return losses.mean()


def evaluate(model: nn.Module, graphs: List, device: str) -> Dict[str, float]:
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
    train_graphs: List,
    val_graphs: List,
    scheduler_name: str,
    device: str = "cuda",
    max_epochs: int = 25,
    patience: int = 8,
) -> Tuple[float, float]:
    input_dim = train_graphs[0].x.shape[1]
    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0, 0.0

    model = SAGERerankerGELU(input_dim=input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=3.69e-5, weight_decay=9.06e-6)
    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)
    steps_per_epoch = len(train_loader)

    # Create scheduler based on name
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        step_scheduler_per_epoch = True
    elif scheduler_name == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=3.69e-4, total_steps=max_epochs * steps_per_epoch)
        step_scheduler_per_epoch = False  # Step per batch
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        step_scheduler_per_epoch = True
    elif scheduler_name == "exponential":
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        step_scheduler_per_epoch = True
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        step_scheduler_per_epoch = True
    elif scheduler_name == "none":
        scheduler = None
        step_scheduler_per_epoch = True
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    loss_fn = PairwiseMarginLoss(margin=0.1)

    best_ndcg = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss = loss_fn(refined, batch.node_labels, batch.batch)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if scheduler is not None and not step_scheduler_per_epoch:
                scheduler.step()

        if scheduler is not None and step_scheduler_per_epoch:
            if scheduler_name == "plateau":
                metrics = evaluate(model, val_pos, device)
                scheduler.step(metrics["ndcg@10"])
            else:
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


def run_ablation(fold_graphs: Dict, device: str) -> List[Dict]:
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    schedulers = [
        "cosine",       # Default (current)
        "onecycle",     # OneCycleLR with warmup
        "step",         # StepLR with decay
        "exponential",  # ExponentialLR
        "plateau",      # ReduceLROnPlateau
        "none",         # No scheduler (constant LR)
    ]

    for sched_name in schedulers:
        logger.info(f"Testing scheduler={sched_name}")
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, sched_name, device)
        results.append({"scheduler": sched_name, "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/experiments/scheduler/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    fold_graphs = load_data(Path(args.graph_dir), EXCLUDED_CRITERIA)
    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    results = run_ablation(fold_graphs, args.device)

    results.sort(key=lambda x: -x.get("ndcg@10", 0))

    logger.info("\n" + "=" * 60)
    logger.info("SCHEDULER ABLATION RESULTS")
    logger.info("=" * 60)
    for i, r in enumerate(results):
        logger.info(f"{i+1}. {r['scheduler']}: nDCG@10={r.get('ndcg@10', 0):.4f}")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
