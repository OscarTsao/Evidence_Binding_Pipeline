#!/usr/bin/env python3
"""Input Feature Normalization Ablation.

Tests:
1. No normalization (baseline)
2. BatchNorm on input features
3. LayerNorm on input features
4. L2 normalization of embeddings
5. Standardization (mean=0, std=1)

Usage:
    python scripts/experiments/input_normalization_ablation.py
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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

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

    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0

    dcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, n_gold)))
    ndcg10 = dcg / idcg if idcg > 0 else 0.0

    return {"mrr": mrr, "ndcg@10": ndcg10}


class SAGERerankerWithNorm(nn.Module):
    """SAGE reranker with configurable input normalization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
        alpha_init: float = 0.65,
        norm_type: str = "none",  # none, batchnorm, layernorm, l2, standardize
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_type = norm_type

        # Input normalization
        if norm_type == "batchnorm":
            self.input_norm = nn.BatchNorm1d(input_dim)
        elif norm_type == "layernorm":
            self.input_norm = nn.LayerNorm(input_dim)
        else:
            self.input_norm = None

        # Input projection for residual
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # SAGE convolutions
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._alpha_logit = nn.Parameter(torch.tensor(self._logit(alpha_init)))

    @staticmethod
    def _logit(p: float) -> float:
        import math
        p = max(min(p, 0.999), 0.001)
        return math.log(p / (1 - p))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_logit)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        # Apply input normalization
        if self.norm_type == "l2":
            x = F.normalize(x, p=2, dim=-1)
        elif self.norm_type == "standardize":
            # Per-feature standardization
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-8
            x = (x - mean) / std
        elif self.input_norm is not None:
            x = self.input_norm(x)

        # Input projection for residual
        residual = self.input_proj(x)

        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            # Residual connection
            if i == 0:
                h_new = h_new + residual
            else:
                h_new = h_new + h

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
            return torch.tensor(0.0, device=scores.device)
        losses = torch.relu(self.margin - scores[pos_mask].unsqueeze(1) + scores[neg_mask].unsqueeze(0))
        return losses.mean()


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
    norm_type: str,
    device: str = "cuda",
    max_epochs: int = 25,
    patience: int = 8,
) -> Tuple[float, float]:
    """Train model with specific normalization."""
    input_dim = train_graphs[0].x.shape[1]
    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0, 0.0

    model = SAGERerankerWithNorm(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.05,
        alpha_init=0.65,
        norm_type=norm_type,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=3.69e-5, weight_decay=9.06e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = PairwiseMarginLoss(margin=0.1)

    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)

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
    """Run normalization ablation."""
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    norm_types = ["none", "batchnorm", "layernorm", "l2", "standardize"]

    for norm_type in norm_types:
        logger.info(f"Testing norm_type={norm_type}")
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, norm_type, device)
        results.append({"norm_type": norm_type, "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/experiments/input_normalization/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    fold_graphs = load_data(Path(args.graph_dir), EXCLUDED_CRITERIA)
    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    results = run_ablation(fold_graphs, args.device)

    # Sort by nDCG@10
    results.sort(key=lambda x: -x.get("ndcg@10", 0))

    logger.info("\n" + "=" * 60)
    logger.info("INPUT NORMALIZATION ABLATION RESULTS")
    logger.info("=" * 60)
    for i, r in enumerate(results):
        logger.info(f"{i+1}. {r['norm_type']}: nDCG@10={r.get('ndcg@10', 0):.4f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
