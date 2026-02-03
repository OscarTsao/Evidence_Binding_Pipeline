#!/usr/bin/env python3
"""Regularization Techniques Ablation.

Tests:
1. DropEdge (edge dropout during training)
2. Node feature dropout
3. Different weight decay values
4. Different gradient clipping values

Usage:
    python scripts/experiments/regularization_ablation.py
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
from torch_geometric.utils import dropout_edge

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

    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0

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


def train_and_evaluate(
    train_graphs: List[Data],
    val_graphs: List[Data],
    config: Dict,
    device: str = "cuda",
    max_epochs: int = 25,
    patience: int = 8,
) -> Tuple[float, float]:
    """Train model with specific regularization config."""
    input_dim = train_graphs[0].x.shape[1]
    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0, 0.0

    gnn_config = GNNModelConfig(
        gnn_type=GNNType.SAGE,
        hidden_dim=128,
        num_layers=2,
        dropout=config.get("dropout", 0.05),
        layer_norm=False,
        residual=True,
    )

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=config.get("dropout", 0.05),
        alpha_init=0.65,
        learn_alpha=True,
        config=gnn_config,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=3.69e-5,
        weight_decay=config.get("weight_decay", 9.06e-6),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = PairwiseMarginLoss(margin=0.1)

    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)

    best_ndcg = 0.0
    patience_counter = 0

    drop_edge_rate = config.get("drop_edge_rate", 0.0)
    input_dropout_rate = config.get("input_dropout_rate", 0.0)
    grad_clip = config.get("grad_clip", 1.0)

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Apply DropEdge
            edge_index = batch.edge_index
            if drop_edge_rate > 0 and model.training:
                edge_index, _ = dropout_edge(edge_index, p=drop_edge_rate, training=True)

            # Apply input dropout
            x = batch.x
            if input_dropout_rate > 0 and model.training:
                x = F.dropout(x, p=input_dropout_rate, training=True)

            refined = model(x, edge_index, batch.reranker_scores, batch.batch)
            loss = loss_fn(refined, batch.node_labels, batch.batch)
            if loss.requires_grad:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
    """Run regularization ablation."""
    results = []

    val_fold = 0
    train_graphs = [g for fid, graphs in fold_graphs.items() if fid != val_fold for g in graphs]
    val_graphs = fold_graphs[val_fold]

    # Baseline
    logger.info("Testing baseline (no extra regularization)")
    ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, {}, device)
    results.append({"config": "baseline", "ndcg@10": ndcg, "alpha": alpha})
    logger.info(f"  nDCG@10: {ndcg:.4f}")

    # DropEdge rates
    for drop_rate in [0.1, 0.2, 0.3]:
        logger.info(f"Testing DropEdge rate={drop_rate}")
        config = {"drop_edge_rate": drop_rate}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"config": f"drop_edge_{drop_rate}", "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    # Input dropout
    for input_drop in [0.1, 0.2]:
        logger.info(f"Testing input dropout={input_drop}")
        config = {"input_dropout_rate": input_drop}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"config": f"input_dropout_{input_drop}", "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    # Weight decay
    for wd in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]:
        logger.info(f"Testing weight_decay={wd}")
        config = {"weight_decay": wd}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"config": f"weight_decay_{wd}", "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    # Gradient clipping
    for gc in [0.5, 1.0, 2.0, 5.0]:
        logger.info(f"Testing grad_clip={gc}")
        config = {"grad_clip": gc}
        ndcg, alpha = train_and_evaluate(train_graphs, val_graphs, config, device)
        results.append({"config": f"grad_clip_{gc}", "ndcg@10": ndcg, "alpha": alpha})
        logger.info(f"  nDCG@10: {ndcg:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"outputs/experiments/regularization/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    fold_graphs = load_data(Path(args.graph_dir), EXCLUDED_CRITERIA)
    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    results = run_ablation(fold_graphs, args.device)

    # Sort by nDCG@10
    results.sort(key=lambda x: -x.get("ndcg@10", 0))

    logger.info("\n" + "=" * 60)
    logger.info("REGULARIZATION ABLATION RESULTS")
    logger.info("=" * 60)
    for i, r in enumerate(results):
        logger.info(f"{i+1}. {r['config']}: nDCG@10={r.get('ndcg@10', 0):.4f}")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
