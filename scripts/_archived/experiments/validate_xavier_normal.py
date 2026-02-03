#!/usr/bin/env python3
"""Validate Xavier normal vs default initialization with full 5-fold CV."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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

    metrics = {"mrr": mrr}
    for k in [5, 10]:
        dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(sorted_gold))) if sorted_gold[i])
        n_gold = int(gold_mask.sum())
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_gold)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def init_weights(module, init_type: str):
    if isinstance(module, nn.Linear):
        if init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SAGERerankerGELU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.05, alpha_init: float = 0.65, use_xavier: bool = False):
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

        if use_xavier:
            self.apply(lambda m: init_weights(m, "xavier_normal"))

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
        pos_mask, neg_mask = labels > 0.5, labels < 0.5
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        losses = torch.relu(self.margin - scores[pos_mask].unsqueeze(1) + scores[neg_mask].unsqueeze(0))
        return losses.mean()


def evaluate(model, graphs, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy() > 0.5
            if not gold_mask.any():
                continue
            refined = model(g.x, g.edge_index, g.reranker_scores)
            all_metrics.append(compute_metrics(gold_mask, refined.cpu().numpy()))
    if not all_metrics:
        return {}
    return {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}


def ensure_cpu(g):
    for key in g.keys():
        if isinstance(g[key], torch.Tensor):
            g[key] = g[key].cpu()
    return g


def train_fold(train_graphs, val_graphs, device, use_xavier=False, max_epochs=25, patience=10):
    input_dim = train_graphs[0].x.shape[1]
    train_pos = [ensure_cpu(g) for g in train_graphs if g.y.item() > 0]
    val_pos = [ensure_cpu(g) for g in val_graphs if g.y.item() > 0]

    model = SAGERerankerGELU(input_dim, use_xavier=use_xavier).to(device)
    optimizer = AdamW(model.parameters(), lr=3.69e-5, weight_decay=9.06e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = PairwiseMarginLoss(margin=0.1)
    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)

    best_ndcg, patience_counter = 0.0, 0
    best_state = None

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_ndcg


def main():
    graph_dir = Path("data/cache/gnn/rebuild_20260120")
    device = "cuda"

    fold_graphs = load_data(graph_dir, EXCLUDED_CRITERIA)
    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")

    # Test both configurations with full 5-fold CV
    for config_name, use_xavier in [("default", False), ("xavier_normal", True)]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {config_name.upper()} (5-Fold CV)")
        logger.info("="*60)

        results = []
        for fold_id in range(5):
            logger.info(f"\n=== Fold {fold_id} ===")
            train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
            val_graphs = fold_graphs[fold_id]
            val_pos = [ensure_cpu(g) for g in val_graphs if g.y.item() > 0]

            model, _ = train_fold(train_graphs, val_graphs, device, use_xavier=use_xavier)

            # Get baseline and refined metrics
            baseline_metrics = []
            refined_metrics = []
            model.eval()
            with torch.no_grad():
                for g in val_pos:
                    g = g.to(device)
                    gold = g.node_labels.cpu().numpy() > 0.5
                    if not gold.any():
                        continue
                    baseline_metrics.append(compute_metrics(gold, g.reranker_scores.cpu().numpy()))
                    refined_metrics.append(compute_metrics(gold, model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()))

            orig_ndcg = np.mean([m["ndcg@10"] for m in baseline_metrics])
            ref_ndcg = np.mean([m["ndcg@10"] for m in refined_metrics])
            ref_mrr = np.mean([m["mrr"] for m in refined_metrics])

            results.append({"fold": fold_id, "orig_ndcg": orig_ndcg, "ndcg@10": ref_ndcg, "mrr": ref_mrr})
            logger.info(f"  Original nDCG@10: {orig_ndcg:.4f}")
            logger.info(f"  Refined nDCG@10: {ref_ndcg:.4f}")
            logger.info(f"  Improvement: +{ref_ndcg - orig_ndcg:.4f}")

        # Summary
        mean_ndcg = np.mean([r["ndcg@10"] for r in results])
        std_ndcg = np.std([r["ndcg@10"] for r in results])
        mean_mrr = np.mean([r["mrr"] for r in results])
        std_mrr = np.std([r["mrr"] for r in results])

        logger.info(f"\n{config_name.upper()} (5-Fold CV)")
        logger.info(f"nDCG@10: {mean_ndcg:.4f} ± {std_ndcg:.4f}")
        logger.info(f"MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")

        # Save
        output_dir = Path(f"outputs/experiments/init_validation/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "config": config_name,
                "use_xavier": use_xavier,
                "folds": results,
                "mean_ndcg": mean_ndcg,
                "std_ndcg": std_ndcg,
                "mean_mrr": mean_mrr
            }, f, indent=2)
        logger.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
