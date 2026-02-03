#!/usr/bin/env python3
"""Test SAGE + Residual + Dynamic-K combination."""

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.constants import EXCLUDED_CRITERIA


def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    sorted_idx = np.argsort(-scores)
    n_gold = gold_mask.sum()
    if n_gold == 0:
        return 0.0
    return gold_mask[sorted_idx[:k]].sum() / n_gold


class SAGEResidualReranker(nn.Module):
    """GraphSAGE reranker with residual connections."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

        # Residual projection if dimensions differ
        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None

    def forward(self, x, edge_index, reranker_scores, batch=None):
        # Save input for residual
        h_in = x
        if self.residual_proj is not None:
            h_in = self.residual_proj(h_in)

        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)

        # Add residual
        h = h + h_in

        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class PairwiseMarginLoss(nn.Module):
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
        pos_mask, neg_mask = labels > 0, labels == 0
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device)
        losses = F.relu(self.margin - scores[pos_mask].unsqueeze(1) + scores[neg_mask].unsqueeze(0))
        return losses.mean()


def select_k_threshold(probs: np.ndarray, tau: float, k_min: int = 2, k_max: int = 10) -> int:
    """DK-A: Select K based on probability threshold."""
    k = np.sum(probs >= tau)
    return max(k_min, min(k, k_max))


def load_graphs(graph_dir: Path):
    """Load graph dataset."""
    with open(graph_dir / "metadata.json") as f:
        metadata = json.load(f)

    fold_graphs = {}
    for fold_id in range(metadata["n_folds"]):
        data = torch.load(graph_dir / f"fold_{fold_id}.pt", weights_only=False)
        graphs = [g for g in data["graphs"]
                  if getattr(g, 'criterion_id', None) not in EXCLUDED_CRITERIA]
        graphs = [g for g in graphs if g.node_labels.sum() > 0]
        fold_graphs[fold_id] = graphs
        print(f"Loaded fold {fold_id}: {len(graphs)} graphs")

    return fold_graphs


def ensure_cpu(g):
    for key in g.keys():
        if isinstance(g[key], torch.Tensor):
            g[key] = g[key].cpu()
    return g


def train_model(train_graphs, val_graphs, device, seed=42):
    """Train SAGE + Residual model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "alpha_init": 0.65, "learn_alpha": True, "lr": 3.69e-5,
        "weight_decay": 9.06e-6, "batch_size": 32, "max_epochs": 25, "patience": 10,
    }

    train_graphs = [ensure_cpu(g) for g in train_graphs]
    input_dim = train_graphs[0].x.shape[1]

    model = SAGEResidualReranker(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        alpha_init=config["alpha_init"],
        learn_alpha=config["learn_alpha"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = PairwiseMarginLoss(margin=0.1)
    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)

    best_ndcg, best_state, patience = -1, None, 0

    for epoch in range(config["max_epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss = loss_fn(refined, batch.node_labels, batch.batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_ndcg = []
        with torch.no_grad():
            for g in val_graphs:
                g = ensure_cpu(g).to(device)
                gold = g.node_labels.cpu().numpy()
                scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
                val_ndcg.append(compute_ndcg_at_k(gold, scores, 10))

        mean_ndcg = np.mean(val_ndcg)
        if mean_ndcg > best_ndcg:
            best_ndcg = mean_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model.to(device)


def main():
    graph_dir = Path("data/cache/gnn/rebuild_20260120")
    output_dir = Path("outputs/sage_residual_dynamick")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading graphs...")
    fold_graphs = load_graphs(graph_dir)
    n_folds = len(fold_graphs)

    # Test configurations
    configs = [
        {"name": "SAGE+Residual (K=10)", "use_dynamic_k": False},
        {"name": "SAGE+Residual + DK (tau=0.3)", "use_dynamic_k": True, "tau": 0.3},
        {"name": "SAGE+Residual + DK (tau=0.4)", "use_dynamic_k": True, "tau": 0.4},
        {"name": "SAGE+Residual + DK (tau=0.5)", "use_dynamic_k": True, "tau": 0.5},
    ]

    results = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*60}")

        fold_ndcg = []
        fold_recall = []
        fold_avg_k = []

        for fold_id in range(n_folds):
            train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
            val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

            # Train model
            model = train_model(train_graphs, val_graphs, device)

            # Evaluate
            model.eval()
            ndcg_list = []
            recall_list = []
            k_list = []

            with torch.no_grad():
                for g in val_graphs:
                    g = g.to(device)
                    gold = g.node_labels.cpu().numpy()
                    scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()

                    if cfg["use_dynamic_k"]:
                        probs = 1 / (1 + np.exp(-scores))
                        k = select_k_threshold(probs, cfg["tau"])
                    else:
                        k = 10

                    k_list.append(k)
                    ndcg_list.append(compute_ndcg_at_k(gold, scores, k))
                    recall_list.append(compute_recall_at_k(gold, scores, k))

            fold_ndcg.append(np.mean(ndcg_list))
            fold_recall.append(np.mean(recall_list))
            fold_avg_k.append(np.mean(k_list))
            print(f"  Fold {fold_id}: nDCG={fold_ndcg[-1]:.4f}, Recall={fold_recall[-1]:.4f}, avg_k={fold_avg_k[-1]:.1f}")

        mean_ndcg = np.mean(fold_ndcg)
        std_ndcg = np.std(fold_ndcg)
        mean_recall = np.mean(fold_recall)
        mean_k = np.mean(fold_avg_k)

        print(f"  Mean: nDCG={mean_ndcg:.4f} ± {std_ndcg:.4f}, Recall={mean_recall:.4f}, avg_k={mean_k:.1f}")

        results[cfg["name"]] = {
            "ndcg_mean": mean_ndcg,
            "ndcg_std": std_ndcg,
            "recall_mean": mean_recall,
            "avg_k": mean_k,
            "fold_ndcg": fold_ndcg,
        }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    baseline_ndcg = results["SAGE+Residual (K=10)"]["ndcg_mean"]
    for name, res in results.items():
        delta = (res["ndcg_mean"] - baseline_ndcg) / baseline_ndcg * 100
        print(f"{name}: nDCG={res['ndcg_mean']:.4f} ± {res['ndcg_std']:.4f}, avg_k={res['avg_k']:.1f} ({delta:+.2f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
