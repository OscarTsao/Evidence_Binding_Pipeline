#!/usr/bin/env python3
"""
Run GraphSAGE with 5-seed ensemble - the best configuration from ablation.
"""

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

    dcg = 0.0
    for i in range(min(k, len(sorted_gold))):
        if sorted_gold[i]:
            dcg += 1.0 / math.log2(i + 2)

    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) == 0:
        return 0.0
    return 1.0 / (gold_positions[0] + 1)


class SAGEReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.05, alpha_init: float = 0.65, learn_alpha: bool = True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)

        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        refined = alpha * reranker_scores + (1 - alpha) * gnn_scores
        return refined


class PairwiseMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels, batch=None):
        if batch is None:
            return self._compute_single(scores, labels)

        total_loss = 0.0
        n_graphs = 0

        unique_batches = batch.unique()
        for b in unique_batches:
            mask = batch == b
            b_scores = scores[mask]
            b_labels = labels[mask]
            loss = self._compute_single(b_scores, b_labels)
            if loss > 0:
                total_loss += loss
                n_graphs += 1

        return total_loss / max(n_graphs, 1)

    def _compute_single(self, scores, labels):
        pos_mask = labels > 0
        neg_mask = labels == 0

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device)

        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        pos_expand = pos_scores.unsqueeze(1)
        neg_expand = neg_scores.unsqueeze(0)

        losses = F.relu(self.margin - pos_expand + neg_expand)
        return losses.mean()


def ensure_cpu(graph):
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph


def load_graphs(graph_dir: Path):
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]
        graphs = [g for g in graphs if getattr(g, 'criterion_id', None) not in EXCLUDED_CRITERIA]
        graphs = [g for g in graphs if g.node_labels.sum() > 0]
        fold_graphs[fold_id] = graphs
        print(f"Loaded fold {fold_id}: {len(graphs)} graphs")

    return fold_graphs


def train_model(train_graphs, val_graphs, config, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_graphs = [ensure_cpu(g) for g in train_graphs]
    input_dim = train_graphs[0].x.shape[1]

    model = SAGEReranker(
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

    best_val_ndcg = -1
    best_state = None
    patience_counter = 0

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
        ndcg_scores = []
        with torch.no_grad():
            for g in val_graphs:
                g = ensure_cpu(g).to(device)
                gold_mask = g.node_labels.cpu().numpy()
                refined = model(g.x, g.edge_index, g.reranker_scores)
                scores = refined.cpu().numpy()
                ndcg_scores.append(compute_ndcg_at_k(gold_mask, scores, 10))

        val_ndcg = np.mean(ndcg_scores)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model


def main():
    graph_dir = Path("data/cache/gnn/rebuild_20260120")
    output_dir = Path("outputs/sage_ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nLoading graphs...")
    fold_graphs = load_graphs(graph_dir)

    config = {
        "hidden_dim": 128,
        "num_layers": 1,
        "dropout": 0.05,
        "alpha_init": 0.65,
        "learn_alpha": True,
        "lr": 3.69e-5,
        "weight_decay": 9.06e-6,
        "batch_size": 32,
        "max_epochs": 25,
        "patience": 10,
    }

    seeds = [42, 123, 456, 789, 1337]
    n_folds = len(fold_graphs)

    print("\n" + "="*60)
    print("GraphSAGE + Pairwise (5-seed ensemble)")
    print("="*60)

    all_fold_results = []

    for fold_id in range(n_folds):
        train_graphs = []
        for fid in range(n_folds):
            if fid != fold_id:
                train_graphs.extend(fold_graphs[fid])
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        # Train with multiple seeds
        seed_predictions = []
        for seed in seeds:
            print(f"  Fold {fold_id}, Seed {seed}...")
            model = train_model(train_graphs, val_graphs, config, device, seed)

            # Get predictions
            model.eval()
            fold_preds = []
            with torch.no_grad():
                for g in val_graphs:
                    g = g.to(device)
                    refined = model(g.x, g.edge_index, g.reranker_scores)
                    fold_preds.append(refined.cpu().numpy())
            seed_predictions.append(fold_preds)

        # Ensemble predictions
        ndcg_scores = []
        mrr_scores = []
        for i, g in enumerate(val_graphs):
            avg_pred = np.mean([seed_predictions[s][i] for s in range(len(seeds))], axis=0)
            gold_mask = g.node_labels.cpu().numpy()
            ndcg_scores.append(compute_ndcg_at_k(gold_mask, avg_pred, 10))
            mrr_scores.append(compute_mrr(gold_mask, avg_pred))

        fold_ndcg = np.mean(ndcg_scores)
        fold_mrr = np.mean(mrr_scores)
        all_fold_results.append({'fold': fold_id, 'ndcg@10': fold_ndcg, 'mrr': fold_mrr})
        print(f"  Fold {fold_id}: nDCG@10={fold_ndcg:.4f}, MRR={fold_mrr:.4f}")

    mean_ndcg = np.mean([r['ndcg@10'] for r in all_fold_results])
    std_ndcg = np.std([r['ndcg@10'] for r in all_fold_results])
    mean_mrr = np.mean([r['mrr'] for r in all_fold_results])

    print(f"\n  Mean: nDCG@10={mean_ndcg:.4f} +/- {std_ndcg:.4f}, MRR={mean_mrr:.4f}")
    print(f"  vs Baseline (0.7428): +{(mean_ndcg - 0.7428) / 0.7428 * 100:.2f}%")

    # Save results
    results = {
        'experiment': 'GraphSAGE + Pairwise (5-seed ensemble)',
        'mean_ndcg@10': mean_ndcg,
        'std_ndcg@10': std_ndcg,
        'mean_mrr': mean_mrr,
        'fold_results': all_fold_results,
        'config': config,
        'seeds': seeds,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = output_dir / f"sage_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
