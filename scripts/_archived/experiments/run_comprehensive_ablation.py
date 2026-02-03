#!/usr/bin/env python3
"""
Comprehensive ablation study: Test ALL possible combinations and improvements.

Experiments:
1. Deeper GraphSAGE (1, 2, 3 layers)
2. Different hidden dimensions (64, 128, 256, 512)
3. Different aggregators (mean, max, lstm)
4. Residual connections
5. Architecture ensemble (GCN + GAT + SAGE)
6. GraphSAGE + Dynamic-K
7. Different dropout rates (0.0, 0.1, 0.2, 0.3)
8. Different alpha init (0.3, 0.5, 0.65, 0.8)
9. Layer normalization
10. Combined best settings

Usage:
    python scripts/experiments/run_comprehensive_ablation.py
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LayerNorm

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.constants import EXCLUDED_CRITERIA


# ============================================================================
# Metrics
# ============================================================================

def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    positions = np.where(sorted_gold)[0]
    return 1.0 / (positions[0] + 1) if len(positions) > 0 else 0.0


# ============================================================================
# Model Variants
# ============================================================================

class FlexibleSAGEReranker(nn.Module):
    """Flexible GraphSAGE with configurable depth, width, residuals, and layer norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.05,
        alpha_init: float = 0.65,
        learn_alpha: bool = True,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        aggregator: str = "mean",  # mean, max, lstm
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Input projection if using residual
        self.input_proj = nn.Linear(input_dim, hidden_dim) if use_residual else None

        # SAGE convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggregator))

        # Layer norms
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
        else:
            self.layer_norms = None

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        # Input projection for residual
        if self.use_residual and self.input_proj is not None:
            residual = self.input_proj(x)

        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)

            if self.use_layer_norm and self.layer_norms is not None:
                h_new = self.layer_norms[i](h_new)

            h_new = self.dropout(h_new)

            # Residual connection (after first layer)
            if self.use_residual and i > 0:
                h_new = h_new + h
            elif self.use_residual and i == 0 and self.input_proj is not None:
                h_new = h_new + residual

            h = h_new

        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class GCNReranker(nn.Module):
    """GCN reranker for ensemble."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = self.dropout(h)
        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * reranker_scores + (1 - alpha) * gnn_scores


class GATReranker(nn.Module):
    """GAT reranker for ensemble."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True, heads=4):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout)])
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        h = x
        for conv in self.convs:
            h = F.elu(conv(h, edge_index))
            h = self.dropout(h)
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


# ============================================================================
# Data Loading
# ============================================================================

def load_graphs(graph_dir: Path):
    with open(graph_dir / "metadata.json") as f:
        metadata = json.load(f)
    fold_graphs = {}
    for fold_id in range(metadata["n_folds"]):
        data = torch.load(graph_dir / f"fold_{fold_id}.pt", weights_only=False)
        graphs = [g for g in data["graphs"] if getattr(g, 'criterion_id', None) not in EXCLUDED_CRITERIA]
        graphs = [g for g in graphs if g.node_labels.sum() > 0]
        fold_graphs[fold_id] = graphs
    return fold_graphs


def ensure_cpu(g):
    for key in g.keys():
        if isinstance(g[key], torch.Tensor):
            g[key] = g[key].cpu()
    return g


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_model(model_class, train_graphs, val_graphs, config, device, seed=42, **model_kwargs):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_graphs = [ensure_cpu(g) for g in train_graphs]
    val_graphs = [ensure_cpu(g) for g in val_graphs]
    input_dim = train_graphs[0].x.shape[1]

    model = model_class(input_dim=input_dim, **model_kwargs).to(device)
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
        val_scores = []
        with torch.no_grad():
            for g in val_graphs:
                g = ensure_cpu(g).to(device)
                scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
                gold = g.node_labels.cpu().numpy()
                val_scores.append(compute_ndcg_at_k(gold, scores, 10))

        mean_ndcg = np.mean(val_scores)
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
    return model.to(device), best_ndcg


def evaluate_model(model, graphs, device):
    model.eval()
    ndcg_scores, mrr_scores = [], []
    with torch.no_grad():
        for g in graphs:
            g = ensure_cpu(g).to(device)
            scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
            gold = g.node_labels.cpu().numpy()
            ndcg_scores.append(compute_ndcg_at_k(gold, scores, 10))
            mrr_scores.append(compute_mrr(gold, scores))
    return np.mean(ndcg_scores), np.mean(mrr_scores)


def run_5fold_experiment(model_class, fold_graphs, config, device, name, **model_kwargs):
    """Run 5-fold CV experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    n_folds = len(fold_graphs)
    fold_results = []

    for fold_id in range(n_folds):
        train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        model, _ = train_model(model_class, train_graphs, val_graphs, config, device, **model_kwargs)
        ndcg, mrr = evaluate_model(model, val_graphs, device)
        fold_results.append({'fold': fold_id, 'ndcg@10': ndcg, 'mrr': mrr})
        print(f"  Fold {fold_id}: nDCG@10={ndcg:.4f}, MRR={mrr:.4f}")

    mean_ndcg = np.mean([r['ndcg@10'] for r in fold_results])
    std_ndcg = np.std([r['ndcg@10'] for r in fold_results])
    print(f"  Mean: {mean_ndcg:.4f} ± {std_ndcg:.4f}")

    return {'name': name, 'mean_ndcg': mean_ndcg, 'std_ndcg': std_ndcg, 'folds': fold_results}


def run_ensemble_experiment(fold_graphs, config, device):
    """Run architecture ensemble experiment."""
    print(f"\n{'='*60}")
    print("Experiment: Architecture Ensemble (GCN + GAT + SAGE)")
    print(f"{'='*60}")

    n_folds = len(fold_graphs)
    fold_results = []

    for fold_id in range(n_folds):
        train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        # Train all three models
        models = {}
        for model_class, name in [(GCNReranker, 'gcn'), (GATReranker, 'gat'), (FlexibleSAGEReranker, 'sage')]:
            model, _ = train_model(
                model_class, train_graphs, val_graphs, config, device,
                hidden_dim=config["hidden_dim"], num_layers=1, dropout=config["dropout"],
                alpha_init=config["alpha_init"], learn_alpha=True
            )
            models[name] = model

        # Ensemble predictions
        ndcg_scores = []
        for g in val_graphs:
            g = ensure_cpu(g).to(device)
            gold = g.node_labels.cpu().numpy()

            # Get predictions from all models
            preds = []
            with torch.no_grad():
                for model in models.values():
                    model.eval()
                    scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
                    preds.append(scores)

            # Average ensemble
            ensemble_scores = np.mean(preds, axis=0)
            ndcg_scores.append(compute_ndcg_at_k(gold, ensemble_scores, 10))

        fold_ndcg = np.mean(ndcg_scores)
        fold_results.append({'fold': fold_id, 'ndcg@10': fold_ndcg})
        print(f"  Fold {fold_id}: nDCG@10={fold_ndcg:.4f}")

    mean_ndcg = np.mean([r['ndcg@10'] for r in fold_results])
    std_ndcg = np.std([r['ndcg@10'] for r in fold_results])
    print(f"  Mean: {mean_ndcg:.4f} ± {std_ndcg:.4f}")

    return {'name': 'Ensemble (GCN+GAT+SAGE)', 'mean_ndcg': mean_ndcg, 'std_ndcg': std_ndcg}


# ============================================================================
# Main
# ============================================================================

def main():
    graph_dir = Path("data/cache/gnn/rebuild_20260120")
    output_dir = Path("outputs/comprehensive_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading graphs...")
    fold_graphs = load_graphs(graph_dir)
    for fid, graphs in fold_graphs.items():
        print(f"  Fold {fid}: {len(graphs)} graphs")

    # Base config
    base_config = {
        "hidden_dim": 128, "lr": 3.69e-5, "weight_decay": 9.06e-6,
        "batch_size": 32, "max_epochs": 25, "patience": 10,
        "dropout": 0.05, "alpha_init": 0.65,
    }

    results = []

    # Baseline: GraphSAGE 1-layer (current best)
    results.append(run_5fold_experiment(
        FlexibleSAGEReranker, fold_graphs, base_config, device,
        "SAGE 1-layer (baseline)", hidden_dim=128, num_layers=1,
        dropout=0.05, alpha_init=0.65, learn_alpha=True
    ))

    # Experiment 1: Deeper networks
    for num_layers in [2, 3]:
        results.append(run_5fold_experiment(
            FlexibleSAGEReranker, fold_graphs, base_config, device,
            f"SAGE {num_layers}-layer", hidden_dim=128, num_layers=num_layers,
            dropout=0.05, alpha_init=0.65, learn_alpha=True
        ))

    # Experiment 2: Different hidden dimensions
    for hidden_dim in [64, 256, 512]:
        results.append(run_5fold_experiment(
            FlexibleSAGEReranker, fold_graphs, base_config, device,
            f"SAGE hidden={hidden_dim}", hidden_dim=hidden_dim, num_layers=1,
            dropout=0.05, alpha_init=0.65, learn_alpha=True
        ))

    # Experiment 3: Residual connections
    results.append(run_5fold_experiment(
        FlexibleSAGEReranker, fold_graphs, base_config, device,
        "SAGE + Residual", hidden_dim=128, num_layers=2,
        dropout=0.05, alpha_init=0.65, learn_alpha=True, use_residual=True
    ))

    # Experiment 4: Layer normalization
    results.append(run_5fold_experiment(
        FlexibleSAGEReranker, fold_graphs, base_config, device,
        "SAGE + LayerNorm", hidden_dim=128, num_layers=2,
        dropout=0.05, alpha_init=0.65, learn_alpha=True, use_layer_norm=True
    ))

    # Experiment 5: Residual + LayerNorm
    results.append(run_5fold_experiment(
        FlexibleSAGEReranker, fold_graphs, base_config, device,
        "SAGE + Residual + LayerNorm", hidden_dim=128, num_layers=2,
        dropout=0.05, alpha_init=0.65, learn_alpha=True, use_residual=True, use_layer_norm=True
    ))

    # Experiment 6: Different dropout rates
    for dropout in [0.0, 0.1, 0.2, 0.3]:
        results.append(run_5fold_experiment(
            FlexibleSAGEReranker, fold_graphs, base_config, device,
            f"SAGE dropout={dropout}", hidden_dim=128, num_layers=1,
            dropout=dropout, alpha_init=0.65, learn_alpha=True
        ))

    # Experiment 7: Different alpha initialization
    for alpha_init in [0.3, 0.5, 0.8]:
        results.append(run_5fold_experiment(
            FlexibleSAGEReranker, fold_graphs, base_config, device,
            f"SAGE alpha_init={alpha_init}", hidden_dim=128, num_layers=1,
            dropout=0.05, alpha_init=alpha_init, learn_alpha=True
        ))

    # Experiment 8: Different aggregators
    for aggr in ["max"]:  # mean is default, lstm not supported in basic SAGEConv
        results.append(run_5fold_experiment(
            FlexibleSAGEReranker, fold_graphs, base_config, device,
            f"SAGE aggr={aggr}", hidden_dim=128, num_layers=1,
            dropout=0.05, alpha_init=0.65, learn_alpha=True, aggregator=aggr
        ))

    # Experiment 9: Architecture ensemble
    ensemble_result = run_ensemble_experiment(fold_graphs, base_config, device)
    results.append(ensemble_result)

    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE ABLATION SUMMARY")
    print("="*80)
    print(f"{'Experiment':<45} {'nDCG@10':<20} {'vs Baseline':<15}")
    print("-"*80)

    baseline_ndcg = results[0]['mean_ndcg']

    # Sort by performance
    sorted_results = sorted(results, key=lambda x: x['mean_ndcg'], reverse=True)

    for r in sorted_results:
        name = r['name']
        ndcg = r['mean_ndcg']
        std = r.get('std_ndcg', 0)
        delta = ((ndcg - baseline_ndcg) / baseline_ndcg * 100)
        marker = "**" if ndcg > baseline_ndcg else ""
        print(f"{marker}{name:<43} {ndcg:.4f} ± {std:.4f}      {delta:+.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"comprehensive_ablation_{timestamp}.json", "w") as f:
        json.dump({
            'baseline_ndcg': baseline_ndcg,
            'results': results,
            'timestamp': timestamp,
        }, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    # Best configuration
    best = sorted_results[0]
    print(f"\nBest: {best['name']} with nDCG@10={best['mean_ndcg']:.4f}")


if __name__ == "__main__":
    main()
