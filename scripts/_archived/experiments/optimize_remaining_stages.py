#!/usr/bin/env python3
"""
Optimize remaining pipeline stages: P2 Dynamic-K, P4 Criterion-Aware GNN.

Tests:
1. GraphSAGE vs GAT for P4 (architecture swap)
2. P2 Dynamic-K threshold/mass tuning
3. Combined P3+P2 integration

Usage:
    python scripts/experiments/optimize_remaining_stages.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/stage_optimization
"""

import argparse
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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

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


def compute_recall_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    sorted_idx = np.argsort(-scores)
    n_gold = gold_mask.sum()
    if n_gold == 0:
        return 0.0
    return gold_mask[sorted_idx[:k]].sum() / n_gold


def compute_hit_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    sorted_idx = np.argsort(-scores)
    return 1.0 if gold_mask[sorted_idx[:k]].sum() > 0 else 0.0


# ============================================================================
# P2 Dynamic-K Selection
# ============================================================================

def select_k_threshold(probs: np.ndarray, tau: float, k_min: int = 2, k_max: int = 10) -> int:
    """DK-A: Select K based on probability threshold."""
    k = np.sum(probs >= tau)
    return max(k_min, min(k, k_max))


def select_k_mass(probs: np.ndarray, gamma: float, k_min: int = 2, k_max: int = 10) -> int:
    """DK-B: Select K based on cumulative mass."""
    sorted_probs = np.sort(probs)[::-1]
    cumsum = np.cumsum(sorted_probs)
    total = sorted_probs.sum()
    if total > 0:
        k = np.searchsorted(cumsum / total, gamma) + 1
    else:
        k = k_min
    return max(k_min, min(k, k_max))


def evaluate_dynamic_k(graphs, scores_list, policy: str, param: float) -> Dict[str, float]:
    """Evaluate dynamic-K selection on graphs."""
    ndcg_scores = []
    recall_scores = []
    hit_scores = []
    k_values = []

    for i, g in enumerate(graphs):
        gold_mask = g.node_labels.cpu().numpy()
        scores = scores_list[i]

        if not gold_mask.any():
            continue

        # Normalize scores to probabilities
        probs = 1 / (1 + np.exp(-scores))  # Sigmoid

        # Select K based on policy
        if policy == "threshold":
            k = select_k_threshold(probs, param)
        elif policy == "mass":
            k = select_k_mass(probs, param)
        else:
            k = 10  # Fixed

        k_values.append(k)
        ndcg_scores.append(compute_ndcg_at_k(gold_mask, scores, k))
        recall_scores.append(compute_recall_at_k(gold_mask, scores, k))
        hit_scores.append(compute_hit_at_k(gold_mask, scores, k))

    return {
        'ndcg': np.mean(ndcg_scores),
        'recall': np.mean(recall_scores),
        'hit_rate': np.mean(hit_scores),
        'avg_k': np.mean(k_values),
    }


# ============================================================================
# GNN Architectures for Reranking (P3 style)
# ============================================================================

class SAGEReranker(nn.Module):
    """GraphSAGE reranker."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.05,
                 alpha_init=0.65, learn_alpha=True):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
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


# ============================================================================
# Training
# ============================================================================

def train_sage_model(train_graphs, val_graphs, config, device, seed=42):
    """Train GraphSAGE reranker."""
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


# ============================================================================
# Main Experiments
# ============================================================================

def run_dynamic_k_optimization(fold_graphs, device):
    """Optimize P2 Dynamic-K thresholds."""
    print("\n" + "="*60)
    print("P2 Dynamic-K Threshold Optimization")
    print("="*60)

    config = {
        "hidden_dim": 128, "num_layers": 1, "dropout": 0.05,
        "alpha_init": 0.65, "learn_alpha": True, "lr": 3.69e-5,
        "weight_decay": 9.06e-6, "batch_size": 32, "max_epochs": 25, "patience": 10,
    }

    n_folds = len(fold_graphs)
    results = []

    # Test different threshold values
    threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    mass_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    for fold_id in range(n_folds):
        print(f"\nFold {fold_id}:")
        train_graphs = [g for fid, graphs in fold_graphs.items() if fid != fold_id for g in graphs]
        val_graphs = [ensure_cpu(g) for g in fold_graphs[fold_id]]

        # Train GraphSAGE
        model = train_sage_model(train_graphs, val_graphs, config, device)

        # Get predictions
        model.eval()
        scores_list = []
        with torch.no_grad():
            for g in val_graphs:
                g = g.to(device)
                scores = model(g.x, g.edge_index, g.reranker_scores).cpu().numpy()
                scores_list.append(scores)

        # Test fixed K=10 (baseline)
        baseline = evaluate_dynamic_k(val_graphs, scores_list, "fixed", 10)
        print(f"  Fixed K=10: nDCG={baseline['ndcg']:.4f}, Recall={baseline['recall']:.4f}")

        # Test threshold policies
        best_threshold = None
        best_threshold_score = 0
        for tau in threshold_values:
            metrics = evaluate_dynamic_k(val_graphs, scores_list, "threshold", tau)
            if metrics['ndcg'] > best_threshold_score:
                best_threshold_score = metrics['ndcg']
                best_threshold = tau
            print(f"  Threshold tau={tau}: nDCG={metrics['ndcg']:.4f}, Recall={metrics['recall']:.4f}, avg_k={metrics['avg_k']:.1f}")

        # Test mass policies
        best_mass = None
        best_mass_score = 0
        for gamma in mass_values:
            metrics = evaluate_dynamic_k(val_graphs, scores_list, "mass", gamma)
            if metrics['ndcg'] > best_mass_score:
                best_mass_score = metrics['ndcg']
                best_mass = gamma
            print(f"  Mass gamma={gamma}: nDCG={metrics['ndcg']:.4f}, Recall={metrics['recall']:.4f}, avg_k={metrics['avg_k']:.1f}")

        results.append({
            'fold': fold_id,
            'baseline_ndcg': baseline['ndcg'],
            'best_threshold': best_threshold,
            'best_threshold_ndcg': best_threshold_score,
            'best_mass': best_mass,
            'best_mass_ndcg': best_mass_score,
        })

    # Summary
    print("\n" + "-"*60)
    print("Dynamic-K Optimization Summary:")
    print(f"  Baseline (K=10): {np.mean([r['baseline_ndcg'] for r in results]):.4f}")
    print(f"  Best Threshold: tau={np.mean([r['best_threshold'] for r in results]):.2f}, nDCG={np.mean([r['best_threshold_ndcg'] for r in results]):.4f}")
    print(f"  Best Mass: gamma={np.mean([r['best_mass'] for r in results]):.2f}, nDCG={np.mean([r['best_mass_ndcg'] for r in results]):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", default="outputs/stage_optimization")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading graphs...")
    fold_graphs = load_graphs(Path(args.graph_dir))

    # Run experiments
    dk_results = run_dynamic_k_optimization(fold_graphs, device)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'dynamic_k': dk_results,
        'timestamp': timestamp,
    }

    with open(output_dir / f"stage_optimization_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
