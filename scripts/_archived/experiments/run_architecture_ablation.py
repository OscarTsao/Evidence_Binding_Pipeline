#!/usr/bin/env python3
"""
Architecture Ablation Study: Compare GCN, GAT, GraphSAGE, and loss functions.

Tests multiple improvements:
1. Different GNN architectures (GCN, GAT, GraphSAGE)
2. Multi-seed ensemble
3. Listwise loss vs pairwise loss
4. Different edge thresholds

Usage:
    python scripts/experiments/run_architecture_ablation.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/architecture_ablation
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from tqdm import tqdm

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.constants import EXCLUDED_CRITERIA


# ============================================================================
# Metrics
# ============================================================================

def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute nDCG@K with binary relevance."""
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
    """Compute Mean Reciprocal Rank."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) == 0:
        return 0.0
    return 1.0 / (gold_positions[0] + 1)


# ============================================================================
# GNN Architectures
# ============================================================================

class GCNReranker(nn.Module):
    """GCN-based reranker (baseline)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.05, alpha_init: float = 0.65, learn_alpha: bool = True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

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


class GATReranker(nn.Module):
    """GAT-based reranker with attention mechanism."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.05, alpha_init: float = 0.65, learn_alpha: bool = True,
                 heads: int = 4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))

        self.score_head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(alpha_init), requires_grad=learn_alpha)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = self.dropout(h)

        gnn_scores = self.score_head(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        refined = alpha * reranker_scores + (1 - alpha) * gnn_scores
        return refined


class SAGEReranker(nn.Module):
    """GraphSAGE-based reranker."""

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


# ============================================================================
# Loss Functions
# ============================================================================

class PairwiseMarginLoss(nn.Module):
    """Pairwise margin ranking loss (current baseline)."""

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, scores, labels, batch=None):
        """Compute pairwise margin loss."""
        if batch is None:
            # Single graph
            return self._compute_single(scores, labels)

        # Batched graphs
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

        # All pairs
        pos_expand = pos_scores.unsqueeze(1)
        neg_expand = neg_scores.unsqueeze(0)

        losses = F.relu(self.margin - pos_expand + neg_expand)
        return losses.mean()


class ListNetLoss(nn.Module):
    """ListNet loss - listwise learning to rank."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, scores, labels, batch=None):
        """Compute ListNet cross-entropy loss."""
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
            if not torch.isnan(loss):
                total_loss += loss
                n_graphs += 1

        return total_loss / max(n_graphs, 1)

    def _compute_single(self, scores, labels):
        if labels.sum() == 0:
            return torch.tensor(0.0, device=scores.device)

        # Convert labels to probability distribution
        labels_float = labels.float()
        y_true = F.softmax(labels_float / self.temperature, dim=0)
        y_pred = F.log_softmax(scores / self.temperature, dim=0)

        # Cross-entropy
        loss = -torch.sum(y_true * y_pred)
        return loss


class ApproxNDCGLoss(nn.Module):
    """Approximate NDCG loss using softmax."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

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
            if not torch.isnan(loss):
                total_loss += loss
                n_graphs += 1

        return total_loss / max(n_graphs, 1)

    def _compute_single(self, scores, labels):
        n = len(scores)
        if labels.sum() == 0 or n < 2:
            return torch.tensor(0.0, device=scores.device)

        # Approximate ranking with softmax
        probs = F.softmax(scores / self.temperature, dim=0)

        # Position weights (discount factors)
        positions = torch.arange(1, n + 1, device=scores.device, dtype=torch.float)
        discounts = 1.0 / torch.log2(positions + 1)

        # Expected DCG
        labels_float = labels.float()
        expected_dcg = torch.sum(probs * labels_float * discounts)

        # Ideal DCG
        sorted_labels, _ = torch.sort(labels_float, descending=True)
        idcg = torch.sum(sorted_labels * discounts[:len(sorted_labels)])

        if idcg == 0:
            return torch.tensor(0.0, device=scores.device)

        # Negative NDCG (to minimize)
        ndcg = expected_dcg / idcg
        return 1.0 - ndcg


# ============================================================================
# Data Loading
# ============================================================================

def load_graphs(graph_dir: Path, exclude_a10: bool = True) -> Dict[int, List]:
    """Load graph dataset."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    exclude_criteria = EXCLUDED_CRITERIA if exclude_a10 else None

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        if exclude_criteria:
            graphs = [g for g in graphs if getattr(g, 'criterion_id', None) not in exclude_criteria]

        # Filter to positives only
        graphs = [g for g in graphs if g.node_labels.sum() > 0]
        fold_graphs[fold_id] = graphs
        print(f"Loaded fold {fold_id}: {len(graphs)} graphs")

    return fold_graphs, metadata


def ensure_cpu(graph):
    """Ensure all tensors in graph are on CPU."""
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph


# ============================================================================
# Training & Evaluation
# ============================================================================

def evaluate(model, graphs, device):
    """Evaluate model on graphs."""
    model.eval()
    ndcg_scores = []
    mrr_scores = []

    with torch.no_grad():
        for g in graphs:
            g = ensure_cpu(g).to(device)
            gold_mask = g.node_labels.cpu().numpy()

            refined = model(g.x, g.edge_index, g.reranker_scores)
            scores = refined.cpu().numpy()

            ndcg_scores.append(compute_ndcg_at_k(gold_mask, scores, 10))
            mrr_scores.append(compute_mrr(gold_mask, scores))

    return {
        'ndcg@10': np.mean(ndcg_scores),
        'mrr': np.mean(mrr_scores),
        'ndcg@10_std': np.std(ndcg_scores),
    }


def train_model(model_class, train_graphs, val_graphs, loss_fn, config, device, seed=42):
    """Train a single model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_graphs = [ensure_cpu(g) for g in train_graphs]
    val_graphs = [ensure_cpu(g) for g in val_graphs]

    input_dim = train_graphs[0].x.shape[1]

    model = model_class(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        alpha_init=config["alpha_init"],
        learn_alpha=config["learn_alpha"],
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
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
        metrics = evaluate(model, val_graphs, device)

        if metrics['ndcg@10'] > best_val_ndcg:
            best_val_ndcg = metrics['ndcg@10']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, best_val_ndcg


def run_5fold_experiment(model_class, loss_fn, fold_graphs, config, device,
                          experiment_name, seeds=[42]):
    """Run 5-fold CV with optional multi-seed ensemble."""
    n_folds = len(fold_graphs)
    all_results = []

    for fold_id in range(n_folds):
        train_graphs = []
        for fid in range(n_folds):
            if fid != fold_id:
                train_graphs.extend(fold_graphs[fid])
        val_graphs = fold_graphs[fold_id]

        # Multi-seed training
        seed_predictions = []
        for seed in seeds:
            model, _ = train_model(
                model_class, train_graphs, val_graphs, loss_fn, config, device, seed
            )

            # Get predictions
            model.eval()
            fold_preds = []
            with torch.no_grad():
                for g in val_graphs:
                    g = ensure_cpu(g).to(device)
                    refined = model(g.x, g.edge_index, g.reranker_scores)
                    fold_preds.append(refined.cpu().numpy())
            seed_predictions.append(fold_preds)

        # Ensemble predictions (average)
        ensemble_preds = []
        for i in range(len(val_graphs)):
            avg_pred = np.mean([seed_predictions[s][i] for s in range(len(seeds))], axis=0)
            ensemble_preds.append(avg_pred)

        # Evaluate ensemble
        ndcg_scores = []
        mrr_scores = []
        for i, g in enumerate(val_graphs):
            gold_mask = g.node_labels.cpu().numpy()
            ndcg_scores.append(compute_ndcg_at_k(gold_mask, ensemble_preds[i], 10))
            mrr_scores.append(compute_mrr(gold_mask, ensemble_preds[i]))

        fold_result = {
            'fold': fold_id,
            'ndcg@10': np.mean(ndcg_scores),
            'mrr': np.mean(mrr_scores),
        }
        all_results.append(fold_result)
        print(f"  Fold {fold_id}: nDCG@10={fold_result['ndcg@10']:.4f}, MRR={fold_result['mrr']:.4f}")

    # Aggregate
    mean_ndcg = np.mean([r['ndcg@10'] for r in all_results])
    std_ndcg = np.std([r['ndcg@10'] for r in all_results])
    mean_mrr = np.mean([r['mrr'] for r in all_results])

    return {
        'experiment': experiment_name,
        'mean_ndcg@10': mean_ndcg,
        'std_ndcg@10': std_ndcg,
        'mean_mrr': mean_mrr,
        'fold_results': all_results,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Architecture Ablation Study")
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default="outputs/architecture_ablation")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\nLoading graphs...")
    fold_graphs, metadata = load_graphs(Path(args.graph_dir))

    # HPO-optimized config (baseline)
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

    # Compute baseline (reranker only)
    print("\n" + "="*60)
    print("BASELINE (Jina-v3 Reranker Only)")
    print("="*60)
    baseline_ndcg = []
    for fold_id, graphs in fold_graphs.items():
        fold_ndcg = []
        for g in graphs:
            gold_mask = g.node_labels.cpu().numpy()
            scores = g.reranker_scores.cpu().numpy()
            fold_ndcg.append(compute_ndcg_at_k(gold_mask, scores, 10))
        baseline_ndcg.append(np.mean(fold_ndcg))
        print(f"  Fold {fold_id}: nDCG@10={np.mean(fold_ndcg):.4f}")

    baseline_mean = np.mean(baseline_ndcg)
    baseline_std = np.std(baseline_ndcg)
    print(f"  Mean: {baseline_mean:.4f} +/- {baseline_std:.4f}")

    results = []
    results.append({
        'experiment': 'Baseline (Jina-v3)',
        'mean_ndcg@10': baseline_mean,
        'std_ndcg@10': baseline_std,
    })

    # Experiment 1: GCN with pairwise loss (current best)
    print("\n" + "="*60)
    print("Experiment 1: GCN + Pairwise Loss (Current Best)")
    print("="*60)
    result = run_5fold_experiment(
        GCNReranker, PairwiseMarginLoss(margin=0.1), fold_graphs, config, device,
        "GCN + Pairwise", seeds=[42]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Experiment 2: GCN with multi-seed ensemble
    print("\n" + "="*60)
    print("Experiment 2: GCN + Pairwise (5-seed ensemble)")
    print("="*60)
    result = run_5fold_experiment(
        GCNReranker, PairwiseMarginLoss(margin=0.1), fold_graphs, config, device,
        "GCN + Pairwise (5-seed)", seeds=[42, 123, 456, 789, 1337]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Experiment 3: GAT
    print("\n" + "="*60)
    print("Experiment 3: GAT + Pairwise Loss")
    print("="*60)
    result = run_5fold_experiment(
        GATReranker, PairwiseMarginLoss(margin=0.1), fold_graphs, config, device,
        "GAT + Pairwise", seeds=[42]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Experiment 4: GraphSAGE
    print("\n" + "="*60)
    print("Experiment 4: GraphSAGE + Pairwise Loss")
    print("="*60)
    result = run_5fold_experiment(
        SAGEReranker, PairwiseMarginLoss(margin=0.1), fold_graphs, config, device,
        "GraphSAGE + Pairwise", seeds=[42]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Experiment 5: GCN + ListNet Loss
    print("\n" + "="*60)
    print("Experiment 5: GCN + ListNet Loss")
    print("="*60)
    result = run_5fold_experiment(
        GCNReranker, ListNetLoss(temperature=1.0), fold_graphs, config, device,
        "GCN + ListNet", seeds=[42]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Experiment 6: GCN + ApproxNDCG Loss
    print("\n" + "="*60)
    print("Experiment 6: GCN + ApproxNDCG Loss")
    print("="*60)
    result = run_5fold_experiment(
        GCNReranker, ApproxNDCGLoss(temperature=1.0), fold_graphs, config, device,
        "GCN + ApproxNDCG", seeds=[42]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Experiment 7: Best architecture + multi-seed
    print("\n" + "="*60)
    print("Experiment 7: GAT + Pairwise (5-seed ensemble)")
    print("="*60)
    result = run_5fold_experiment(
        GATReranker, PairwiseMarginLoss(margin=0.1), fold_graphs, config, device,
        "GAT + Pairwise (5-seed)", seeds=[42, 123, 456, 789, 1337]
    )
    results.append(result)
    print(f"  Mean: {result['mean_ndcg@10']:.4f} +/- {result['std_ndcg@10']:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Experiment':<35} {'nDCG@10':<20} {'vs Baseline':<15}")
    print("-"*70)

    for r in results:
        name = r['experiment']
        ndcg = r['mean_ndcg@10']
        std = r.get('std_ndcg@10', 0)
        improvement = ((ndcg - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
        print(f"{name:<35} {ndcg:.4f} +/- {std:.4f}    {improvement:+.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"ablation_results_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump({
            'baseline': {'mean_ndcg@10': baseline_mean, 'std': baseline_std},
            'experiments': results,
            'config': config,
            'timestamp': timestamp,
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Find best
    best = max(results, key=lambda x: x['mean_ndcg@10'])
    print(f"\nBest: {best['experiment']} with nDCG@10={best['mean_ndcg@10']:.4f}")


if __name__ == "__main__":
    main()


# Additional experiment: GraphSAGE + 5-seed ensemble
if __name__ == "__main__":
    pass  # Run from command line
