#!/usr/bin/env python3
"""
Single-Split GNN HPO: Fast HPO using single train/val split.

Uses fold 0 as validation, folds 1-4 as training for speed.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from tqdm import tqdm

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


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

    for i, is_gold in enumerate(sorted_gold):
        if is_gold:
            return 1.0 / (i + 1)
    return 0.0


class P3GraphReranker(nn.Module):
    """P3 Graph Reranker with configurable hyperparameters."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        alpha_init: float = 0.7,
        learn_alpha: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.output_proj = nn.Linear(hidden_dim, 1)

        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        original_scores = data.reranker_scores

        h = self.input_proj(x)
        h = torch.relu(h)
        h = self.dropout_layer(h)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout_layer(h)

        gnn_score = self.output_proj(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        final_score = alpha * original_scores + (1 - alpha) * gnn_score

        return final_score


def deep_clone_to_cpu(graph: Data) -> Data:
    """Create a deep copy of graph with all tensors on CPU."""
    new_data = Data()
    for key, value in graph:
        if isinstance(value, torch.Tensor):
            new_data[key] = value.detach().cpu().clone()
        else:
            new_data[key] = value
    return new_data


def load_graph_dataset(
    graph_dir: Path,
    exclude_criteria: List[str] = ["A.10"],
) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache, ensuring all on CPU."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False, map_location='cpu')
        graphs = data["graphs"]

        processed = []
        for g in graphs:
            g_cpu = deep_clone_to_cpu(g)
            if exclude_criteria:
                if getattr(g_cpu, 'criterion_id', None) in exclude_criteria:
                    continue
            processed.append(g_cpu)

        print(f"Fold {fold_id}: {len(processed)} graphs", flush=True)
        fold_graphs[fold_id] = processed

    return fold_graphs, metadata


def train_epoch_batched(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch using batched DataLoader."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        scores = model(batch)
        labels = batch.node_labels.float()

        batch_idx = batch.batch
        unique_batches = batch_idx.unique()

        loss = 0.0
        n_valid = 0

        for bid in unique_batches:
            mask = batch_idx == bid
            graph_scores = scores[mask]
            graph_labels = labels[mask]

            pos_mask = graph_labels > 0.5
            neg_mask = graph_labels < 0.5

            if pos_mask.any() and neg_mask.any():
                pos_scores = graph_scores[pos_mask]
                neg_scores = graph_scores[neg_mask]

                n_pos = min(len(pos_scores), 10)
                n_neg = min(len(neg_scores), 10)

                for ps in pos_scores[:n_pos]:
                    for ns in neg_scores[:n_neg]:
                        loss += torch.relu(1.0 - ps + ns)
                n_valid += n_pos * n_neg

        if n_valid > 0:
            loss = loss / n_valid
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    val_graphs: List[Data],
    device: torch.device,
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    ndcg_scores = []
    mrr_scores = []

    with torch.no_grad():
        for graph in val_graphs:
            graph_gpu = graph.clone().to(device)
            scores = model(graph_gpu).cpu().numpy()
            gold_mask = graph.node_labels.cpu().numpy()

            if gold_mask.sum() > 0:
                ndcg = compute_ndcg_at_k(gold_mask, scores, top_k)
                mrr = compute_mrr(gold_mask, scores)

                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'n_queries': len(ndcg_scores),
    }


def evaluate_baseline(val_graphs: List[Data], top_k: int = 10) -> Dict[str, float]:
    """Evaluate baseline (original reranker scores)."""
    ndcg_scores = []
    mrr_scores = []

    for graph in val_graphs:
        scores = graph.reranker_scores.cpu().numpy()
        gold_mask = graph.node_labels.cpu().numpy()

        if gold_mask.sum() > 0:
            ndcg = compute_ndcg_at_k(gold_mask, scores, top_k)
            mrr = compute_mrr(gold_mask, scores)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'n_queries': len(ndcg_scores),
    }


def get_input_dim(graphs: List[Data]) -> int:
    """Get input dimension from graphs."""
    for g in graphs:
        if hasattr(g, 'x') and g.x is not None:
            return g.x.shape[1]
    return 1


def objective(
    trial: optuna.Trial,
    train_graphs: List[Data],
    val_graphs: List[Data],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Optuna objective with single split for fast HPO."""

    # Hyperparameter search space
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
    alpha_init = trial.suggest_float('alpha_init', 0.1, 0.95, step=0.05)
    learn_alpha = trial.suggest_categorical('learn_alpha', [True, False])

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    n_epochs = trial.suggest_int('n_epochs', 5, 30, step=5)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    input_dim = get_input_dim(train_graphs)

    train_loader = DataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = P3GraphReranker(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        alpha_init=alpha_init,
        learn_alpha=learn_alpha,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_ndcg = 0.0
    patience_counter = 0
    patience = 5

    for epoch in range(n_epochs):
        train_epoch_batched(model, train_loader, optimizer, device)

        # Evaluate every epoch for early stopping
        metrics = evaluate(model, val_graphs, device)

        if metrics['ndcg@10'] > best_ndcg:
            best_ndcg = metrics['ndcg@10']
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        # Report for pruning
        trial.report(metrics['ndcg@10'], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_ndcg


def main():
    parser = argparse.ArgumentParser(description='Single-Split GNN HPO (fast)')
    parser.add_argument('--graph_dir', type=str,
                        default='data/cache/gnn/rebuild_20260120')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/hpo/gnn_single_split')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_fold', type=int, default=0,
                        help='Fold to use as validation (default: 0)')
    parser.add_argument('--timeout', type=int, default=None)

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("\n=== Loading Graphs ===", flush=True)
    fold_graphs, metadata = load_graph_dataset(graph_dir)

    # Split into train/val using single fold
    val_fold = args.val_fold
    train_graphs = []
    val_graphs = []

    for fold_id, graphs in fold_graphs.items():
        if fold_id == val_fold:
            val_graphs.extend(graphs)
        else:
            train_graphs.extend(graphs)

    print(f"\nTrain: {len(train_graphs)} graphs", flush=True)
    print(f"Val: {len(val_graphs)} graphs (fold {val_fold})", flush=True)

    # Baseline evaluation
    baseline_metrics = evaluate_baseline(val_graphs)
    print(f"\nBaseline nDCG@10: {baseline_metrics['ndcg@10']:.4f}", flush=True)

    print("\n" + "="*60, flush=True)
    print("=== SINGLE-SPLIT GNN HPO ===", flush=True)
    print("="*60, flush=True)
    print(f"\nTrials: {args.n_trials}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=15)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)

    study = optuna.create_study(
        study_name="gnn_hpo_single_split",
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, train_graphs, val_graphs, device, args.batch_size),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'baseline_ndcg@10': baseline_metrics['ndcg@10'],
        'improvement': study.best_value - baseline_metrics['ndcg@10'],
        'improvement_pct': (study.best_value - baseline_metrics['ndcg@10']) / baseline_metrics['ndcg@10'] * 100,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'val_fold': val_fold,
        'n_train': len(train_graphs),
        'n_val': len(val_graphs),
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(results, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'trials.csv', index=False)

    print(f"\n{'='*60}", flush=True)
    print(f"=== HPO COMPLETE ===", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total trials: {len(study.trials)}", flush=True)
    print(f"Completed: {results['n_complete']}", flush=True)
    print(f"Pruned: {results['n_pruned']}", flush=True)
    print(f"\nBaseline nDCG@10: {baseline_metrics['ndcg@10']:.4f}", flush=True)
    print(f"Best nDCG@10:     {study.best_value:.4f}", flush=True)
    print(f"Improvement:      +{results['improvement']:.4f} (+{results['improvement_pct']:.2f}%)", flush=True)
    print(f"\nBest params: {json.dumps(study.best_params, indent=2)}", flush=True)
    print(f"\nResults saved to: {output_dir}", flush=True)


if __name__ == '__main__':
    main()
