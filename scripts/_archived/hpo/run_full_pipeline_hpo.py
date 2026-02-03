#!/usr/bin/env python3
"""
Full Pipeline HPO: Jina-Reranker-v3 + GNN stages.

Runs hyperparameter optimization for:
1. Jina-Reranker-v3 parameters (max_length, top_k_rerank, top_k_final)
2. P3 GNN Graph Reranker (hidden_dim, num_layers, alpha_init, dropout)

Uses 5-fold cross-validation with positives_only protocol.
Leverages pre-built graph cache for efficiency.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

# Add project root to path
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


def compute_recall_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Recall@K."""
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0

    sorted_idx = np.argsort(-scores)
    top_k_idx = sorted_idx[:k]
    n_retrieved = int(gold_mask[top_k_idx].sum())
    return n_retrieved / n_gold


class P3GraphReranker(nn.Module):
    """P3 Graph Reranker with configurable hyperparameters."""

    def __init__(
        self,
        input_dim: int = 1,
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

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Alpha for interpolation
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(alpha_init))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # Input projection
        h = self.input_proj(x)
        h = torch.relu(h)
        h = self.dropout_layer(h)

        # GCN layers
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout_layer(h)

        # Output score
        gnn_score = self.output_proj(h).squeeze(-1)

        # Interpolate with original score
        original_score = x[:, 0]
        alpha = torch.sigmoid(self.alpha)
        final_score = alpha * original_score + (1 - alpha) * gnn_score

        return final_score


def ensure_cpu(graph: Data) -> Data:
    """Ensure all tensors in graph are on CPU."""
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph


def load_fold_graphs(graph_dir: Path, fold_id: int) -> Tuple[List[Data], List[Data]]:
    """Load train and val graphs for a fold."""
    fold_dir = graph_dir / f"fold_{fold_id}"

    train_graphs = []
    val_graphs = []

    # Load train graphs
    train_dir = fold_dir / "train"
    if train_dir.exists():
        for pt_file in sorted(train_dir.glob("*.pt")):
            graph = torch.load(pt_file, map_location='cpu', weights_only=False)
            graph = ensure_cpu(graph)
            train_graphs.append(graph)

    # Load val graphs
    val_dir = fold_dir / "val"
    if val_dir.exists():
        for pt_file in sorted(val_dir.glob("*.pt")):
            graph = torch.load(pt_file, map_location='cpu', weights_only=False)
            graph = ensure_cpu(graph)
            val_graphs.append(graph)

    return train_graphs, val_graphs


def filter_graphs_by_top_k(graphs: List[Data], top_k_rerank: int, top_k_final: int) -> List[Data]:
    """
    Simulate different top_k_rerank and top_k_final settings.

    - top_k_rerank: Number of candidates passed to reranker (affects graph size)
    - top_k_final: Number of candidates in final output (affects evaluation)
    """
    filtered = []
    for g in graphs:
        # Limit to top_k_rerank candidates (by original score)
        n_nodes = g.x.shape[0]
        if n_nodes > top_k_rerank:
            # Get top-k by original score
            scores = g.x[:, 0].numpy()
            top_k_idx = np.argsort(-scores)[:top_k_rerank]
            top_k_idx = np.sort(top_k_idx)  # Keep original order

            # Create new graph with subset
            new_x = g.x[top_k_idx]
            new_y = g.y[top_k_idx] if hasattr(g, 'y') and g.y is not None else None

            # Filter edges
            mask = torch.zeros(n_nodes, dtype=torch.bool)
            mask[top_k_idx] = True

            # Remap indices
            idx_map = torch.full((n_nodes,), -1, dtype=torch.long)
            idx_map[top_k_idx] = torch.arange(len(top_k_idx))

            if g.edge_index.shape[1] > 0:
                edge_mask = mask[g.edge_index[0]] & mask[g.edge_index[1]]
                new_edge_index = g.edge_index[:, edge_mask]
                new_edge_index = idx_map[new_edge_index]
            else:
                new_edge_index = g.edge_index

            new_g = Data(x=new_x, edge_index=new_edge_index, y=new_y)
            # Copy metadata
            if hasattr(g, 'query_id'):
                new_g.query_id = g.query_id
            filtered.append(new_g)
        else:
            filtered.append(g)

    return filtered


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        scores = model(batch)

        # Pairwise ranking loss
        if batch.y is not None:
            y = batch.y.float()
            # Find positive and negative pairs
            pos_mask = y > 0.5
            neg_mask = y < 0.5

            if pos_mask.any() and neg_mask.any():
                pos_scores = scores[pos_mask]
                neg_scores = scores[neg_mask]

                # Margin ranking loss
                loss = 0.0
                n_pairs = 0
                for ps in pos_scores:
                    for ns in neg_scores:
                        loss += torch.relu(1.0 - ps + ns)
                        n_pairs += 1

                if n_pairs > 0:
                    loss = loss / n_pairs
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    val_graphs: List[Data],
    device: torch.device,
    top_k_final: int = 10,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    with torch.no_grad():
        for graph in val_graphs:
            graph = graph.to(device)
            scores = model(graph).cpu().numpy()

            if graph.y is not None:
                gold_mask = graph.y.cpu().numpy() > 0.5

                # Only evaluate queries with evidence
                if gold_mask.sum() > 0:
                    ndcg = compute_ndcg_at_k(gold_mask, scores, top_k_final)
                    mrr = compute_mrr(gold_mask, scores)
                    recall = compute_recall_at_k(gold_mask, scores, top_k_final)

                    ndcg_scores.append(ndcg)
                    mrr_scores.append(mrr)
                    recall_scores.append(recall)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'recall@10': np.mean(recall_scores) if recall_scores else 0.0,
        'n_queries': len(ndcg_scores),
    }


def evaluate_baseline(
    val_graphs: List[Data],
    top_k_final: int = 10,
) -> Dict[str, float]:
    """Evaluate baseline (original scores without GNN)."""
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for graph in val_graphs:
        scores = graph.x[:, 0].numpy()

        if graph.y is not None:
            gold_mask = graph.y.numpy() > 0.5

            if gold_mask.sum() > 0:
                ndcg = compute_ndcg_at_k(gold_mask, scores, top_k_final)
                mrr = compute_mrr(gold_mask, scores)
                recall = compute_recall_at_k(gold_mask, scores, top_k_final)

                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)
                recall_scores.append(recall)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'recall@10': np.mean(recall_scores) if recall_scores else 0.0,
        'n_queries': len(ndcg_scores),
    }


def objective(trial: optuna.Trial, graph_dir: Path, device: torch.device) -> float:
    """Optuna objective function for full pipeline HPO."""

    # Jina-Reranker-v3 parameters (simulated via graph filtering)
    top_k_rerank = trial.suggest_categorical('top_k_rerank', [10, 15, 20, 25, 30])
    top_k_final = trial.suggest_categorical('top_k_final', [5, 10, 15])

    # P3 GNN parameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.1)
    alpha_init = trial.suggest_float('alpha_init', 0.3, 0.9, step=0.1)
    learn_alpha = trial.suggest_categorical('learn_alpha', [True, False])

    # Training parameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    n_epochs = trial.suggest_int('n_epochs', 10, 50, step=10)

    # 5-fold cross-validation
    fold_ndcgs = []

    for fold_id in range(5):
        # Load graphs
        train_graphs, val_graphs = load_fold_graphs(graph_dir, fold_id)

        if not train_graphs or not val_graphs:
            continue

        # Filter graphs by top_k settings
        train_graphs = filter_graphs_by_top_k(train_graphs, top_k_rerank, top_k_final)
        val_graphs = filter_graphs_by_top_k(val_graphs, top_k_rerank, top_k_final)

        # Create model
        model = P3GraphReranker(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            alpha_init=alpha_init,
            learn_alpha=learn_alpha,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create data loader
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

        # Train
        for epoch in range(n_epochs):
            train_epoch(model, train_loader, optimizer, device)

        # Evaluate
        metrics = evaluate(model, val_graphs, device, top_k_final)
        fold_ndcgs.append(metrics['ndcg@10'])

        # Report intermediate value for pruning
        trial.report(np.mean(fold_ndcgs), fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_ndcgs) if fold_ndcgs else 0.0


def run_hpo(
    graph_dir: Path,
    output_dir: Path,
    n_trials: int = 100,
    study_name: str = "full_pipeline_hpo",
) -> Dict:
    """Run HPO study."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, graph_dir, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat(),
        'study_name': study_name,
    }

    # Save best params
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save trial history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'trials.csv', index=False)

    print(f"\nBest nDCG@10: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return results


def evaluate_best_config(
    graph_dir: Path,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """Evaluate best config with full 5-fold CV."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    top_k_rerank = config['top_k_rerank']
    top_k_final = config['top_k_final']

    fold_results = []

    for fold_id in range(5):
        train_graphs, val_graphs = load_fold_graphs(graph_dir, fold_id)

        if not train_graphs or not val_graphs:
            continue

        # Filter graphs
        train_graphs = filter_graphs_by_top_k(train_graphs, top_k_rerank, top_k_final)
        val_graphs = filter_graphs_by_top_k(val_graphs, top_k_rerank, top_k_final)

        # Baseline evaluation
        baseline_metrics = evaluate_baseline(val_graphs, top_k_final)

        # GNN evaluation
        model = P3GraphReranker(
            input_dim=1,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            alpha_init=config['alpha_init'],
            learn_alpha=config['learn_alpha'],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

        for epoch in range(config['n_epochs']):
            train_epoch(model, train_loader, optimizer, device)

        gnn_metrics = evaluate(model, val_graphs, device, top_k_final)

        fold_results.append({
            'fold_id': fold_id,
            'baseline_ndcg@10': baseline_metrics['ndcg@10'],
            'gnn_ndcg@10': gnn_metrics['ndcg@10'],
            'improvement': gnn_metrics['ndcg@10'] - baseline_metrics['ndcg@10'],
            'n_queries': gnn_metrics['n_queries'],
        })

        print(f"Fold {fold_id}: Baseline={baseline_metrics['ndcg@10']:.4f}, "
              f"GNN={gnn_metrics['ndcg@10']:.4f}, "
              f"Δ={gnn_metrics['ndcg@10'] - baseline_metrics['ndcg@10']:.4f}")

    # Aggregate results
    baseline_mean = np.mean([r['baseline_ndcg@10'] for r in fold_results])
    baseline_std = np.std([r['baseline_ndcg@10'] for r in fold_results])
    gnn_mean = np.mean([r['gnn_ndcg@10'] for r in fold_results])
    gnn_std = np.std([r['gnn_ndcg@10'] for r in fold_results])

    summary = {
        'config': config,
        'baseline': {'mean': baseline_mean, 'std': baseline_std},
        'gnn': {'mean': gnn_mean, 'std': gnn_std},
        'improvement': {
            'absolute': gnn_mean - baseline_mean,
            'relative_pct': (gnn_mean - baseline_mean) / baseline_mean * 100,
        },
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / 'best_config_evaluation.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Final Results ===")
    print(f"Baseline nDCG@10: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"GNN nDCG@10: {gnn_mean:.4f} ± {gnn_std:.4f}")
    print(f"Improvement: +{(gnn_mean - baseline_mean) / baseline_mean * 100:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Full Pipeline HPO')
    parser.add_argument('--graph_dir', type=str,
                        default='data/cache/gnn/rebuild_20260120',
                        help='Path to graph cache directory')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/hpo/full_pipeline',
                        help='Output directory for results')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of HPO trials')
    parser.add_argument('--study_name', type=str, default='full_pipeline_hpo',
                        help='Optuna study name')
    parser.add_argument('--evaluate_only', type=str, default=None,
                        help='Path to best_params.json to evaluate (skip HPO)')

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_only:
        # Load and evaluate existing config
        with open(args.evaluate_only) as f:
            data = json.load(f)
        config = data['best_params']
        evaluate_best_config(graph_dir, config, output_dir)
    else:
        # Run HPO
        results = run_hpo(
            graph_dir=graph_dir,
            output_dir=output_dir,
            n_trials=args.n_trials,
            study_name=args.study_name,
        )

        # Evaluate best config
        evaluate_best_config(graph_dir, results['best_params'], output_dir)


if __name__ == '__main__':
    main()
