#!/usr/bin/env python3
"""
Optimized GNN HPO: Maximizes GPU utilization with batched training.

Uses DataLoader with proper device handling for efficient batch processing.
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
            # Deep clone to ensure all on CPU
            g_cpu = deep_clone_to_cpu(g)
            if exclude_criteria:
                if getattr(g_cpu, 'criterion_id', None) in exclude_criteria:
                    continue
            processed.append(g_cpu)

        print(f"Fold {fold_id}: {len(processed)} graphs")
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

        # Per-graph loss with batch indices
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

                # Sample pairs for efficiency
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
    recall_scores = []

    with torch.no_grad():
        for graph in val_graphs:
            # Clone to avoid modifying original (to() modifies in place)
            graph_gpu = graph.clone().to(device)
            scores = model(graph_gpu).cpu().numpy()
            gold_mask = graph.node_labels.cpu().numpy()

            if gold_mask.sum() > 0:
                ndcg = compute_ndcg_at_k(gold_mask, scores, top_k)
                mrr = compute_mrr(gold_mask, scores)
                recall = compute_recall_at_k(gold_mask, scores, top_k)

                ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)
                recall_scores.append(recall)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'recall@10': np.mean(recall_scores) if recall_scores else 0.0,
        'n_queries': len(ndcg_scores),
    }


def evaluate_baseline(val_graphs: List[Data], top_k: int = 10) -> Dict[str, float]:
    """Evaluate baseline (original reranker scores)."""
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for graph in val_graphs:
        scores = graph.reranker_scores.cpu().numpy()
        gold_mask = graph.node_labels.cpu().numpy()

        if gold_mask.sum() > 0:
            ndcg = compute_ndcg_at_k(gold_mask, scores, top_k)
            mrr = compute_mrr(gold_mask, scores)
            recall = compute_recall_at_k(gold_mask, scores, top_k)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            recall_scores.append(recall)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'recall@10': np.mean(recall_scores) if recall_scores else 0.0,
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
    fold_graphs: Dict[int, List[Data]],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Optuna objective with batched training for GPU efficiency."""

    # Hyperparameter search space
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
    alpha_init = trial.suggest_float('alpha_init', 0.1, 0.95, step=0.05)
    learn_alpha = trial.suggest_categorical('learn_alpha', [True, False])

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    n_epochs = trial.suggest_int('n_epochs', 3, 15, step=3)  # Reduced for faster trials
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    fold_ndcgs = []
    input_dim = get_input_dim(list(fold_graphs.values())[0])

    for val_fold in range(5):
        train_graphs = []
        val_graphs = []

        for fold_id, graphs in fold_graphs.items():
            if fold_id == val_fold:
                val_graphs.extend(graphs)
            else:
                train_graphs.extend(graphs)

        if not train_graphs or not val_graphs:
            continue

        # Create DataLoader with batching
        train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
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

        # Training with batched DataLoader
        for epoch in range(n_epochs):
            train_epoch_batched(model, train_loader, optimizer, device)

        # Evaluate
        metrics = evaluate(model, val_graphs, device)
        fold_ndcgs.append(metrics['ndcg@10'])

        # Report intermediate value for pruning
        trial.report(np.mean(fold_ndcgs), val_fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_ndcgs) if fold_ndcgs else 0.0


def run_full_evaluation(
    best_params: Dict,
    fold_graphs: Dict[int, List[Data]],
    device: torch.device,
    output_dir: Path,
    batch_size: int = 64,
) -> Dict:
    """Run full 5-fold evaluation with best params."""
    print("\n=== Full 5-Fold Evaluation with Best Params ===")
    print(f"Params: {json.dumps(best_params, indent=2)}")

    input_dim = get_input_dim(list(fold_graphs.values())[0])
    fold_results = []

    for val_fold in range(5):
        train_graphs = []
        val_graphs = []

        for fold_id, graphs in fold_graphs.items():
            if fold_id == val_fold:
                val_graphs.extend(graphs)
            else:
                train_graphs.extend(graphs)

        if not train_graphs or not val_graphs:
            continue

        baseline_metrics = evaluate_baseline(val_graphs)

        train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        model = P3GraphReranker(
            input_dim=input_dim,
            hidden_dim=best_params['hidden_dim'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            alpha_init=best_params['alpha_init'],
            learn_alpha=best_params['learn_alpha'],
        ).to(device)

        weight_decay = best_params.get('weight_decay', 1e-4)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=weight_decay
        )

        print(f"\nFold {val_fold}: Training on {len(train_graphs)} graphs...")
        for epoch in tqdm(range(best_params['n_epochs']), desc=f"Fold {val_fold}"):
            train_epoch_batched(model, train_loader, optimizer, device)

        gnn_metrics = evaluate(model, val_graphs, device)

        fold_results.append({
            'fold_id': val_fold,
            'n_train': len(train_graphs),
            'n_val': len(val_graphs),
            'baseline_ndcg@10': baseline_metrics['ndcg@10'],
            'baseline_mrr': baseline_metrics['mrr'],
            'baseline_recall@10': baseline_metrics['recall@10'],
            'gnn_ndcg@10': gnn_metrics['ndcg@10'],
            'gnn_mrr': gnn_metrics['mrr'],
            'gnn_recall@10': gnn_metrics['recall@10'],
            'improvement': gnn_metrics['ndcg@10'] - baseline_metrics['ndcg@10'],
            'n_queries': gnn_metrics['n_queries'],
        })

        print(f"Fold {val_fold}: Baseline={baseline_metrics['ndcg@10']:.4f}, "
              f"GNN={gnn_metrics['ndcg@10']:.4f}, Δ={gnn_metrics['ndcg@10'] - baseline_metrics['ndcg@10']:+.4f}")

    # Aggregate
    baseline_ndcg_mean = np.mean([r['baseline_ndcg@10'] for r in fold_results])
    baseline_ndcg_std = np.std([r['baseline_ndcg@10'] for r in fold_results])
    gnn_ndcg_mean = np.mean([r['gnn_ndcg@10'] for r in fold_results])
    gnn_ndcg_std = np.std([r['gnn_ndcg@10'] for r in fold_results])

    baseline_mrr_mean = np.mean([r['baseline_mrr'] for r in fold_results])
    gnn_mrr_mean = np.mean([r['gnn_mrr'] for r in fold_results])

    baseline_recall_mean = np.mean([r['baseline_recall@10'] for r in fold_results])
    gnn_recall_mean = np.mean([r['gnn_recall@10'] for r in fold_results])

    summary = {
        'best_params': best_params,
        'baseline': {
            'ndcg@10': {'mean': baseline_ndcg_mean, 'std': baseline_ndcg_std},
            'mrr': {'mean': baseline_mrr_mean},
            'recall@10': {'mean': baseline_recall_mean},
        },
        'gnn': {
            'ndcg@10': {'mean': gnn_ndcg_mean, 'std': gnn_ndcg_std},
            'mrr': {'mean': gnn_mrr_mean},
            'recall@10': {'mean': gnn_recall_mean},
        },
        'improvement': {
            'ndcg@10': {
                'absolute': gnn_ndcg_mean - baseline_ndcg_mean,
                'relative_pct': (gnn_ndcg_mean - baseline_ndcg_mean) / baseline_ndcg_mean * 100,
            },
            'mrr': {
                'absolute': gnn_mrr_mean - baseline_mrr_mean,
                'relative_pct': (gnn_mrr_mean - baseline_mrr_mean) / baseline_mrr_mean * 100,
            },
        },
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat(),
        'training_protocol': 'batched_full_data',
        'batch_size': batch_size,
    }

    output_path = output_dir / 'best_config_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"=== FINAL RESULTS ===")
    print(f"{'='*60}")
    print(f"\nBaseline: nDCG@10 = {baseline_ndcg_mean:.4f} ± {baseline_ndcg_std:.4f}")
    print(f"GNN:      nDCG@10 = {gnn_ndcg_mean:.4f} ± {gnn_ndcg_std:.4f}")
    print(f"Improvement: +{(gnn_ndcg_mean - baseline_ndcg_mean) / baseline_ndcg_mean * 100:.2f}%")
    print(f"\nResults saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Optimized GNN HPO (batched training)')
    parser.add_argument('--graph_dir', type=str,
                        default='data/cache/gnn/rebuild_20260120')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/hpo/gnn_optimized')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (higher = more GPU usage)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds')
    parser.add_argument('--evaluate_only', type=str, default=None)

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n=== Loading Graphs ===")
    fold_graphs, metadata = load_graph_dataset(graph_dir)

    total_graphs = sum(len(g) for g in fold_graphs.values())
    print(f"\nTotal graphs: {total_graphs}")
    print(f"Batch size: {args.batch_size}")

    if args.evaluate_only:
        with open(args.evaluate_only) as f:
            data = json.load(f)
        best_params = data.get('best_params', data)
        run_full_evaluation(best_params, fold_graphs, device, output_dir, args.batch_size)
    else:
        print("\n" + "="*60)
        print("=== OPTIMIZED GNN HPO (Batched Training) ===")
        print("="*60)
        print(f"\nTrials: {args.n_trials}")
        print(f"Batch size: {args.batch_size}")
        print(f"\nSearch space:")
        print(f"  - hidden_dim: [32, 64, 128, 256, 512]")
        print(f"  - num_layers: [1, 2, 3, 4]")
        print(f"  - dropout: [0.0, ..., 0.5]")
        print(f"  - alpha_init: [0.1, ..., 0.95]")
        print(f"  - learn_alpha: [True, False]")
        print(f"  - lr: [1e-5, 1e-2]")
        print(f"  - n_epochs: [3, 6, 9, 12, 15]")
        print(f"  - weight_decay: [1e-6, 1e-3]")

        sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=2)

        study = optuna.create_study(
            study_name="gnn_hpo_optimized",
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            lambda trial: objective(trial, fold_graphs, device, args.batch_size),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'timestamp': datetime.now().isoformat(),
            'batch_size': args.batch_size,
        }

        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(results, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_dir / 'trials.csv', index=False)

        print(f"\n{'='*60}")
        print(f"=== HPO COMPLETE ===")
        print(f"{'='*60}")
        print(f"Total trials: {len(study.trials)}")
        print(f"Best nDCG@10: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        run_full_evaluation(study.best_params, fold_graphs, device, output_dir, args.batch_size)


if __name__ == '__main__':
    main()
