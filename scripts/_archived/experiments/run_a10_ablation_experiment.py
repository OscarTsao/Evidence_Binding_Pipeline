#!/usr/bin/env python3
"""Run A.10 ablation experiment: Compare training with/without A.10.

This experiment tests whether including A.10 (SPECIAL_CASE) in training
helps or hurts performance on criteria A.1-A.9.

Usage:
    python scripts/experiments/run_a10_ablation_experiment.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/a10_ablation_experiment
"""

import argparse
import json
import logging
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading and Filtering
# ============================================================================

def load_graph_dataset(graph_dir: Path) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        fold_graphs[fold_id] = data["graphs"]
        logger.info(f"Loaded fold {fold_id}: {len(data['graphs'])} graphs")

    return fold_graphs, metadata


def filter_graphs_by_criteria(
    graphs: List[Data],
    exclude_criteria: List[str] = None,
    include_criteria: List[str] = None,
) -> List[Data]:
    """Filter graphs by criterion ID."""
    filtered = []
    for g in graphs:
        criterion = getattr(g, 'criterion_id', None)
        if criterion is None:
            continue

        if exclude_criteria and criterion in exclude_criteria:
            continue
        if include_criteria and criterion not in include_criteria:
            continue

        filtered.append(g)

    return filtered


def get_criterion_distribution(graphs: List[Data]) -> Dict[str, int]:
    """Get distribution of criteria in graphs."""
    dist = {}
    for g in graphs:
        crit = getattr(g, 'criterion_id', 'unknown')
        dist[crit] = dist.get(crit, 0) + 1
    return dict(sorted(dist.items()))


# ============================================================================
# P3 Graph Reranker Model (simplified for this experiment)
# ============================================================================

class SimpleGraphReranker(nn.Module):
    """Simplified P3 Graph Reranker for ablation experiments."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        alpha_init: float = 0.7,
    ):
        super().__init__()

        from torch_geometric.nn import GATConv, global_mean_pool

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, reranker_scores, batch=None):
        # GNN encoding
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)

        # Score prediction
        gnn_scores = self.score_head(h).squeeze(-1)

        # Blend with original scores
        alpha = torch.sigmoid(self.alpha)
        refined = alpha * reranker_scores + (1 - alpha) * gnn_scores

        return refined


# ============================================================================
# Training and Evaluation
# ============================================================================

def compute_ranking_metrics(gold_mask: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute ranking metrics."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    metrics = {}

    # MRR
    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0
    metrics["mrr"] = mrr

    # nDCG@K and Recall@K
    for k in [1, 3, 5, 10]:
        # nDCG
        dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(sorted_gold))) if sorted_gold[i])
        n_gold = int(gold_mask.sum())
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_gold)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

        # Recall
        recall = sorted_gold[:k].sum() / n_gold if n_gold > 0 else 0.0
        metrics[f"recall@{k}"] = recall

    return metrics


def ensure_cpu(graph):
    """Ensure all tensors in graph are on CPU."""
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph


def evaluate_model(
    model: Optional[nn.Module],
    graphs: List[Data],
    device: str = "cuda",
    use_refined: bool = True,
    eval_criteria: List[str] = None,
) -> Dict:
    """Evaluate model on graphs, optionally filtered by criteria."""
    if model is not None:
        model.eval()

    all_metrics = []
    per_criterion = {}

    with torch.no_grad():
        for g in graphs:
            criterion = getattr(g, 'criterion_id', None)

            # Filter by evaluation criteria
            if eval_criteria and criterion not in eval_criteria:
                continue

            g = ensure_cpu(g)
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy() > 0.5

            if not gold_mask.any():
                continue

            if use_refined and model is not None:
                refined = model(g.x, g.edge_index, g.reranker_scores)
                scores = refined.cpu().numpy()
            else:
                scores = g.reranker_scores.cpu().numpy()

            metrics = compute_ranking_metrics(gold_mask, scores)
            all_metrics.append(metrics)

            # Per-criterion tracking
            if criterion not in per_criterion:
                per_criterion[criterion] = []
            per_criterion[criterion].append(metrics)

    if not all_metrics:
        return {"overall": {}, "per_criterion": {}}

    # Aggregate overall metrics
    overall = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        overall[key] = float(np.mean(values))
        overall[f"{key}_std"] = float(np.std(values))
    overall["n_queries"] = len(all_metrics)

    # Aggregate per-criterion metrics
    criterion_agg = {}
    for crit, metrics_list in per_criterion.items():
        if metrics_list:
            crit_agg = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list]
                crit_agg[key] = float(np.mean(values))
            crit_agg["n_queries"] = len(metrics_list)
            criterion_agg[crit] = crit_agg

    return {"overall": overall, "per_criterion": criterion_agg}


def train_model(
    model: nn.Module,
    train_graphs: List[Data],
    val_graphs: List[Data],
    config: Dict,
    device: str = "cuda",
    eval_criteria: List[str] = None,
) -> Tuple[Dict, float]:
    """Train model and return best state."""
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    # Only train on graphs with evidence
    train_pos = [ensure_cpu(g) for g in train_graphs if g.y.item() > 0]
    val_pos = [ensure_cpu(g) for g in val_graphs if g.y.item() > 0]

    logger.info(f"  Training: {len(train_pos)} graphs with evidence")
    logger.info(f"  Validation: {len(val_pos)} graphs with evidence")

    train_loader = DataLoader(train_pos, batch_size=config["batch_size"], shuffle=True)

    best_ndcg = -float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)

            # Pairwise ranking loss
            gold = batch.node_labels > 0.5
            pos_mask = gold.float()
            neg_mask = (~gold).float()

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_scores = (refined * pos_mask).sum() / pos_mask.sum()
                neg_scores = (refined * neg_mask).sum() / neg_mask.sum()
                loss = torch.relu(config["margin"] - pos_scores + neg_scores)
            else:
                loss = torch.tensor(0.0, device=device)

            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate on validation (filtered to eval_criteria)
        val_metrics = evaluate_model(model, val_pos, device, use_refined=True, eval_criteria=eval_criteria)
        val_ndcg = val_metrics["overall"].get("ndcg@10", 0)

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.4f}, val_ndcg@10={val_ndcg:.4f}")

        if patience_counter >= config["patience"]:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    return best_state, best_ndcg


def run_experiment(
    fold_graphs: Dict[int, List],
    config: Dict,
    output_dir: Path,
    device: str = "cuda",
    n_folds: int = 2,
) -> Dict:
    """Run the full ablation experiment."""

    results = {
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
    }

    # Criteria definitions
    ALL_CRITERIA = [f"A.{i}" for i in range(1, 11)]
    A1_A9 = [f"A.{i}" for i in range(1, 10)]

    # Get input dimension
    sample_graph = fold_graphs[0][0]
    input_dim = sample_graph.x.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    # ========================================================================
    # Experiment 1: Baseline (original reranker scores, no GNN)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Baseline (Jina-Reranker-v3 scores, no GNN)")
    logger.info("=" * 70)

    all_graphs = []
    for fid in range(min(n_folds, len(fold_graphs))):
        all_graphs.extend(fold_graphs[fid])

    # Evaluate baseline on A.1-A.9 only
    graphs_a1_a9 = filter_graphs_by_criteria(all_graphs, exclude_criteria=["A.10"])
    baseline_metrics = evaluate_model(None, graphs_a1_a9, device, use_refined=False, eval_criteria=A1_A9)

    results["experiments"]["baseline_jina"] = {
        "description": "Jina-Reranker-v3 baseline (no GNN)",
        "train_criteria": "N/A",
        "eval_criteria": A1_A9,
        "n_eval_queries": baseline_metrics["overall"].get("n_queries", 0),
        "metrics": baseline_metrics,
    }

    logger.info(f"Baseline nDCG@10 (A.1-A.9): {baseline_metrics['overall'].get('ndcg@10', 0):.4f}")

    # ========================================================================
    # Experiment 2: Train with ALL criteria (including A.10), eval on A.1-A.9
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: Train WITH A.10, evaluate on A.1-A.9")
    logger.info("=" * 70)

    fold_results_with_a10 = []

    for fold_id in range(min(n_folds, len(fold_graphs))):
        logger.info(f"\n--- Fold {fold_id} (train WITH A.10) ---")

        # Split data
        val_graphs = fold_graphs[fold_id]
        train_graphs = []
        for fid in range(len(fold_graphs)):
            if fid != fold_id:
                train_graphs.extend(fold_graphs[fid])

        # Log distribution
        train_dist = get_criterion_distribution(train_graphs)
        logger.info(f"  Train distribution: {train_dist}")

        # Initialize model
        model = SimpleGraphReranker(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            alpha_init=config["alpha_init"],
        )

        # Train (with all criteria)
        best_state, best_ndcg = train_model(
            model, train_graphs, val_graphs, config, device,
            eval_criteria=A1_A9,  # Evaluate only on A.1-A.9
        )

        # Load best and evaluate
        model.load_state_dict(best_state)
        val_graphs_a1_a9 = filter_graphs_by_criteria(val_graphs, exclude_criteria=["A.10"])
        val_pos = [g for g in val_graphs_a1_a9 if g.y.item() > 0]

        fold_metrics = evaluate_model(model, val_pos, device, use_refined=True, eval_criteria=A1_A9)
        fold_results_with_a10.append(fold_metrics)

        logger.info(f"  Fold {fold_id} nDCG@10 (A.1-A.9): {fold_metrics['overall'].get('ndcg@10', 0):.4f}")

    # Aggregate
    agg_with_a10 = aggregate_fold_results(fold_results_with_a10)
    results["experiments"]["train_with_a10"] = {
        "description": "Train with ALL criteria (including A.10), evaluate on A.1-A.9",
        "train_criteria": ALL_CRITERIA,
        "eval_criteria": A1_A9,
        "fold_results": [f["overall"] for f in fold_results_with_a10],
        "aggregated": agg_with_a10,
    }

    logger.info(f"\nAggregated (train WITH A.10): nDCG@10 = {agg_with_a10.get('ndcg@10_mean', 0):.4f} +/- {agg_with_a10.get('ndcg@10_std', 0):.4f}")

    # ========================================================================
    # Experiment 3: Train WITHOUT A.10, eval on A.1-A.9
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: Train WITHOUT A.10, evaluate on A.1-A.9")
    logger.info("=" * 70)

    fold_results_without_a10 = []

    for fold_id in range(min(n_folds, len(fold_graphs))):
        logger.info(f"\n--- Fold {fold_id} (train WITHOUT A.10) ---")

        # Split data
        val_graphs = fold_graphs[fold_id]
        train_graphs = []
        for fid in range(len(fold_graphs)):
            if fid != fold_id:
                train_graphs.extend(fold_graphs[fid])

        # FILTER: Remove A.10 from training
        train_graphs_no_a10 = filter_graphs_by_criteria(train_graphs, exclude_criteria=["A.10"])

        train_dist = get_criterion_distribution(train_graphs_no_a10)
        logger.info(f"  Train distribution (no A.10): {train_dist}")

        # Initialize model
        model = SimpleGraphReranker(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            alpha_init=config["alpha_init"],
        )

        # Train (WITHOUT A.10)
        best_state, best_ndcg = train_model(
            model, train_graphs_no_a10, val_graphs, config, device,
            eval_criteria=A1_A9,
        )

        # Load best and evaluate
        model.load_state_dict(best_state)
        val_graphs_a1_a9 = filter_graphs_by_criteria(val_graphs, exclude_criteria=["A.10"])
        val_pos = [g for g in val_graphs_a1_a9 if g.y.item() > 0]

        fold_metrics = evaluate_model(model, val_pos, device, use_refined=True, eval_criteria=A1_A9)
        fold_results_without_a10.append(fold_metrics)

        logger.info(f"  Fold {fold_id} nDCG@10 (A.1-A.9): {fold_metrics['overall'].get('ndcg@10', 0):.4f}")

    # Aggregate
    agg_without_a10 = aggregate_fold_results(fold_results_without_a10)
    results["experiments"]["train_without_a10"] = {
        "description": "Train WITHOUT A.10, evaluate on A.1-A.9",
        "train_criteria": A1_A9,
        "eval_criteria": A1_A9,
        "fold_results": [f["overall"] for f in fold_results_without_a10],
        "aggregated": agg_without_a10,
    }

    logger.info(f"\nAggregated (train WITHOUT A.10): nDCG@10 = {agg_without_a10.get('ndcg@10_mean', 0):.4f} +/- {agg_without_a10.get('ndcg@10_std', 0):.4f}")

    # ========================================================================
    # Summary Comparison
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: A.10 Training Impact")
    logger.info("=" * 70)

    baseline_ndcg = baseline_metrics["overall"].get("ndcg@10", 0)
    with_a10_ndcg = agg_with_a10.get("ndcg@10_mean", 0)
    without_a10_ndcg = agg_without_a10.get("ndcg@10_mean", 0)

    delta = without_a10_ndcg - with_a10_ndcg

    results["summary"] = {
        "baseline_jina_ndcg10": baseline_ndcg,
        "train_with_a10_ndcg10": with_a10_ndcg,
        "train_with_a10_ndcg10_std": agg_with_a10.get("ndcg@10_std", 0),
        "train_without_a10_ndcg10": without_a10_ndcg,
        "train_without_a10_ndcg10_std": agg_without_a10.get("ndcg@10_std", 0),
        "delta_removing_a10": delta,
        "conclusion": "REMOVING A.10 HELPS" if delta > 0 else "KEEPING A.10 HELPS" if delta < 0 else "NO DIFFERENCE",
    }

    logger.info(f"""
Results (evaluated on A.1-A.9 only):

  1. Baseline (Jina-v3):           nDCG@10 = {baseline_ndcg:.4f}
  2. GNN train WITH A.10:          nDCG@10 = {with_a10_ndcg:.4f} +/- {agg_with_a10.get('ndcg@10_std', 0):.4f}
  3. GNN train WITHOUT A.10:       nDCG@10 = {without_a10_ndcg:.4f} +/- {agg_without_a10.get('ndcg@10_std', 0):.4f}

  Delta (removing A.10):           {delta:+.4f}

  CONCLUSION: {results['summary']['conclusion']}
""")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "a10_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {results_path}")

    return results


def aggregate_fold_results(fold_results: List[Dict]) -> Dict:
    """Aggregate metrics across folds."""
    if not fold_results:
        return {}

    metrics_to_agg = ["mrr", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
                      "recall@1", "recall@3", "recall@5", "recall@10"]

    agg = {}
    for metric in metrics_to_agg:
        values = [f["overall"].get(metric, 0) for f in fold_results]
        agg[f"{metric}_mean"] = float(np.mean(values))
        agg[f"{metric}_std"] = float(np.std(values))

    agg["n_folds"] = len(fold_results)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default="outputs/a10_ablation_experiment")
    parser.add_argument("--n_folds", type=int, default=2, help="Number of folds to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Training config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_init", type=float, default=0.7)
    parser.add_argument("--margin", type=float, default=0.1)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp

    config = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha_init": args.alpha_init,
        "margin": args.margin,
    }

    logger.info(f"Loading graphs from {args.graph_dir}")
    fold_graphs, metadata = load_graph_dataset(Path(args.graph_dir))

    results = run_experiment(
        fold_graphs=fold_graphs,
        config=config,
        output_dir=output_dir,
        device=args.device,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
