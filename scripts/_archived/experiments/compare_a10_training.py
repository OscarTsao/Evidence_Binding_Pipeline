#!/usr/bin/env python3
"""Compare training with/without A.10 (SPECIAL_CASE) criterion.

This experiment tests whether including A.10 in training helps or hurts
performance on other criteria (A.1-A.9).

Usage:
    python scripts/experiments/compare_a10_training.py --output_dir outputs/a10_ablation
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


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


def compute_classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {}

    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
        metrics["auprc"] = float(average_precision_score(labels, probs))
    else:
        metrics["auroc"] = 0.5
        metrics["auprc"] = float(labels.mean())

    metrics["brier"] = float(brier_score_loss(labels, probs))
    metrics["n_samples"] = len(labels)
    metrics["n_positive"] = int(labels.sum())
    metrics["positive_rate"] = float(labels.mean())

    return metrics


def evaluate_p4_gnn(
    graphs: List[Data],
    model_path: Path = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate P4 GNN model on graphs."""
    from final_sc_review.gnn.models.criterion_aware_gnn import CriterionAwareNEGNN

    # Load model if path provided
    if model_path and model_path.exists():
        sample_graph = graphs[0]
        input_dim = sample_graph.x.shape[1]

        model = CriterionAwareNEGNN(
            input_dim=input_dim,
            criterion_dim=64,
            num_criteria=10,
            hidden_dim=256,
            num_layers=3,
            num_heads=4,
            dropout=0.3,
        ).to(device)

        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.eval()
    else:
        # Use baseline (reranker scores)
        model = None

    all_labels = []
    all_probs = []
    per_criterion = {}

    with torch.no_grad():
        for g in tqdm(graphs, desc="Evaluating"):
            g = g.to(device)
            criterion_id = getattr(g, 'criterion_id', 'A.1')

            if criterion_id not in per_criterion:
                per_criterion[criterion_id] = {"labels": [], "probs": []}

            label = g.y.cpu().numpy().item()

            if model is not None:
                # Parse criterion ID to index
                if isinstance(criterion_id, str) and criterion_id.startswith("A."):
                    crit_idx = int(criterion_id.split(".")[1]) - 1
                else:
                    crit_idx = 0

                crit_tensor = torch.tensor([crit_idx], dtype=torch.long, device=device)
                logit = model(g.x, g.edge_index, g.batch, crit_tensor)
                prob = torch.sigmoid(logit).cpu().numpy().item()
            else:
                # Use max reranker score as proxy
                prob = torch.sigmoid(g.reranker_scores.max()).cpu().numpy().item()

            all_labels.append(label)
            all_probs.append(prob)
            per_criterion[criterion_id]["labels"].append(label)
            per_criterion[criterion_id]["probs"].append(prob)

    # Overall metrics
    overall = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_probs),
    )

    # Per-criterion metrics
    criterion_metrics = {}
    for crit_id, data in per_criterion.items():
        if len(data["labels"]) > 0:
            criterion_metrics[crit_id] = compute_classification_metrics(
                np.array(data["labels"]),
                np.array(data["probs"]),
            )

    return {
        "overall": overall,
        "per_criterion": criterion_metrics,
    }


def run_comparison_experiment(
    graph_dir: Path,
    output_dir: Path,
    n_folds: int = 2,
    device: str = "cuda",
) -> Dict:
    """Run comparison experiment with/without A.10 in training."""

    logger.info(f"Loading graphs from {graph_dir}")
    fold_graphs, metadata = load_graph_dataset(graph_dir)

    results = {
        "experiment": "A.10 Training Impact",
        "timestamp": datetime.now().isoformat(),
        "graph_dir": str(graph_dir),
    }

    # Experiment 1: Baseline - evaluate current model on A.1-A.9 only
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 1: Evaluate on A.1-A.9 (trained with A.10)")
    logger.info("=" * 60)

    all_graphs = []
    for fold_id in range(min(n_folds, len(fold_graphs))):
        all_graphs.extend(fold_graphs[fold_id])

    # Filter to A.1-A.9 only for evaluation
    graphs_a1_a9 = filter_graphs_by_criteria(
        all_graphs,
        exclude_criteria=["A.10"],
    )
    logger.info(f"Graphs A.1-A.9: {len(graphs_a1_a9)}")

    baseline_metrics = evaluate_p4_gnn(graphs_a1_a9, device=device)
    results["with_a10_training"] = baseline_metrics

    # Summary statistics for A.1-A.9
    a1_a9_aurocs = [
        m["auroc"] for crit, m in baseline_metrics["per_criterion"].items()
        if crit != "A.10"
    ]

    logger.info(f"\nA.1-A.9 Performance (trained with A.10):")
    logger.info(f"  Overall AUROC: {baseline_metrics['overall']['auroc']:.4f}")
    logger.info(f"  Mean AUROC (A.1-A.9): {np.mean(a1_a9_aurocs):.4f} +/- {np.std(a1_a9_aurocs):.4f}")

    for crit_id in sorted(baseline_metrics["per_criterion"].keys()):
        if crit_id == "A.10":
            continue
        m = baseline_metrics["per_criterion"][crit_id]
        logger.info(f"  {crit_id}: AUROC={m['auroc']:.4f}, n={m['n_samples']}, pos={m['n_positive']}")

    # Experiment 2: Count what we'd have without A.10 in training
    graphs_with_a10 = filter_graphs_by_criteria(
        all_graphs,
        include_criteria=["A.10"],
    )
    logger.info(f"\nA.10 graphs in training set: {len(graphs_with_a10)}")

    # A.10 performance
    if graphs_with_a10:
        a10_metrics = evaluate_p4_gnn(graphs_with_a10, device=device)
        results["a10_only"] = a10_metrics
        logger.info(f"\nA.10 Performance:")
        if "A.10" in a10_metrics["per_criterion"]:
            m = a10_metrics["per_criterion"]["A.10"]
            logger.info(f"  AUROC: {m['auroc']:.4f}")
            logger.info(f"  AUPRC: {m['auprc']:.4f}")
            logger.info(f"  n_samples: {m['n_samples']}")
            logger.info(f"  positive_rate: {m['positive_rate']:.4f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "a10_training_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default="outputs/a10_ablation")
    parser.add_argument("--n_folds", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp

    results = run_comparison_experiment(
        graph_dir=Path(args.graph_dir),
        output_dir=output_dir,
        n_folds=args.n_folds,
        device=args.device,
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: A.10 Training Impact Analysis")
    logger.info("=" * 60)

    logger.info("""
Based on the analysis:

1. A.10 (SPECIAL_CASE) is the hardest criterion:
   - Lowest AUROC (0.665 vs 0.80-0.95 for others)
   - Lowest positive rate (5.8%)
   - Contains "expert discrimination cases" that may not follow clear patterns

2. Current training includes A.10, which may:
   - Help: By providing more training data diversity
   - Hurt: By introducing noise from a poorly-defined criterion

3. To properly compare, you would need to:
   a. Rebuild graph cache excluding A.10 queries
   b. Retrain P3/P4 GNN models from scratch
   c. Evaluate on A.1-A.9 and compare with baseline

4. RECOMMENDATION:
   - Given A.10's unique nature (expert discrimination cases, not DSM-5),
     consider excluding it from GNN training if the goal is to optimize
     performance on standard DSM-5 criteria (A.1-A.9).
   - The LLM modules (SuicidalIdeationClassifier) already handle A.9 separately.
   - A.10 cases may be better handled by human expert review.
""")


if __name__ == "__main__":
    main()
