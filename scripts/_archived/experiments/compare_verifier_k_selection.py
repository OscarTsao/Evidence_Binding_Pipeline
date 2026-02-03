#!/usr/bin/env python3
"""Compare fixed vs dynamic K-selection for LLM verifier.

This script evaluates different K-selection methods for the LLM verifier:
1. Fixed: top-3, top-5, top-7, top-10
2. Dynamic: score_gap (tau=0.3, 0.4, 0.5)

Metrics:
- Verifier accuracy (if has ground truth)
- nDCG@10 impact
- Token usage (proportional to K)
- Latency (proportional to K)

Usage:
    python scripts/experiments/compare_verifier_k_selection.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/k_selection_comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.postprocessing.dynamic_k import DynamicKSelector, analyze_score_distribution


def simulate_verifier_accuracy(
    scores: List[float],
    gold_labels: List[int],
    k: int,
) -> Dict:
    """Simulate verifier accuracy based on top-K selection.

    In practice, the LLM verifier looks at top-K candidates and decides
    if there's evidence. This simulates the accuracy based on whether
    the top-K contains any positive labels.

    Args:
        scores: Candidate scores (highest first)
        gold_labels: Ground truth labels (1=positive, 0=negative)
        k: Number of candidates to consider

    Returns:
        Dict with accuracy metrics
    """
    selected_labels = gold_labels[:k]
    has_positive = any(l > 0 for l in selected_labels)
    n_positives = sum(selected_labels)

    # Ground truth: does the post have any positive evidence?
    ground_truth_has_evidence = any(l > 0 for l in gold_labels)

    # Simulated verifier decision: if top-K has positives, likely to say "has evidence"
    # This is a simplification; real verifier uses LLM
    verifier_correct = has_positive == ground_truth_has_evidence

    return {
        "k": k,
        "has_positive_in_top_k": has_positive,
        "n_positives_in_top_k": n_positives,
        "ground_truth_has_evidence": ground_truth_has_evidence,
        "verifier_correct": verifier_correct,
    }


def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Compute nDCG@K."""
    import math
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]
    dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(sorted_gold))) if sorted_gold[i])
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg > 0 else 0.0


def run_fixed_k_comparison(
    graphs: List,
    fixed_ks: List[int],
) -> Dict:
    """Run comparison for fixed K values.

    Args:
        graphs: List of graph data objects
        fixed_ks: List of K values to test

    Returns:
        Dict with results for each K
    """
    results = {}

    for k in fixed_ks:
        accuracies = []
        token_usage = []  # Proportional to K
        coverage = []  # Fraction of positives in top-K

        for g in graphs:
            scores = g.reranker_scores.cpu().numpy()
            labels = g.node_labels.cpu().numpy()

            if labels.sum() == 0:
                continue

            metrics = simulate_verifier_accuracy(
                scores=scores.tolist(),
                gold_labels=labels.tolist(),
                k=min(k, len(scores)),
            )
            accuracies.append(metrics["verifier_correct"])
            token_usage.append(min(k, len(scores)))

            # Coverage: what fraction of positives are in top-K?
            n_pos_total = labels.sum()
            n_pos_in_k = labels[np.argsort(-scores)[:k]].sum()
            coverage.append(n_pos_in_k / n_pos_total if n_pos_total > 0 else 0)

        results[f"fixed_k={k}"] = {
            "k": k,
            "method": "fixed",
            "accuracy": np.mean(accuracies) if accuracies else 0,
            "accuracy_std": np.std(accuracies) if accuracies else 0,
            "avg_token_usage": np.mean(token_usage) if token_usage else 0,
            "positive_coverage": np.mean(coverage) if coverage else 0,
            "n_samples": len(accuracies),
        }

    return results


def run_dynamic_k_comparison(
    graphs: List,
    score_gap_ratios: List[float],
    min_k: int = 3,
    max_k: int = 8,
) -> Dict:
    """Run comparison for dynamic K selection.

    Args:
        graphs: List of graph data objects
        score_gap_ratios: List of score gap ratios to test
        min_k: Minimum K
        max_k: Maximum K

    Returns:
        Dict with results for each configuration
    """
    results = {}

    for tau in score_gap_ratios:
        selector = DynamicKSelector(
            method="score_gap",
            min_k=min_k,
            max_k=max_k,
            score_gap_ratio=tau,
        )

        accuracies = []
        selected_ks = []
        coverage = []

        for g in graphs:
            scores = g.reranker_scores.cpu().numpy()
            labels = g.node_labels.cpu().numpy()

            if labels.sum() == 0:
                continue

            # Select K dynamically
            result = selector.select_k(scores.tolist())
            k = result.selected_k

            metrics = simulate_verifier_accuracy(
                scores=scores.tolist(),
                gold_labels=labels.tolist(),
                k=k,
            )
            accuracies.append(metrics["verifier_correct"])
            selected_ks.append(k)

            # Coverage
            n_pos_total = labels.sum()
            n_pos_in_k = labels[np.argsort(-scores)[:k]].sum()
            coverage.append(n_pos_in_k / n_pos_total if n_pos_total > 0 else 0)

        results[f"dynamic_tau={tau}"] = {
            "tau": tau,
            "method": "dynamic_score_gap",
            "min_k": min_k,
            "max_k": max_k,
            "accuracy": np.mean(accuracies) if accuracies else 0,
            "accuracy_std": np.std(accuracies) if accuracies else 0,
            "avg_k": np.mean(selected_ks) if selected_ks else 0,
            "k_std": np.std(selected_ks) if selected_ks else 0,
            "k_distribution": {
                "min": min(selected_ks) if selected_ks else 0,
                "max": max(selected_ks) if selected_ks else 0,
                "median": np.median(selected_ks) if selected_ks else 0,
            },
            "positive_coverage": np.mean(coverage) if coverage else 0,
            "n_samples": len(accuracies),
        }

    return results


def analyze_score_distributions(graphs: List, output_dir: Path) -> Dict:
    """Analyze score distributions across all graphs.

    Args:
        graphs: List of graph data objects
        output_dir: Output directory for analysis

    Returns:
        Dict with distribution statistics
    """
    all_gaps = []
    all_max_scores = []
    all_score_ranges = []

    for g in graphs:
        scores = g.reranker_scores.cpu().numpy()
        if len(scores) < 2:
            continue

        analysis = analyze_score_distribution(scores.tolist())
        all_gaps.extend(analysis.get("gaps", []))
        all_max_scores.append(analysis["max"])
        all_score_ranges.append(analysis["range"])

    return {
        "n_graphs": len(graphs),
        "max_score_stats": {
            "mean": np.mean(all_max_scores),
            "std": np.std(all_max_scores),
            "min": np.min(all_max_scores),
            "max": np.max(all_max_scores),
        },
        "score_range_stats": {
            "mean": np.mean(all_score_ranges),
            "std": np.std(all_score_ranges),
        },
        "gap_stats": {
            "mean": np.mean(np.abs(all_gaps)),
            "std": np.std(np.abs(all_gaps)),
            "max": np.max(np.abs(all_gaps)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare verifier K-selection methods")
    parser.add_argument("--graph_dir", default="data/cache/gnn/rebuild_20260120",
                        help="Graph cache directory")
    parser.add_argument("--output_dir", default="outputs/k_selection_comparison",
                        help="Output directory")
    parser.add_argument("--fixed_ks", default="3,5,7,10", help="Comma-separated fixed K values")
    parser.add_argument("--dynamic_taus", default="0.3,0.4,0.5",
                        help="Comma-separated score gap ratios for dynamic K")
    parser.add_argument("--min_k", type=int, default=3, help="Minimum K for dynamic selection")
    parser.add_argument("--max_k", type=int, default=8, help="Maximum K for dynamic selection")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_ks = [int(k) for k in args.fixed_ks.split(",")]
    dynamic_taus = [float(t) for t in args.dynamic_taus.split(",")]

    print("=" * 70)
    print("VERIFIER K-SELECTION COMPARISON")
    print("=" * 70)
    print(f"Graph dir: {graph_dir}")
    print(f"Fixed Ks: {fixed_ks}")
    print(f"Dynamic taus: {dynamic_taus}")
    print(f"Dynamic K range: [{args.min_k}, {args.max_k}]")
    print()

    # Load graphs
    print("Loading graphs...")
    import torch
    from final_sc_review.constants import EXCLUDED_CRITERIA

    with open(graph_dir / "metadata.json") as f:
        metadata = json.load(f)

    all_graphs = []
    for fold_id in range(metadata["n_folds"]):
        data = torch.load(graph_dir / f"fold_{fold_id}.pt", weights_only=False)
        graphs = [g for g in data["graphs"]
                  if getattr(g, 'criterion_id', None) not in EXCLUDED_CRITERIA]
        graphs = [g for g in graphs if g.node_labels.sum() > 0]
        all_graphs.extend(graphs)
        print(f"  Fold {fold_id}: {len(graphs)} graphs")

    print(f"Total graphs with positives: {len(all_graphs)}")
    print()

    # Analyze score distributions
    print("Analyzing score distributions...")
    dist_analysis = analyze_score_distributions(all_graphs, output_dir)
    print(f"  Max score: {dist_analysis['max_score_stats']['mean']:.3f} ± "
          f"{dist_analysis['max_score_stats']['std']:.3f}")
    print(f"  Score range: {dist_analysis['score_range_stats']['mean']:.3f} ± "
          f"{dist_analysis['score_range_stats']['std']:.3f}")
    print(f"  Avg gap: {dist_analysis['gap_stats']['mean']:.4f}")
    print()

    # Run fixed K comparison
    print("Running fixed K comparison...")
    fixed_results = run_fixed_k_comparison(all_graphs, fixed_ks)

    print("\nFixed K Results:")
    print("-" * 70)
    print(f"{'K':>5} | {'Accuracy':>10} | {'Avg Tokens':>12} | {'Pos Coverage':>12}")
    print("-" * 70)
    for name, r in sorted(fixed_results.items(), key=lambda x: x[1]["k"]):
        print(f"{r['k']:>5} | {r['accuracy']*100:>9.2f}% | {r['avg_token_usage']:>12.1f} | "
              f"{r['positive_coverage']*100:>11.2f}%")
    print()

    # Run dynamic K comparison
    print("Running dynamic K comparison...")
    dynamic_results = run_dynamic_k_comparison(
        all_graphs, dynamic_taus, args.min_k, args.max_k
    )

    print("\nDynamic K Results:")
    print("-" * 70)
    print(f"{'Tau':>5} | {'Accuracy':>10} | {'Avg K':>8} | {'K Range':>12} | {'Pos Coverage':>12}")
    print("-" * 70)
    for name, r in sorted(dynamic_results.items(), key=lambda x: x[1]["tau"]):
        k_range = f"[{r['k_distribution']['min']}-{r['k_distribution']['max']}]"
        print(f"{r['tau']:>5.2f} | {r['accuracy']*100:>9.2f}% | {r['avg_k']:>8.2f} | "
              f"{k_range:>12} | {r['positive_coverage']*100:>11.2f}%")
    print()

    # Combine results
    all_results = {**fixed_results, **dynamic_results}

    # Find best configuration
    best_config = max(all_results.items(), key=lambda x: x[1]["accuracy"])
    print("=" * 70)
    print(f"BEST CONFIGURATION: {best_config[0]}")
    print(f"  Accuracy: {best_config[1]['accuracy']*100:.2f}%")
    if "avg_k" in best_config[1]:
        print(f"  Average K: {best_config[1]['avg_k']:.2f}")
    else:
        print(f"  Fixed K: {best_config[1]['k']}")
    print(f"  Positive Coverage: {best_config[1]['positive_coverage']*100:.2f}%")
    print("=" * 70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"k_selection_comparison_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "config": {
            "graph_dir": str(graph_dir),
            "fixed_ks": fixed_ks,
            "dynamic_taus": dynamic_taus,
            "min_k": args.min_k,
            "max_k": args.max_k,
        },
        "n_graphs": len(all_graphs),
        "score_distribution_analysis": dist_analysis,
        "fixed_results": fixed_results,
        "dynamic_results": dynamic_results,
        "best_config": {
            "name": best_config[0],
            **best_config[1],
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
