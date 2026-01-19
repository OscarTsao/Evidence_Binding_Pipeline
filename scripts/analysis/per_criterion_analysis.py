#!/usr/bin/env python3
"""
Analyze per-criterion performance for STEP 6: A.10 Criterion Study.

Uses per-query rankings from baseline evaluation to compute metrics
for each criterion (A.1 through A.10) and identify performance patterns.

Usage:
    python scripts/analysis/per_criterion_analysis.py \
        --per_query_rankings outputs/repro_baseline/nv_embed_v2/test_results_per_query_rankings.jsonl \
        --output outputs/analysis/per_criterion_analysis.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from final_sc_review.metrics.ranking import (
    recall_at_k,
    mrr_at_k,
    map_at_k,
    ndcg_at_k,
)
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_per_query_rankings(rankings_path: Path) -> List[Dict]:
    """Load per-query rankings from JSONL file."""
    rankings = []
    with open(rankings_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rankings.append(json.loads(line))
    return rankings


def extract_criterion_from_query_id(query_id: str) -> str:
    """Extract criterion ID from query_id (format: post_id_criterion)."""
    parts = query_id.split("_")
    # Criterion is usually the last part (e.g., A.1, A.2, etc.)
    return parts[-1]


def compute_metrics_for_criterion(
    rankings: List[Dict],
    k_values: List[int] = [1, 3, 5, 10, 20],
) -> Dict:
    """Compute metrics for a set of rankings."""
    if not rankings:
        return {}

    metrics = {}

    for k in k_values:
        recall_scores = []
        mrr_scores = []
        map_scores = []
        ndcg_scores = []

        for ranking in rankings:
            ranked_uids = ranking["ranked_uids"]
            gold_uids = set(ranking["gold_uids"])

            if len(gold_uids) == 0:
                continue  # Skip queries with no positives

            # Create relevance array
            relevance = np.array([1 if uid in gold_uids else 0 for uid in ranked_uids])

            # Create scores (descending order: highest score = rank 1)
            scores = np.arange(len(ranked_uids), 0, -1, dtype=float)

            # Compute metrics
            recall_scores.append(recall_at_k(relevance, scores, k))
            mrr_scores.append(mrr_at_k(relevance, scores, k))
            map_scores.append(map_at_k(relevance, scores, k))
            ndcg_scores.append(ndcg_at_k(relevance, scores, k))

        # Aggregate
        if recall_scores:
            metrics[f"recall@{k}"] = {
                "mean": float(np.mean(recall_scores)),
                "std": float(np.std(recall_scores)),
                "min": float(np.min(recall_scores)),
                "max": float(np.max(recall_scores)),
            }
            metrics[f"mrr@{k}"] = {
                "mean": float(np.mean(mrr_scores)),
                "std": float(np.std(mrr_scores)),
            }
            metrics[f"map@{k}"] = {
                "mean": float(np.mean(map_scores)),
                "std": float(np.std(map_scores)),
            }
            metrics[f"ndcg@{k}"] = {
                "mean": float(np.mean(ndcg_scores)),
                "std": float(np.std(ndcg_scores)),
            }

    return metrics


def analyze_per_criterion(
    rankings: List[Dict],
    k_values: List[int] = [1, 3, 5, 10, 20],
) -> Dict:
    """Analyze performance per criterion."""
    logger.info("Analyzing per-criterion performance...")

    # Group rankings by criterion
    criterion_rankings = defaultdict(list)
    for ranking in rankings:
        query_id = ranking["query_id"]
        criterion = extract_criterion_from_query_id(query_id)
        criterion_rankings[criterion].append(ranking)

    logger.info(f"Found {len(criterion_rankings)} criteria")

    # Compute metrics per criterion
    results = {}
    for criterion, crit_rankings in sorted(criterion_rankings.items()):
        logger.info(f"  Analyzing {criterion}: {len(crit_rankings)} queries")

        # Count queries with/without positives
        queries_total = len(crit_rankings)
        queries_with_positives = sum(1 for r in crit_rankings if len(r["gold_uids"]) > 0)

        # Compute metrics
        metrics = compute_metrics_for_criterion(crit_rankings, k_values)

        # Compute baseline rate (fraction with evidence)
        baseline_rate = queries_with_positives / queries_total if queries_total > 0 else 0.0

        results[criterion] = {
            "n_queries_total": queries_total,
            "n_queries_with_positives": queries_with_positives,
            "baseline_rate": baseline_rate,
            "metrics": metrics,
        }

    return results


def rank_criteria_by_difficulty(results: Dict) -> List[Tuple[str, float]]:
    """Rank criteria by difficulty (lower nDCG@10 = harder)."""
    criterion_ndcg = []
    for criterion, data in results.items():
        ndcg_10 = data["metrics"].get("ndcg@10", {}).get("mean", 0.0)
        criterion_ndcg.append((criterion, ndcg_10))

    # Sort by nDCG@10 (ascending = hardest first)
    return sorted(criterion_ndcg, key=lambda x: x[1])


def identify_a10_challenges(results: Dict) -> Dict:
    """Identify specific challenges for criterion A.10."""
    if "A.10" not in results:
        return {"error": "A.10 not found in results"}

    a10_data = results["A.10"]

    # Compare A.10 with average of other criteria
    other_criteria = [c for c in results.keys() if c != "A.10"]

    if not other_criteria:
        return {"error": "No other criteria to compare"}

    # Compute average metrics across other criteria
    avg_metrics = {}
    for k in [1, 3, 5, 10, 20]:
        for metric_family in ["recall", "mrr", "map", "ndcg"]:
            metric_name = f"{metric_family}@{k}"
            values = [
                results[c]["metrics"].get(metric_name, {}).get("mean", 0.0)
                for c in other_criteria
            ]
            avg_metrics[metric_name] = np.mean(values) if values else 0.0

    # Compute differences
    differences = {}
    for metric_name, avg_value in avg_metrics.items():
        a10_value = a10_data["metrics"].get(metric_name, {}).get("mean", 0.0)
        differences[metric_name] = {
            "a10": a10_value,
            "others_avg": avg_value,
            "diff": a10_value - avg_value,
            "relative_diff": (a10_value - avg_value) / avg_value if avg_value > 0 else 0.0,
        }

    # Identify if A.10 is significantly harder
    ndcg_10_diff = differences.get("ndcg@10", {}).get("diff", 0.0)
    is_harder = ndcg_10_diff < -0.05  # More than 5% lower

    return {
        "a10_metrics": a10_data["metrics"],
        "others_avg_metrics": avg_metrics,
        "differences": differences,
        "is_harder": bool(is_harder),
        "hardness_level": "significantly harder" if ndcg_10_diff < -0.10 else "moderately harder" if is_harder else "comparable",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-criterion performance (STEP 6)"
    )
    parser.add_argument(
        "--per_query_rankings",
        type=Path,
        required=True,
        help="Path to per-query rankings JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for analysis JSON",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="K values for metrics",
    )

    args = parser.parse_args()

    # Load rankings
    logger.info(f"Loading per-query rankings from {args.per_query_rankings}")
    rankings = load_per_query_rankings(args.per_query_rankings)
    logger.info(f"  Loaded {len(rankings)} query rankings")

    # Analyze per criterion
    per_criterion_results = analyze_per_criterion(rankings, k_values=args.ks)

    # Rank by difficulty
    logger.info("\nRanking criteria by difficulty...")
    ranked = rank_criteria_by_difficulty(per_criterion_results)

    print("\n" + "=" * 60)
    print("CRITERIA RANKED BY DIFFICULTY (hardest first)")
    print("=" * 60)
    for rank, (criterion, ndcg) in enumerate(ranked, 1):
        n_queries = per_criterion_results[criterion]["n_queries_with_positives"]
        print(f"{rank}. {criterion:6s}: nDCG@10 = {ndcg:.4f} ({n_queries} queries)")
    print("=" * 60)

    # A.10 analysis
    logger.info("\nAnalyzing A.10 specifically...")
    a10_analysis = identify_a10_challenges(per_criterion_results)

    print("\n" + "=" * 60)
    print("A.10 CRITERION ANALYSIS")
    print("=" * 60)
    print(f"Hardness level: {a10_analysis.get('hardness_level', 'unknown')}")
    print(f"Is harder than average: {a10_analysis.get('is_harder', False)}")

    if "differences" in a10_analysis:
        print("\nKey metrics comparison:")
        for metric in ["ndcg@10", "recall@10", "mrr@10", "map@10"]:
            diff_data = a10_analysis["differences"].get(metric, {})
            a10_val = diff_data.get("a10", 0.0)
            others_val = diff_data.get("others_avg", 0.0)
            diff = diff_data.get("diff", 0.0)
            print(f"  {metric:12s}: A.10={a10_val:.4f}, Others={others_val:.4f}, Diff={diff:+.4f}")
    print("=" * 60)

    # Save results
    output_data = {
        "timestamp": "2026-01-19",
        "n_queries_total": len(rankings),
        "n_criteria": len(per_criterion_results),
        "k_values": args.ks,
        "per_criterion_results": per_criterion_results,
        "difficulty_ranking": [{"rank": i+1, "criterion": c, "ndcg@10": ndcg} for i, (c, ndcg) in enumerate(ranked)],
        "a10_analysis": a10_analysis,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Analysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
