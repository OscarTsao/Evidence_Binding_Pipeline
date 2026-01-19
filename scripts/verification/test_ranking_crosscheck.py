#!/usr/bin/env python3
"""Test script for ranking metrics cross-check functionality.

This script creates synthetic per-query rankings to verify that the
cross_check_ranking_metrics() function works correctly.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from metric_crosscheck import cross_check_ranking_metrics


def create_test_data():
    """Create synthetic test data for cross-checking."""

    # Create 10 synthetic queries with known rankings
    per_query_rankings = []

    for i in range(10):
        # Each query has 20 candidate sentences
        # First 3 sentences are relevant (gold)
        ranked_uids = [f"sent_{j}" for j in range(20)]
        gold_uids = [f"sent_{j}" for j in range(3)]

        per_query_rankings.append({
            "query_id": f"query_{i}",
            "ranked_uids": ranked_uids,
            "gold_uids": gold_uids,
        })

    # Compute expected metrics manually
    # For this simple case where gold items are always at positions 0, 1, 2:
    # recall@1 = 1/3 = 0.3333
    # recall@3 = 3/3 = 1.0
    # recall@5 = 3/3 = 1.0
    # recall@10 = 3/3 = 1.0
    # recall@20 = 3/3 = 1.0
    #
    # mrr@1 = 1/1 = 1.0 (first relevant at rank 1)
    # mrr@3 = 1/1 = 1.0
    # mrr@5 = 1/1 = 1.0
    # mrr@10 = 1/1 = 1.0
    # mrr@20 = 1/1 = 1.0
    #
    # map@1 = prec@1 = 1/1 = 1.0 (first item is relevant)
    # map@3 = (1/1 + 2/2 + 3/3) / 3 = 1.0
    # map@5 = 1.0
    # map@10 = 1.0
    # map@20 = 1.0
    #
    # For nDCG, with perfect ranking at top 3:
    # ndcg@1 = 1.0 / 1.0 = 1.0
    # ndcg@3 = (1 + 1/log2(3) + 1/log2(4)) / ideal_dcg = 1.0
    # etc.

    expected_metrics = {
        "recall@1": 0.3333,
        "recall@3": 1.0,
        "recall@5": 1.0,
        "recall@10": 1.0,
        "recall@20": 1.0,
        "mrr@1": 1.0,
        "mrr@3": 1.0,
        "mrr@5": 1.0,
        "mrr@10": 1.0,
        "mrr@20": 1.0,
        "map@1": 1.0,  # Fixed: first item is relevant, so prec@1 = 1.0
        "map@3": 1.0,
        "map@5": 1.0,
        "map@10": 1.0,
        "map@20": 1.0,
        "ndcg@1": 1.0,
        "ndcg@3": 1.0,
        "ndcg@5": 1.0,
        "ndcg@10": 1.0,
        "ndcg@20": 1.0,
    }

    return per_query_rankings, expected_metrics


def main():
    print("="*60)
    print("TESTING RANKING METRICS CROSS-CHECK FUNCTIONALITY")
    print("="*60)

    # Create test data
    print("\n1. Creating synthetic test data...")
    per_query_rankings, expected_metrics = create_test_data()
    print(f"   Created {len(per_query_rankings)} synthetic queries")
    print(f"   Each query has 20 candidates, 3 are relevant (at positions 0, 1, 2)")

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save per-query rankings
        rankings_file = tmpdir / "per_query_rankings.jsonl"
        with open(rankings_file, 'w') as f:
            for ranking in per_query_rankings:
                f.write(json.dumps(ranking) + '\n')

        # Create pipeline results (use expected metrics as "pipeline" results)
        results_file = tmpdir / "test_results.json"
        pipeline_results = {
            "reranked": expected_metrics,
            "eval_split": "test",
        }
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)

        # Run cross-check
        print("\n2. Running cross-check...")
        output_file = tmpdir / "crosscheck_results.json"

        try:
            results = cross_check_ranking_metrics(
                results_json=results_file,
                per_query_rankings_jsonl=rankings_file,
                output_file=output_file,
                tolerance=0.01,
                ks=[1, 3, 5, 10, 20],
            )

            print("\n3. Verification Results:")
            if results['all_metrics_match']:
                print("   ✅ SUCCESS: All metrics matched within tolerance!")
                print(f"\n   Checked {results['n_queries_checked']} queries")
                print(f"   Tolerance: ±{results['tolerance']}")

                # Show a few example comparisons
                print("\n   Sample comparisons:")
                for comp in results['comparison'][:5]:
                    metric = comp['metric']
                    pipeline_val = comp['pipeline']
                    recomputed_val = comp['recomputed']
                    diff = comp['diff']
                    print(f"     {metric:<12} Pipeline: {pipeline_val:.4f}  "
                          f"Recomputed: {recomputed_val:.4f}  "
                          f"Diff: {diff:.6f}")

                return 0
            else:
                print("   ❌ FAILED: Some metrics did not match")
                for comp in results['comparison']:
                    if not comp['match']:
                        metric = comp['metric']
                        pipeline_val = comp.get('pipeline')
                        recomputed_val = comp.get('recomputed')
                        diff = comp.get('diff')

                        if pipeline_val is None:
                            print(f"     {metric}: Pipeline N/A, "
                                  f"Recomputed {recomputed_val:.4f}")
                        elif diff is None:
                            print(f"     {metric}: Error computing diff")
                        else:
                            print(f"     {metric}: "
                                  f"Pipeline {pipeline_val:.4f} != "
                                  f"Recomputed {recomputed_val:.4f} "
                                  f"(diff: {diff:.6f})")
                return 1

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
