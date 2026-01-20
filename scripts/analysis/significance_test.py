#!/usr/bin/env python3
"""Statistical significance testing for method comparison.

Performs paired bootstrap significance tests comparing:
- Proposed pipeline vs baselines
- With vs without GNN modules (ablation)

Usage:
    python scripts/analysis/significance_test.py \
        --proposed outputs/final_research_eval/20260118_031312_complete/per_query.csv \
        --baseline outputs/baselines/*/per_query_bm25.csv \
        --output outputs/significance/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """Perform paired bootstrap significance test.

    Tests if mean(scores_a) > mean(scores_b) is statistically significant.

    Args:
        scores_a: Scores from method A (per query)
        scores_b: Scores from method B (per query)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Dictionary with test results
    """
    assert len(scores_a) == len(scores_b), "Score arrays must have same length"

    rng = np.random.default_rng(seed)
    n = len(scores_a)

    # Observed difference
    diff = scores_a - scores_b
    observed_diff = np.mean(diff)

    # Bootstrap distribution under null (centered)
    diff_centered = diff - np.mean(diff)
    boot_diffs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_diff = np.mean(diff_centered[indices])
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    # P-value (two-sided)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

    # Confidence interval for difference
    ci_lower = np.percentile(boot_diffs + observed_diff, 2.5)
    ci_upper = np.percentile(boot_diffs + observed_diff, 97.5)

    return {
        "observed_difference": float(observed_diff),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "effect_size_cohens_d": float(cohens_d),
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }


def compare_methods(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metrics: List[str],
    method_a_name: str,
    method_b_name: str,
) -> Dict:
    """Compare two methods across multiple metrics.

    Args:
        df_a: Per-query results for method A
        df_b: Per-query results for method B
        metrics: List of metric column names to compare
        method_a_name: Name of method A
        method_b_name: Name of method B

    Returns:
        Dictionary with comparison results
    """
    # Align by query
    merge_cols = ["post_id", "criterion_id"]

    # Check if merge columns exist
    if not all(c in df_a.columns for c in merge_cols):
        logger.warning("Cannot align by query - using index alignment")
        merged = pd.DataFrame({
            "a_" + col: df_a[col].values for col in metrics if col in df_a.columns
        })
        for col in metrics:
            if col in df_b.columns:
                merged["b_" + col] = df_b[col].values
    else:
        merged = df_a.merge(df_b, on=merge_cols, suffixes=("_a", "_b"))

    results = {
        "comparison": f"{method_a_name} vs {method_b_name}",
        "n_paired_samples": len(merged),
        "metrics": {},
    }

    for metric in metrics:
        col_a = f"{metric}_a" if f"{metric}_a" in merged.columns else metric
        col_b = f"{metric}_b" if f"{metric}_b" in merged.columns else metric

        if col_a not in merged.columns or col_b not in merged.columns:
            logger.warning(f"Metric {metric} not found in both dataframes")
            continue

        scores_a = merged[col_a].dropna().values
        scores_b = merged[col_b].dropna().values

        # Ensure same length after dropna
        min_len = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:min_len]
        scores_b = scores_b[:min_len]

        if len(scores_a) == 0:
            logger.warning(f"No valid samples for {metric}")
            continue

        logger.info(f"Testing {metric}...")
        test_result = paired_bootstrap_test(scores_a, scores_b)

        results["metrics"][metric] = {
            "method_a_mean": float(np.mean(scores_a)),
            "method_b_mean": float(np.mean(scores_b)),
            **test_result,
        }

        sig = "**" if test_result["significant_01"] else "*" if test_result["significant_05"] else ""
        logger.info(f"  {metric}: {np.mean(scores_a):.4f} vs {np.mean(scores_b):.4f} "
                   f"(diff={test_result['observed_difference']:.4f}, p={test_result['p_value']:.4f}{sig})")

    return results


def generate_significance_table(results: Dict) -> pd.DataFrame:
    """Generate significance test summary table."""
    rows = []

    for metric, data in results.get("metrics", {}).items():
        rows.append({
            "metric": metric,
            "method_a_mean": data["method_a_mean"],
            "method_b_mean": data["method_b_mean"],
            "difference": data["observed_difference"],
            "ci_95_lower": data["ci_95_lower"],
            "ci_95_upper": data["ci_95_upper"],
            "p_value": data["p_value"],
            "significant": "*" if data["significant_05"] else "",
            "effect_size": data["effect_size_cohens_d"],
        })

    return pd.DataFrame(rows)


def generate_report(
    results: List[Dict],
    output_dir: Path,
) -> None:
    """Generate significance test report."""
    report = f"""# Statistical Significance Report

Generated: {datetime.now().isoformat()}

---

## Methodology

- **Test**: Paired bootstrap significance test
- **Bootstrap iterations**: 10,000
- **Significance level**: α = 0.05 (two-sided)
- **Effect size**: Cohen's d

---

## Results Summary

"""

    for comparison in results:
        report += f"\n### {comparison['comparison']}\n\n"
        report += f"Paired samples: {comparison['n_paired_samples']}\n\n"
        report += "| Metric | Method A | Method B | Diff | 95% CI | p-value | Sig | d |\n"
        report += "|--------|----------|----------|------|--------|---------|-----|---|\n"

        for metric, data in comparison.get("metrics", {}).items():
            sig = "**" if data["significant_01"] else "*" if data["significant_05"] else ""
            report += (
                f"| {metric} | {data['method_a_mean']:.4f} | {data['method_b_mean']:.4f} | "
                f"{data['observed_difference']:+.4f} | [{data['ci_95_lower']:.4f}, {data['ci_95_upper']:.4f}] | "
                f"{data['p_value']:.4f} | {sig} | {data['effect_size_cohens_d']:.2f} |\n"
            )

    report += """
---

## Interpretation

- **p < 0.05**: Statistically significant difference (marked with *)
- **p < 0.01**: Highly significant difference (marked with **)
- **Cohen's d**: Effect size (0.2=small, 0.5=medium, 0.8=large)
- **95% CI**: Confidence interval for the difference

---

## Notes

- Tests are paired by query (same post_id + criterion_id)
- Bootstrap uses 10,000 resamples with fixed seed (42) for reproducibility
- Two-sided test: tests for any difference (not directional)
"""

    with open(output_dir / "significance_report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'significance_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument(
        "--proposed",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Per-query results for proposed method"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="+",
        default=[],
        help="Per-query results for baseline method(s)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/significance"),
        help="Output directory"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["evidence_recall_at_k", "mrr"],
        help="Metrics to compare"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("STATISTICAL SIGNIFICANCE TESTING")
    logger.info("=" * 80)

    # Load proposed method results
    if not args.proposed.exists():
        logger.error(f"Proposed results not found: {args.proposed}")
        return 1

    df_proposed = pd.read_csv(args.proposed)
    logger.info(f"Loaded proposed results: {len(df_proposed)} queries")

    # Filter to positive queries for ranking metrics
    df_proposed_pos = df_proposed[df_proposed["has_evidence_gold"] == 1]

    all_results = []

    # Compare against each baseline
    for baseline_path in args.baseline:
        if not baseline_path.exists():
            logger.warning(f"Baseline not found: {baseline_path}")
            continue

        baseline_name = baseline_path.stem.replace("per_query_", "")
        df_baseline = pd.read_csv(baseline_path)
        df_baseline_pos = df_baseline[df_baseline.get("has_evidence_gold", df_baseline.get("has_evidence", 0)) == 1]

        logger.info(f"\nComparing vs {baseline_name}...")

        results = compare_methods(
            df_a=df_proposed_pos,
            df_b=df_baseline_pos,
            metrics=args.metrics,
            method_a_name="Proposed",
            method_b_name=baseline_name,
        )
        all_results.append(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "significance_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate tables
    for results in all_results:
        table = generate_significance_table(results)
        comparison_name = results["comparison"].replace(" ", "_").replace("vs", "vs")
        table.to_csv(output_dir / f"significance_{comparison_name}.csv", index=False)

    # Generate report
    generate_report(all_results, output_dir)

    logger.info("=" * 80)
    logger.info("SIGNIFICANCE TESTING COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
