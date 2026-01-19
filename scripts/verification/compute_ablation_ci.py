#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for ablation study results.

This script loads the existing ablation study results and computes
bootstrap 95% confidence intervals for key metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        values: Array of values (e.g., per-fold metrics)
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    np.random.seed(random_state)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean = np.mean(values)
    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)

    return float(mean), float(lower), float(upper)


def compute_ablation_ci_from_existing(
    ablation_file: Path,
    output_file: Path,
    n_bootstrap: int = 10000
):
    """Compute bootstrap CI for existing ablation results.

    Args:
        ablation_file: Path to ablation_study.json
        output_file: Path to save results with CI
        n_bootstrap: Number of bootstrap samples
    """
    logger.info(f"Loading ablation results from: {ablation_file}")

    with open(ablation_file) as f:
        ablation_data = json.load(f)

    ablations = ablation_data.get("ablations", {})

    logger.info(f"Found {len(ablations)} ablation configurations")

    # For the existing data, we only have summary statistics, not per-fold values
    # So we'll compute CI assuming values from summary.json fold_results
    # This is a limitation - ideally we'd have per-fold ablation results

    # Instead, let's create a framework for when we have per-fold data
    results_with_ci = {
        "ablations": {},
        "comparisons": {},
        "metadata": {
            "n_bootstrap": n_bootstrap,
            "confidence_level": 0.95,
            "note": "Bootstrap CI computed from per-fold values (when available)"
        }
    }

    # Copy existing ablations
    for config_name, config_data in ablations.items():
        results_with_ci["ablations"][config_name] = config_data.copy()

        # Add placeholder for CI (would compute from per-fold data)
        results_with_ci["ablations"][config_name]["ci_note"] = (
            "CI requires per-fold ablation results. "
            "Current values are from single evaluation."
        )

    # Compute pairwise comparisons (deltas)
    comparisons = []

    # Compare each config to baseline
    baseline = ablations.get("A0_baseline", {})
    baseline_auroc = baseline.get("auroc", 0.5)

    for config_name, config_data in ablations.items():
        if config_name == "A0_baseline":
            continue

        config_auroc = config_data.get("auroc", 0.0)
        delta_auroc = config_auroc - baseline_auroc

        comparisons.append({
            "comparison": f"{config_name} vs A0_baseline",
            "metric": "auroc",
            "baseline_value": baseline_auroc,
            "config_value": config_auroc,
            "delta": delta_auroc,
            "delta_ci_note": "Requires per-fold data for bootstrap test"
        })

    # Incremental comparisons (A1->A2, A2->A3, etc.)
    config_order = ["A0_baseline", "A1_p4_only", "A2_p4_calibrated",
                    "A3_with_gate", "A4_full_system"]

    for i in range(len(config_order) - 1):
        curr_name = config_order[i]
        next_name = config_order[i + 1]

        if curr_name in ablations and next_name in ablations:
            curr_auroc = ablations[curr_name].get("auroc", 0.0)
            next_auroc = ablations[next_name].get("auroc", 0.0)
            delta = next_auroc - curr_auroc

            comparisons.append({
                "comparison": f"{next_name} vs {curr_name}",
                "metric": "auroc",
                "baseline_value": curr_auroc,
                "config_value": next_auroc,
                "delta": delta,
                "delta_ci_note": "Requires per-fold data for paired bootstrap test"
            })

    results_with_ci["comparisons"] = comparisons

    # Save results
    logger.info(f"Saving results with CI framework to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results_with_ci, f, indent=2)

    logger.info("✅ Ablation CI framework created")
    logger.info("⚠️  Note: Full bootstrap CI requires per-fold ablation results")
    logger.info("   Current ablation_study.json has single summary values only")

    return results_with_ci


def compute_ci_from_clinical_folds(
    summary_file: Path,
    output_file: Path,
    n_bootstrap: int = 10000
):
    """Compute bootstrap CI from clinical evaluation fold results.

    This demonstrates the full bootstrap approach using actual per-fold data.

    Args:
        summary_file: Path to clinical summary.json with fold_results
        output_file: Path to save CI results
        n_bootstrap: Number of bootstrap samples
    """
    logger.info(f"Loading clinical evaluation from: {summary_file}")

    with open(summary_file) as f:
        summary = json.load(f)

    fold_results = summary.get("fold_results", [])
    n_folds = len(fold_results)

    logger.info(f"Found {n_folds} folds")

    # Extract per-fold metrics
    metrics_to_analyze = [
        ("ne_gate", "auroc"),
        ("ne_gate", "auprc"),
        ("deployment", "screening_sensitivity"),
        ("deployment", "alert_precision"),
        ("deployment", "screening_fn_per_1000"),
        ("dynamic_k", "mean_k"),
    ]

    results = {
        "n_folds": n_folds,
        "n_bootstrap": n_bootstrap,
        "confidence_level": 0.95,
        "metrics": {}
    }

    for category, metric_name in metrics_to_analyze:
        # Extract values across folds
        values = []
        for fold in fold_results:
            test_metrics = fold.get("test_metrics", {})
            category_metrics = test_metrics.get(category, {})
            value = category_metrics.get(metric_name)

            if value is not None and not np.isnan(value):
                values.append(value)

        if len(values) > 0:
            values = np.array(values)
            mean, lower, upper = bootstrap_ci(values, n_bootstrap=n_bootstrap)

            metric_key = f"{category}.{metric_name}"
            results["metrics"][metric_key] = {
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_range": upper - lower,
                "n_folds": len(values),
                "fold_values": values.tolist()
            }

            logger.info(f"  {metric_key}: {mean:.4f} [{lower:.4f}, {upper:.4f}]")

    # Save results
    logger.info(f"Saving bootstrap CI results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Computed bootstrap CI for {len(results['metrics'])} metrics")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute bootstrap CI for ablations")
    parser.add_argument(
        "--ablation_file",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/verification/ablation_study.json"),
        help="Path to ablation_study.json"
    )
    parser.add_argument(
        "--clinical_summary",
        type=Path,
        default=Path("outputs/clinical_high_recall/20260118_015913/summary.json"),
        help="Path to clinical summary.json for CI demonstration"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/version_a_full_audit/ci_results"),
        help="Output directory for CI results"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("BOOTSTRAP CONFIDENCE INTERVAL COMPUTATION")
    logger.info("="*80)

    # Compute CI framework for ablations (limited by data availability)
    logger.info("\n1. Ablation Study CI Framework")
    logger.info("-" * 80)
    ablation_output = args.output_dir / "ablation_ci_framework.json"
    compute_ablation_ci_from_existing(
        args.ablation_file,
        ablation_output,
        args.n_bootstrap
    )

    # Demonstrate full bootstrap CI using clinical fold results
    logger.info("\n2. Clinical Evaluation Bootstrap CI (Demonstration)")
    logger.info("-" * 80)
    clinical_output = args.output_dir / "clinical_bootstrap_ci.json"
    compute_ci_from_clinical_folds(
        args.clinical_summary,
        clinical_output,
        args.n_bootstrap
    )

    logger.info("\n" + "="*80)
    logger.info("✅ BOOTSTRAP CI COMPUTATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutputs saved to: {args.output_dir}")
    logger.info("  - ablation_ci_framework.json (framework only - needs per-fold data)")
    logger.info("  - clinical_bootstrap_ci.json (full bootstrap CI from fold results)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
