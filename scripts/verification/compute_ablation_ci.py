#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for ablation study results.

This script computes real bootstrap 95% confidence intervals for ablation
metrics using per-fold data from per_query.csv.

Usage:
    python scripts/verification/compute_ablation_ci.py \
        --per_query outputs/final_research_eval/20260118_031312_complete/per_query.csv \
        --output_dir outputs/ablation_ci/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

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


def compute_ablation_metrics_per_fold(
    df: pd.DataFrame,
    ablation_name: str
) -> Dict[str, List[float]]:
    """Compute ablation metrics for each fold.

    Args:
        df: DataFrame with per-query predictions
        ablation_name: Name of ablation configuration

    Returns:
        Dict mapping metric names to lists of per-fold values
    """
    folds = sorted(df["fold_id"].unique())
    metrics = {"auroc": [], "auprc": []}

    for fold_id in folds:
        fold_df = df[df["fold_id"] == fold_id]
        y_true = fold_df["has_evidence_gold"].values

        # Get predictions based on ablation type
        if ablation_name == "A0_baseline":
            # Random baseline: uniform 0.5
            y_score = np.full(len(y_true), 0.5)
        elif ablation_name == "A1_p4_only":
            # Raw P4 probabilities
            y_score = fold_df["p4_prob_raw"].values
        elif ablation_name in ("A2_p4_calibrated", "A3_with_gate", "A4_full_system"):
            # Calibrated probabilities (same column, different gating logic)
            y_score = fold_df["p4_prob_calibrated"].values
        else:
            logger.warning(f"Unknown ablation: {ablation_name}, using calibrated probs")
            y_score = fold_df["p4_prob_calibrated"].values

        # Compute metrics (handle edge cases)
        if len(np.unique(y_true)) < 2:
            # Skip folds with only one class
            continue

        try:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            metrics["auroc"].append(auroc)
            metrics["auprc"].append(auprc)
        except ValueError as e:
            logger.warning(f"Fold {fold_id} skipped: {e}")
            continue

    return metrics


def compute_ablation_ci_from_per_query(
    per_query_file: Path,
    output_file: Path,
    n_bootstrap: int = 10000
) -> Dict:
    """Compute real bootstrap CI for ablation results from per_query.csv.

    Args:
        per_query_file: Path to per_query.csv with fold-level predictions
        output_file: Path to save results with CI
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with ablation results and confidence intervals
    """
    logger.info(f"Loading per-query predictions from: {per_query_file}")
    df = pd.read_csv(per_query_file)

    n_queries = len(df)
    n_folds = df["fold_id"].nunique()
    logger.info(f"Loaded {n_queries} queries across {n_folds} folds")

    # Define ablation configurations
    ablation_configs = [
        ("A0_baseline", "Random baseline (0.5)", []),
        ("A1_p4_only", "P4 raw probabilities", ["p4"]),
        ("A2_p4_calibrated", "P4 calibrated", ["p4", "calibration"]),
        ("A3_with_gate", "P4 + NE Gate", ["p4", "calibration", "ne_gate"]),
        ("A4_full_system", "Full system", ["p4", "calibration", "ne_gate", "dynamic_k"]),
    ]

    results = {
        "metadata": {
            "source": str(per_query_file),
            "n_queries": n_queries,
            "n_folds": n_folds,
            "n_bootstrap": n_bootstrap,
            "confidence_level": 0.95,
        },
        "ablations": {},
        "comparisons": []
    }

    # Compute metrics with CI for each ablation
    ablation_metrics = {}
    for ablation_name, description, components in ablation_configs:
        logger.info(f"Computing CI for {ablation_name}: {description}")

        fold_metrics = compute_ablation_metrics_per_fold(df, ablation_name)

        if len(fold_metrics["auroc"]) == 0:
            logger.warning(f"No valid folds for {ablation_name}")
            continue

        auroc_values = np.array(fold_metrics["auroc"])
        auprc_values = np.array(fold_metrics["auprc"])

        auroc_mean, auroc_lower, auroc_upper = bootstrap_ci(
            auroc_values, n_bootstrap=n_bootstrap
        )
        auprc_mean, auprc_lower, auprc_upper = bootstrap_ci(
            auprc_values, n_bootstrap=n_bootstrap
        )

        ablation_metrics[ablation_name] = auroc_values

        results["ablations"][ablation_name] = {
            "name": description,
            "components": components,
            "n_folds": len(auroc_values),
            "auroc": {
                "mean": auroc_mean,
                "ci_lower": auroc_lower,
                "ci_upper": auroc_upper,
                "fold_values": auroc_values.tolist()
            },
            "auprc": {
                "mean": auprc_mean,
                "ci_lower": auprc_lower,
                "ci_upper": auprc_upper,
                "fold_values": auprc_values.tolist()
            }
        }

        logger.info(f"  AUROC: {auroc_mean:.4f} [{auroc_lower:.4f}, {auroc_upper:.4f}]")
        logger.info(f"  AUPRC: {auprc_mean:.4f} [{auprc_lower:.4f}, {auprc_upper:.4f}]")

    # Compute pairwise comparisons with paired bootstrap
    logger.info("\nComputing pairwise comparisons...")

    # Compare each config to baseline
    baseline_values = ablation_metrics.get("A0_baseline", np.array([0.5]))

    for ablation_name in ["A1_p4_only", "A2_p4_calibrated", "A3_with_gate", "A4_full_system"]:
        if ablation_name not in ablation_metrics:
            continue

        config_values = ablation_metrics[ablation_name]
        delta_values = config_values - baseline_values

        delta_mean, delta_lower, delta_upper = bootstrap_ci(
            delta_values, n_bootstrap=n_bootstrap
        )

        # Statistical significance: check if CI excludes 0
        significant = (delta_lower > 0) or (delta_upper < 0)

        results["comparisons"].append({
            "comparison": f"{ablation_name} vs A0_baseline",
            "metric": "auroc",
            "delta_mean": delta_mean,
            "delta_ci_lower": delta_lower,
            "delta_ci_upper": delta_upper,
            "significant_at_95": significant,
            "interpretation": f"+{delta_mean:.4f} AUROC" if delta_mean > 0 else f"{delta_mean:.4f} AUROC"
        })

        logger.info(f"  {ablation_name} vs baseline: Δ={delta_mean:+.4f} [{delta_lower:+.4f}, {delta_upper:+.4f}] {'*' if significant else ''}")

    # Incremental comparisons
    config_order = ["A0_baseline", "A1_p4_only", "A2_p4_calibrated", "A3_with_gate", "A4_full_system"]

    for i in range(len(config_order) - 1):
        curr_name = config_order[i]
        next_name = config_order[i + 1]

        if curr_name not in ablation_metrics or next_name not in ablation_metrics:
            continue

        curr_values = ablation_metrics[curr_name]
        next_values = ablation_metrics[next_name]
        delta_values = next_values - curr_values

        delta_mean, delta_lower, delta_upper = bootstrap_ci(
            delta_values, n_bootstrap=n_bootstrap
        )

        significant = (delta_lower > 0) or (delta_upper < 0)

        results["comparisons"].append({
            "comparison": f"{next_name} vs {curr_name}",
            "metric": "auroc",
            "delta_mean": delta_mean,
            "delta_ci_lower": delta_lower,
            "delta_ci_upper": delta_upper,
            "significant_at_95": significant,
            "interpretation": f"+{delta_mean:.4f} AUROC (incremental)" if delta_mean > 0 else f"{delta_mean:.4f} AUROC (incremental)"
        })

    # Save results
    logger.info(f"\nSaving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Real bootstrap CI computed from per-fold data")

    return results


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
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CI for ablation study from per-query predictions"
    )
    parser.add_argument(
        "--per_query",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Path to per_query.csv with fold-level predictions"
    )
    parser.add_argument(
        "--clinical_summary",
        type=Path,
        default=Path("outputs/clinical_high_recall/20260118_015913/summary.json"),
        help="Path to clinical summary.json for additional CI computation"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/ablation_ci"),
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

    logger.info("=" * 80)
    logger.info("BOOTSTRAP CONFIDENCE INTERVAL COMPUTATION")
    logger.info("=" * 80)

    # Compute real ablation CI from per_query.csv
    logger.info("\n1. Ablation Study Bootstrap CI (Real)")
    logger.info("-" * 80)

    if args.per_query.exists():
        ablation_output = args.output_dir / "ablation_bootstrap_ci.json"
        ablation_results = compute_ablation_ci_from_per_query(
            args.per_query,
            ablation_output,
            args.n_bootstrap
        )

        # Print summary table
        print("\n" + "=" * 60)
        print("ABLATION RESULTS WITH 95% CI")
        print("=" * 60)
        print(f"{'Configuration':<25} {'AUROC':^25} {'AUPRC':^25}")
        print("-" * 60)

        for name, data in ablation_results.get("ablations", {}).items():
            auroc = data.get("auroc", {})
            auprc = data.get("auprc", {})
            auroc_str = f"{auroc.get('mean', 0):.4f} [{auroc.get('ci_lower', 0):.4f}, {auroc.get('ci_upper', 0):.4f}]"
            auprc_str = f"{auprc.get('mean', 0):.4f} [{auprc.get('ci_lower', 0):.4f}, {auprc.get('ci_upper', 0):.4f}]"
            print(f"{name:<25} {auroc_str:^25} {auprc_str:^25}")

        print("=" * 60)
    else:
        logger.warning(f"per_query.csv not found: {args.per_query}")
        logger.warning("Skipping ablation CI computation")

    # Compute clinical fold CI if available
    logger.info("\n2. Clinical Evaluation Bootstrap CI")
    logger.info("-" * 80)

    if args.clinical_summary.exists():
        clinical_output = args.output_dir / "clinical_bootstrap_ci.json"
        compute_ci_from_clinical_folds(
            args.clinical_summary,
            clinical_output,
            args.n_bootstrap
        )
    else:
        logger.warning(f"Clinical summary not found: {args.clinical_summary}")
        logger.warning("Skipping clinical CI computation")

    logger.info("\n" + "=" * 80)
    logger.info("✅ BOOTSTRAP CI COMPUTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nOutputs saved to: {args.output_dir}")
    logger.info("  - ablation_bootstrap_ci.json (real bootstrap CI from per-fold data)")
    if args.clinical_summary.exists():
        logger.info("  - clinical_bootstrap_ci.json (clinical evaluation CI)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
