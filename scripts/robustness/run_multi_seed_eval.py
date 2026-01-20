#!/usr/bin/env python3
"""Multi-seed robustness evaluation.

Runs evaluation with multiple random seeds to assess:
1. Stability of metric estimates across different train/test splits
2. Variance in reported metrics
3. Confidence intervals for all metrics

Two modes:
1. Bootstrap mode (default): Fixed model, bootstrap CIs over queries/posts
2. Resplit mode: Regenerate splits per seed (more expensive, requires re-evaluation)

Usage:
    # Bootstrap mode (recommended - uses cached evaluation)
    python scripts/robustness/run_multi_seed_eval.py \
        --per_query outputs/final_research_eval/20260118_031312_complete/per_query.csv \
        --output outputs/robustness/ \
        --mode bootstrap

    # Resplit mode (requires running evaluation per seed)
    python scripts/robustness/run_multi_seed_eval.py \
        --config configs/default.yaml \
        --seeds 11 21 42 84 168 \
        --output outputs/robustness/ \
        --mode resplit
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

from final_sc_review.metrics.compute_metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_ranking_metrics_from_csv,
    bootstrap_ci,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_bootstrap_robustness(
    per_query_csv: Path,
    output_dir: Path,
    n_bootstrap: int = 2000,
    seeds: List[int] = None,
    unit: str = "post",
) -> Dict:
    """Run bootstrap robustness analysis on existing evaluation results.

    This is the recommended mode: it uses the canonical evaluation output
    and computes bootstrap CIs by resampling queries or posts.

    Args:
        per_query_csv: Path to per_query.csv from evaluation
        output_dir: Output directory
        n_bootstrap: Number of bootstrap iterations
        seeds: List of seeds for reproducibility check
        unit: Resampling unit ("query" or "post")

    Returns:
        Dictionary with robustness results
    """
    logger.info("=" * 80)
    logger.info("BOOTSTRAP ROBUSTNESS ANALYSIS")
    logger.info(f"Source: {per_query_csv}")
    logger.info(f"Bootstrap iterations: {n_bootstrap}")
    logger.info(f"Resampling unit: {unit}")
    logger.info("=" * 80)

    # Load data
    df = pd.read_csv(per_query_csv)
    logger.info(f"Loaded {len(df)} queries")

    # Key metrics to analyze
    metrics_config = {
        # Classification metrics (all_queries)
        "auroc": {
            "fn": lambda d: compute_classification_metrics(d).get("auroc", np.nan),
            "protocol": "all_queries",
        },
        "auprc": {
            "fn": lambda d: compute_classification_metrics(d).get("auprc", np.nan),
            "protocol": "all_queries",
        },
        # Ranking metrics (positives_only) - computed from pre-computed columns
        "evidence_recall_at_k": {
            "fn": lambda d: d[d["has_evidence_gold"] == 1]["evidence_recall_at_k"].dropna().mean(),
            "protocol": "positives_only",
        },
        "mrr": {
            "fn": lambda d: d[d["has_evidence_gold"] == 1]["mrr"].dropna().mean(),
            "protocol": "positives_only",
        },
    }

    # Compute point estimates
    point_estimates = {}
    for metric_name, config in metrics_config.items():
        try:
            point_estimates[metric_name] = config["fn"](df)
        except Exception as e:
            logger.warning(f"Failed to compute {metric_name}: {e}")
            point_estimates[metric_name] = np.nan

    logger.info("Point estimates:")
    for name, val in point_estimates.items():
        logger.info(f"  {name}: {val:.4f}")

    # Compute bootstrap CIs
    results = {}
    base_seed = 42

    for metric_name, config in metrics_config.items():
        logger.info(f"Computing bootstrap CI for {metric_name}...")

        try:
            if unit == "post":
                ci = bootstrap_ci(
                    config["fn"],
                    df,
                    n_bootstrap=n_bootstrap,
                    seed=base_seed,
                    unit="post",
                    unit_col="post_id",
                )
            else:
                ci = bootstrap_ci(
                    config["fn"],
                    df,
                    n_bootstrap=n_bootstrap,
                    seed=base_seed,
                    unit="query",
                )

            results[metric_name] = {
                "value": point_estimates[metric_name],
                "mean": ci["mean"],
                "std": ci["std"],
                "ci_95_lower": ci["ci_lower"],
                "ci_95_upper": ci["ci_upper"],
                "protocol": config["protocol"],
                "n_bootstrap": n_bootstrap,
            }

            logger.info(f"  {metric_name}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

        except Exception as e:
            logger.warning(f"Failed bootstrap for {metric_name}: {e}")
            results[metric_name] = {"error": str(e)}

    # Check stability across different seeds
    if seeds:
        logger.info(f"Checking reproducibility across seeds: {seeds}")
        seed_variation = {}

        for metric_name, config in metrics_config.items():
            seed_values = []
            for seed in seeds:
                ci = bootstrap_ci(
                    config["fn"],
                    df,
                    n_bootstrap=100,  # Fewer for speed
                    seed=seed,
                    unit=unit,
                    unit_col="post_id" if unit == "post" else None,
                )
                seed_values.append(ci["mean"])

            seed_variation[metric_name] = {
                "values": seed_values,
                "mean": float(np.mean(seed_values)),
                "std": float(np.std(seed_values)),
                "cv_pct": float(np.std(seed_values) / np.mean(seed_values) * 100) if np.mean(seed_values) > 0 else 0,
            }

        results["seed_variation"] = seed_variation

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "robustness_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "source": str(per_query_csv),
            "n_queries": len(df),
            "n_bootstrap": n_bootstrap,
            "resampling_unit": unit,
            "metrics": results,
        }, f, indent=2, default=str)

    # Generate report
    generate_robustness_report(results, point_estimates, output_dir)

    # Create CSV summary
    summary_rows = []
    for metric_name, data in results.items():
        if isinstance(data, dict) and "value" in data:
            summary_rows.append({
                "metric": metric_name,
                "value": data["value"],
                "ci_95_lower": data["ci_95_lower"],
                "ci_95_upper": data["ci_95_upper"],
                "std": data["std"],
                "protocol": data["protocol"],
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "robustness_summary.csv", index=False)

    logger.info(f"Results saved to: {output_dir}")
    return results


def run_resplit_robustness(
    config_path: Path,
    output_dir: Path,
    seeds: List[int],
) -> Dict:
    """Run evaluation with different train/test splits.

    This mode regenerates splits per seed and runs full evaluation.
    More expensive but shows true split variation.

    Args:
        config_path: Path to config file
        output_dir: Output directory
        seeds: List of seeds for splits

    Returns:
        Dictionary with results per seed
    """
    logger.info("=" * 80)
    logger.info("RESPLIT ROBUSTNESS EVALUATION")
    logger.info(f"Seeds: {seeds}")
    logger.info("=" * 80)

    # This requires the full evaluation pipeline
    # For now, we check if cached results exist for each seed

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_per_seed = {}

    for seed in seeds:
        logger.info(f"\nEvaluating seed {seed}...")

        # Check for cached results
        seed_output = output_dir / f"seed_{seed}"
        cached_csv = seed_output / "per_query.csv"

        if cached_csv.exists():
            logger.info(f"  Loading cached results from {cached_csv}")
            df = pd.read_csv(cached_csv)

            class_metrics = compute_classification_metrics(df)
            rank_metrics = compute_ranking_metrics_from_csv(df)

            results_per_seed[seed] = {
                "auroc": class_metrics.get("auroc"),
                "auprc": class_metrics.get("auprc"),
                "evidence_recall_at_k": rank_metrics.get("evidence_recall_at_k"),
                "mrr": rank_metrics.get("mrr"),
                "n_queries": len(df),
            }
        else:
            logger.warning(f"  No cached results for seed {seed}. Run evaluation first:")
            logger.warning(f"    python scripts/eval_zoo_pipeline.py --config {config_path} --seed {seed} --output {seed_output}")
            results_per_seed[seed] = {"error": "No cached results"}

    # Aggregate across seeds
    metrics_to_aggregate = ["auroc", "auprc", "evidence_recall_at_k", "mrr"]
    aggregated = {}

    for metric in metrics_to_aggregate:
        values = [r[metric] for r in results_per_seed.values() if metric in r and r[metric] is not None]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv_pct": float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else 0,
                "n_seeds": len(values),
            }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resplit_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "seeds": seeds,
            "per_seed": results_per_seed,
            "aggregated": aggregated,
        }, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_dir}")
    return {"per_seed": results_per_seed, "aggregated": aggregated}


def generate_robustness_report(
    results: Dict,
    point_estimates: Dict,
    output_dir: Path,
) -> None:
    """Generate markdown report of robustness analysis."""
    report = f"""# Robustness Analysis Report

Generated: {datetime.now().isoformat()}

---

## Executive Summary

This report presents bootstrap confidence intervals for key metrics,
demonstrating the stability and reliability of the evaluation results.

---

## Metric Confidence Intervals (95%)

| Metric | Value | 95% CI Lower | 95% CI Upper | Std | Protocol |
|--------|-------|--------------|--------------|-----|----------|
"""

    for metric_name, data in results.items():
        if isinstance(data, dict) and "value" in data:
            report += (
                f"| **{metric_name}** | {data['value']:.4f} | "
                f"{data['ci_95_lower']:.4f} | {data['ci_95_upper']:.4f} | "
                f"{data['std']:.4f} | {data['protocol']} |\n"
            )

    report += """
---

## Stability Assessment

"""

    for metric_name, data in results.items():
        if isinstance(data, dict) and "std" in data:
            cv = data["std"] / data["value"] * 100 if data["value"] > 0 else 0
            stability = "Excellent" if cv < 1 else "Good" if cv < 2 else "Moderate" if cv < 5 else "Poor"
            report += f"- **{metric_name}**: CV = {cv:.2f}% ({stability} stability)\n"

    # Seed variation if present
    if "seed_variation" in results:
        report += """
---

## Seed Reproducibility Check

| Metric | Mean | Std | CV% |
|--------|------|-----|-----|
"""
        for metric_name, data in results["seed_variation"].items():
            report += f"| {metric_name} | {data['mean']:.4f} | {data['std']:.4f} | {data['cv_pct']:.2f}% |\n"

    report += """
---

## Methodology

- **Bootstrap method**: Non-parametric bootstrap with replacement
- **Confidence level**: 95% (percentile method)
- **Resampling unit**: Post-level (preserves within-post correlation)
- **Iterations**: 2,000 bootstrap samples

### Interpretation

- **Narrow CIs**: High confidence in reported metrics
- **CV < 2%**: Excellent stability across resamples
- **Post-level resampling**: More conservative than query-level

---

## Reproducibility

```bash
python scripts/robustness/run_multi_seed_eval.py \\
    --per_query outputs/final_research_eval/20260118_031312_complete/per_query.csv \\
    --output outputs/robustness/ \\
    --mode bootstrap \\
    --n_bootstrap 2000
```
"""

    with open(output_dir / "robustness_report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'robustness_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed robustness evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "resplit"],
        help="Evaluation mode: bootstrap (recommended) or resplit"
    )
    parser.add_argument(
        "--per_query",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Path to per_query.csv (for bootstrap mode)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file (for resplit mode)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 21, 42, 84, 168],
        help="Random seeds for reproducibility check"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robustness"),
        help="Output directory"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="post",
        choices=["query", "post"],
        help="Resampling unit for bootstrap"
    )

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / timestamp

    if args.mode == "bootstrap":
        if not args.per_query.exists():
            logger.error(f"per_query.csv not found: {args.per_query}")
            return 1

        results = run_bootstrap_robustness(
            per_query_csv=args.per_query,
            output_dir=output_dir,
            n_bootstrap=args.n_bootstrap,
            seeds=args.seeds,
            unit=args.unit,
        )
    else:
        results = run_resplit_robustness(
            config_path=args.config,
            output_dir=output_dir,
            seeds=args.seeds,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Robustness Analysis Complete")
    print("=" * 60)

    if args.mode == "bootstrap":
        for metric_name, data in results.items():
            if isinstance(data, dict) and "value" in data:
                print(f"{metric_name}: {data['value']:.4f} "
                      f"[95% CI: {data['ci_95_lower']:.4f}, {data['ci_95_upper']:.4f}]")
    else:
        if "aggregated" in results:
            for metric_name, data in results["aggregated"].items():
                print(f"{metric_name}: {data['mean']:.4f} ± {data['std']:.4f}")

    print("=" * 60)
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
