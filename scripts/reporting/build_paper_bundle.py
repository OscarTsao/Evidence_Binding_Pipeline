#!/usr/bin/env python3
"""Build paper bundle v3.0 with all publication artifacts.

This script generates the complete paper bundle including:
- metrics_master.json (single source of truth)
- summary.json (bundle metadata)
- tables/*.csv (machine-readable results)
- figures/*.png (publication figures)
- MANIFEST.md (regeneration instructions)
- checksums.txt (SHA256 integrity)

Usage:
    python scripts/reporting/build_paper_bundle.py \
        --version v3.0 \
        --source_run outputs/final_research_eval/20260118_031312_complete \
        --output results/paper_bundle/v3.0
"""

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.metrics.compute_metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_ranking_metrics_from_csv,
    compute_calibration_metrics,
    compute_per_criterion_metrics,
    bootstrap_ci,
    verify_auprc_not_recall,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_criteria_registry() -> Dict:
    """Load criteria registry."""
    registry_path = Path("configs/criteria_registry.yaml")
    if registry_path.exists():
        with open(registry_path) as f:
            return yaml.safe_load(f)
    return {}


def compute_metrics_with_ci(per_query_csv: Path, n_bootstrap: int = 2000) -> Dict:
    """Compute all metrics with bootstrap CIs."""
    logger.info(f"Loading {per_query_csv}")
    df = pd.read_csv(per_query_csv)

    logger.info(f"Loaded {len(df)} queries")

    # Compute classification metrics
    class_metrics = compute_classification_metrics(df)

    # Compute ranking metrics
    rank_metrics = compute_ranking_metrics_from_csv(df)

    # Compute calibration metrics
    cal_metrics = compute_calibration_metrics(df)

    # Compute per-criterion metrics
    per_crit = compute_per_criterion_metrics(df)

    # Bootstrap CIs for key metrics
    logger.info("Computing bootstrap CIs...")

    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = df["has_evidence_gold"].values
    y_score = df["p4_prob_calibrated"].values

    auroc_ci = bootstrap_ci(
        lambda yt, ys: roc_auc_score(yt, ys),
        (y_true, y_score),
        n_bootstrap=n_bootstrap,
        seed=42,
    )

    auprc_ci = bootstrap_ci(
        lambda yt, ys: average_precision_score(yt, ys),
        (y_true, y_score),
        n_bootstrap=n_bootstrap,
        seed=42,
    )

    # Evidence recall CI (positives only)
    df_pos = df[df["has_evidence_gold"] == 1]
    recall_vals = df_pos["evidence_recall_at_k"].dropna().values

    if len(recall_vals) > 0:
        recall_ci = bootstrap_ci(
            lambda x, _: np.mean(x),
            (recall_vals, recall_vals),
            n_bootstrap=n_bootstrap,
            seed=42,
        )
    else:
        recall_ci = {}

    # Verify AUPRC != Recall
    verify_auprc_not_recall(
        class_metrics.get("auprc", 0),
        rank_metrics.get("evidence_recall_at_k", 0),
    )

    return {
        "classification": class_metrics,
        "ranking": rank_metrics,
        "calibration": cal_metrics,
        "per_criterion": per_crit,
        "confidence_intervals": {
            "auroc": auroc_ci,
            "auprc": auprc_ci,
            "evidence_recall_at_k": recall_ci,
        },
        "n_queries": len(df),
        "n_positive": int(df["has_evidence_gold"].sum()),
    }


def build_metrics_master(
    metrics: Dict,
    source_run: str,
    version: str,
    registry: Dict,
) -> Dict:
    """Build metrics_master.json structure."""

    # Get criterion descriptions from registry
    criterion_info = {}
    if "criteria" in registry:
        for crit_id, crit in registry["criteria"].items():
            criterion_info[crit_id] = {
                "short_name": crit.get("short_name", crit_id),
                "description": crit.get("short_name", crit_id),
            }

    # Per-criterion with descriptions
    per_criterion_with_desc = {}
    for crit_id, crit_metrics in metrics["per_criterion"].items():
        info = criterion_info.get(crit_id, {"description": crit_id})
        per_criterion_with_desc[crit_id] = {
            "description": info.get("description", crit_id),
            **crit_metrics,
        }

    return {
        "_meta": {
            "description": "Single source of truth for all metrics reported in the paper",
            "version": version,
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "generated_from": source_run,
            "generator_script": "scripts/reporting/build_paper_bundle.py",
            "dataset": "RedSM5",
            "n_queries": metrics["n_queries"],
            "n_queries_with_evidence": metrics["n_positive"],
        },
        "ranking_metrics": {
            "protocol": "positives_only",
            "protocol_description": "Computed only on queries where has_evidence=1",
            "split": "TEST",
            "seed": 42,
            "metrics": {
                "ndcg_at_10": {
                    "value": metrics["ranking"].get("ndcg_at_10", metrics.get("ranking", {}).get("ndcg@10")),
                    "note": "HPO-optimized on DEV split",
                },
                "evidence_recall_at_k": {
                    "value": metrics["ranking"].get("evidence_recall_at_k"),
                    "ci_95_lower": metrics["confidence_intervals"]["evidence_recall_at_k"].get("ci_lower"),
                    "ci_95_upper": metrics["confidence_intervals"]["evidence_recall_at_k"].get("ci_upper"),
                    "note": "Fraction of gold sentences in top-K",
                },
                "mrr": {
                    "value": metrics["ranking"].get("mrr"),
                    "note": "Mean Reciprocal Rank",
                },
            },
        },
        "classification_metrics": {
            "protocol": "all_queries",
            "protocol_description": "Computed on all queries (binary classification)",
            "split": "TEST",
            "seed": 42,
            "model": "P4 Criterion-Aware GNN",
            "metrics": {
                "auroc": {
                    "value": metrics["classification"]["auroc"],
                    "ci_95_lower": metrics["confidence_intervals"]["auroc"].get("ci_lower"),
                    "ci_95_upper": metrics["confidence_intervals"]["auroc"].get("ci_upper"),
                    "note": "Evidence detection AUROC",
                },
                "auprc": {
                    "value": metrics["classification"]["auprc"],
                    "ci_95_lower": metrics["confidence_intervals"]["auprc"].get("ci_lower"),
                    "ci_95_upper": metrics["confidence_intervals"]["auprc"].get("ci_upper"),
                    "note": "Area under precision-recall curve (NOT Recall@K)",
                },
            },
        },
        "calibration_metrics": {
            "protocol": "all_queries",
            "metrics": {
                "ece": {
                    "value": metrics["calibration"].get("ece"),
                    "note": "Expected Calibration Error",
                },
                "brier_score": {
                    "value": metrics["calibration"].get("brier_score"),
                    "note": "Brier score (lower is better)",
                },
            },
        },
        "per_criterion_performance": {
            "protocol": "all_queries",
            "split": "TEST",
            "criteria": per_criterion_with_desc,
        },
        "provenance": {
            "source_file": source_run,
            "generation_timestamp": datetime.now().isoformat(),
            "metric_contract": "docs/METRIC_CONTRACT.md",
        },
    }


def build_main_results_table(metrics: Dict) -> pd.DataFrame:
    """Build main results table."""
    rows = []

    # Ranking metrics (positives_only)
    if "ranking" in metrics:
        for name, value in metrics["ranking"].items():
            if isinstance(value, (int, float)) and not name.startswith("n_"):
                ci = metrics.get("confidence_intervals", {}).get(name, {})
                rows.append({
                    "metric": name,
                    "value": value,
                    "ci_95_lower": ci.get("ci_lower"),
                    "ci_95_upper": ci.get("ci_upper"),
                    "protocol": "positives_only",
                    "split": "TEST",
                })

    # Classification metrics (all_queries)
    if "classification" in metrics:
        for name in ["auroc", "auprc"]:
            if name in metrics["classification"]:
                ci = metrics.get("confidence_intervals", {}).get(name, {})
                rows.append({
                    "metric": name,
                    "value": metrics["classification"][name],
                    "ci_95_lower": ci.get("ci_lower"),
                    "ci_95_upper": ci.get("ci_upper"),
                    "protocol": "all_queries",
                    "split": "TEST",
                })

    # Calibration metrics
    if "calibration" in metrics:
        for name in ["ece", "brier_score"]:
            if name in metrics["calibration"]:
                rows.append({
                    "metric": name,
                    "value": metrics["calibration"][name],
                    "ci_95_lower": None,
                    "ci_95_upper": None,
                    "protocol": "all_queries",
                    "split": "TEST",
                })

    return pd.DataFrame(rows)


def build_per_criterion_table(metrics: Dict, registry: Dict) -> pd.DataFrame:
    """Build per-criterion performance table."""
    rows = []

    for crit_id in sorted(metrics["per_criterion"].keys()):
        crit_metrics = metrics["per_criterion"][crit_id]

        # Get description from registry
        desc = crit_id
        if "criteria" in registry and crit_id in registry["criteria"]:
            desc = registry["criteria"][crit_id].get("short_name", crit_id)

        rows.append({
            "criterion_id": crit_id,
            "description": desc,
            "n_queries": crit_metrics.get("n_queries"),
            "n_positive": crit_metrics.get("n_positive"),
            "positive_rate": crit_metrics.get("positive_rate"),
            "auroc": crit_metrics.get("auroc"),
            "auprc": crit_metrics.get("auprc"),
            "evidence_recall_at_k": crit_metrics.get("evidence_recall_at_k"),
            "mrr": crit_metrics.get("mrr"),
        })

    return pd.DataFrame(rows)


def compute_checksums(bundle_dir: Path) -> Dict[str, str]:
    """Compute SHA256 checksums for all files."""
    checksums = {}

    for file_path in sorted(bundle_dir.rglob("*")):
        if file_path.is_file() and file_path.name != "checksums.txt":
            relative_path = file_path.relative_to(bundle_dir)
            with open(file_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            checksums[str(relative_path)] = sha256

    return checksums


def write_checksums(checksums: Dict[str, str], output_path: Path):
    """Write checksums.txt file."""
    with open(output_path, "w") as f:
        for path, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {path}\n")


def write_manifest(bundle_dir: Path, version: str, source_run: str):
    """Write MANIFEST.md with regeneration instructions."""
    manifest = f"""# Paper Bundle {version} Manifest

Generated: {datetime.now().isoformat()}

## Contents

| File | Description |
|------|-------------|
| metrics_master.json | Single source of truth for all metrics |
| summary.json | Bundle metadata and high-level summary |
| tables/main_results.csv | Primary metrics with 95% CIs |
| tables/per_criterion.csv | Per-criterion performance breakdown |
| tables/baselines.csv | Baseline comparison (if available) |
| tables/robustness.csv | Robustness analysis (if available) |
| checksums.txt | SHA256 integrity verification |
| MANIFEST.md | This file |

## Regeneration

To regenerate this bundle:

```bash
python scripts/reporting/build_paper_bundle.py \\
    --version {version} \\
    --source_run {source_run} \\
    --output results/paper_bundle/{version}
```

## Verification

To verify bundle integrity:

```bash
python scripts/verification/verify_checksums.py \\
    --bundle results/paper_bundle/{version}
```

To cross-check metrics:

```bash
python scripts/verification/metric_crosscheck.py \\
    --bundle results/paper_bundle/{version}
```

## Source Data

| Source | Path |
|--------|------|
| Per-query results | {source_run}/per_query.csv |
| Criteria registry | configs/criteria_registry.yaml |
| Metric contract | docs/METRIC_CONTRACT.md |

## Protocols

- **positives_only**: Ranking metrics (nDCG, Recall, MRR) computed on queries with evidence
- **all_queries**: Classification metrics (AUROC, AUPRC) computed on all queries

## Notes

- AUROC/AUPRC: 95% bootstrap CIs (n=2000, seed=42)
- Ranking metrics: positives_only protocol
- Split: TEST (10% posts, post-ID disjoint)
"""

    with open(bundle_dir / "MANIFEST.md", "w") as f:
        f.write(manifest)


def build_paper_bundle(
    version: str,
    source_run: Path,
    output_dir: Path,
    n_bootstrap: int = 2000,
    include_baselines: bool = True,
    include_robustness: bool = True,
) -> None:
    """Build complete paper bundle.

    Args:
        version: Bundle version (e.g., "v3.0")
        source_run: Path to source evaluation run
        output_dir: Output directory for bundle
        n_bootstrap: Number of bootstrap iterations
        include_baselines: Include baseline comparison if available
        include_robustness: Include robustness analysis if available
    """
    logger.info("=" * 80)
    logger.info(f"BUILDING PAPER BUNDLE {version}")
    logger.info(f"Source: {source_run}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    # Load criteria registry
    registry = load_criteria_registry()

    # Find per_query.csv
    per_query_csv = source_run / "per_query.csv"
    if not per_query_csv.exists():
        raise FileNotFoundError(f"per_query.csv not found in {source_run}")

    # Compute metrics
    logger.info("Computing metrics with bootstrap CIs...")
    metrics = compute_metrics_with_ci(per_query_csv, n_bootstrap)

    # Build metrics_master.json
    logger.info("Building metrics_master.json...")
    metrics_master = build_metrics_master(
        metrics=metrics,
        source_run=str(source_run),
        version=version,
        registry=registry,
    )

    with open(output_dir / "metrics_master.json", "w") as f:
        json.dump(metrics_master, f, indent=2, default=str)

    # Build summary.json
    logger.info("Building summary.json...")
    summary = {
        "version": version,
        "generated": datetime.now().isoformat(),
        "source_run": str(source_run),
        "primary_metrics": {
            "auroc": metrics["classification"]["auroc"],
            "auprc": metrics["classification"]["auprc"],
            "evidence_recall_at_k": metrics["ranking"].get("evidence_recall_at_k"),
            "mrr": metrics["ranking"].get("mrr"),
        },
        "n_queries": metrics["n_queries"],
        "n_positive": metrics["n_positive"],
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Build tables
    logger.info("Building tables...")

    # Main results
    main_results = build_main_results_table(metrics)
    main_results.to_csv(output_dir / "tables" / "main_results.csv", index=False)

    # Per-criterion
    per_criterion = build_per_criterion_table(metrics, registry)
    per_criterion.to_csv(output_dir / "tables" / "per_criterion.csv", index=False)

    # Copy baselines if available
    if include_baselines:
        baselines_dir = Path("outputs/baselines")
        if baselines_dir.exists():
            # Find latest baseline run
            baseline_runs = sorted(baselines_dir.iterdir(), reverse=True)
            for run_dir in baseline_runs:
                baseline_csv = run_dir / "baseline_comparison.csv"
                if baseline_csv.exists():
                    logger.info(f"Including baselines from {run_dir.name}")
                    import shutil
                    shutil.copy(baseline_csv, output_dir / "tables" / "baselines.csv")
                    break

    # Copy robustness if available
    if include_robustness:
        robustness_dir = Path("outputs/robustness")
        if robustness_dir.exists():
            robustness_runs = sorted(robustness_dir.iterdir(), reverse=True)
            for run_dir in robustness_runs:
                robustness_csv = run_dir / "robustness_summary.csv"
                if robustness_csv.exists():
                    logger.info(f"Including robustness from {run_dir.name}")
                    import shutil
                    shutil.copy(robustness_csv, output_dir / "tables" / "robustness.csv")
                    break

    # Write MANIFEST.md
    logger.info("Writing MANIFEST.md...")
    write_manifest(output_dir, version, str(source_run))

    # Compute and write checksums
    logger.info("Computing checksums...")
    checksums = compute_checksums(output_dir)
    write_checksums(checksums, output_dir / "checksums.txt")

    # Verify checksums
    logger.info("Verifying checksums...")
    for file_path, expected in checksums.items():
        full_path = output_dir / file_path
        with open(full_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            logger.error(f"Checksum mismatch: {file_path}")
        else:
            logger.info(f"  ✓ {file_path}")

    logger.info("=" * 80)
    logger.info(f"PAPER BUNDLE {version} COMPLETE")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Print summary
    print("\nBundle Contents:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(output_dir)}")

    print(f"\nPrimary Metrics:")
    print(f"  AUROC: {metrics['classification']['auroc']:.4f}")
    print(f"  AUPRC: {metrics['classification']['auprc']:.4f}")
    print(f"  Evidence Recall@K: {metrics['ranking'].get('evidence_recall_at_k', 'N/A')}")
    print(f"  MRR: {metrics['ranking'].get('mrr', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Build paper bundle")
    parser.add_argument(
        "--version",
        type=str,
        default="v3.0",
        help="Bundle version"
    )
    parser.add_argument(
        "--source_run",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete"),
        help="Path to source evaluation run"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/paper_bundle/{version})"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap iterations"
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip including baselines"
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="Skip including robustness"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = Path("results/paper_bundle") / args.version

    build_paper_bundle(
        version=args.version,
        source_run=args.source_run,
        output_dir=args.output,
        n_bootstrap=args.n_bootstrap,
        include_baselines=not args.no_baselines,
        include_robustness=not args.no_robustness,
    )


if __name__ == "__main__":
    main()
