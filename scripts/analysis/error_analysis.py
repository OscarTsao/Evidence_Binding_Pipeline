#!/usr/bin/env python3
"""Error analysis for evidence retrieval pipeline.

Analyzes failure modes and generates insights for:
1. False negatives (missed evidence)
2. False positives (retrieved non-evidence)
3. Per-criterion error patterns
4. Post-level difficulty analysis

Usage:
    python scripts/analysis/error_analysis.py \
        --predictions outputs/eval/per_query.csv \
        --groundtruth data/groundtruth/evidence_sentence_groundtruth.csv \
        --output outputs/analysis/
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load per-query predictions."""
    df = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(df)} predictions")
    return df


def load_groundtruth(groundtruth_path: Path) -> pd.DataFrame:
    """Load groundtruth labels."""
    df = pd.read_csv(groundtruth_path)
    logger.info(f"Loaded {len(df)} groundtruth rows")
    return df


def analyze_false_negatives(
    predictions: pd.DataFrame,
    groundtruth: pd.DataFrame,
    threshold: float = 0.5
) -> Dict:
    """Analyze false negatives (missed evidence).

    Args:
        predictions: Per-query predictions with recall scores
        groundtruth: Groundtruth labels
        threshold: Recall threshold for considering a query "failed"

    Returns:
        Analysis results
    """
    logger.info("Analyzing false negatives...")

    # Filter queries with evidence
    with_evidence = predictions[predictions["has_evidence"] == 1].copy()

    if "recall@10" not in with_evidence.columns:
        logger.warning("recall@10 not found in predictions")
        return {"error": "Missing recall@10 column"}

    # Identify failures (low recall)
    failures = with_evidence[with_evidence["recall@10"] < threshold]

    analysis = {
        "total_with_evidence": len(with_evidence),
        "n_failures": len(failures),
        "failure_rate": len(failures) / len(with_evidence) if len(with_evidence) > 0 else 0,
        "threshold": threshold,
    }

    if len(failures) > 0:
        # Analyze by criterion
        criterion_failures = failures.groupby("criterion_id").size()
        criterion_totals = with_evidence.groupby("criterion_id").size()
        criterion_rates = (criterion_failures / criterion_totals).fillna(0)

        analysis["per_criterion_failure_rates"] = criterion_rates.to_dict()
        analysis["worst_criterion"] = criterion_rates.idxmax() if len(criterion_rates) > 0 else None

        # Analyze by n_gold (number of evidence sentences)
        if "n_gold" in failures.columns:
            gold_dist = failures["n_gold"].describe().to_dict()
            analysis["n_gold_stats"] = {k: float(v) for k, v in gold_dist.items()}

    return analysis


def analyze_false_positives(
    predictions: pd.DataFrame,
    groundtruth: pd.DataFrame,
) -> Dict:
    """Analyze false positives (retrieved non-evidence).

    This would require access to the actual retrieved sentences,
    which may not be in the predictions file.
    """
    logger.info("Analyzing false positives...")

    # Filter queries without evidence
    no_evidence = predictions[predictions["has_evidence"] == 0].copy()

    analysis = {
        "total_no_evidence": len(no_evidence),
        "note": "Full FP analysis requires access to retrieved sentences"
    }

    # If we have retrieval counts
    if "n_retrieved" in no_evidence.columns:
        avg_retrieved = no_evidence["n_retrieved"].mean()
        analysis["avg_retrieved_for_no_evidence"] = float(avg_retrieved)

    return analysis


def analyze_per_criterion(
    predictions: pd.DataFrame,
    groundtruth: pd.DataFrame,
) -> Dict:
    """Per-criterion stratified analysis."""
    logger.info("Analyzing per-criterion performance...")

    with_evidence = predictions[predictions["has_evidence"] == 1].copy()

    if len(with_evidence) == 0:
        return {"error": "No queries with evidence"}

    # Group by criterion
    criterion_stats = []
    for criterion_id, group in with_evidence.groupby("criterion_id"):
        stats = {
            "criterion_id": criterion_id,
            "n_queries": len(group),
        }

        for metric in ["recall@10", "ndcg@10", "mrr"]:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    stats[f"{metric}_mean"] = float(values.mean())
                    stats[f"{metric}_std"] = float(values.std())
                    stats[f"{metric}_min"] = float(values.min())
                    stats[f"{metric}_max"] = float(values.max())

        criterion_stats.append(stats)

    # Sort by performance
    criterion_df = pd.DataFrame(criterion_stats)
    if "ndcg@10_mean" in criterion_df.columns:
        criterion_df = criterion_df.sort_values("ndcg@10_mean", ascending=False)

    return {
        "per_criterion": criterion_df.to_dict("records"),
        "best_criterion": criterion_df.iloc[0]["criterion_id"] if len(criterion_df) > 0 else None,
        "worst_criterion": criterion_df.iloc[-1]["criterion_id"] if len(criterion_df) > 0 else None,
    }


def analyze_difficulty(
    predictions: pd.DataFrame,
    groundtruth: pd.DataFrame,
) -> Dict:
    """Analyze query difficulty based on performance."""
    logger.info("Analyzing query difficulty...")

    with_evidence = predictions[predictions["has_evidence"] == 1].copy()

    if "ndcg@10" not in with_evidence.columns:
        return {"error": "Missing ndcg@10 column"}

    # Categorize by difficulty
    with_evidence["difficulty"] = pd.cut(
        with_evidence["ndcg@10"],
        bins=[0, 0.5, 0.7, 0.9, 1.0],
        labels=["Hard", "Medium", "Easy", "Very Easy"]
    )

    difficulty_dist = with_evidence["difficulty"].value_counts().to_dict()
    difficulty_dist = {str(k): int(v) for k, v in difficulty_dist.items()}

    # Analyze hard cases
    hard_cases = with_evidence[with_evidence["ndcg@10"] < 0.5]

    analysis = {
        "difficulty_distribution": difficulty_dist,
        "n_hard_cases": len(hard_cases),
        "hard_case_rate": len(hard_cases) / len(with_evidence) if len(with_evidence) > 0 else 0,
    }

    # Hard cases by criterion
    if len(hard_cases) > 0:
        hard_by_criterion = hard_cases.groupby("criterion_id").size().to_dict()
        analysis["hard_cases_by_criterion"] = {str(k): int(v) for k, v in hard_by_criterion.items()}

    return analysis


def generate_error_report(
    fn_analysis: Dict,
    fp_analysis: Dict,
    criterion_analysis: Dict,
    difficulty_analysis: Dict,
    output_dir: Path,
) -> None:
    """Generate comprehensive error analysis report."""
    report = f"""# Error Analysis Report

Generated: {datetime.now().isoformat()}

---

## Executive Summary

- **Total queries with evidence:** {fn_analysis.get('total_with_evidence', 'N/A')}
- **Failure rate (Recall@10 < 0.5):** {fn_analysis.get('failure_rate', 0)*100:.1f}%
- **Hard case rate (nDCG@10 < 0.5):** {difficulty_analysis.get('hard_case_rate', 0)*100:.1f}%

---

## 1. False Negative Analysis

False negatives occur when the system misses relevant evidence sentences.

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total queries with evidence | {fn_analysis.get('total_with_evidence', 'N/A')} |
| Number of failures | {fn_analysis.get('n_failures', 'N/A')} |
| Failure rate | {fn_analysis.get('failure_rate', 0)*100:.2f}% |

### Per-Criterion Failure Rates

"""

    if "per_criterion_failure_rates" in fn_analysis:
        for criterion, rate in sorted(fn_analysis["per_criterion_failure_rates"].items()):
            report += f"- **{criterion}**: {rate*100:.1f}%\n"

    report += f"""
### Worst Performing Criterion

**{fn_analysis.get('worst_criterion', 'N/A')}** - This criterion has the highest failure rate.

---

## 2. Per-Criterion Performance

"""

    if "per_criterion" in criterion_analysis:
        report += "| Criterion | n_queries | nDCG@10 | Recall@10 | MRR |\n"
        report += "|-----------|-----------|---------|-----------|-----|\n"
        for c in criterion_analysis["per_criterion"]:
            ndcg = c.get("ndcg@10_mean", 0)
            recall = c.get("recall@10_mean", 0)
            mrr = c.get("mrr_mean", 0)
            report += f"| {c['criterion_id']} | {c['n_queries']} | {ndcg:.4f} | {recall:.4f} | {mrr:.4f} |\n"

    report += f"""
### Key Findings

- **Best criterion:** {criterion_analysis.get('best_criterion', 'N/A')}
- **Worst criterion:** {criterion_analysis.get('worst_criterion', 'N/A')}

---

## 3. Difficulty Analysis

"""

    if "difficulty_distribution" in difficulty_analysis:
        report += "| Difficulty | Count |\n"
        report += "|------------|-------|\n"
        for diff, count in difficulty_analysis["difficulty_distribution"].items():
            report += f"| {diff} | {count} |\n"

    report += f"""
### Hard Cases by Criterion

"""

    if "hard_cases_by_criterion" in difficulty_analysis:
        for criterion, count in sorted(difficulty_analysis["hard_cases_by_criterion"].items()):
            report += f"- **{criterion}**: {count} hard cases\n"

    report += """
---

## 4. Recommendations

Based on this analysis:

1. **Focus on hard criteria**: Criteria with high failure rates may benefit from:
   - Domain-specific fine-tuning
   - Additional training data
   - Criteria-specific prompting

2. **Error patterns**: Common failure modes may indicate:
   - Lexical mismatch between criteria and evidence
   - Implicit evidence requiring inference
   - Rare or unusual expressions

3. **Model improvements**: Consider:
   - Ensemble methods for hard cases
   - Human-in-the-loop for uncertain predictions
   - Active learning on failure cases

---

## Methodology

- False negatives: Queries with Recall@10 < 0.5
- Hard cases: Queries with nDCG@10 < 0.5
- Analysis performed on test split (seed=42)
"""

    with open(output_dir / "error_analysis_report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'error_analysis_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Error analysis")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Per-query predictions CSV"
    )
    parser.add_argument(
        "--groundtruth",
        type=Path,
        default=Path("data/groundtruth/evidence_sentence_groundtruth.csv"),
        help="Groundtruth CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis"),
        help="Output directory"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    predictions = load_predictions(args.predictions)
    groundtruth = load_groundtruth(args.groundtruth)

    # Run analyses
    fn_analysis = analyze_false_negatives(predictions, groundtruth)
    fp_analysis = analyze_false_positives(predictions, groundtruth)
    criterion_analysis = analyze_per_criterion(predictions, groundtruth)
    difficulty_analysis = analyze_difficulty(predictions, groundtruth)

    # Save JSON results
    results = {
        "timestamp": datetime.now().isoformat(),
        "false_negatives": fn_analysis,
        "false_positives": fp_analysis,
        "per_criterion": criterion_analysis,
        "difficulty": difficulty_analysis,
    }

    with open(args.output / "error_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    generate_error_report(
        fn_analysis=fn_analysis,
        fp_analysis=fp_analysis,
        criterion_analysis=criterion_analysis,
        difficulty_analysis=difficulty_analysis,
        output_dir=args.output,
    )

    logger.info("Error analysis complete")


if __name__ == "__main__":
    main()
