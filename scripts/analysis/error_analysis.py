#!/usr/bin/env python3
"""Systematic qualitative error analysis for publication.

Generates:
1. Stratified error analysis by criterion, post length, evidence density
2. Confusion matrices (overall and per-criterion)
3. Failure mode categorization
4. Representative error examples (anonymized)

Usage:
    python scripts/analysis/error_analysis.py \
        --per_query outputs/final_research_eval/20260118_031312_complete/per_query.csv \
        --output outputs/error_analysis/
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Criterion registry for readable names (per ReDSM5 taxonomy)
CRITERION_NAMES = {
    "A.1": "Depressed Mood",
    "A.2": "Anhedonia",
    "A.3": "Weight/Appetite Change",
    "A.4": "Sleep Disturbance",
    "A.5": "Psychomotor Changes",
    "A.6": "Fatigue/Loss of Energy",
    "A.7": "Worthlessness/Guilt",
    "A.8": "Concentration Difficulty",
    "A.9": "Suicidal Ideation",
    "A.10": "SPECIAL_CASE",  # Per ReDSM5: expert discrimination cases
}


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute confusion matrix and derived metrics."""
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "total": int(total),
    }


def categorize_failure_mode(row: pd.Series, threshold: float = 0.5) -> str:
    """Categorize the type of failure for a query."""
    gold = row["has_evidence_gold"]
    prob = row["p4_prob_calibrated"]
    pred = 1 if prob >= threshold else 0
    recall = row.get("evidence_recall_at_k", 0)

    if gold == 1 and pred == 0:
        # False negative - missed evidence
        if prob < 0.2:
            return "FN_HIGH_CONFIDENCE"  # Model was very confident there's no evidence
        elif prob < 0.4:
            return "FN_MODERATE_CONFIDENCE"
        else:
            return "FN_BORDERLINE"  # Close to threshold

    elif gold == 0 and pred == 1:
        # False positive - hallucinated evidence
        if prob > 0.8:
            return "FP_HIGH_CONFIDENCE"  # Model was very confident there is evidence
        elif prob > 0.6:
            return "FP_MODERATE_CONFIDENCE"
        else:
            return "FP_BORDERLINE"

    elif gold == 1 and pred == 1:
        # True positive - check ranking quality
        if pd.notna(recall):
            if recall < 0.3:
                return "TP_POOR_RANKING"  # Detected but ranked poorly
            elif recall < 0.7:
                return "TP_MODERATE_RANKING"
            else:
                return "TP_GOOD_RANKING"
        return "TP_NO_RANKING_INFO"

    else:
        # True negative
        return "TN"


def analyze_errors_by_criterion(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Analyze error rates stratified by criterion."""
    results = []

    for criterion_id in sorted(df["criterion_id"].unique()):
        subset = df[df["criterion_id"] == criterion_id]
        y_true = subset["has_evidence_gold"].values
        y_pred = (subset["p4_prob_calibrated"] >= threshold).astype(int).values

        cm = compute_confusion_matrix(y_true, y_pred)

        # Compute ranking metrics for positives
        pos_subset = subset[subset["has_evidence_gold"] == 1]
        mean_recall = pos_subset["evidence_recall_at_k"].mean() if len(pos_subset) > 0 else 0
        mean_mrr = pos_subset["mrr"].mean() if len(pos_subset) > 0 else 0

        results.append({
            "criterion_id": criterion_id,
            "criterion_name": CRITERION_NAMES.get(criterion_id, criterion_id),
            "n_queries": len(subset),
            "n_positive": int(y_true.sum()),
            "positive_rate": float(y_true.mean()),
            **cm,
            "mean_recall_at_k": float(mean_recall) if pd.notna(mean_recall) else 0,
            "mean_mrr": float(mean_mrr) if pd.notna(mean_mrr) else 0,
        })

    return pd.DataFrame(results)


def analyze_errors_by_evidence_density(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Analyze error rates by evidence density (positive rate per post)."""
    # Compute evidence density per post
    post_evidence = df.groupby("post_id")["has_evidence_gold"].agg(["sum", "count"])
    post_evidence["density"] = post_evidence["sum"] / post_evidence["count"]

    # Create density bins
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
    labels = ["0-10%", "10-20%", "20-30%", "30-50%", "50-100%"]
    post_evidence["density_bin"] = pd.cut(post_evidence["density"], bins=bins, labels=labels, include_lowest=True)

    # Merge back
    df_with_density = df.merge(post_evidence[["density_bin"]], left_on="post_id", right_index=True)

    results = []
    for density_bin in labels:
        subset = df_with_density[df_with_density["density_bin"] == density_bin]
        if len(subset) == 0:
            continue

        y_true = subset["has_evidence_gold"].values
        y_pred = (subset["p4_prob_calibrated"] >= threshold).astype(int).values

        cm = compute_confusion_matrix(y_true, y_pred)

        results.append({
            "evidence_density_bin": density_bin,
            "n_queries": len(subset),
            "n_posts": subset["post_id"].nunique(),
            **cm,
        })

    return pd.DataFrame(results)


def analyze_failure_modes(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Categorize and count failure modes."""
    df = df.copy()
    df["failure_mode"] = df.apply(lambda row: categorize_failure_mode(row, threshold), axis=1)

    # Count by failure mode
    mode_counts = df["failure_mode"].value_counts().to_dict()

    # Count by failure mode and criterion
    mode_by_criterion = df.groupby(["criterion_id", "failure_mode"]).size().unstack(fill_value=0)

    # Get summary stats for each failure mode
    failure_summary = []
    for mode in df["failure_mode"].unique():
        subset = df[df["failure_mode"] == mode]
        failure_summary.append({
            "failure_mode": mode,
            "count": len(subset),
            "percentage": len(subset) / len(df) * 100,
            "mean_prob": subset["p4_prob_calibrated"].mean(),
            "std_prob": subset["p4_prob_calibrated"].std(),
            "criteria_affected": subset["criterion_id"].nunique(),
            "top_criteria": subset["criterion_id"].value_counts().head(3).to_dict(),
        })

    return pd.DataFrame(failure_summary), mode_by_criterion


def generate_anonymized_examples(
    df: pd.DataFrame,
    n_examples_per_mode: int = 3,
    threshold: float = 0.5,
) -> List[Dict]:
    """Generate anonymized representative examples for each failure mode.

    Note: This generates STATISTICAL examples only, not actual post content.
    Actual content cannot be shared due to privacy constraints.
    """
    df = df.copy()
    df["failure_mode"] = df.apply(lambda row: categorize_failure_mode(row, threshold), axis=1)

    examples = []

    for mode in df["failure_mode"].unique():
        if mode == "TN":
            continue  # Skip true negatives

        subset = df[df["failure_mode"] == mode]

        # Sample representative queries (by statistics, not content)
        sampled = subset.sample(min(n_examples_per_mode, len(subset)), random_state=42)

        for _, row in sampled.iterrows():
            examples.append({
                "failure_mode": mode,
                "criterion_id": row["criterion_id"],
                "criterion_name": CRITERION_NAMES.get(row["criterion_id"], row["criterion_id"]),
                "has_evidence_gold": int(row["has_evidence_gold"]),
                "predicted_prob": round(float(row["p4_prob_calibrated"]), 4),
                "n_candidates": int(row["n_candidates"]),
                "evidence_recall_at_k": round(float(row["evidence_recall_at_k"]), 4) if pd.notna(row["evidence_recall_at_k"]) else None,
                "mrr": round(float(row["mrr"]), 4) if pd.notna(row["mrr"]) else None,
                # Anonymized identifiers
                "example_id": f"{mode}_{row['criterion_id']}_{hash(row['post_id']) % 10000:04d}",
                "note": "Content not shown due to privacy constraints. Statistics only.",
            })

    return examples


def generate_report(
    criterion_analysis: pd.DataFrame,
    density_analysis: pd.DataFrame,
    failure_summary: pd.DataFrame,
    overall_cm: Dict,
    output_dir: Path,
) -> None:
    """Generate markdown error analysis report."""
    report = f"""# Error Analysis Report

Generated: {datetime.now().isoformat()}

---

## Executive Summary

This report provides systematic qualitative error analysis for the evidence binding pipeline,
stratified by criterion, evidence density, and failure mode category.

### Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | {overall_cm['accuracy']:.4f} |
| Precision | {overall_cm['precision']:.4f} |
| Recall | {overall_cm['recall']:.4f} |
| F1 Score | {overall_cm['f1']:.4f} |
| Specificity | {overall_cm['specificity']:.4f} |

### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | {overall_cm['tp']} (TP) | {overall_cm['fn']} (FN) |
| **Actual Negative** | {overall_cm['fp']} (FP) | {overall_cm['tn']} (TN) |

---

## 1. Error Analysis by DSM-5 Criterion

| Criterion | Name | N | Pos Rate | Precision | Recall | F1 | FP | FN |
|-----------|------|---|----------|-----------|--------|----|----|----|
"""

    for _, row in criterion_analysis.iterrows():
        report += (
            f"| {row['criterion_id']} | {row['criterion_name']} | {row['n_queries']} | "
            f"{row['positive_rate']:.1%} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['fp']} | {row['fn']} |\n"
        )

    report += """
### Key Observations

"""
    # Find worst performing criteria
    worst_f1 = criterion_analysis.nsmallest(3, "f1")
    for _, row in worst_f1.iterrows():
        report += f"- **{row['criterion_id']} ({row['criterion_name']})**: F1={row['f1']:.3f} - "
        if row['precision'] < row['recall']:
            report += f"Low precision ({row['precision']:.3f}), high FP rate\n"
        else:
            report += f"Low recall ({row['recall']:.3f}), high FN rate\n"

    report += """
---

## 2. Error Analysis by Evidence Density

Evidence density = fraction of criteria with evidence per post.

| Density Bin | N Queries | N Posts | Precision | Recall | F1 |
|-------------|-----------|---------|-----------|--------|----|
"""

    for _, row in density_analysis.iterrows():
        report += (
            f"| {row['evidence_density_bin']} | {row['n_queries']} | {row['n_posts']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |\n"
        )

    report += """
---

## 3. Failure Mode Analysis

| Failure Mode | Count | % | Mean Prob | Top Criteria |
|--------------|-------|---|-----------|--------------|
"""

    for _, row in failure_summary.sort_values("count", ascending=False).iterrows():
        top_crit = ", ".join([f"{k}({v})" for k, v in list(row["top_criteria"].items())[:2]])
        report += (
            f"| {row['failure_mode']} | {row['count']} | {row['percentage']:.1f}% | "
            f"{row['mean_prob']:.3f} | {top_crit} |\n"
        )

    report += """
### Failure Mode Definitions

- **FN_HIGH_CONFIDENCE**: Model predicted <0.2 probability but evidence exists
- **FN_MODERATE_CONFIDENCE**: Model predicted 0.2-0.4 probability but evidence exists
- **FN_BORDERLINE**: Model predicted 0.4-0.5 probability but evidence exists
- **FP_HIGH_CONFIDENCE**: Model predicted >0.8 probability but no evidence
- **FP_MODERATE_CONFIDENCE**: Model predicted 0.6-0.8 probability but no evidence
- **FP_BORDERLINE**: Model predicted 0.5-0.6 probability but no evidence
- **TP_POOR_RANKING**: Correctly detected evidence but Recall@K < 0.3
- **TP_MODERATE_RANKING**: Correctly detected evidence, Recall@K 0.3-0.7
- **TP_GOOD_RANKING**: Correctly detected evidence, Recall@K > 0.7

---

## 4. Clinical Implications

### High-Risk Failure Modes

1. **FN on A.9 (Suicidal Ideation)**: Any false negative on suicide-related content is safety-critical.
   The model should be calibrated for high recall on this criterion.

2. **FP_HIGH_CONFIDENCE**: Cases where the model is very confident but wrong suggest
   systematic bias or confounding patterns in the training data.

3. **TP_POOR_RANKING**: Even when evidence is detected, poor ranking means clinicians
   may need to review more candidates than necessary.

### Recommendations

1. Consider threshold adjustment per criterion based on clinical priority
2. A.9 (Suicidal Ideation) should use lower threshold to maximize recall
3. A.10 (SPECIAL_CASE) has lowest performance - heterogeneous expert cases are harder to detect

---

## 5. Reproducibility

```bash
python scripts/analysis/error_analysis.py \\
    --per_query outputs/final_research_eval/20260118_031312_complete/per_query.csv \\
    --output outputs/error_analysis/
```

---

## Notes

- All examples are anonymized; actual post content cannot be shared due to privacy constraints
- Statistics computed using threshold=0.5 for binary classification
- Error analysis is stratified to identify systematic patterns, not individual failures
"""

    with open(output_dir / "ERROR_ANALYSIS_REPORT.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'ERROR_ANALYSIS_REPORT.md'}")


def main():
    parser = argparse.ArgumentParser(description="Systematic error analysis")
    parser.add_argument(
        "--per_query",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Path to per_query.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/error_analysis"),
        help="Output directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("SYSTEMATIC ERROR ANALYSIS")
    logger.info("=" * 80)

    # Load data
    if not args.per_query.exists():
        logger.error(f"per_query.csv not found: {args.per_query}")
        return 1

    df = pd.read_csv(args.per_query)
    logger.info(f"Loaded {len(df)} queries from {args.per_query}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Compute overall confusion matrix
    logger.info("Computing overall confusion matrix...")
    y_true = df["has_evidence_gold"].values
    y_pred = (df["p4_prob_calibrated"] >= args.threshold).astype(int).values
    overall_cm = compute_confusion_matrix(y_true, y_pred)
    with open(args.output / "confusion_matrix_overall.json", "w") as f:
        json.dump(overall_cm, f, indent=2)

    # 2. Analyze by criterion
    logger.info("Analyzing errors by criterion...")
    criterion_analysis = analyze_errors_by_criterion(df, args.threshold)
    criterion_analysis.to_csv(args.output / "errors_by_criterion.csv", index=False)

    # 3. Analyze by evidence density
    logger.info("Analyzing errors by evidence density...")
    density_analysis = analyze_errors_by_evidence_density(df, args.threshold)
    density_analysis.to_csv(args.output / "errors_by_density.csv", index=False)

    # 4. Analyze failure modes
    logger.info("Categorizing failure modes...")
    failure_summary, mode_by_criterion = analyze_failure_modes(df, args.threshold)
    failure_summary.to_csv(args.output / "failure_modes.csv", index=False)
    mode_by_criterion.to_csv(args.output / "failure_modes_by_criterion.csv")

    # 5. Generate anonymized examples
    logger.info("Generating anonymized examples...")
    examples = generate_anonymized_examples(df, n_examples_per_mode=5, threshold=args.threshold)
    with open(args.output / "anonymized_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    # 6. Generate report
    logger.info("Generating report...")
    generate_report(criterion_analysis, density_analysis, failure_summary, overall_cm, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOverall Performance (threshold={args.threshold}):")
    print(f"  Accuracy:  {overall_cm['accuracy']:.4f}")
    print(f"  Precision: {overall_cm['precision']:.4f}")
    print(f"  Recall:    {overall_cm['recall']:.4f}")
    print(f"  F1:        {overall_cm['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {overall_cm['tp']:5d}  FP: {overall_cm['fp']:5d}")
    print(f"  FN: {overall_cm['fn']:5d}  TN: {overall_cm['tn']:5d}")
    print(f"\nTop Failure Modes:")
    for _, row in failure_summary.nlargest(5, "count").iterrows():
        if row["failure_mode"] != "TN":
            print(f"  {row['failure_mode']}: {row['count']} ({row['percentage']:.1f}%)")
    print(f"\nResults saved to: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
