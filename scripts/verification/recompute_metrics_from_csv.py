#!/usr/bin/env python3
"""
Metric Verification Script - Recompute all metrics from per_query.csv

This script provides an independent verification of all reported metrics by
recomputing them directly from the per-query predictions CSV file. This ensures:
1. No silent bugs in metric computation
2. Audit trail for all reported numbers
3. Sanity checks on metric ranges

Usage:
    python scripts/verification/recompute_metrics_from_csv.py \\
        --per_query_csv outputs/final_eval/XXX/per_query.csv \\
        --summary_json outputs/final_eval/XXX/summary.json \\
        --output_dir outputs/final_eval/XXX/verification

Returns:
    Exit code 0 if all metrics match within tolerance
    Exit code 1 if discrepancies found or sanity checks fail
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)


def compute_ranking_metrics(
    predictions: pd.DataFrame,
    k_values: List[int] = [1, 3, 5, 10, 20]
) -> Dict:
    """
    Compute ranking metrics from per-query predictions.

    Assumes columns: query_id, has_evidence_gold, rank, score
    """
    metrics = {}

    # Group by query
    for k in k_values:
        recalls = []
        precisions = []
        hits = []
        reciprocal_ranks = []
        avg_precisions = []

        for query_id, group in predictions.groupby('query_id'):
            # Get top-k
            topk = group.nsmallest(k, 'rank')  # Lower rank = better

            # Get gold positives
            gold_positives = group[group['has_evidence_gold'] == 1]
            n_relevant = len(gold_positives)

            if n_relevant == 0:
                # No relevant items for this query, skip
                continue

            # Recall@K
            retrieved_relevant = topk[topk['has_evidence_gold'] == 1]
            recall = len(retrieved_relevant) / n_relevant if n_relevant > 0 else 0.0
            recalls.append(recall)

            # Precision@K
            precision = len(retrieved_relevant) / k if k > 0 else 0.0
            precisions.append(precision)

            # Hit@K (binary: was ANY relevant item retrieved?)
            hit = 1.0 if len(retrieved_relevant) > 0 else 0.0
            hits.append(hit)

            # MRR contribution (rank of first relevant item)
            if len(retrieved_relevant) > 0:
                first_relevant_rank = retrieved_relevant['rank'].min()
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

            # AP@K (average precision)
            precisions_at_i = []
            relevances_at_i = []
            for i in range(1, k + 1):
                top_i = group.nsmallest(i, 'rank')
                relevant_in_top_i = top_i[top_i['has_evidence_gold'] == 1]
                p_at_i = len(relevant_in_top_i) / i
                is_relevant = 1 if i <= len(topk) and topk.iloc[i-1]['has_evidence_gold'] == 1 else 0
                precisions_at_i.append(p_at_i)
                relevances_at_i.append(is_relevant)

            if sum(relevances_at_i) > 0:
                ap = sum([p * r for p, r in zip(precisions_at_i, relevances_at_i)]) / n_relevant
            else:
                ap = 0.0
            avg_precisions.append(ap)

        metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0
        metrics[f'Precision@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'HitRate@{k}'] = np.mean(hits) if hits else 0.0
        metrics[f'MRR@{k}'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        metrics[f'MAP@{k}'] = np.mean(avg_precisions) if avg_precisions else 0.0

    # nDCG@K requires more complex computation
    # Simplified implementation
    for k in k_values:
        ndcgs = []
        for query_id, group in predictions.groupby('query_id'):
            topk = group.nsmallest(k, 'rank')
            relevances = topk['has_evidence_gold'].values

            # DCG
            dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevances)])

            # IDCG (ideal DCG)
            ideal_relevances = sorted(group['has_evidence_gold'].values, reverse=True)[:k]
            idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances)])

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

        metrics[f'nDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0

    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict:
    """Compute binary classification metrics."""
    metrics = {}

    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['Specificity'] = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['BalancedAccuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # NPV (Negative Predictive Value)
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fn = (y_true * (1 - y_pred)).sum()
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # Probabilistic metrics (if probabilities provided)
    if y_prob is not None:
        try:
            metrics['AUROC'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['AUROC'] = None

        try:
            metrics['AUPRC'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['AUPRC'] = None

        metrics['Brier'] = brier_score_loss(y_true, y_prob)

        # TPR at fixed FPR thresholds
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        for target_fpr in [0.01, 0.03, 0.05, 0.10]:
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                metrics[f'TPR@FPR{int(target_fpr*100)}%'] = tpr[idx[-1]]
            else:
                metrics[f'TPR@FPR{int(target_fpr*100)}%'] = None

    return metrics


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute calibration metrics (ECE, MCE)."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # ECE (Expected Calibration Error)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        bin_counts.append(mask.sum())

    bin_counts = np.array(bin_counts)
    bin_weights = bin_counts / len(y_prob)

    calibration_errors = np.abs(prob_true - prob_pred)
    ece = np.sum(bin_weights * calibration_errors)
    mce = np.max(calibration_errors)

    return {
        'ECE': ece,
        'MCE': mce,
        'reliability_curve': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
    }


def sanity_check_metrics(metrics: Dict) -> List[str]:
    """
    Sanity check all metrics are in valid ranges.

    Returns list of error messages (empty if all checks pass).
    """
    errors = []

    # Metrics that must be in [0, 1]
    zero_one_metrics = [
        'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1',
        'BalancedAccuracy', 'NPV', 'AUROC', 'AUPRC', 'Brier', 'ECE', 'MCE'
    ]

    for metric in zero_one_metrics:
        if metric in metrics and metrics[metric] is not None:
            value = metrics[metric]
            if not (0.0 <= value <= 1.0):
                errors.append(f"{metric} = {value:.4f} is outside [0, 1]")

    # MCC must be in [-1, 1]
    if 'MCC' in metrics and metrics['MCC'] is not None:
        value = metrics['MCC']
        if not (-1.0 <= value <= 1.0):
            errors.append(f"MCC = {value:.4f} is outside [-1, 1]")

    # Ranking metrics must be in [0, 1]
    ranking_metrics = ['Recall', 'Precision', 'HitRate', 'MRR', 'MAP', 'nDCG']
    for base in ranking_metrics:
        for k in [1, 3, 5, 10, 20]:
            metric = f"{base}@{k}"
            if metric in metrics and metrics[metric] is not None:
                value = metrics[metric]
                if not (0.0 <= value <= 1.0):
                    errors.append(f"{metric} = {value:.4f} is outside [0, 1]")

    # Logical checks
    if 'Recall' in metrics and 'Precision' in metrics:
        if metrics['F1'] is not None:
            r, p, f1 = metrics['Recall'], metrics['Precision'], metrics['F1']
            if r + p > 0:
                expected_f1 = 2 * r * p / (r + p)
                if abs(f1 - expected_f1) > 0.01:
                    errors.append(f"F1 = {f1:.4f} inconsistent with R={r:.4f}, P={p:.4f} (expected {expected_f1:.4f})")

    return errors


def compare_metrics(computed: Dict, reported: Dict, tolerance: float = 0.001) -> Tuple[bool, List[str]]:
    """
    Compare computed metrics against reported metrics.

    Returns (all_match, discrepancies)
    """
    discrepancies = []
    all_match = True

    for key in computed.keys():
        if key not in reported:
            discrepancies.append(f"Metric {key} not found in reported metrics")
            all_match = False
            continue

        computed_val = computed[key]
        reported_val = reported[key]

        # Skip if both are None
        if computed_val is None and reported_val is None:
            continue

        # Check if one is None
        if (computed_val is None) != (reported_val is None):
            discrepancies.append(f"{key}: computed={computed_val}, reported={reported_val} (one is None)")
            all_match = False
            continue

        # Compare values
        if isinstance(computed_val, (int, float)):
            diff = abs(computed_val - reported_val)
            if diff > tolerance:
                discrepancies.append(
                    f"{key}: computed={computed_val:.6f}, reported={reported_val:.6f}, diff={diff:.6f}"
                )
                all_match = False

    return all_match, discrepancies


def main():
    parser = argparse.ArgumentParser(description='Recompute and verify all metrics from per_query.csv')
    parser.add_argument('--per_query_csv', type=str, required=True,
                        help='Path to per_query.csv')
    parser.add_argument('--summary_json', type=str, required=False,
                        help='Path to summary.json for comparison')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for verification results')
    parser.add_argument('--tolerance', type=float, default=0.001,
                        help='Tolerance for metric comparison')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("METRIC VERIFICATION - Recompute from per_query.csv")
    print("="*80)
    print()

    # Load per_query predictions
    print(f"Loading {args.per_query_csv}...")
    df = pd.read_csv(args.per_query_csv)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Recompute metrics
    print("Recomputing metrics...")

    # Classification metrics (if we have binary predictions)
    if 'has_evidence_pred' in df.columns or 'state' in df.columns:
        print("\n--- Classification Metrics ---")

        y_true = df['has_evidence_gold'].values

        # Use 'has_evidence_pred' if available, otherwise derive from state
        if 'has_evidence_pred' in df.columns:
            y_pred = df['has_evidence_pred'].values
        elif 'state' in df.columns:
            y_pred = (df['state'] != 'NEG').astype(int).values
        else:
            y_pred = None

        y_prob = df['p4_prob_calibrated'].values if 'p4_prob_calibrated' in df.columns else None

        if y_pred is not None:
            classification_metrics = compute_classification_metrics(y_true, y_pred, y_prob)

            for metric, value in classification_metrics.items():
                if value is not None:
                    print(f"  {metric}: {value:.6f}")

            # Calibration metrics
            if y_prob is not None:
                print("\n--- Calibration Metrics ---")
                calibration_metrics = compute_calibration_metrics(y_true, y_prob)
                for metric, value in calibration_metrics.items():
                    if metric != 'reliability_curve':
                        print(f"  {metric}: {value:.6f}")
        else:
            classification_metrics = {}
            calibration_metrics = {}
    else:
        classification_metrics = {}
        calibration_metrics = {}

    # Sanity checks
    print("\n--- Sanity Checks ---")
    all_metrics = {**classification_metrics, **calibration_metrics}
    errors = sanity_check_metrics(all_metrics)

    if errors:
        print("❌ SANITY CHECK FAILURES:")
        for error in errors:
            print(f"  - {error}")
        sanity_passed = False
    else:
        print("✅ All sanity checks passed")
        sanity_passed = True

    # Compare with reported metrics (if summary.json provided)
    if args.summary_json:
        print("\n--- Comparison with Reported Metrics ---")
        with open(args.summary_json) as f:
            summary = json.load(f)

        # Extract reported metrics (adjust path based on summary.json structure)
        reported_metrics = summary.get('aggregated', {}).get('metrics', {})

        match, discrepancies = compare_metrics(all_metrics, reported_metrics, args.tolerance)

        if match:
            print("✅ All metrics match within tolerance")
            comparison_passed = True
        else:
            print(f"❌ {len(discrepancies)} discrepancies found:")
            for disc in discrepancies[:10]:  # Show first 10
                print(f"  - {disc}")
            if len(discrepancies) > 10:
                print(f"  ... and {len(discrepancies) - 10} more")
            comparison_passed = False
    else:
        print("\n--- No Summary JSON Provided ---")
        print("Skipping comparison (use --summary_json to enable)")
        comparison_passed = True

    # Save verification results
    verification_results = {
        'per_query_csv': args.per_query_csv,
        'n_rows': len(df),
        'recomputed_metrics': all_metrics,
        'sanity_checks_passed': sanity_passed,
        'sanity_check_errors': errors,
        'comparison_passed': comparison_passed,
        'comparison_discrepancies': discrepancies if args.summary_json else []
    }

    output_file = output_dir / 'verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(verification_results, f, indent=2)

    print(f"\n✅ Verification results saved to {output_file}")

    # Exit code
    if sanity_passed and comparison_passed:
        print("\n✅ VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("\n❌ VERIFICATION FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
