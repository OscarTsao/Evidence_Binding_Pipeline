#!/usr/bin/env python3
"""Independent metric implementation for cross-checking pipeline metrics.

This module implements all metrics from scratch using only numpy/sklearn
to verify correctness of the main pipeline's metric calculations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    matthews_corrcoef,
    balanced_accuracy_score
)


def recall_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate Recall@K.

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        Recall@K value
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    top_k_relevance = relevance[sorted_indices[:k]]

    recall = top_k_relevance.sum() / relevance.sum()
    return float(recall)


def precision_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate Precision@K.

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        Precision@K value
    """
    if len(relevance) == 0 or k == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    top_k_relevance = relevance[sorted_indices[:k]]

    precision = top_k_relevance.sum() / min(k, len(relevance))
    return float(precision)


def hit_rate_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate HitRate@K (binary: 1 if any relevant item in top-K).

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        HitRate@K value (0 or 1)
    """
    if len(relevance) == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    top_k_relevance = relevance[sorted_indices[:k]]

    return float(top_k_relevance.sum() > 0)


def mrr(relevance: np.ndarray, scores: np.ndarray) -> float:
    """Calculate Mean Reciprocal Rank (for single query).

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]

    Returns:
        Reciprocal rank of first relevant item
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices]

    # Find first relevant item
    first_relevant = np.where(sorted_relevance > 0)[0]
    if len(first_relevant) == 0:
        return 0.0

    # Rank is 1-indexed
    rank = first_relevant[0] + 1
    return float(1.0 / rank)


def dcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate Discounted Cumulative Gain@K.

    Args:
        relevance: Relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        DCG@K value
    """
    if len(relevance) == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    top_k_relevance = relevance[sorted_indices[:k]]

    # Compute DCG
    gains = top_k_relevance
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    return float(dcg)


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K.

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        nDCG@K value
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return 0.0

    # Actual DCG
    dcg = dcg_at_k(relevance, scores, k)

    # Ideal DCG (sort by relevance)
    ideal_dcg = dcg_at_k(relevance, relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return float(dcg / ideal_dcg)


def average_precision(relevance: np.ndarray, scores: np.ndarray) -> float:
    """Calculate Average Precision (for single query).

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]

    Returns:
        Average Precision value
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices]

    # Compute precision at each relevant position
    precisions = []
    for k in range(1, len(sorted_relevance) + 1):
        if sorted_relevance[k-1] > 0:
            prec = sorted_relevance[:k].sum() / k
            precisions.append(prec)

    if len(precisions) == 0:
        return 0.0

    return float(np.mean(precisions))


def map_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate Average Precision@K (for single query).

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        AP@K value
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices[:k]]  # Only consider top-k

    # Compute precision at each relevant position within top-k
    precisions = []
    for i in range(len(sorted_relevance)):
        if sorted_relevance[i] > 0:
            prec = sorted_relevance[:i+1].sum() / (i + 1)
            precisions.append(prec)

    if len(precisions) == 0:
        return 0.0

    # Normalize by min(k, total relevant)
    # Standard MAP@K normalizes by number of relevant items in top-k
    return float(np.mean(precisions))


def mrr_at_k(relevance: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Calculate Reciprocal Rank@K (for single query).

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k: Cut-off rank

    Returns:
        RR@K value (0 if no relevant item in top-k)
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return 0.0

    # Sort by scores (descending)
    sorted_indices = np.argsort(-scores)
    sorted_relevance = relevance[sorted_indices[:k]]  # Only consider top-k

    # Find first relevant item in top-k
    first_relevant = np.where(sorted_relevance > 0)[0]
    if len(first_relevant) == 0:
        return 0.0

    # Rank is 1-indexed
    rank = first_relevant[0] + 1
    return float(1.0 / rank)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """Calculate TPR at a specific FPR threshold.

    Args:
        labels: Binary labels
        scores: Prediction scores
        target_fpr: Target false positive rate

    Returns:
        (tpr, threshold) at target FPR
    """
    if len(labels) == 0:
        return 0.0, 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find closest FPR
    idx = np.where(fpr >= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, thresholds[-1]

    idx = idx[0]
    return float(tpr[idx]), float(thresholds[idx])


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error.

    Args:
        labels: Binary labels
        probs: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value
    """
    if len(labels) == 0:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.sum() / len(probs)

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Calculate Brier score.

    Args:
        labels: Binary labels
        probs: Predicted probabilities

    Returns:
        Brier score
    """
    if len(labels) == 0:
        return 0.0

    return float(np.mean((probs - labels) ** 2))


def multi_label_exact_match(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate exact match rate for multi-label classification.

    Args:
        y_true: True labels [n_samples, n_labels]
        y_pred: Predicted labels [n_samples, n_labels]

    Returns:
        Exact match rate
    """
    if len(y_true) == 0:
        return 0.0

    matches = np.all(y_true == y_pred, axis=1)
    return float(matches.mean())


def multi_label_hamming_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Hamming score (1 - Hamming loss).

    Args:
        y_true: True labels [n_samples, n_labels]
        y_pred: Predicted labels [n_samples, n_labels]

    Returns:
        Hamming score
    """
    if len(y_true) == 0:
        return 0.0

    # Hamming score = fraction of correct labels
    return float(np.mean(y_true == y_pred))


def cross_check_metrics(per_query_csvs: List[Path], pipeline_summary_json: Path, output_file: Path):
    """Cross-check metrics from per-query predictions against pipeline summary.

    Args:
        per_query_csvs: List of paths to per-query predictions CSVs (all folds)
        pipeline_summary_json: Path to pipeline's summary.json
        output_file: Path to save cross-check results
    """
    print(f"{'='*80}")
    print("METRIC CROSS-CHECK")
    print(f"{'='*80}\n")

    # Load and concatenate all folds
    print(f"Loading per-query predictions from {len(per_query_csvs)} folds...")
    dfs = []
    for csv_path in per_query_csvs:
        fold_df = pd.read_csv(csv_path)
        print(f"  {csv_path.name}: {len(fold_df)} queries")
        dfs.append(fold_df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal queries across all folds: {len(df)}\n")

    print(f"Loading pipeline summary from: {pipeline_summary_json}")
    with open(pipeline_summary_json) as f:
        pipeline = json.load(f)
    print(f"Loaded pipeline summary with {len(pipeline.get('fold_results', []))} folds\n")

    # Cross-check classification metrics (NE gate)
    print("Cross-checking NE gate metrics...")
    probs = df['p4_prob_calibrated'].values
    labels = df['has_evidence_gold'].values

    # AUROC - concatenated approach
    auroc_check_concat = roc_auc_score(labels, probs)

    # AUROC - per-fold average (same as pipeline)
    auroc_per_fold = []
    for fold_id in df['fold_id'].unique():
        fold_mask = df['fold_id'] == fold_id
        fold_labels = labels[fold_mask]
        fold_probs = probs[fold_mask]
        auroc_per_fold.append(roc_auc_score(fold_labels, fold_probs))
    auroc_check_avg = np.mean(auroc_per_fold)

    auroc_pipeline = pipeline['aggregated_metrics']['ne_gate.auroc']['mean']
    auroc_match = np.abs(auroc_check_avg - auroc_pipeline) < 0.01

    print(f"  AUROC (concatenated): {auroc_check_concat:.4f}")
    print(f"  AUROC (per-fold avg): {auroc_check_avg:.4f} vs {auroc_pipeline:.4f} (pipeline) "
          f"{'✅' if auroc_match else '❌'}")

    # AUPRC - concatenated approach
    auprc_check_concat = average_precision_score(labels, probs)

    # AUPRC - per-fold average (same as pipeline)
    auprc_per_fold = []
    for fold_id in df['fold_id'].unique():
        fold_mask = df['fold_id'] == fold_id
        fold_labels = labels[fold_mask]
        fold_probs = probs[fold_mask]
        auprc_per_fold.append(average_precision_score(fold_labels, fold_probs))
    auprc_check_avg = np.mean(auprc_per_fold)

    auprc_pipeline = pipeline['aggregated_metrics']['ne_gate.auprc']['mean']
    auprc_match = np.abs(auprc_check_avg - auprc_pipeline) < 0.01

    print(f"  AUPRC (concatenated): {auprc_check_concat:.4f}")
    print(f"  AUPRC (per-fold avg): {auprc_check_avg:.4f} vs {auprc_pipeline:.4f} (pipeline) "
          f"{'✅' if auprc_match else '❌'}")

    # TPR@FPR
    print("\n  TPR@FPR:")
    for target_fpr in [0.01, 0.03, 0.05, 0.10]:
        tpr, thresh = tpr_at_fpr(labels, probs, target_fpr)
        print(f"    TPR@{target_fpr:.0%}FPR: {tpr:.4f} (threshold: {thresh:.4f})")

    # ECE
    ece_check = expected_calibration_error(labels, probs)
    print(f"\n  ECE: {ece_check:.4f}")

    # Brier score
    brier_check = brier_score(labels, probs)
    print(f"  Brier score: {brier_check:.4f}")

    # Cross-check deployment metrics
    print("\nCross-checking deployment metrics...")

    # Screening tier
    screening_tp = ((df['state'] != 'NEG') & (df['has_evidence_gold'] == 1)).sum()
    screening_fn = ((df['state'] == 'NEG') & (df['has_evidence_gold'] == 1)).sum()
    screening_sensitivity = screening_tp / labels.sum() if labels.sum() > 0 else 0

    sensitivity_pipeline = pipeline['aggregated_metrics']['deployment.screening_sensitivity']['mean']
    sens_match = np.abs(screening_sensitivity - sensitivity_pipeline) < 0.01

    print(f"  Screening sensitivity: {screening_sensitivity:.4f} (check) vs "
          f"{sensitivity_pipeline:.4f} (pipeline) {'✅' if sens_match else '❌'}")

    # Alert tier
    alert_tp = ((df['state'] == 'POS') & (df['has_evidence_gold'] == 1)).sum()
    alert_fp = ((df['state'] == 'POS') & (df['has_evidence_gold'] == 0)).sum()
    alert_precision = alert_tp / (alert_tp + alert_fp) if (alert_tp + alert_fp) > 0 else 0

    precision_pipeline = pipeline['aggregated_metrics']['deployment.alert_precision']['mean']
    prec_match = np.abs(alert_precision - precision_pipeline) < 0.01

    print(f"  Alert precision: {alert_precision:.4f} (check) vs "
          f"{precision_pipeline:.4f} (pipeline) {'✅' if prec_match else '❌'}")

    # Summary
    print(f"\n{'='*80}")
    print("CROSS-CHECK SUMMARY")
    print(f"{'='*80}\n")

    all_checks_passed = auroc_match and auprc_match and sens_match and prec_match

    if all_checks_passed:
        print("✅ ALL METRIC CHECKS PASSED")
        status = "PASS"
    else:
        print("❌ SOME METRIC CHECKS FAILED")
        status = "FAIL"

    # Save results (convert numpy types to Python types for JSON serialization)
    results = {
        'status': status,
        'checks': {
            'auroc': {
                'check_concatenated': float(auroc_check_concat),
                'check_per_fold_avg': float(auroc_check_avg),
                'pipeline': float(auroc_pipeline),
                'passed': bool(auroc_match)
            },
            'auprc': {
                'check_concatenated': float(auprc_check_concat),
                'check_per_fold_avg': float(auprc_check_avg),
                'pipeline': float(auprc_pipeline),
                'passed': bool(auprc_match)
            },
            'screening_sensitivity': {'check': float(screening_sensitivity), 'pipeline': float(sensitivity_pipeline), 'passed': bool(sens_match)},
            'alert_precision': {'check': float(alert_precision), 'pipeline': float(precision_pipeline), 'passed': bool(prec_match)}
        },
        'ece': float(ece_check),
        'brier_score': float(brier_check),
        'note': 'AUROC and AUPRC use per-fold averaging for comparison (same as pipeline). Concatenated values also shown for reference.'
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return 0 if all_checks_passed else 1


def cross_check_ranking_metrics(
    results_json: Path,
    per_query_rankings_jsonl: Path,
    output_file: Path,
    tolerance: float = 0.001,
    ks: List[int] = None,
) -> Dict:
    """Cross-check ranking metrics from zoo pipeline results.

    This function independently recomputes all ranking metrics from per-query
    rankings and compares them with the pipeline-reported metrics.

    Args:
        results_json: Path to zoo pipeline test_results.json
        per_query_rankings_jsonl: Path to per-query rankings JSONL file
        output_file: Path to save cross-check results
        tolerance: Maximum allowed difference for metrics to match
        ks: List of K values to check (default: [1, 3, 5, 10, 20])

    Returns:
        Dictionary with cross-check results and verification status
    """
    from datetime import datetime
    from collections import defaultdict

    if ks is None:
        ks = [1, 3, 5, 10, 20]

    # Load pipeline results
    with open(results_json, 'r') as f:
        pipeline_results = json.load(f)

    # Load per-query rankings
    per_query_rankings = []
    with open(per_query_rankings_jsonl, 'r') as f:
        for line in f:
            per_query_rankings.append(json.loads(line))

    # Recompute metrics independently for each K
    recomputed_metrics = defaultdict(list)

    for query_data in per_query_rankings:
        ranked_uids = query_data['ranked_uids']
        gold_uids = set(query_data['gold_uids'])

        # Skip queries with no positives (if pipeline did the same)
        if not gold_uids:
            continue

        # Create binary relevance array
        relevance = np.array([1 if uid in gold_uids else 0 for uid in ranked_uids])
        # Scores are just the rank positions (higher rank = higher score)
        scores = np.arange(len(ranked_uids), 0, -1, dtype=float)

        # Compute metrics at each K
        for k in ks:
            # Recall@K
            recall_k = recall_at_k(relevance, scores, k)
            recomputed_metrics[f'recall@{k}'].append(recall_k)

            # MRR@K
            mrr_k = mrr_at_k(relevance, scores, k)
            recomputed_metrics[f'mrr@{k}'].append(mrr_k)

            # MAP@K
            map_k = map_at_k(relevance, scores, k)
            recomputed_metrics[f'map@{k}'].append(map_k)

            # nDCG@K
            ndcg_k = ndcg_at_k(relevance, scores, k)
            recomputed_metrics[f'ndcg@{k}'].append(ndcg_k)

    # Aggregate metrics (mean across queries)
    aggregated_metrics = {}
    for metric_name, values in recomputed_metrics.items():
        aggregated_metrics[metric_name] = float(np.mean(values))

    # Compare with pipeline results
    comparison_results = []
    all_match = True

    # Check reranked metrics (most important)
    pipeline_reranked = pipeline_results.get('reranked', {})

    for metric_name, recomputed_value in sorted(aggregated_metrics.items()):
        pipeline_value = pipeline_reranked.get(metric_name)

        if pipeline_value is None:
            comparison_results.append({
                'metric': metric_name,
                'pipeline': None,
                'recomputed': recomputed_value,
                'diff': None,
                'match': False,
                'status': 'MISSING_IN_PIPELINE'
            })
            all_match = False
        else:
            diff = abs(recomputed_value - pipeline_value)
            match = diff <= tolerance
            if not match:
                all_match = False

            comparison_results.append({
                'metric': metric_name,
                'pipeline': pipeline_value,
                'recomputed': recomputed_value,
                'diff': diff,
                'match': match,
                'status': 'MATCH' if match else 'MISMATCH'
            })

    # Prepare output
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'results_json': str(results_json),
        'per_query_rankings_jsonl': str(per_query_rankings_jsonl),
        'n_queries_checked': len(per_query_rankings),
        'tolerance': tolerance,
        'ks_checked': ks,
        'all_metrics_match': all_match,
        'comparison': comparison_results,
    }

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("RANKING METRICS CROSS-CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Queries checked: {len(per_query_rankings)}")
    print(f"Tolerance: ±{tolerance}")
    print(f"Overall status: {'✅ ALL MATCH' if all_match else '❌ MISMATCHES FOUND'}")
    print(f"\n{'Metric':<15} {'Pipeline':<12} {'Recomputed':<12} {'Diff':<10} {'Status'}")
    print("-" * 60)

    for result in comparison_results:
        metric = result['metric']
        pipeline_val = result['pipeline']
        recomputed_val = result['recomputed']
        diff = result['diff']
        status = result['status']

        if pipeline_val is None:
            print(f"{metric:<15} {'N/A':<12} {recomputed_val:<12.4f} {'N/A':<10} {status}")
        else:
            status_symbol = '✅' if result['match'] else '❌'
            print(f"{metric:<15} {pipeline_val:<12.4f} {recomputed_val:<12.4f} {diff:<10.6f} {status_symbol} {status}")

    print(f"\nResults saved to: {output_file}")

    return output_data


def cross_check_bundle(bundle_dir: Path, output_file: Path = None) -> int:
    """Cross-check metrics in a paper bundle.

    Verifies that metrics_master.json values are consistent with the source
    per_query.csv data using the canonical compute_metrics module.

    Args:
        bundle_dir: Path to paper bundle directory
        output_file: Optional path to save cross-check results

    Returns:
        0 if all checks pass, 1 otherwise
    """
    print(f"{'='*80}")
    print("PAPER BUNDLE METRIC CROSS-CHECK")
    print(f"{'='*80}\n")

    # Load metrics_master.json
    metrics_master_path = bundle_dir / "metrics_master.json"
    if not metrics_master_path.exists():
        print(f"ERROR: metrics_master.json not found in {bundle_dir}")
        return 1

    with open(metrics_master_path) as f:
        metrics_master = json.load(f)

    print(f"Loaded metrics_master.json from: {metrics_master_path}")

    # Get source per_query.csv path
    source_csv = metrics_master.get("metadata", {}).get("source_per_query_csv")
    if not source_csv:
        print("WARNING: source_per_query_csv not in metadata, trying default location")
        source_csv = "outputs/final_research_eval/20260118_031312_complete/per_query.csv"

    source_csv_path = Path(source_csv)
    if not source_csv_path.is_absolute():
        # Try relative to current dir
        if not source_csv_path.exists():
            source_csv_path = bundle_dir.parent.parent.parent / source_csv

    if not source_csv_path.exists():
        print(f"ERROR: Source per_query.csv not found: {source_csv_path}")
        print("Cannot verify metrics without source data.")
        return 1

    print(f"Loading source data from: {source_csv_path}")
    df = pd.read_csv(source_csv_path)
    print(f"Loaded {len(df)} queries\n")

    # Verify key metrics
    checks = []
    tolerance = 0.0001

    # Classification metrics (all_queries protocol)
    if "has_evidence_gold" in df.columns:
        labels = df["has_evidence_gold"].values

        # Try to find probability column
        prob_col = None
        for col in ["p4_prob_calibrated", "p4_prob", "prob", "score"]:
            if col in df.columns:
                prob_col = col
                break

        if prob_col:
            probs = df[prob_col].values
            valid_mask = ~np.isnan(probs)

            # AUROC
            if valid_mask.sum() > 0:
                auroc_computed = roc_auc_score(labels[valid_mask], probs[valid_mask])
                # Support both naming conventions
                class_metrics = metrics_master.get("classification_metrics", metrics_master.get("classification", {}))
                class_metrics = class_metrics.get("metrics", class_metrics)  # Handle nested structure
                auroc_reported = class_metrics.get("auroc", {}).get("value")

                if auroc_reported is not None:
                    diff = abs(auroc_computed - auroc_reported)
                    match = diff <= tolerance
                    checks.append({
                        "metric": "AUROC",
                        "computed": auroc_computed,
                        "reported": auroc_reported,
                        "diff": diff,
                        "match": match,
                    })
                    print(f"AUROC: {auroc_computed:.4f} (computed) vs {auroc_reported:.4f} (reported) "
                          f"{'✅' if match else '❌'}")

                # AUPRC - CRITICAL: verify it's NOT confused with Recall@K
                auprc_computed = average_precision_score(labels[valid_mask], probs[valid_mask])
                auprc_reported = class_metrics.get("auprc", {}).get("value")

                if auprc_reported is not None:
                    diff = abs(auprc_computed - auprc_reported)
                    match = diff <= tolerance
                    checks.append({
                        "metric": "AUPRC",
                        "computed": auprc_computed,
                        "reported": auprc_reported,
                        "diff": diff,
                        "match": match,
                    })
                    print(f"AUPRC: {auprc_computed:.4f} (computed) vs {auprc_reported:.4f} (reported) "
                          f"{'✅' if match else '❌'}")

                    # Safety check: AUPRC should NOT equal any Recall@K value
                    ranking_section = metrics_master.get("ranking_metrics", metrics_master.get("ranking", {}))
                    ranking_metrics = ranking_section.get("metrics", ranking_section)
                    for metric_name, metric_data in ranking_metrics.items():
                        if "recall" in metric_name.lower():
                            recall_val = metric_data.get("value", 0)
                            if abs(auprc_computed - recall_val) < 0.0001:
                                print(f"⚠️  WARNING: AUPRC ({auprc_computed:.4f}) equals {metric_name} ({recall_val:.4f})!")
                                print("    This may indicate metric confusion. AUPRC and Recall@K are different metrics.")

    # Ranking metrics (positives_only protocol)
    pos_mask = df["has_evidence_gold"] == 1 if "has_evidence_gold" in df.columns else df.index
    df_pos = df[pos_mask]

    # Support both naming conventions
    ranking_section = metrics_master.get("ranking_metrics", metrics_master.get("ranking", {}))
    ranking_metrics_dict = ranking_section.get("metrics", ranking_section)

    for metric_name in ["evidence_recall_at_k", "mrr", "ndcg_at_k"]:
        if metric_name in df_pos.columns:
            computed = df_pos[metric_name].dropna().mean()
            reported = ranking_metrics_dict.get(metric_name, {}).get("value")

            if reported is not None:
                diff = abs(computed - reported)
                match = diff <= tolerance
                checks.append({
                    "metric": metric_name,
                    "computed": computed,
                    "reported": reported,
                    "diff": diff,
                    "match": match,
                })
                print(f"{metric_name}: {computed:.4f} (computed) vs {reported:.4f} (reported) "
                      f"{'✅' if match else '❌'}")

    # Summary
    print(f"\n{'='*80}")
    print("CROSS-CHECK SUMMARY")
    print(f"{'='*80}\n")

    all_passed = all(c["match"] for c in checks) if checks else False
    n_passed = sum(1 for c in checks if c["match"])
    n_total = len(checks)

    if all_passed:
        print(f"✅ ALL {n_total} METRIC CHECKS PASSED")
        status = "PASS"
    else:
        print(f"❌ {n_total - n_passed}/{n_total} METRIC CHECKS FAILED")
        status = "FAIL"

    # Save results if output specified
    if output_file:
        results = {
            "status": status,
            "bundle_dir": str(bundle_dir),
            "source_csv": str(source_csv_path),
            "n_queries": len(df),
            "checks": checks,
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return 0 if all_passed else 1


def main():
    parser = argparse.ArgumentParser(description="Cross-check metrics")
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Legacy mode (fold-based)
    fold_parser = subparsers.add_parser("folds", help="Cross-check fold-based results")
    fold_parser.add_argument(
        "--fold_results_dir",
        type=Path,
        required=True,
        help="Directory containing fold_*_predictions.csv files"
    )
    fold_parser.add_argument(
        "--pipeline_summary",
        type=Path,
        required=True,
        help="Path to pipeline summary.json"
    )
    fold_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for cross-check results"
    )

    # Bundle mode (v3.0+)
    bundle_parser = subparsers.add_parser("bundle", help="Cross-check paper bundle")
    bundle_parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to paper bundle directory"
    )
    bundle_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON file for cross-check results"
    )

    # Direct bundle argument for backward compatibility
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to paper bundle directory (shortcut for 'bundle' mode)"
    )

    args = parser.parse_args()

    # Handle direct --bundle argument
    if args.bundle and not args.mode:
        return cross_check_bundle(args.bundle)

    if args.mode == "bundle":
        return cross_check_bundle(args.bundle, args.output)

    elif args.mode == "folds":
        # Find all fold CSV files
        per_query_csvs = sorted(args.fold_results_dir.glob("fold_*_predictions.csv"))

        if not per_query_csvs:
            print(f"ERROR: No fold_*_predictions.csv files found in {args.fold_results_dir}")
            return 1

        return cross_check_metrics(
            per_query_csvs,
            args.pipeline_summary,
            args.output
        )

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
