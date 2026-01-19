"""Independent reference implementation of metrics for cross-validation.

This module provides pure function implementations of all metrics,
independent of the main codebase, for cross-checking correctness.

CRITICAL: This module must NOT import from final_sc_review.metrics
to ensure independence.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    matthews_corrcoef,
)


def recall_at_k_reference(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int
) -> float:
    """Reference implementation of Recall@K.

    Args:
        y_true: Binary relevance labels [n_items]
        y_score: Scores [n_items]
        k: Cutoff

    Returns:
        Recall@K
    """
    n_relevant = y_true.sum()
    if n_relevant == 0:
        return 0.0

    # Get top-K
    top_k_indices = np.argsort(y_score)[::-1][:k]
    n_relevant_at_k = y_true[top_k_indices].sum()

    return n_relevant_at_k / n_relevant


def precision_at_k_reference(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int
) -> float:
    """Reference implementation of Precision@K.

    Args:
        y_true: Binary relevance labels [n_items]
        y_score: Scores [n_items]
        k: Cutoff

    Returns:
        Precision@K
    """
    if k == 0:
        return 0.0

    # Get top-K
    top_k_indices = np.argsort(y_score)[::-1][:k]
    n_relevant_at_k = y_true[top_k_indices].sum()

    return n_relevant_at_k / k


def mrr_reference(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Reference implementation of MRR (Mean Reciprocal Rank).

    Args:
        y_true: Binary relevance labels [n_items]
        y_score: Scores [n_items]

    Returns:
        MRR (reciprocal rank of first relevant item)
    """
    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_indices]

    # Find first relevant
    relevant_indices = np.where(sorted_labels == 1)[0]

    if len(relevant_indices) == 0:
        return 0.0

    first_relevant_rank = relevant_indices[0] + 1  # 1-indexed
    return 1.0 / first_relevant_rank


def ndcg_at_k_reference(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int
) -> float:
    """Reference implementation of nDCG@K.

    Args:
        y_true: Binary or graded relevance labels [n_items]
        y_score: Scores [n_items]
        k: Cutoff

    Returns:
        nDCG@K
    """
    # DCG@K
    top_k_indices = np.argsort(y_score)[::-1][:k]
    top_k_labels = y_true[top_k_indices]

    # DCG = sum(rel_i / log2(i + 2)) for i in [0, k-1]
    positions = np.arange(1, len(top_k_labels) + 1)
    dcg = np.sum(top_k_labels / np.log2(positions + 1))

    # Ideal DCG (sort labels descending)
    ideal_labels = np.sort(y_true)[::-1][:k]
    ideal_positions = np.arange(1, len(ideal_labels) + 1)
    idcg = np.sum(ideal_labels / np.log2(ideal_positions + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def map_at_k_reference(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int
) -> float:
    """Reference implementation of MAP@K (Mean Average Precision).

    Args:
        y_true: Binary relevance labels [n_items]
        y_score: Scores [n_items]
        k: Cutoff

    Returns:
        MAP@K
    """
    # Get top-K
    top_k_indices = np.argsort(y_score)[::-1][:k]
    top_k_labels = y_true[top_k_indices]

    if top_k_labels.sum() == 0:
        return 0.0

    # Compute precision at each relevant position
    precisions = []
    n_relevant = 0

    for i, label in enumerate(top_k_labels):
        if label == 1:
            n_relevant += 1
            precision_at_i = n_relevant / (i + 1)
            precisions.append(precision_at_i)

    return np.mean(precisions) if precisions else 0.0


def tpr_at_fpr_reference(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float
) -> Tuple[float, float]:
    """Reference implementation of TPR@FPR.

    Args:
        y_true: Binary labels [n_samples]
        y_score: Scores [n_samples]
        target_fpr: Target FPR (e.g., 0.05 for 5%)

    Returns:
        (tpr, threshold) at target FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Find FPR closest to target
    idx = np.argmin(np.abs(fpr - target_fpr))

    return tpr[idx], thresholds[idx]


def expected_calibration_error_reference(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Reference implementation of ECE (Expected Calibration Error).

    Args:
        y_true: Binary labels [n_samples]
        y_prob: Predicted probabilities [n_samples]
        n_bins: Number of bins

    Returns:
        ECE
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Find samples in bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        if in_bin.sum() == 0:
            continue

        # Confidence = mean predicted probability in bin
        confidence = y_prob[in_bin].mean()

        # Accuracy = fraction of correct predictions in bin
        accuracy = y_true[in_bin].mean()

        # Weighted absolute difference
        bin_weight = in_bin.sum() / n_samples
        ece += bin_weight * np.abs(confidence - accuracy)

    return ece


def multilabel_metrics_reference(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Reference implementation of multi-label metrics.

    Args:
        y_true: Binary matrix [n_samples, n_labels]
        y_pred: Binary matrix [n_samples, n_labels]

    Returns:
        Dictionary with multi-label metrics
    """
    metrics = {}

    # Exact match rate
    exact_match = (y_true == y_pred).all(axis=1).mean()
    metrics["exact_match"] = float(exact_match)

    # Subset accuracy (same as exact match for binary)
    metrics["subset_accuracy"] = float(exact_match)

    # Hamming score (1 - Hamming loss)
    hamming_score = (y_true == y_pred).mean()
    metrics["hamming_score"] = float(hamming_score)

    # Micro-averaged F1
    tp = (y_true & y_pred).sum()
    fp = (~y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    metrics["micro_precision"] = float(micro_precision)
    metrics["micro_recall"] = float(micro_recall)
    metrics["micro_f1"] = float(micro_f1)

    # Macro-averaged F1 (per-label average)
    n_labels = y_true.shape[1]
    per_label_f1 = []

    for i in range(n_labels):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        tp_i = (y_true_i & y_pred_i).sum()
        fp_i = (~y_true_i & y_pred_i).sum()
        fn_i = (y_true_i & ~y_pred_i).sum()

        prec_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        rec_i = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        f1_i = (
            2 * prec_i * rec_i / (prec_i + rec_i)
            if (prec_i + rec_i) > 0 else 0.0
        )

        per_label_f1.append(f1_i)

    metrics["macro_f1"] = float(np.mean(per_label_f1))

    return metrics


def cross_check_metrics(
    existing_func,
    reference_func,
    *args,
    **kwargs
) -> Tuple[bool, float, float]:
    """Cross-check existing metric implementation against reference.

    Args:
        existing_func: Existing metric function
        reference_func: Reference metric function
        *args, **kwargs: Arguments to both functions

    Returns:
        (match, existing_value, reference_value)
    """
    existing_value = existing_func(*args, **kwargs)
    reference_value = reference_func(*args, **kwargs)

    # Check if values match within tolerance
    if isinstance(existing_value, float) and isinstance(reference_value, float):
        match = np.isclose(existing_value, reference_value, rtol=1e-5, atol=1e-8)
    else:
        match = existing_value == reference_value

    return match, existing_value, reference_value
