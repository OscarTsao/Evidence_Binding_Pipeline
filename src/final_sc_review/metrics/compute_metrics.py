"""Canonical metric computation module.

This is the SINGLE SOURCE OF TRUTH for all metric computations in the paper.

Protocols:
- positives_only: Ranking metrics computed only on queries where has_evidence=1
- all_queries: Classification metrics computed on all queries

Metrics:
- Ranking: nDCG@K, Recall@K, Precision@K, MRR, MAP@K
- Classification: AUROC, AUPRC (Average Precision), Accuracy, Precision, Recall, F1
- Calibration: Brier score, ECE (Expected Calibration Error), MCE

All metrics computed here match the definitions in docs/METRIC_CONTRACT.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from final_sc_review.metrics.ranking import (
    evidence_coverage,
    f1_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class MetricResult:
    """Container for a single metric with optional CI."""

    name: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    protocol: str = "all_queries"
    n_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "name": self.name,
            "value": self.value,
            "protocol": self.protocol,
        }
        if self.ci_lower is not None:
            d["ci_95_lower"] = self.ci_lower
        if self.ci_upper is not None:
            d["ci_95_upper"] = self.ci_upper
        if self.n_samples is not None:
            d["n_samples"] = self.n_samples
        return d


@dataclass
class MetricBundle:
    """Bundle of all computed metrics."""

    ranking: Dict[str, MetricResult] = field(default_factory=dict)
    classification: Dict[str, MetricResult] = field(default_factory=dict)
    calibration: Dict[str, MetricResult] = field(default_factory=dict)
    per_criterion: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary."""
        return {
            "ranking_metrics": {
                "protocol": "positives_only",
                "metrics": {k: v.to_dict() for k, v in self.ranking.items()}
            },
            "classification_metrics": {
                "protocol": "all_queries",
                "metrics": {k: v.to_dict() for k, v in self.classification.items()}
            },
            "calibration_metrics": {
                "protocol": "all_queries",
                "metrics": {k: v.to_dict() for k, v in self.calibration.items()}
            },
            "per_criterion_performance": self.per_criterion,
            "metadata": self.metadata,
        }


# =============================================================================
# RANKING METRICS (positives_only protocol)
# =============================================================================


def compute_ranking_metrics(
    per_query_df: pd.DataFrame,
    k_list: Sequence[int] = (1, 3, 5, 10),
    gold_col: str = "has_evidence_gold",
    ranked_col: str = "ranked_ids",
    gold_ids_col: str = "gold_ids",
) -> Dict[str, float]:
    """Compute ranking metrics using positives_only protocol.

    Args:
        per_query_df: DataFrame with per-query results
        k_list: List of K values for @K metrics
        gold_col: Column indicating if query has gold evidence
        ranked_col: Column with ranked sentence IDs (list)
        gold_ids_col: Column with gold sentence IDs (list)

    Returns:
        Dictionary of metric_name -> value

    Note:
        Only queries where gold_col == 1 are included (positives_only protocol).
        This is the correct protocol for ranking metrics per METRIC_CONTRACT.md.
    """
    # Filter to positive queries only
    df_pos = per_query_df[per_query_df[gold_col] == 1].copy()

    if len(df_pos) == 0:
        return {"error": "No positive queries found"}

    metrics = {
        "n_queries_positives_only": len(df_pos),
        "n_queries_total": len(per_query_df),
    }

    # Check if we have ranking data
    if ranked_col not in df_pos.columns or gold_ids_col not in df_pos.columns:
        # Use pre-computed metrics if available
        for k in k_list:
            recall_col = f"evidence_recall_at_k" if k == 10 else f"recall@{k}"
            if recall_col in df_pos.columns:
                metrics[f"recall@{k}"] = float(df_pos[recall_col].mean())
        if "mrr" in df_pos.columns:
            metrics["mrr"] = float(df_pos["mrr"].mean())
        return metrics

    # Compute ranking metrics
    for k in k_list:
        recall_vals = []
        ndcg_vals = []
        mrr_vals = []
        map_vals = []
        precision_vals = []

        for _, row in df_pos.iterrows():
            gold_ids = row[gold_ids_col]
            ranked_ids = row[ranked_col]

            if not gold_ids or not ranked_ids:
                continue

            recall_vals.append(recall_at_k(gold_ids, ranked_ids, k))
            ndcg_vals.append(ndcg_at_k(gold_ids, ranked_ids, k))
            mrr_vals.append(mrr_at_k(gold_ids, ranked_ids, k))
            map_vals.append(map_at_k(gold_ids, ranked_ids, k))

            # Precision@K
            hits = len(set(ranked_ids[:k]) & set(gold_ids))
            precision_vals.append(hits / k)

        if recall_vals:
            metrics[f"recall@{k}"] = float(np.mean(recall_vals))
            metrics[f"ndcg@{k}"] = float(np.mean(ndcg_vals))
            metrics[f"precision@{k}"] = float(np.mean(precision_vals))

        if k == max(k_list) and mrr_vals:
            metrics["mrr"] = float(np.mean(mrr_vals))
            metrics[f"map@{k}"] = float(np.mean(map_vals))

    return metrics


def compute_ranking_metrics_from_csv(
    per_query_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compute ranking metrics from per_query.csv format.

    This function handles the specific format of per_query.csv where
    ranking metrics are pre-computed per row.

    Args:
        per_query_df: DataFrame from per_query.csv

    Returns:
        Dictionary of ranking metrics
    """
    # Filter to positive queries only (positives_only protocol)
    df_pos = per_query_df[per_query_df["has_evidence_gold"] == 1].copy()

    if len(df_pos) == 0:
        return {"error": "No positive queries found"}

    metrics = {
        "n_queries_positives_only": len(df_pos),
        "n_queries_total": len(per_query_df),
        "positive_rate": len(df_pos) / len(per_query_df),
    }

    # Evidence Recall@K (using selected_k from Dynamic-K)
    if "evidence_recall_at_k" in df_pos.columns:
        valid_recall = df_pos["evidence_recall_at_k"].dropna()
        if len(valid_recall) > 0:
            metrics["evidence_recall_at_k"] = float(valid_recall.mean())

    # MRR
    if "mrr" in df_pos.columns:
        valid_mrr = df_pos["mrr"].dropna()
        if len(valid_mrr) > 0:
            metrics["mrr"] = float(valid_mrr.mean())

    # Evidence Precision@K
    if "evidence_precision_at_k" in df_pos.columns:
        valid_prec = df_pos["evidence_precision_at_k"].dropna()
        if len(valid_prec) > 0:
            metrics["evidence_precision_at_k"] = float(valid_prec.mean())

    return metrics


# =============================================================================
# CLASSIFICATION METRICS (all_queries protocol)
# =============================================================================


def compute_classification_metrics(
    per_query_df: pd.DataFrame,
    prob_col: str = "p4_prob_calibrated",
    gold_col: str = "has_evidence_gold",
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute classification metrics using all_queries protocol.

    Args:
        per_query_df: DataFrame with per-query results
        prob_col: Column with predicted probability
        gold_col: Column with gold label (0/1)
        threshold: Decision threshold (if None, reports only AUROC/AUPRC)

    Returns:
        Dictionary of metric_name -> value

    CRITICAL: AUPRC is computed using sklearn.average_precision_score,
    which is the correct implementation of Area Under Precision-Recall Curve.
    This is NOT the same as Recall@K!
    """
    y_true = per_query_df[gold_col].values
    y_score = per_query_df[prob_col].values

    # Remove any NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid_mask]
    y_score = y_score[valid_mask]

    metrics = {
        "n_queries_all": len(y_true),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "positive_rate": float(y_true.mean()),
    }

    # Check for valid data
    if len(np.unique(y_true)) < 2:
        metrics["error"] = "Only one class present in y_true"
        return metrics

    # AUROC - Area Under ROC Curve
    metrics["auroc"] = float(roc_auc_score(y_true, y_score))

    # AUPRC - Area Under Precision-Recall Curve
    # CRITICAL: This uses sklearn.average_precision_score which computes
    # the area under the precision-recall curve. This is NOT Recall@K!
    metrics["auprc"] = float(average_precision_score(y_true, y_score))

    # Brier score (for calibration)
    metrics["brier_score"] = float(brier_score_loss(y_true, y_score))

    # TPR at various FPR thresholds (useful for clinical applications)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    for target_fpr in [0.01, 0.03, 0.05, 0.10]:
        # Find TPR at target FPR
        idx = np.searchsorted(fpr, target_fpr)
        if idx < len(tpr):
            metrics[f"tpr_at_fpr_{int(target_fpr*100)}pct"] = float(tpr[idx])

    # If threshold provided, compute threshold-based metrics
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        metrics["threshold"] = threshold
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        # Specificity
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # NPV (Negative Predictive Value)
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    return metrics


# =============================================================================
# CALIBRATION METRICS
# =============================================================================


def compute_calibration_metrics(
    per_query_df: pd.DataFrame,
    prob_col: str = "p4_prob_calibrated",
    gold_col: str = "has_evidence_gold",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration metrics.

    Args:
        per_query_df: DataFrame with per-query results
        prob_col: Column with predicted probability
        gold_col: Column with gold label (0/1)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with ECE, MCE, Brier, and reliability curve data
    """
    y_true = per_query_df[gold_col].values
    y_prob = per_query_df[prob_col].values

    # Remove NaN
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    metrics = {
        "n_samples": len(y_true),
    }

    # Brier score
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # Compute calibration curve (reliability diagram)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    prob_true = []
    prob_pred = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            prob_true.append(float(y_true[mask].mean()))
            prob_pred.append(float(y_prob[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            prob_true.append(None)
            prob_pred.append(None)
            bin_counts.append(0)

    # Filter out empty bins for ECE/MCE computation
    valid_bins = [(pt, pp, c) for pt, pp, c in zip(prob_true, prob_pred, bin_counts) if pt is not None]

    if valid_bins:
        # ECE: Expected Calibration Error (weighted average of |prob_true - prob_pred|)
        total_samples = sum(c for _, _, c in valid_bins)
        ece = sum(abs(pt - pp) * c / total_samples for pt, pp, c in valid_bins)
        metrics["ece"] = float(ece)

        # MCE: Maximum Calibration Error
        mce = max(abs(pt - pp) for pt, pp, _ in valid_bins)
        metrics["mce"] = float(mce)

    # Reliability curve data
    metrics["reliability_curve"] = {
        "prob_true": [p for p in prob_true if p is not None],
        "prob_pred": [p for p in prob_pred if p is not None],
        "bin_counts": [c for c in bin_counts if c > 0],
    }

    return metrics


# =============================================================================
# MULTI-LABEL F1 METRICS
# =============================================================================


def compute_multilabel_f1(
    per_query_df: pd.DataFrame,
    prob_col: str = "p4_prob_calibrated",
    gold_col: str = "has_evidence_gold",
    criterion_col: str = "criterion_id",
    post_col: str = "post_id",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Micro and Macro F1 for multi-label criterion classification.

    In the multi-label setting, each post can have multiple criteria with evidence.
    This function computes:
    - Micro F1: Global TP/FP/FN across all post-criterion pairs
    - Macro F1: Average F1 per criterion

    Args:
        per_query_df: DataFrame with per-query (post x criterion) results
        prob_col: Column with predicted probability
        gold_col: Column with gold label (0/1)
        criterion_col: Column with criterion ID
        post_col: Column with post ID
        threshold: Decision threshold for classification

    Returns:
        Dictionary with micro_f1, macro_f1, and per-criterion F1
    """
    y_true = per_query_df[gold_col].values
    y_score = per_query_df[prob_col].values
    y_pred = (y_score >= threshold).astype(int)

    # Remove NaN
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    df_valid = per_query_df[valid_mask].copy()

    # Micro F1: global counts
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Macro F1: average F1 per criterion
    criterion_f1s = []
    per_criterion_f1 = {}

    for criterion_id in df_valid[criterion_col].unique():
        crit_mask = df_valid[criterion_col] == criterion_id
        crit_true = y_true[crit_mask]
        crit_pred = y_pred[crit_mask]

        crit_tp = ((crit_true == 1) & (crit_pred == 1)).sum()
        crit_fp = ((crit_true == 0) & (crit_pred == 1)).sum()
        crit_fn = ((crit_true == 1) & (crit_pred == 0)).sum()

        crit_prec = crit_tp / (crit_tp + crit_fp) if (crit_tp + crit_fp) > 0 else 0.0
        crit_rec = crit_tp / (crit_tp + crit_fn) if (crit_tp + crit_fn) > 0 else 0.0
        crit_f1 = (
            2 * crit_prec * crit_rec / (crit_prec + crit_rec)
            if (crit_prec + crit_rec) > 0
            else 0.0
        )

        criterion_f1s.append(crit_f1)
        per_criterion_f1[criterion_id] = crit_f1

    macro_f1 = float(np.mean(criterion_f1s)) if criterion_f1s else 0.0

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "threshold": threshold,
        "n_criteria": len(criterion_f1s),
        "per_criterion_f1": per_criterion_f1,
    }


# =============================================================================
# RELIABILITY DIAGRAM GENERATION
# =============================================================================


def plot_reliability_diagram(
    per_query_df: pd.DataFrame,
    prob_col: str = "p4_prob_calibrated",
    gold_col: str = "has_evidence_gold",
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> Dict[str, Any]:
    """Generate and optionally save a reliability diagram.

    A reliability diagram (calibration plot) shows how well predicted
    probabilities match empirical frequencies.

    Args:
        per_query_df: DataFrame with per-query results
        prob_col: Column with predicted probability
        gold_col: Column with gold label (0/1)
        n_bins: Number of bins for calibration curve
        output_path: If provided, save plot to this path
        title: Plot title

    Returns:
        Dictionary with calibration data and ECE/MCE metrics
    """
    # Get calibration data
    cal_metrics = compute_calibration_metrics(
        per_query_df, prob_col, gold_col, n_bins
    )

    if output_path:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Left plot: Reliability diagram
            curve = cal_metrics["reliability_curve"]
            prob_true = curve["prob_true"]
            prob_pred = curve["prob_pred"]

            ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            ax1.scatter(prob_pred, prob_true, s=50, alpha=0.7, label="Model")
            ax1.plot(prob_pred, prob_true, "b-", alpha=0.5)
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title(f"{title}\nECE={cal_metrics.get('ece', 0):.4f}")
            ax1.legend()
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3)

            # Right plot: Histogram of predictions
            y_prob = per_query_df[prob_col].dropna().values
            ax2.hist(y_prob, bins=n_bins, edgecolor="black", alpha=0.7)
            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Count")
            ax2.set_title("Prediction Distribution")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            cal_metrics["plot_saved"] = output_path
        except ImportError:
            cal_metrics["plot_error"] = "matplotlib not available"

    return cal_metrics


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================


def bootstrap_ci(
    metric_fn: Callable,
    data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    unit: str = "query",
    unit_col: Optional[str] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        metric_fn: Function that computes the metric from data
        data: DataFrame or tuple of (y_true, y_score)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (0.95 for 95% CI)
        seed: Random seed for reproducibility
        unit: Resampling unit ("query" or "post")
        unit_col: Column name for resampling unit (if unit="post")

    Returns:
        Dictionary with mean, ci_lower, ci_upper, std
    """
    rng = np.random.default_rng(seed)

    if isinstance(data, tuple):
        # (y_true, y_score) format
        y_true, y_score = data
        n = len(y_true)

        boot_values = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            try:
                val = metric_fn(y_true[indices], y_score[indices])
                boot_values.append(val)
            except Exception:
                continue
    else:
        # DataFrame format
        if unit == "post" and unit_col:
            # Resample by post_id
            units = data[unit_col].unique()
            n_units = len(units)

            boot_values = []
            for _ in range(n_bootstrap):
                sampled_units = rng.choice(units, size=n_units, replace=True)
                df_boot = pd.concat([data[data[unit_col] == u] for u in sampled_units], ignore_index=True)
                try:
                    val = metric_fn(df_boot)
                    boot_values.append(val)
                except Exception:
                    continue
        else:
            # Resample by query
            n = len(data)
            boot_values = []
            for _ in range(n_bootstrap):
                df_boot = data.sample(n=n, replace=True, random_state=rng.integers(0, 2**31))
                try:
                    val = metric_fn(df_boot)
                    boot_values.append(val)
                except Exception:
                    continue

    if not boot_values:
        return {"error": "No successful bootstrap iterations"}

    boot_values = np.array(boot_values)
    alpha = 1 - confidence

    return {
        "mean": float(np.mean(boot_values)),
        "std": float(np.std(boot_values)),
        "ci_lower": float(np.percentile(boot_values, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(boot_values, 100 * (1 - alpha / 2))),
        "n_bootstrap": len(boot_values),
    }


# =============================================================================
# PER-CRITERION METRICS
# =============================================================================


def compute_per_criterion_metrics(
    per_query_df: pd.DataFrame,
    prob_col: str = "p4_prob_calibrated",
    gold_col: str = "has_evidence_gold",
    criterion_col: str = "criterion_id",
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by criterion.

    Args:
        per_query_df: DataFrame with per-query results
        prob_col: Column with predicted probability
        gold_col: Column with gold label
        criterion_col: Column with criterion ID

    Returns:
        Dictionary of criterion_id -> metrics
    """
    results = {}

    for criterion_id in sorted(per_query_df[criterion_col].unique()):
        df_crit = per_query_df[per_query_df[criterion_col] == criterion_id]

        y_true = df_crit[gold_col].values
        y_score = df_crit[prob_col].values

        # Remove NaN
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_score))
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]

        crit_metrics = {
            "n_queries": len(y_true),
            "n_positive": int(y_true.sum()),
            "positive_rate": float(y_true.mean()) if len(y_true) > 0 else 0.0,
        }

        # Only compute if we have both classes
        if len(np.unique(y_true)) >= 2:
            crit_metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            crit_metrics["auprc"] = float(average_precision_score(y_true, y_score))

        # Ranking metrics for positive queries
        df_pos = df_crit[df_crit[gold_col] == 1]
        if len(df_pos) > 0:
            if "evidence_recall_at_k" in df_pos.columns:
                valid_recall = df_pos["evidence_recall_at_k"].dropna()
                if len(valid_recall) > 0:
                    crit_metrics["evidence_recall_at_k"] = float(valid_recall.mean())
            if "mrr" in df_pos.columns:
                valid_mrr = df_pos["mrr"].dropna()
                if len(valid_mrr) > 0:
                    crit_metrics["mrr"] = float(valid_mrr.mean())

        results[criterion_id] = crit_metrics

    return results


# =============================================================================
# MAIN COMPUTE ALL FUNCTION
# =============================================================================


def compute_all_metrics(
    per_query_csv: Union[str, pd.DataFrame],
    prob_col: str = "p4_prob_calibrated",
    gold_col: str = "has_evidence_gold",
    compute_cis: bool = True,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> MetricBundle:
    """Compute all metrics from per_query.csv.

    This is the main entry point for metric computation.

    Args:
        per_query_csv: Path to CSV or DataFrame
        prob_col: Column with predicted probability
        gold_col: Column with gold label
        compute_cis: Whether to compute bootstrap CIs
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        MetricBundle with all computed metrics
    """
    # Load data
    if isinstance(per_query_csv, str):
        df = pd.read_csv(per_query_csv)
    else:
        df = per_query_csv.copy()

    bundle = MetricBundle()
    bundle.metadata = {
        "n_queries_total": len(df),
        "n_queries_with_evidence": int(df[gold_col].sum()),
        "seed": seed,
    }

    # Classification metrics (all_queries protocol)
    class_metrics = compute_classification_metrics(df, prob_col, gold_col)

    for metric_name in ["auroc", "auprc", "brier_score"]:
        if metric_name in class_metrics:
            result = MetricResult(
                name=metric_name,
                value=class_metrics[metric_name],
                protocol="all_queries",
                n_samples=class_metrics.get("n_queries_all"),
            )

            # Compute CI if requested
            if compute_cis and metric_name in ["auroc", "auprc"]:
                if metric_name == "auroc":
                    ci = bootstrap_ci(
                        lambda yt, ys: roc_auc_score(yt, ys),
                        (df[gold_col].values, df[prob_col].values),
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                else:
                    ci = bootstrap_ci(
                        lambda yt, ys: average_precision_score(yt, ys),
                        (df[gold_col].values, df[prob_col].values),
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                result.ci_lower = ci.get("ci_lower")
                result.ci_upper = ci.get("ci_upper")

            bundle.classification[metric_name] = result

    # Add TPR@FPR metrics
    for fpr in [1, 3, 5, 10]:
        key = f"tpr_at_fpr_{fpr}pct"
        if key in class_metrics:
            bundle.classification[key] = MetricResult(
                name=key,
                value=class_metrics[key],
                protocol="all_queries",
            )

    # Ranking metrics (positives_only protocol)
    rank_metrics = compute_ranking_metrics_from_csv(df)

    for metric_name in ["evidence_recall_at_k", "mrr", "evidence_precision_at_k"]:
        if metric_name in rank_metrics:
            result = MetricResult(
                name=metric_name,
                value=rank_metrics[metric_name],
                protocol="positives_only",
                n_samples=rank_metrics.get("n_queries_positives_only"),
            )

            # Compute CI for key ranking metrics
            if compute_cis and metric_name == "evidence_recall_at_k":
                df_pos = df[df[gold_col] == 1]
                if metric_name in df_pos.columns:
                    valid_vals = df_pos[metric_name].dropna().values
                    if len(valid_vals) > 0:
                        ci = bootstrap_ci(
                            lambda x, _: np.mean(x),
                            (valid_vals, valid_vals),
                            n_bootstrap=n_bootstrap,
                            seed=seed,
                        )
                        result.ci_lower = ci.get("ci_lower")
                        result.ci_upper = ci.get("ci_upper")

            bundle.ranking[metric_name] = result

    # Calibration metrics
    cal_metrics = compute_calibration_metrics(df, prob_col, gold_col)
    for metric_name in ["ece", "mce", "brier_score"]:
        if metric_name in cal_metrics:
            bundle.calibration[metric_name] = MetricResult(
                name=metric_name,
                value=cal_metrics[metric_name],
                protocol="all_queries",
            )

    # Store reliability curve
    if "reliability_curve" in cal_metrics:
        bundle.metadata["reliability_curve"] = cal_metrics["reliability_curve"]

    # Per-criterion metrics
    bundle.per_criterion = compute_per_criterion_metrics(df, prob_col, gold_col)

    return bundle


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================


def verify_auprc_not_recall(auprc: float, recall_at_10: float, tolerance: float = 0.01) -> bool:
    """Verify AUPRC is not accidentally set to Recall@10.

    This is a safety check to prevent the common error of confusing these metrics.

    Args:
        auprc: Computed AUPRC value
        recall_at_10: Computed Recall@10 value
        tolerance: How close values can be before flagging

    Returns:
        True if values are different (safe), False if suspiciously similar
    """
    if abs(auprc - recall_at_10) < tolerance:
        import warnings
        warnings.warn(
            f"AUPRC ({auprc:.4f}) is very close to Recall@10 ({recall_at_10:.4f}). "
            "This may indicate metric confusion. AUPRC should be computed using "
            "sklearn.average_precision_score, NOT recall_at_k."
        )
        return False
    return True


def crosscheck_metrics(
    bundle: MetricBundle,
    expected: Dict[str, float],
    tolerance: float = 0.001,
) -> Tuple[bool, List[str]]:
    """Cross-check computed metrics against expected values.

    Args:
        bundle: Computed metric bundle
        expected: Dictionary of expected metric values
        tolerance: Acceptable difference

    Returns:
        Tuple of (all_passed, list of discrepancy messages)
    """
    discrepancies = []

    # Check classification metrics
    for name, expected_val in expected.items():
        if name in bundle.classification:
            computed = bundle.classification[name].value
            if abs(computed - expected_val) > tolerance:
                discrepancies.append(
                    f"{name}: expected {expected_val:.4f}, computed {computed:.4f}"
                )
        elif name in bundle.ranking:
            computed = bundle.ranking[name].value
            if abs(computed - expected_val) > tolerance:
                discrepancies.append(
                    f"{name}: expected {expected_val:.4f}, computed {computed:.4f}"
                )

    return len(discrepancies) == 0, discrepancies
