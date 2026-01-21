"""Metrics package for S-C retrieval evaluation.

Exports:
- Ranking metrics: recall_at_k, precision_at_k, mrr_at_k, map_at_k, ndcg_at_k, f1_at_k
- Evidence metrics: evidence_coverage
- K policy: compute_k_eff, get_paper_k_values, K_PRIMARY, K_EXTENDED
- Evaluation: evaluate_rankings, dual_evaluate, paper_evaluate
- Canonical compute: compute_all_metrics, compute_classification_metrics, etc.
- Multi-label: compute_multilabel_f1 (micro/macro F1)
- Calibration: plot_reliability_diagram

Metric Registry:
    Use METRIC_REGISTRY to get a complete list of available metrics
    and their properties (protocol, requires_gold_ids, etc.)
"""

from final_sc_review.metrics.ranking import (
    evidence_coverage,
    f1_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from final_sc_review.metrics.k_policy import (
    compute_k_eff,
    get_paper_k_values,
    K_PRIMARY,
    K_EXTENDED,
    K_CEILING,
)
from final_sc_review.metrics.retrieval_eval import (
    evaluate_rankings,
    dual_evaluate,
    paper_evaluate,
    evaluate_with_k_eff,
    format_dual_metrics,
)
from final_sc_review.metrics.compute_metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_ranking_metrics,
    compute_ranking_metrics_from_csv,
    compute_calibration_metrics,
    compute_per_criterion_metrics,
    compute_multilabel_f1,
    plot_reliability_diagram,
    bootstrap_ci,
    verify_auprc_not_recall,
    crosscheck_metrics,
    MetricResult,
    MetricBundle,
)


# =============================================================================
# METRICS REGISTRY
# =============================================================================

METRIC_REGISTRY = {
    # Ranking metrics (positives_only protocol)
    "recall@k": {
        "function": recall_at_k,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Fraction of relevant items in top-K",
    },
    "precision@k": {
        "function": precision_at_k,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Fraction of top-K items that are relevant",
    },
    "mrr@k": {
        "function": mrr_at_k,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Reciprocal rank of first relevant item",
    },
    "map@k": {
        "function": map_at_k,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Mean average precision up to rank K",
    },
    "ndcg@k": {
        "function": ndcg_at_k,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Normalized discounted cumulative gain",
    },
    "f1@k": {
        "function": f1_at_k,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Harmonic mean of Precision@K and Recall@K",
    },
    "evidence_coverage": {
        "function": evidence_coverage,
        "protocol": "positives_only",
        "requires_gold_ids": True,
        "description": "Fraction of evidence sentences retrieved (alias for Recall@K)",
    },
    # Classification metrics (all_queries protocol)
    "auroc": {
        "function": "sklearn.roc_auc_score",
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Area Under ROC Curve",
    },
    "auprc": {
        "function": "sklearn.average_precision_score",
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Area Under Precision-Recall Curve (NOT Recall@K!)",
    },
    "micro_f1": {
        "function": compute_multilabel_f1,
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Micro-averaged F1 across all post-criterion pairs",
    },
    "macro_f1": {
        "function": compute_multilabel_f1,
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Macro-averaged F1 (mean F1 per criterion)",
    },
    # Calibration metrics
    "ece": {
        "function": compute_calibration_metrics,
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Expected Calibration Error",
    },
    "mce": {
        "function": compute_calibration_metrics,
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Maximum Calibration Error",
    },
    "brier_score": {
        "function": compute_calibration_metrics,
        "protocol": "all_queries",
        "requires_gold_ids": False,
        "description": "Brier Score (mean squared error of probabilities)",
    },
}


def list_metrics() -> list:
    """List all available metric names."""
    return sorted(METRIC_REGISTRY.keys())


def get_metric_info(name: str) -> dict:
    """Get information about a metric."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list_metrics()}")
    return METRIC_REGISTRY[name]


__all__ = [
    # Ranking metrics
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "map_at_k",
    "ndcg_at_k",
    "f1_at_k",
    "evidence_coverage",
    # K policy
    "compute_k_eff",
    "get_paper_k_values",
    "K_PRIMARY",
    "K_EXTENDED",
    "K_CEILING",
    # Evaluation
    "evaluate_rankings",
    "dual_evaluate",
    "paper_evaluate",
    "evaluate_with_k_eff",
    "format_dual_metrics",
    # Canonical compute (v3.0)
    "compute_all_metrics",
    "compute_classification_metrics",
    "compute_ranking_metrics",
    "compute_ranking_metrics_from_csv",
    "compute_calibration_metrics",
    "compute_per_criterion_metrics",
    "compute_multilabel_f1",
    "plot_reliability_diagram",
    "bootstrap_ci",
    "verify_auprc_not_recall",
    "crosscheck_metrics",
    "MetricResult",
    "MetricBundle",
    # Registry
    "METRIC_REGISTRY",
    "list_metrics",
    "get_metric_info",
]
