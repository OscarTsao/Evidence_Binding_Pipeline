"""Metrics package for S-C retrieval evaluation.

Exports:
- Ranking metrics: recall_at_k, mrr_at_k, map_at_k, ndcg_at_k
- K policy: compute_k_eff, get_paper_k_values, K_PRIMARY, K_EXTENDED
- Evaluation: evaluate_rankings, dual_evaluate, paper_evaluate
- Canonical compute: compute_all_metrics, compute_classification_metrics, etc.
"""

from final_sc_review.metrics.ranking import (
    recall_at_k,
    mrr_at_k,
    map_at_k,
    ndcg_at_k,
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
    bootstrap_ci,
    verify_auprc_not_recall,
    crosscheck_metrics,
    MetricResult,
    MetricBundle,
)

__all__ = [
    # Ranking metrics
    "recall_at_k",
    "mrr_at_k",
    "map_at_k",
    "ndcg_at_k",
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
    "bootstrap_ci",
    "verify_auprc_not_recall",
    "crosscheck_metrics",
    "MetricResult",
    "MetricBundle",
]
