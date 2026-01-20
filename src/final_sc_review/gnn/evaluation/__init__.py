"""GNN assessment infrastructure.

Components:
- Cross-validation orchestration
- Metric computation
- Threshold optimization
- P3 evaluation utilities
"""

from final_sc_review.gnn.evaluation.cv import CrossValidator
from final_sc_review.gnn.evaluation.metrics import NEGateMetrics, DynamicKMetrics
from final_sc_review.gnn.evaluation.p3_utils import (
    load_p3_model,
    run_p3_inference,
    compute_ranking_metrics,
    evaluate_graphs,
    load_graph_dataset,
    ensure_cpu,
)

__all__ = [
    "CrossValidator",
    "NEGateMetrics",
    "DynamicKMetrics",
    # P3 utilities
    "load_p3_model",
    "run_p3_inference",
    "compute_ranking_metrics",
    "evaluate_graphs",
    "load_graph_dataset",
    "ensure_cpu",
]
