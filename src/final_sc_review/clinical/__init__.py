"""Clinical high-recall deployment module.

This module implements a clinically-validated, high-recall evidence retrieval system
with rigorous verification to prevent data leakage and ensure reproducibility.

Key components:
- ThreeStateGate: NEG/UNCERTAIN/POS classification with nested threshold selection
- ClinicalDynamicK: Per-state dynamic-K policies for evidence extraction
- Evaluation script: scripts/clinical/run_clinical_high_recall_eval.py
"""

from final_sc_review.clinical.config import ClinicalConfig, ThresholdConfig, DynamicKStateConfig
from final_sc_review.clinical.three_state_gate import ThreeStateGate, GateDecision
from final_sc_review.clinical.dynamic_k import ClinicalDynamicK
from final_sc_review.clinical.model_inference import ClinicalModelInference

__all__ = [
    "ClinicalConfig",
    "ThresholdConfig",
    "DynamicKStateConfig",
    "ThreeStateGate",
    "GateDecision",
    "ClinicalDynamicK",
    "ClinicalModelInference",
]
