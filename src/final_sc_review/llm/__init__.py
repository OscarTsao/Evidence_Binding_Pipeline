"""LLM integration modules for evidence retrieval pipeline.

This package provides:
- LLM-based reranking with bias controls
- LLM verification for UNCERTAIN cases
- A.9 Suicidal Ideation classification (high-stakes clinical safety)
- Hybrid NE detection (GNN confidence + LLM fallback)

All modules include:
- Position bias mitigation (A/B vs B/A swapping)
- Self-consistency checks (multiple runs)
- Caching for efficiency

Note on criterion naming:
- A.9 = Suicidal Ideation (DSM-5)
- A.10 = SPECIAL_CASE (expert discrimination cases, per ReDSM5 taxonomy)

Module Status (Updated 2026-01-30):
==============================================================
| Module                  | Status           | Performance              |
|-------------------------|------------------|--------------------------|
| LLMVerifier             | PRODUCTION-READY | 87% acc, 0.8855 AUROC    |
| SuicidalIdeationClassifier (A.9) | PRODUCTION-READY | 75% acc, 0.8902 AUROC, 91% recall |
| HybridNEDetector        | PRODUCTION-READY | GNN + LLM hybrid         |
| LLMReranker             | EXPERIMENTAL     | High position bias (not recommended) |
| HybridPipeline          | PRODUCTION-READY | Supports NE detection + dynamic-K |

Recommendation: Use LLMVerifier and SuicidalIdeationClassifier for production.
               Use Jina-Reranker-v3 instead of LLMReranker.
               Use HybridNEDetector for no-evidence detection (replaces P1 NE Gate).
"""

from .base import extract_json_from_response, JSONExtractionError, LLMBase
from .reranker import LLMReranker
from .verifier import LLMVerifier
from .suicidal_ideation_classifier import SuicidalIdeationClassifier, A10Classifier  # A10Classifier is deprecated alias
from .ne_detector import HybridNEDetector, NEDetectionResult

__all__ = [
    # Base utilities
    'extract_json_from_response',
    'JSONExtractionError',
    'LLMBase',
    # Modules
    'LLMReranker',
    'LLMVerifier',
    'SuicidalIdeationClassifier',
    'A10Classifier',
    'HybridNEDetector',
    'NEDetectionResult',
]

# Data loader
from .data_loader import LLMEvaluationDataLoader

__all__.append('LLMEvaluationDataLoader')

# Phase 4: Production hybrid pipeline
from .hybrid_pipeline import HybridPipeline

__all__.append('HybridPipeline')
