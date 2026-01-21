"""LLM integration modules for evidence retrieval pipeline.

This package provides:
- LLM-based reranking with bias controls
- LLM verification for UNCERTAIN cases
- A.9 Suicidal Ideation classification (high-stakes clinical safety)

All modules include:
- Position bias mitigation (A/B vs B/A swapping)
- Self-consistency checks (multiple runs)
- Caching for efficiency

Note on criterion naming:
- A.9 = Suicidal Ideation (DSM-5)
- A.10 = SPECIAL_CASE (expert discrimination cases, per ReDSM5 taxonomy)
"""

from .reranker import LLMReranker
from .verifier import LLMVerifier
from .suicidal_ideation_classifier import SuicidalIdeationClassifier, A10Classifier  # A10Classifier is deprecated alias

__all__ = ['LLMReranker', 'LLMVerifier', 'SuicidalIdeationClassifier', 'A10Classifier']

# Data loader
from .data_loader import LLMEvaluationDataLoader

__all__.append('LLMEvaluationDataLoader')

# Phase 4: Production hybrid pipeline
from .hybrid_pipeline import HybridPipeline

__all__.append('HybridPipeline')
