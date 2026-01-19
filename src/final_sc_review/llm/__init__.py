"""LLM integration modules for evidence retrieval pipeline.

This package provides:
- LLM-based reranking with bias controls
- LLM verification for UNCERTAIN cases
- A.10-specific classification

All modules include:
- Position bias mitigation (A/B vs B/A swapping)
- Self-consistency checks (multiple runs)
- Caching for efficiency
"""

from .reranker import LLMReranker
from .verifier import LLMVerifier
from .a10_classifier import A10Classifier

__all__ = ['LLMReranker', 'LLMVerifier', 'A10Classifier']

# Data loader
from .data_loader import LLMEvaluationDataLoader

__all__.append('LLMEvaluationDataLoader')

# Phase 4: Production hybrid pipeline
from .hybrid_pipeline import HybridPipeline

__all__.append('HybridPipeline')
