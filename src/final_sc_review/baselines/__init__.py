"""Baseline retrieval methods for comparison.

This module provides baseline implementations for evidence retrieval:
- BM25: Lexical retrieval using Okapi BM25
- TF-IDF: Term frequency-inverse document frequency with cosine similarity
- E5-base: Dense bi-encoder baseline (intfloat/e5-base-v2)
- Contriever: Unsupervised dense retrieval (facebook/contriever)
- Random: Random ranking baseline

All baselines implement the same interface:
    score(query: str, sentences: List[str]) -> np.ndarray

Usage:
    from final_sc_review.baselines import get_baseline

    baseline = get_baseline("e5-base")
    scores = baseline.score(query, sentences)
"""

from final_sc_review.baselines.base import (
    BaselineRetriever,
    get_baseline,
    list_baselines,
)

__all__ = [
    "BaselineRetriever",
    "get_baseline",
    "list_baselines",
]
