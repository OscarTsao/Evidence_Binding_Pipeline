"""Baseline retrieval methods for comparison.

This module provides baseline implementations for evidence retrieval:

Lexical Baselines:
- BM25: Okapi BM25 lexical retrieval
- TF-IDF: Term frequency-inverse document frequency with cosine similarity
- Random: Random ranking baseline (reference)

Dense Bi-encoder Baselines:
- E5-base: intfloat/e5-base-v2 bi-encoder
- Contriever: facebook/contriever unsupervised dense retrieval
- BGE: BAAI/bge-base-en-v1.5 bi-encoder

Cross-encoder Baseline:
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 for joint scoring

Linear Model Baseline:
- Linear: TF-IDF with interaction features (LogReg/SVM-style)

LLM Embedding Baseline:
- LLM-embed: Instruction-tuned embedding model (e5-large-v2)

All baselines implement the same interface:
    score(query: str, sentences: List[str]) -> np.ndarray

Usage:
    from final_sc_review.baselines import get_baseline, list_baselines

    # List available baselines
    print(list_baselines())

    # Use a baseline
    baseline = get_baseline("bge")
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
