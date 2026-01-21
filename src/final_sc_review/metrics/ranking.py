"""Ranking metrics for retrieval evaluation.

All metrics follow standard IR definitions:
- Recall@K: Fraction of relevant items retrieved in top-K
- Precision@K: Fraction of top-K items that are relevant
- MRR@K: Reciprocal rank of first relevant item
- MAP@K: Mean average precision up to rank K
- nDCG@K: Normalized discounted cumulative gain

All metrics use binary relevance (relevant/not relevant).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def recall_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """Recall@K: fraction of gold items appearing in top-K.

    Args:
        gold_ids: Set of relevant item IDs
        ranked_ids: Ranked list of retrieved item IDs
        k: Cutoff position

    Returns:
        Recall value in [0, 1]
    """
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = set(ranked_ids[:k]) & gold
    return len(hits) / len(gold)


def precision_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """Precision@K: fraction of top-K items that are relevant.

    Args:
        gold_ids: Set of relevant item IDs
        ranked_ids: Ranked list of retrieved item IDs
        k: Cutoff position

    Returns:
        Precision value in [0, 1]
    """
    gold = set(gold_ids)
    if not ranked_ids or k <= 0:
        return 0.0
    actual_k = min(k, len(ranked_ids))
    hits = len(set(ranked_ids[:actual_k]) & gold)
    return hits / actual_k


def mrr_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """MRR@K: reciprocal rank of first relevant item in top-K."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            return 1.0 / idx
    return 0.0


def map_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """MAP@K: mean average precision with binary relevance."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        if sent_id in gold:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / min(len(gold), k)


def ndcg_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """nDCG@K with binary relevance."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, sent_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if sent_id in gold else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evidence_coverage(
    gold_ids: Iterable[str],
    ranked_ids: Sequence[str],
    k: int,
) -> float:
    """Evidence Coverage@K: fraction of unique evidence sentences retrieved.

    This is the same as Recall@K but emphasizes coverage of distinct evidence.

    Args:
        gold_ids: Set of gold evidence sentence IDs
        ranked_ids: Ranked list of retrieved sentence IDs
        k: Cutoff position

    Returns:
        Coverage value in [0, 1]
    """
    return recall_at_k(gold_ids, ranked_ids, k)


def f1_at_k(gold_ids: Iterable[str], ranked_ids: Sequence[str], k: int) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K.

    Args:
        gold_ids: Set of relevant item IDs
        ranked_ids: Ranked list of retrieved item IDs
        k: Cutoff position

    Returns:
        F1 value in [0, 1]
    """
    p = precision_at_k(gold_ids, ranked_ids, k)
    r = recall_at_k(gold_ids, ranked_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
