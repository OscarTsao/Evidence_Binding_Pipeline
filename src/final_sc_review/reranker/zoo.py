"""Reranker Zoo: Simplified interface for Jina-Reranker-v3.

This module provides the Jina-Reranker-v3 reranker, which is the best
performing reranker from HPO (nDCG@10 = 0.8658 with NV-Embed-v2).

Hardware optimizations:
- BF16 with FP16 fallback (AMP-style)
- TF32 for tensor cores
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from final_sc_review.utils.logging import get_logger
from final_sc_review.utils.gpu_optimize import (
    get_optimal_dtype,
    get_device,
    auto_init as gpu_auto_init,
)

logger = get_logger(__name__)

# Apply hardware optimizations on module load
gpu_auto_init()


@dataclass
class RerankerConfig:
    """Configuration for a reranker."""
    name: str
    model_id: str
    reranker_type: str = "listwise"
    max_length: int = 1024
    batch_size: int = 128
    use_fp16: bool = True
    trust_remote_code: bool = True
    query_instruction: str = ""
    listwise_max_docs: int = 32
    use_torch_compile: bool = False  # Enable for ~1.4x speedup (requires warmup)


@dataclass
class RerankerResult:
    """Result from reranking a single query."""
    sent_uid: str
    text: str
    score: float
    rank: int


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    def __init__(self, config: RerankerConfig, device: Optional[str] = None):
        self.config = config
        self.device = get_device(device)
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the reranker model."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # [(sent_uid, text), ...]
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        """Rerank candidates for a query."""
        pass

    def rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Tuple[str, str]]]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankerResult]]:
        """Batch rerank multiple queries."""
        return [
            self.rerank(query, candidates, top_k)
            for query, candidates in queries_and_candidates
        ]


class JinaRerankerV3(BaseReranker):
    """Jina-Reranker-v3 listwise reranker.

    This is the best performing reranker from HPO testing.
    Uses listwise scoring for efficient batch processing.
    """

    def __init__(self, config: RerankerConfig, device: Optional[str] = None):
        super().__init__(config, device)
        self.tokenizer = None

    @staticmethod
    def _extract_scores_from_logits(
        logits: np.ndarray,
        expected_len: Optional[int] = None,
    ) -> List[float]:
        """Extract scores from model logits with consistent handling.

        Args:
            logits: Model output logits (various shapes possible)
            expected_len: If provided, validate output length matches

        Returns:
            List of float scores

        Raises:
            ValueError: If expected_len provided and doesn't match
        """
        if logits.ndim == 1:
            scores = logits.tolist()
        elif logits.ndim == 2 and logits.shape[1] == 1:
            scores = logits.squeeze(-1).tolist()
        else:
            scores = logits[:, 0].tolist()

        if not isinstance(scores, list):
            scores = [float(scores)]

        # Validate length if expected
        if expected_len is not None and len(scores) != expected_len:
            raise ValueError(
                f"Score extraction produced {len(scores)} scores, "
                f"expected {expected_len}"
            )

        return scores

    def load_model(self) -> None:
        if self.model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading Jina-Reranker-v3: {self.config.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = get_optimal_dtype() if self.config.use_fp16 else torch.float32

        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.config.trust_remote_code,
        }

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.train(False)

        # Optional torch.compile for ~1.4x speedup
        if self.config.use_torch_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info(f"  Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"  torch.compile failed, using eager mode: {e}")

        logger.info(f"  Loaded Jina-Reranker-v3 with dtype={dtype}")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[RerankerResult]:
        if self.model is None:
            self.load_model()

        if not candidates:
            return []

        if self.config.query_instruction:
            query = self.config.query_instruction + query

        all_scores = []
        batch_size = min(self.config.listwise_max_docs, len(candidates))

        for i in range(0, len(candidates), batch_size):
            batch_candidates = candidates[i:i + batch_size]
            batch_texts = [text for _, text in batch_candidates]

            inputs = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.float().cpu().numpy()
                scores = self._extract_scores_from_logits(
                    logits, expected_len=len(batch_texts)
                )

            all_scores.extend(scores)

        results = []
        for i, (sent_uid, text) in enumerate(candidates):
            results.append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(all_scores[i]),
                rank=0,
            ))

        results.sort(key=lambda x: -x.score)
        for i, r in enumerate(results):
            r.rank = i + 1

        if top_k:
            results = results[:top_k]

        return results

    def rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Tuple[str, str]]]],
        top_k: Optional[int] = None,
    ) -> List[List[RerankerResult]]:
        """Batch rerank multiple queries efficiently."""
        if self.model is None:
            self.load_model()

        if not queries_and_candidates:
            return []

        all_queries = []
        all_texts = []
        candidate_info = []

        for query_idx, (query, candidates) in enumerate(queries_and_candidates):
            if not candidates:
                continue

            q = self.config.query_instruction + query if self.config.query_instruction else query

            for sent_uid, text in candidates:
                all_queries.append(q)
                all_texts.append(text)
                candidate_info.append((sent_uid, text, query_idx))

        if not all_queries:
            return [[] for _ in queries_and_candidates]

        all_scores = []
        batch_size = self.config.batch_size

        for i in range(0, len(all_queries), batch_size):
            batch_queries = all_queries[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.float().cpu().numpy()
                scores = self._extract_scores_from_logits(
                    logits, expected_len=len(batch_texts)
                )
                all_scores.extend(scores)

        all_results = [[] for _ in queries_and_candidates]

        for i, (sent_uid, text, query_idx) in enumerate(candidate_info):
            all_results[query_idx].append(RerankerResult(
                sent_uid=sent_uid,
                text=text,
                score=float(all_scores[i]),
                rank=0,
            ))

        for results in all_results:
            results.sort(key=lambda x: -x.score)
            for i, r in enumerate(results):
                r.rank = i + 1
            if top_k:
                results[:] = results[:top_k]

        return all_results


# Default configuration for Jina-Reranker-v3
JINA_RERANKER_V3_CONFIG = RerankerConfig(
    name="jina-reranker-v3",
    model_id="jinaai/jina-reranker-v3",
    reranker_type="listwise",
    max_length=1024,
    batch_size=128,
    listwise_max_docs=32,
    trust_remote_code=True,
)


class RerankerZoo:
    """Factory for Jina-Reranker-v3.

    Simplified zoo that only supports Jina-Reranker-v3 (the best performing reranker).
    """

    DEFAULT_RERANKERS = [JINA_RERANKER_V3_CONFIG]

    def __init__(
        self,
        configs: Optional[List[RerankerConfig]] = None,
        device: Optional[str] = None,
    ):
        self.device = get_device(device)
        self.configs = configs or self.DEFAULT_RERANKERS
        self._rerankers: Dict[str, BaseReranker] = {}

    def get_reranker(self, name: str = "jina-reranker-v3") -> BaseReranker:
        """Get the Jina-Reranker-v3 reranker."""
        if name in self._rerankers:
            return self._rerankers[name]

        config = None
        for c in self.configs:
            if c.name == name:
                config = c
                break

        if config is None:
            raise ValueError(
                f"Unknown reranker: {name}. "
                f"This simplified zoo only supports: {self.list_rerankers()}"
            )

        reranker = JinaRerankerV3(config, self.device)
        self._rerankers[name] = reranker
        return reranker

    def list_rerankers(self) -> List[str]:
        """List available reranker names."""
        return [c.name for c in self.configs]

    def get_config(self, name: str) -> RerankerConfig:
        """Get config for a reranker."""
        for c in self.configs:
            if c.name == name:
                return c
        raise ValueError(f"Unknown reranker: {name}")
