"""Retriever Zoo: Simplified interface for NV-Embed-v2 retriever.

This module provides the NV-Embed-v2 retriever which loads pre-computed
embeddings from cache. The embeddings must be generated using the
scripts/encode_nv_embed.py script in the 'nv-embed-v2' conda environment.

Two-Environment Architecture:
1. nv-embed-v2 env: Run scripts/encode_nv_embed.py to cache embeddings
2. llmhe env: Run pipeline which loads cached embeddings via this module
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from final_sc_review.data.schemas import Sentence
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Standard result format for all retrievers."""
    sent_uid: str
    text: str
    score: float
    component_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for a retriever."""
    name: str
    model_id: str
    retriever_type: str = "dense"
    max_length: int = 512
    batch_size: int = 8
    use_fp16: bool = True
    pooling: str = "cls"
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    trust_remote_code: bool = True


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, config: RetrieverConfig, sentences: List[Sentence], cache_dir: Path):
        self.config = config
        self.sentences = sentences
        self.cache_dir = cache_dir / config.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.post_to_indices: Dict[str, List[int]] = {}
        self.sent_uid_to_index: Dict[str, int] = {}
        for idx, sent in enumerate(sentences):
            self.post_to_indices.setdefault(sent.post_id, []).append(idx)
            self.sent_uid_to_index[sent.sent_uid] = idx

    @abstractmethod
    def encode_corpus(self, rebuild: bool = False) -> None:
        """Encode and cache corpus embeddings."""
        pass

    @abstractmethod
    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k: int = 100,
    ) -> List[RetrievalResult]:
        """Retrieve candidates from within a specific post."""
        pass

    def _corpus_fingerprint(self) -> str:
        """Compute fingerprint of corpus for cache validation."""
        h = hashlib.sha256()
        for sent in self.sentences:
            h.update(f"{sent.post_id}|{sent.sid}|{sent.text}".encode())
        return h.hexdigest()[:16]


class NVEmbedRetriever(BaseRetriever):
    """NV-Embed-v2 retriever that loads pre-computed embeddings.

    This retriever expects embeddings to be pre-computed using
    scripts/encode_nv_embed.py in the 'nv-embed-v2' conda environment.

    For query encoding, it uses sentence-transformers which works
    in the llmhe environment.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        sentences: List[Sentence],
        cache_dir: Path,
        device: Optional[str] = None,
    ):
        super().__init__(config, sentences, cache_dir)
        self.device = device or "cuda"
        self.embeddings: Optional[np.ndarray] = None
        self.model = None
        self._uid_to_idx: Optional[Dict[str, int]] = None

    def _load_query_encoder(self):
        """Load a compatible model for query encoding."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        # Use a compatible model for query encoding
        # The corpus embeddings are from NV-Embed-v2, but we need a model
        # that produces compatible embeddings for queries
        logger.info("Loading query encoder for NV-Embed-v2 compatibility")

        # Try to load NV-Embed-v2 via sentence-transformers
        # If it fails due to compatibility, fall back to loading embeddings only
        try:
            self.model = SentenceTransformer(
                self.config.model_id,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info("  Loaded NV-Embed-v2 for query encoding")
        except Exception as e:
            logger.warning(f"Could not load NV-Embed-v2 for queries: {e}")
            logger.warning("Query encoding will not be available")
            self.model = None

    def encode_corpus(self, rebuild: bool = False) -> None:
        """Load pre-computed embeddings from cache.

        The embeddings must be pre-computed using scripts/encode_nv_embed.py
        in the 'nv-embed-v2' conda environment.
        """
        embeddings_path = self.cache_dir / "embeddings.npy"
        fingerprint_path = self.cache_dir / "fingerprint.json"
        uid_mapping_path = self.cache_dir / "uid_to_idx.json"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"NV-Embed-v2 embeddings not found at {embeddings_path}. "
                f"Please run: conda activate nv-embed-v2 && python scripts/encode_nv_embed.py"
            )

        # Validate fingerprint
        if fingerprint_path.exists():
            with open(fingerprint_path) as f:
                meta = json.load(f)
            expected_fp = self._corpus_fingerprint()
            if meta.get("corpus") != expected_fp:
                raise ValueError(
                    f"Corpus fingerprint mismatch. Cached: {meta.get('corpus')}, Current: {expected_fp}. "
                    f"Please re-run: conda activate nv-embed-v2 && python scripts/encode_nv_embed.py --rebuild"
                )

        logger.info(f"Loading cached NV-Embed-v2 embeddings from {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        logger.info(f"  Loaded embeddings: {self.embeddings.shape}")

        # Load UID mapping
        if uid_mapping_path.exists():
            with open(uid_mapping_path) as f:
                self._uid_to_idx = json.load(f)
        else:
            # Build from sentences
            self._uid_to_idx = {s.sent_uid: i for i, s in enumerate(self.sentences)}

    def retrieve_within_post(
        self,
        query: str,
        post_id: str,
        top_k: int = 100,
    ) -> List[RetrievalResult]:
        """Retrieve candidates from within a specific post."""
        if self.embeddings is None:
            self.encode_corpus()

        indices = self.post_to_indices.get(post_id, [])
        if not indices:
            return []

        # Encode query
        self._load_query_encoder()

        if self.model is None:
            raise RuntimeError(
                "Query encoder not available. NV-Embed-v2 requires the nv-embed-v2 "
                "conda environment for query encoding."
            )

        query_text = self.config.query_prefix + query if self.config.query_prefix else query

        # Use sentence-transformers encode
        query_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Get within-post candidates
        candidate_embs = self.embeddings[indices]
        scores = candidate_embs @ query_emb

        ranked_indices = np.argsort(-scores)[:top_k]
        results = []
        for rank_idx in ranked_indices:
            idx = indices[rank_idx]
            sent = self.sentences[idx]
            results.append(RetrievalResult(
                sent_uid=sent.sent_uid,
                text=sent.text,
                score=float(scores[rank_idx]),
                component_scores={"dense": float(scores[rank_idx])},
            ))
        return results


# Default configuration for NV-Embed-v2
NV_EMBED_V2_CONFIG = RetrieverConfig(
    name="nv-embed-v2",
    model_id="nvidia/NV-Embed-v2",
    retriever_type="dense",
    max_length=512,
    batch_size=8,
    query_prefix="Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: ",
    trust_remote_code=True,
)


class RetrieverZoo:
    """Factory for NV-Embed-v2 retriever.

    Simplified zoo that only supports NV-Embed-v2 (the best performing retriever).
    """

    DEFAULT_RETRIEVERS = [NV_EMBED_V2_CONFIG]

    def __init__(
        self,
        sentences: List[Sentence],
        cache_dir: Path,
        configs: Optional[List[RetrieverConfig]] = None,
        device: Optional[str] = None,
    ):
        self.sentences = sentences
        self.cache_dir = cache_dir / "retriever_zoo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.configs = configs or self.DEFAULT_RETRIEVERS
        self._retrievers: Dict[str, BaseRetriever] = {}

    def get_retriever(self, name: str = "nv-embed-v2") -> BaseRetriever:
        """Get the NV-Embed-v2 retriever."""
        if name in self._retrievers:
            return self._retrievers[name]

        config = None
        for c in self.configs:
            if c.name == name:
                config = c
                break

        if config is None:
            raise ValueError(
                f"Unknown retriever: {name}. "
                f"This simplified zoo only supports: {self.list_retrievers()}"
            )

        retriever = NVEmbedRetriever(
            config=config,
            sentences=self.sentences,
            cache_dir=self.cache_dir,
            device=self.device,
        )
        self._retrievers[name] = retriever
        return retriever

    def list_retrievers(self) -> List[str]:
        """List available retriever names."""
        return [c.name for c in self.configs]

    def encode_all(self, rebuild: bool = False) -> None:
        """Load cached embeddings for all retrievers."""
        for config in self.configs:
            logger.info(f"Loading embeddings for {config.name}")
            try:
                retriever = self.get_retriever(config.name)
                retriever.encode_corpus(rebuild=rebuild)
            except Exception as e:
                logger.warning(f"Failed to load {config.name}: {e}")
