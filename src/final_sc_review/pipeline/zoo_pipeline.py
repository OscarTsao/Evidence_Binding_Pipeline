"""Zoo-based pipeline using NV-Embed-v2 retriever and Jina-Reranker-v3.

Best Pipeline Configuration (from HPO):
- Retriever: NV-Embed-v2 (nDCG@10 = 0.8658)
- Reranker: Jina-Reranker-v3

Two-Environment Architecture:
1. nv-embed-v2 env: Run scripts/encode_nv_embed.py to cache corpus embeddings
2. llmhe env: Run this pipeline which loads cached embeddings and performs reranking
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from final_sc_review.data.io import Sentence
from final_sc_review.retriever.zoo import RetrieverZoo, BaseRetriever
from final_sc_review.reranker.zoo import RerankerZoo, BaseReranker
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ZooPipelineConfig:
    """Configuration for zoo-based pipeline."""
    # Model selection
    retriever_name: str = "nv-embed-v2"
    reranker_name: str = "jina-reranker-v3"

    # Retrieval parameters
    top_k_retriever: int = 24
    top_k_final: int = 10

    # Fusion parameters (for hybrid retrievers)
    use_sparse: bool = False
    use_colbert: bool = False
    dense_weight: float = 1.0
    sparse_weight: float = 0.0
    colbert_weight: float = 0.0
    fusion_method: str = "rrf"  # "weighted_sum" or "rrf"
    score_normalization: str = "none"
    rrf_k: int = 60

    # Device
    device: Optional[str] = None


class ZooPipeline:
    """Pipeline using retriever and reranker from zoo."""

    def __init__(
        self,
        sentences: List[Sentence],
        cache_dir: Path,
        config: ZooPipelineConfig,
    ):
        self.sentences = sentences
        self.cache_dir = cache_dir
        self.config = config

        # Build lookup maps
        self.sent_uid_to_idx: Dict[str, int] = {
            s.sent_uid: i for i, s in enumerate(sentences)
        }
        self.post_to_indices: Dict[str, List[int]] = {}
        for i, s in enumerate(sentences):
            self.post_to_indices.setdefault(s.post_id, []).append(i)

        # Initialize zoos
        self._retriever_zoo = RetrieverZoo(
            sentences=sentences,
            cache_dir=cache_dir,
            device=config.device,
        )
        self._reranker_zoo = RerankerZoo()

        # Lazy-loaded models
        self._retriever: Optional[BaseRetriever] = None
        self._reranker: Optional[BaseReranker] = None

    @property
    def retriever(self) -> BaseRetriever:
        """Get or load the retriever."""
        if self._retriever is None:
            logger.info(f"Loading retriever: {self.config.retriever_name}")
            self._retriever = self._retriever_zoo.get_retriever(self.config.retriever_name)
        return self._retriever

    @property
    def reranker(self) -> BaseReranker:
        """Get or load the reranker."""
        if self._reranker is None:
            logger.info(f"Loading reranker: {self.config.reranker_name}")
            self._reranker = self._reranker_zoo.get_reranker(self.config.reranker_name)
        return self._reranker

    def retrieve(
        self,
        query: str,
        post_id: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, str, float]]:
        """Retrieve and rerank sentences for a query.

        Args:
            query: The query text (criterion text)
            post_id: The post ID to retrieve from
            top_k: Number of results to return (default: config.top_k_final)

        Returns:
            List of (sent_uid, sentence_text, score) tuples
        """
        top_k = top_k or self.config.top_k_final

        # Get candidate indices for this post
        candidate_indices = self.post_to_indices.get(post_id, [])
        if not candidate_indices:
            logger.warning(f"No sentences found for post_id: {post_id}")
            return []

        # Stage 1: Retrieval
        retrieval_results = self.retriever.retrieve_within_post(
            query=query,
            post_id=post_id,
            top_k=self.config.top_k_retriever,
        )

        if not retrieval_results:
            return []

        # Stage 2: Reranking (optional)
        if self.config.reranker_name is None:
            # No reranker - return retrieval results directly
            results = [
                (r.sent_uid, r.text, r.score)
                for r in retrieval_results[:top_k]
            ]
        else:
            # Extract candidate info for reranking
            candidates = [(r.sent_uid, r.text) for r in retrieval_results]

            # Rerank
            rerank_results = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=top_k,
            )

            # Format results
            results = [
                (r.sent_uid, r.text, r.score)
                for r in rerank_results
            ]

        return results

    def retrieve_batch(
        self,
        queries: List[Tuple[str, str]],  # List of (query, post_id)
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[str, str, float]]]:
        """Batch retrieve for multiple queries.

        Args:
            queries: List of (query_text, post_id) tuples
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        return [
            self.retrieve(query, post_id, top_k)
            for query, post_id in queries
        ]


class ConfigValidationError(ValueError):
    """Raised when config validation fails."""
    pass


# Valid model names for validation
VALID_RETRIEVERS = ["nv-embed-v2"]
VALID_RERANKERS = ["jina-reranker-v3"]


def load_zoo_pipeline_from_config(config_path: Path) -> ZooPipeline:
    """Load zoo pipeline from YAML config file with validation.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configured ZooPipeline instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If config values are invalid
    """
    import yaml
    from final_sc_review.data.io import load_sentence_corpus

    # Validate file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate config structure
    if not cfg or not isinstance(cfg, dict):
        raise ConfigValidationError(
            f"Invalid YAML config: {config_path} (empty or not a dictionary)"
        )

    # Validate required paths
    if "paths" not in cfg:
        raise ConfigValidationError("Config missing required 'paths' section")

    paths_cfg = cfg["paths"]
    for key in ["sentence_corpus", "cache_dir"]:
        if key not in paths_cfg:
            raise ConfigValidationError(f"Config paths missing required key: {key}")

    # Extract model names from config
    models_cfg = cfg.get("models", {})
    retriever_cfg = cfg.get("retriever", {})

    # Default to best HPO models
    retriever_name = models_cfg.get("retriever_name", "nv-embed-v2")
    reranker_name = models_cfg.get("reranker_name", "jina-reranker-v3")

    # Validate model names BEFORE loading corpus (fail-fast)
    if retriever_name not in VALID_RETRIEVERS:
        raise ConfigValidationError(
            f"Unknown retriever: {retriever_name}. "
            f"Valid options: {VALID_RETRIEVERS}"
        )

    if reranker_name not in VALID_RERANKERS:
        raise ConfigValidationError(
            f"Unknown reranker: {reranker_name}. "
            f"Valid options: {VALID_RERANKERS}"
        )

    # Extract and validate numeric parameters BEFORE loading corpus
    top_k_retriever = retriever_cfg.get("top_k_retriever", 24)
    top_k_final = retriever_cfg.get("top_k_final", 10)
    rrf_k = retriever_cfg.get("rrf_k", 60)

    if not isinstance(top_k_retriever, int) or top_k_retriever <= 0:
        raise ConfigValidationError(
            f"top_k_retriever must be positive integer, got {top_k_retriever}"
        )

    if not isinstance(top_k_final, int) or top_k_final <= 0:
        raise ConfigValidationError(
            f"top_k_final must be positive integer, got {top_k_final}"
        )

    if top_k_final > top_k_retriever:
        raise ConfigValidationError(
            f"top_k_final ({top_k_final}) cannot exceed "
            f"top_k_retriever ({top_k_retriever})"
        )

    if not isinstance(rrf_k, (int, float)) or rrf_k <= 0:
        raise ConfigValidationError(
            f"rrf_k must be positive number, got {rrf_k}"
        )

    # Load corpus after validation passes (expensive operation)
    sentences = load_sentence_corpus(Path(paths_cfg["sentence_corpus"]))
    cache_dir = Path(paths_cfg["cache_dir"])

    pipeline_config = ZooPipelineConfig(
        retriever_name=retriever_name,
        reranker_name=reranker_name,
        top_k_retriever=top_k_retriever,
        top_k_final=top_k_final,
        use_sparse=retriever_cfg.get("use_sparse", False),
        use_colbert=retriever_cfg.get("use_colbert", False),
        dense_weight=retriever_cfg.get("dense_weight", 1.0),
        sparse_weight=retriever_cfg.get("sparse_weight", 0.0),
        colbert_weight=retriever_cfg.get("colbert_weight", 0.0),
        fusion_method=retriever_cfg.get("fusion_method", "rrf"),
        score_normalization=retriever_cfg.get("score_normalization", "none"),
        rrf_k=rrf_k,
        device=cfg.get("device"),
    )

    return ZooPipeline(
        sentences=sentences,
        cache_dir=cache_dir,
        config=pipeline_config,
    )
