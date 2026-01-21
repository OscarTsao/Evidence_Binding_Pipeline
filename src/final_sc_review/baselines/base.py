"""Base interface and factory for baseline retrievers.

All baselines implement:
- score(query, sentences) -> scores array
- retrieve_within_post(query, post_id, sentences, top_k) -> ranked results
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class BaselineRetriever(ABC):
    """Abstract base class for baseline retrievers."""

    name: str = "base"
    requires_gpu: bool = False

    @abstractmethod
    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences against a query.

        Args:
            query: Query text (criterion description)
            sentences: List of candidate sentences

        Returns:
            Array of scores, higher is more relevant
        """
        pass

    def retrieve_within_post(
        self,
        query: str,
        sentences: List[str],
        sentence_ids: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """Retrieve top-K sentences from a post.

        Args:
            query: Query text
            sentences: List of candidate sentences
            sentence_ids: Corresponding sentence IDs (sent_uid)
            top_k: Number of results to return

        Returns:
            List of (sent_uid, sentence, score) tuples, sorted by score descending
        """
        if not sentences:
            return []

        scores = self.score(query, sentences)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append((
                sentence_ids[idx],
                sentences[idx],
                float(scores[idx])
            ))

        return results


# =============================================================================
# LEXICAL BASELINES
# =============================================================================


class BM25Baseline(BaselineRetriever):
    """BM25 (Okapi) lexical retrieval baseline."""

    name = "bm25"
    requires_gpu = False

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25.

        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_cls = BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 required: pip install rank-bm25")

        self.k1 = k1
        self.b = b
        self._corpus_cache: Dict[str, Any] = {}

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using BM25."""
        if not sentences:
            return np.array([])

        # Tokenize
        tokenized_corpus = [s.lower().split() for s in sentences]
        tokenized_query = query.lower().split()

        # Build BM25 index
        bm25 = self._bm25_cls(tokenized_corpus, k1=self.k1, b=self.b)
        scores = bm25.get_scores(tokenized_query)

        return np.array(scores)


class TfidfBaseline(BaselineRetriever):
    """TF-IDF + cosine similarity baseline."""

    name = "tfidf"
    requires_gpu = False

    def __init__(self, **vectorizer_kwargs):
        """Initialize TF-IDF.

        Args:
            vectorizer_kwargs: Arguments for TfidfVectorizer
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self._vectorizer_cls = TfidfVectorizer
            self._cosine_similarity = cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        self.vectorizer_kwargs = vectorizer_kwargs

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF cosine similarity."""
        if not sentences:
            return np.array([])

        # Fit vectorizer on corpus + query
        vectorizer = self._vectorizer_cls(**self.vectorizer_kwargs)
        all_texts = sentences + [query]
        vectors = vectorizer.fit_transform(all_texts)

        # Compute cosine similarity
        query_vector = vectors[-1]
        doc_vectors = vectors[:-1]
        scores = self._cosine_similarity(query_vector, doc_vectors)[0]

        return np.array(scores)


class RandomBaseline(BaselineRetriever):
    """Random ranking baseline for reference."""

    name = "random"
    requires_gpu = False

    def __init__(self, seed: int = 42):
        """Initialize random baseline.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Return random scores."""
        if not sentences:
            return np.array([])

        return self.rng.random(len(sentences))


# =============================================================================
# DENSE BASELINES
# =============================================================================


class E5Baseline(BaselineRetriever):
    """E5-base-v2 bi-encoder baseline.

    Uses intfloat/e5-base-v2 for dense retrieval.
    Reference: https://huggingface.co/intfloat/e5-base-v2
    """

    name = "e5-base"
    requires_gpu = True

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize E5 baseline.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            cache_dir: Directory to cache embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def _get_model(self):
        """Lazy load model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading E5 model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts with E5-specific prefix.

        E5 models require:
        - Query prefix: "query: "
        - Passage prefix: "passage: "
        """
        model = self._get_model()

        # Add E5-specific prefixes
        if is_query:
            prefixed = [f"query: {t}" for t in texts]
        else:
            prefixed = [f"passage: {t}" for t in texts]

        embeddings = model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using E5 embeddings."""
        if not sentences:
            return np.array([])

        # Encode query and sentences
        query_emb = self._encode([query], is_query=True)[0]
        sent_embs = self._encode(sentences, is_query=False)

        # Cosine similarity (embeddings are normalized)
        scores = sent_embs @ query_emb

        return scores


class ContrieverBaseline(BaselineRetriever):
    """Contriever bi-encoder baseline.

    Uses facebook/contriever for unsupervised dense retrieval.
    Reference: https://huggingface.co/facebook/contriever
    """

    name = "contriever"
    requires_gpu = True

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize Contriever baseline.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            cache_dir: Directory to cache embeddings
        """
        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def _get_model(self):
        """Lazy load model."""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"Loading Contriever model: {self.model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

        return self._model, self._tokenizer

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings."""
        import torch

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts with Contriever."""
        import torch

        model, tokenizer = self._get_model()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using Contriever embeddings."""
        if not sentences:
            return np.array([])

        # Encode query and sentences
        query_emb = self._encode([query])[0]
        sent_embs = self._encode(sentences)

        # Cosine similarity (embeddings are normalized)
        scores = sent_embs @ query_emb

        return scores


class BGEBaseline(BaselineRetriever):
    """BGE (BAAI General Embedding) bi-encoder baseline.

    Uses BAAI/bge-base-en-v1.5 for dense retrieval.
    Reference: https://huggingface.co/BAAI/bge-base-en-v1.5
    """

    name = "bge"
    requires_gpu = True

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize BGE baseline.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu)
            cache_dir: Directory to cache embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self._model = None

    def _get_model(self):
        """Lazy load model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading BGE model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts with BGE-specific prefix.

        BGE models require instruction prefix for queries.
        """
        model = self._get_model()

        # Add BGE-specific prefix for queries
        if is_query:
            prefixed = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        else:
            prefixed = texts

        embeddings = model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using BGE embeddings."""
        if not sentences:
            return np.array([])

        # Encode query and sentences
        query_emb = self._encode([query], is_query=True)[0]
        sent_embs = self._encode(sentences, is_query=False)

        # Cosine similarity (embeddings are normalized)
        scores = sent_embs @ query_emb

        return scores


# =============================================================================
# CROSS-ENCODER BASELINES
# =============================================================================


class CrossEncoderBaseline(BaselineRetriever):
    """Cross-encoder reranking baseline.

    Uses a cross-encoder model to jointly encode query-sentence pairs.
    More accurate than bi-encoders but slower.
    Reference: https://www.sbert.net/docs/cross_encoder/pretrained_models.html
    """

    name = "cross-encoder"
    requires_gpu = True

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize cross-encoder baseline.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device to use (cuda/cpu)
            batch_size: Batch size for inference
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        import torch

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazy load model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using cross-encoder."""
        if not sentences:
            return np.array([])

        model = self._get_model()

        # Create query-sentence pairs
        pairs = [[query, sent] for sent in sentences]

        # Score pairs
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        return np.array(scores)


# =============================================================================
# LINEAR MODEL BASELINES
# =============================================================================


class LinearModelBaseline(BaselineRetriever):
    """Linear model (LogReg/SVM) on TF-IDF features baseline.

    Trains a classifier on TF-IDF features to score query-sentence relevance.
    Uses the concatenation of query and sentence TF-IDF vectors.
    """

    name = "linear"
    requires_gpu = False

    def __init__(
        self,
        model_type: str = "logistic",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        """Initialize linear model baseline.

        Args:
            model_type: 'logistic' for LogisticRegression or 'svm' for LinearSVC
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import LinearSVC
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = None
        self._cosine_similarity = cosine_similarity

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using TF-IDF cosine similarity with query expansion.

        For baseline purposes, we use TF-IDF cosine similarity with
        query-sentence interaction features (element-wise product).
        """
        if not sentences:
            return np.array([])

        from sklearn.feature_extraction.text import TfidfVectorizer

        # Fit vectorizer on corpus + query
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        all_texts = sentences + [query]
        vectors = vectorizer.fit_transform(all_texts)

        # Compute cosine similarity
        query_vector = vectors[-1]
        doc_vectors = vectors[:-1]
        base_scores = self._cosine_similarity(query_vector, doc_vectors)[0]

        # Add interaction features: element-wise product magnitude
        interaction_scores = np.array([
            np.sum(query_vector.toarray() * doc_vectors[i].toarray())
            for i in range(len(sentences))
        ])

        # Combine scores (weighted average)
        scores = 0.7 * base_scores + 0.3 * interaction_scores

        return scores


# =============================================================================
# LLM BASELINES
# =============================================================================


class NoRetrievalLLMBaseline(BaselineRetriever):
    """No-retrieval LLM baseline.

    Uses an LLM to directly score sentence-criterion relevance without
    any retrieval mechanism. This tests whether an LLM can identify
    relevant sentences purely from its parametric knowledge.

    NOTE: This is a lightweight implementation using sentence similarity
    with an instruction-tuned model, not a full LLM API call per sentence.
    For full LLM evaluation, use the separate LLM evaluation scripts.
    """

    name = "llm-embed"
    requires_gpu = True

    def __init__(
        self,
        model_name: str = "intfloat/e5-mistral-7b-instruct",
        device: Optional[str] = None,
        use_lightweight: bool = True,
    ):
        """Initialize LLM embedding baseline.

        Args:
            model_name: Instruction-tuned embedding model
            device: Device to use (cuda/cpu)
            use_lightweight: Use a smaller model for faster inference
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_lightweight = use_lightweight

        # Use a lighter model if requested (for faster evaluation)
        if use_lightweight:
            self.model_name = "intfloat/e5-large-v2"
        else:
            self.model_name = model_name

        self._model = None

    def _get_model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("sentence-transformers required: pip install sentence-transformers")

            logger.info(f"Loading LLM embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts with instruction prefix."""
        model = self._get_model()

        # Use instruction prefix for better alignment
        if is_query:
            prefixed = [f"query: {t}" for t in texts]
        else:
            prefixed = [f"passage: {t}" for t in texts]

        embeddings = model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings

    def score(self, query: str, sentences: List[str]) -> np.ndarray:
        """Score sentences using LLM embeddings."""
        if not sentences:
            return np.array([])

        # Encode query and sentences
        query_emb = self._encode([query], is_query=True)[0]
        sent_embs = self._encode(sentences, is_query=False)

        # Cosine similarity (embeddings are normalized)
        scores = sent_embs @ query_emb

        return scores


# =============================================================================
# FACTORY
# =============================================================================


BASELINE_REGISTRY: Dict[str, type] = {
    # Lexical baselines
    "bm25": BM25Baseline,
    "tfidf": TfidfBaseline,
    "random": RandomBaseline,
    # Dense bi-encoder baselines
    "e5-base": E5Baseline,
    "e5": E5Baseline,  # Alias
    "contriever": ContrieverBaseline,
    "bge": BGEBaseline,
    "bge-base": BGEBaseline,  # Alias
    # Cross-encoder baseline
    "cross-encoder": CrossEncoderBaseline,
    "crossencoder": CrossEncoderBaseline,  # Alias
    # Linear model baseline
    "linear": LinearModelBaseline,
    "logistic": LinearModelBaseline,  # Alias
    # LLM embedding baseline
    "llm-embed": NoRetrievalLLMBaseline,
    "llm": NoRetrievalLLMBaseline,  # Alias
}


def get_baseline(name: str, **kwargs) -> BaselineRetriever:
    """Get a baseline retriever by name.

    Args:
        name: Baseline name (bm25, tfidf, random, e5-base, contriever)
        kwargs: Arguments to pass to baseline constructor

    Returns:
        Initialized baseline retriever
    """
    name_lower = name.lower()
    if name_lower not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")

    return BASELINE_REGISTRY[name_lower](**kwargs)


def list_baselines() -> List[str]:
    """List available baseline names."""
    return sorted(set(BASELINE_REGISTRY.keys()))
