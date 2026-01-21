"""Hybrid pipeline combining P4 GNN with LLM modules for production deployment.

This module implements Phase 4: Production Integration
- P4 GNN for fast baseline predictions (80% of queries)
- LLM modules for hard cases (20% of queries):
  - A.9 (Suicidal Ideation) → SuicidalIdeationClassifier
  - UNCERTAIN cases (P ∈ [0.4, 0.6]) → LLM Verifier
  - Optional: Top-10 → LLM Reranker

Note on criterion naming:
- A.9 = Suicidal Ideation (DSM-5)
- A.10 = SPECIAL_CASE (expert discrimination cases, per ReDSM5 taxonomy)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import LLMBase
from .reranker import LLMReranker
from .verifier import LLMVerifier
from .suicidal_ideation_classifier import SuicidalIdeationClassifier

logger = logging.getLogger(__name__)


class HybridPipeline:
    """Hybrid pipeline with P4 GNN + LLM modules."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        cache_dir: Optional[Path] = None,
        enable_llm_reranker: bool = False,
        enable_llm_verifier: bool = True,
        enable_suicidal_ideation_classifier: bool = True,
        uncertain_threshold_low: float = 0.4,
        uncertain_threshold_high: float = 0.6,
    ):
        """Initialize hybrid pipeline.

        Args:
            model_name: LLM model name
            load_in_4bit: Use 4-bit quantization
            cache_dir: Cache directory
            enable_llm_reranker: Enable LLM reranker (optional)
            enable_llm_verifier: Enable LLM verifier for UNCERTAIN cases
            enable_suicidal_ideation_classifier: Enable A.9 suicidal ideation classifier
            uncertain_threshold_low: Lower bound for UNCERTAIN (default: 0.4)
            uncertain_threshold_high: Upper bound for UNCERTAIN (default: 0.6)
        """
        self.enable_llm_reranker = enable_llm_reranker
        self.enable_llm_verifier = enable_llm_verifier
        self.enable_suicidal_ideation_classifier = enable_suicidal_ideation_classifier
        self.uncertain_threshold_low = uncertain_threshold_low
        self.uncertain_threshold_high = uncertain_threshold_high

        model_kwargs = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "load_in_4bit": load_in_4bit,
        }

        # Initialize LLM modules (lazy loading)
        self._llm_reranker = None
        self._llm_verifier = None
        self._suicidal_ideation_classifier = None
        self._model_kwargs = model_kwargs

        logger.info(f"Initialized HybridPipeline:")
        logger.info(f"  LLM model: {model_name}")
        logger.info(f"  LLM reranker: {enable_llm_reranker}")
        logger.info(f"  LLM verifier: {enable_llm_verifier}")
        logger.info(f"  A.9 Suicidal Ideation classifier: {enable_suicidal_ideation_classifier}")

    @property
    def llm_reranker(self) -> LLMReranker:
        """Get or create LLM reranker."""
        if self._llm_reranker is None:
            logger.info("Loading LLM Reranker...")
            self._llm_reranker = LLMReranker(**self._model_kwargs)
        return self._llm_reranker

    @property
    def llm_verifier(self) -> LLMVerifier:
        """Get or create LLM verifier."""
        if self._llm_verifier is None:
            logger.info("Loading LLM Verifier...")
            self._llm_verifier = LLMVerifier(**self._model_kwargs)
        return self._llm_verifier

    @property
    def suicidal_ideation_classifier(self) -> SuicidalIdeationClassifier:
        """Get or create A.9 Suicidal Ideation classifier."""
        if self._suicidal_ideation_classifier is None:
            logger.info("Loading A.9 Suicidal Ideation Classifier...")
            self._suicidal_ideation_classifier = SuicidalIdeationClassifier(**self._model_kwargs)
        return self._suicidal_ideation_classifier

    # Backward compatibility alias
    @property
    def a10_classifier(self) -> SuicidalIdeationClassifier:
        """Deprecated alias for suicidal_ideation_classifier."""
        logger.warning("a10_classifier is deprecated, use suicidal_ideation_classifier instead")
        return self.suicidal_ideation_classifier

    def predict(
        self,
        post_text: str,
        criterion_id: str,
        criterion_text: str,
        p4_prob: float,
        candidates: List[Dict],
        state: str = None,
    ) -> Dict:
        """Make prediction using hybrid pipeline.

        Args:
            post_text: Full post text
            criterion_id: Criterion ID (e.g., "A.1")
            criterion_text: Criterion description
            p4_prob: P4 GNN probability (calibrated)
            candidates: List of candidate sentences
            state: 3-state gate output (NEG/UNCERTAIN/POS) - if None, inferred from p4_prob

        Returns:
            Dict with:
                - final_prob: Final probability (may be updated by LLM)
                - final_state: Final state
                - used_llm: Whether LLM was used
                - llm_module: Which LLM module was used (if any)
                - llm_metadata: Metadata from LLM module
                - reranked_candidates: Reranked candidates (if reranker used)
        """
        # Infer state if not provided
        if state is None:
            if p4_prob >= self.uncertain_threshold_high:
                state = "POS"
            elif p4_prob <= self.uncertain_threshold_low:
                state = "NEG"
            else:
                state = "UNCERTAIN"

        result = {
            "final_prob": p4_prob,
            "final_state": state,
            "used_llm": False,
            "llm_module": None,
            "llm_metadata": {},
            "reranked_candidates": candidates,
        }

        # Check if A.9 (Suicidal Ideation) - requires special handling
        # Note: A.10 is SPECIAL_CASE (expert discrimination), NOT suicidal ideation
        is_suicidal_ideation = criterion_id == "A.9"

        # Route to appropriate LLM module
        if is_suicidal_ideation and self.enable_suicidal_ideation_classifier:
            # Use A.9 Suicidal Ideation classifier
            logger.debug(f"Routing to A.9 Suicidal Ideation Classifier (criterion={criterion_id})")

            classification = self.suicidal_ideation_classifier.classify(
                post_text=post_text,
                self_consistency_runs=3
            )

            # Update probability based on LLM classification
            # Conservative: if LLM detects SI, set prob to 0.9
            if classification["has_suicidal_ideation"]:
                result["final_prob"] = max(p4_prob, 0.9)  # Boost but don't lower
                result["final_state"] = "POS" if result["final_prob"] >= 0.7 else "UNCERTAIN"

            result["used_llm"] = True
            result["llm_module"] = "suicidal_ideation_classifier"
            result["llm_metadata"] = classification

        elif state == "UNCERTAIN" and self.enable_llm_verifier:
            # Use LLM verifier for UNCERTAIN cases
            logger.debug(f"Routing to LLM Verifier (prob={p4_prob:.3f}, state={state})")

            # Get top-K evidence sentences
            evidence_sentences = [c["sentence"] for c in candidates[:5]]

            verification = self.llm_verifier.verify(
                post_text=post_text,
                criterion_text=criterion_text,
                evidence_sentences=evidence_sentences,
                self_consistency_runs=3
            )

            # Update probability based on LLM verification
            if verification["has_evidence"]:
                # LLM says has evidence, boost probability
                boost = verification["confidence"] * 0.3  # Max boost of 0.3
                result["final_prob"] = min(p4_prob + boost, 0.95)
            else:
                # LLM says no evidence, lower probability
                penalty = verification["confidence"] * 0.2  # Max penalty of 0.2
                result["final_prob"] = max(p4_prob - penalty, 0.05)

            # Update state
            if result["final_prob"] >= self.uncertain_threshold_high:
                result["final_state"] = "POS"
            elif result["final_prob"] <= self.uncertain_threshold_low:
                result["final_state"] = "NEG"
            else:
                result["final_state"] = "UNCERTAIN"

            result["used_llm"] = True
            result["llm_module"] = "verifier"
            result["llm_metadata"] = verification

        # Optional: LLM reranker
        if self.enable_llm_reranker and len(candidates) >= 5:
            logger.debug(f"Applying LLM Reranker")

            reranked, metadata = self.llm_reranker.rerank(
                post_text=post_text,
                criterion_text=criterion_text,
                candidates=candidates,
                top_k=5,
                check_position_bias=False  # Skip bias check in production
            )

            result["reranked_candidates"] = reranked

            if not result["used_llm"]:
                result["used_llm"] = True
                result["llm_module"] = "reranker"
                result["llm_metadata"] = metadata
            else:
                # Already used another LLM module, append reranker metadata
                result["llm_metadata"]["reranker"] = metadata

        return result

    def get_usage_stats(self) -> Dict:
        """Get LLM usage statistics.

        Returns:
            Dict with LLM usage counts and percentages
        """
        # This would track usage in production
        # For now, return expected distribution
        return {
            "total_queries": 14770,
            "llm_used": 3694,  # 25% of queries
            "llm_usage_rate": 0.25,
            "breakdown": {
                "suicidal_ideation_classifier": 1477,  # ~10% (A.9 queries)
                "verifier": 2201,  # 14.9% (UNCERTAIN cases)
                "reranker": 0 if not self.enable_llm_reranker else 14770,  # All if enabled
            }
        }

    def get_expected_latency(self) -> Dict:
        """Get expected latency statistics.

        Returns:
            Dict with latency estimates
        """
        p4_latency = 5  # ms
        llm_latency = 300  # ms
        llm_usage_rate = 0.25  # 25% use LLM

        avg_latency = p4_latency + (llm_usage_rate * llm_latency)

        return {
            "p4_gnn_latency_ms": p4_latency,
            "llm_latency_ms": llm_latency,
            "llm_usage_rate": llm_usage_rate,
            "average_latency_ms": avg_latency,
            "p50_latency_ms": p4_latency,  # 75% don't use LLM
            "p90_latency_ms": llm_latency,  # 25% use LLM
            "p99_latency_ms": llm_latency,
        }

    def get_expected_cost(self, queries_per_day: int = 10000) -> Dict:
        """Get expected cost statistics.

        Args:
            queries_per_day: Expected query volume

        Returns:
            Dict with cost estimates
        """
        llm_usage_rate = 0.25
        llm_queries_per_day = queries_per_day * llm_usage_rate

        # Local model (electricity cost only)
        gpu_power_watts = 450  # RTX 5090 TDP
        electricity_cost_per_kwh = 0.15  # USD
        hours_per_day = llm_queries_per_day / (20 * 3600)  # 20 queries/sec
        daily_kwh = (gpu_power_watts / 1000) * hours_per_day
        daily_electricity_cost = daily_kwh * electricity_cost_per_kwh

        return {
            "model_type": "local (Qwen2.5-7B)",
            "queries_per_day": queries_per_day,
            "llm_queries_per_day": llm_queries_per_day,
            "llm_usage_rate": llm_usage_rate,
            "daily_electricity_cost_usd": daily_electricity_cost,
            "monthly_cost_usd": daily_electricity_cost * 30,
            "yearly_cost_usd": daily_electricity_cost * 365,
            "cost_per_1000_queries_usd": (daily_electricity_cost / queries_per_day) * 1000,
        }
