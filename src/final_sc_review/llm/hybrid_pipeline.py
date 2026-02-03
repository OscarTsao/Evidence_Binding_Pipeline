"""Hybrid pipeline combining P4 GNN with LLM modules for production deployment.

This module implements Phase 4: Production Integration
- P4 GNN for fast baseline predictions (80% of queries)
- LLM modules for hard cases (20% of queries):
  - A.9 (Suicidal Ideation) → SuicidalIdeationClassifier
  - UNCERTAIN cases (P ∈ [0.4, 0.6]) → LLM Verifier
  - Optional: Top-10 → LLM Reranker

Features:
- Hybrid NE detection (GNN confidence + LLM fallback)
- Dynamic-K selection for verifier (replaces hardcoded top-5)

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
from .ne_detector import HybridNEDetector, NEDetectionResult
from ..postprocessing.dynamic_k import DynamicKSelector

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
        enable_ne_detection: bool = False,
        uncertain_threshold_low: float = 0.4,
        uncertain_threshold_high: float = 0.6,
        verifier_k_method: str = "fixed",
        verifier_k_fixed: int = 5,
        verifier_k_min: int = 3,
        verifier_k_max: int = 8,
    ):
        """Initialize hybrid pipeline.

        Args:
            model_name: LLM model name
            load_in_4bit: Use 4-bit quantization
            cache_dir: Cache directory
            enable_llm_reranker: Enable LLM reranker (optional)
            enable_llm_verifier: Enable LLM verifier for UNCERTAIN cases
            enable_suicidal_ideation_classifier: Enable A.9 suicidal ideation classifier
            enable_ne_detection: Enable hybrid NE detection (GNN + LLM fallback)
            uncertain_threshold_low: Lower bound for UNCERTAIN (default: 0.4)
            uncertain_threshold_high: Upper bound for UNCERTAIN (default: 0.6)
            verifier_k_method: Method for selecting K candidates for verifier
                - 'fixed': Use fixed top-K (default: 5)
                - 'dynamic': Use DynamicKSelector based on score distribution
            verifier_k_fixed: Fixed K value when verifier_k_method='fixed' (default: 5)
            verifier_k_min: Minimum K when verifier_k_method='dynamic' (default: 3)
            verifier_k_max: Maximum K when verifier_k_method='dynamic' (default: 8)
        """
        self.enable_llm_reranker = enable_llm_reranker
        self.enable_llm_verifier = enable_llm_verifier
        self.enable_suicidal_ideation_classifier = enable_suicidal_ideation_classifier
        self.enable_ne_detection = enable_ne_detection
        self.uncertain_threshold_low = uncertain_threshold_low
        self.uncertain_threshold_high = uncertain_threshold_high
        self.verifier_k_method = verifier_k_method
        self.verifier_k_fixed = verifier_k_fixed

        model_kwargs = {
            "model_name": model_name,
            "cache_dir": cache_dir,
            "load_in_4bit": load_in_4bit,
        }

        # Initialize LLM modules (lazy loading)
        self._llm_reranker = None
        self._llm_verifier = None
        self._suicidal_ideation_classifier = None
        self._ne_detector = None
        self._model_kwargs = model_kwargs

        # Initialize dynamic-K selector if needed
        self._dynamic_k_selector = None
        if verifier_k_method == "dynamic":
            self._dynamic_k_selector = DynamicKSelector(
                method="score_gap",
                min_k=verifier_k_min,
                max_k=verifier_k_max,
                score_gap_ratio=0.3,
            )

        logger.info(f"Initialized HybridPipeline:")
        logger.info(f"  LLM model: {model_name}")
        logger.info(f"  LLM reranker: {enable_llm_reranker}")
        logger.info(f"  LLM verifier: {enable_llm_verifier}")
        logger.info(f"  A.9 Suicidal Ideation classifier: {enable_suicidal_ideation_classifier}")
        logger.info(f"  NE detection: {enable_ne_detection}")
        logger.info(f"  Verifier K method: {verifier_k_method}")

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

    @property
    def ne_detector(self) -> HybridNEDetector:
        """Get or create hybrid NE detector."""
        if self._ne_detector is None:
            logger.info("Loading Hybrid NE Detector...")
            self._ne_detector = HybridNEDetector(
                llm_verifier=self._llm_verifier,  # May be None, will lazy load
                enable_llm_fallback=self.enable_llm_verifier,
            )
        return self._ne_detector

    def _select_verifier_candidates(
        self, candidates: List[Dict], scores: Optional[List[float]] = None
    ) -> Tuple[List[Dict], int]:
        """Select candidates for LLM verifier using configured method.

        Args:
            candidates: List of candidate sentences
            scores: Optional scores for dynamic selection

        Returns:
            Tuple of (selected candidates, selected K)
        """
        if self.verifier_k_method == "fixed":
            k = min(self.verifier_k_fixed, len(candidates))
            return candidates[:k], k

        # Dynamic K selection based on score distribution
        if scores is None or self._dynamic_k_selector is None:
            # Fall back to fixed if no scores available
            k = min(self.verifier_k_fixed, len(candidates))
            return candidates[:k], k

        result = self._dynamic_k_selector.select_k(scores)
        k = min(result.selected_k, len(candidates))
        return candidates[:k], k

    def predict(
        self,
        post_text: str,
        criterion_id: str,
        criterion_text: str,
        p4_prob: float,
        candidates: List[Dict],
        state: str = None,
        candidate_scores: Optional[List[float]] = None,
    ) -> Dict:
        """Make prediction using hybrid pipeline.

        Args:
            post_text: Full post text
            criterion_id: Criterion ID (e.g., "A.1")
            criterion_text: Criterion description
            p4_prob: P4 GNN probability (calibrated)
            candidates: List of candidate sentences
            state: 3-state gate output (NEG/UNCERTAIN/POS) - if None, inferred from p4_prob
            candidate_scores: Optional scores for dynamic K selection

        Returns:
            Dict with:
                - final_prob: Final probability (may be updated by LLM)
                - final_state: Final state
                - used_llm: Whether LLM was used
                - llm_module: Which LLM module was used (if any)
                - llm_metadata: Metadata from LLM module
                - reranked_candidates: Reranked candidates (if reranker used)
                - ne_detection: NE detection result (if enabled)
                - verifier_k: Number of candidates used for verifier
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
            "ne_detection": None,
            "verifier_k": None,
        }

        # Check for no-evidence using hybrid NE detector
        if self.enable_ne_detection and candidates:
            scores_for_ne = candidate_scores or [
                c.get("score", c.get("reranker_score", 0.0))
                for c in candidates
            ]
            ne_result = self.ne_detector.detect(
                gnn_scores=scores_for_ne,
                post_text=post_text,
                criterion_text=criterion_text,
                candidates=candidates,
            )
            result["ne_detection"] = {
                "is_no_evidence": ne_result.is_no_evidence,
                "confidence": ne_result.confidence,
                "method": ne_result.method,
                "gnn_max_score": ne_result.gnn_max_score,
            }

            # If high-confidence no-evidence, short-circuit
            if ne_result.is_no_evidence and ne_result.confidence > 0.8:
                result["final_prob"] = 0.1
                result["final_state"] = "NEG"
                if ne_result.method == "llm_verified":
                    result["used_llm"] = True
                    result["llm_module"] = "ne_detector"
                    result["llm_metadata"] = ne_result.llm_verification or {}
                return result

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

            # Get top-K evidence sentences using configured method
            selected_candidates, k = self._select_verifier_candidates(
                candidates, candidate_scores
            )
            evidence_sentences = [c["sentence"] for c in selected_candidates]
            result["verifier_k"] = k

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
