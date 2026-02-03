"""Hybrid No-Evidence (NE) Detector.

Combines GNN confidence with LLM fallback for detecting no-evidence cases.
Since P1 NE Gate had poor performance (AUROC=0.577), this hybrid approach
uses GNN score confidence for fast decisions and LLM verification for
uncertain cases.

Strategy:
- High GNN score (>0.5): Confident has evidence -> skip LLM
- Very low GNN score (<0.1): Confident no evidence -> skip LLM
- Uncertain (0.1-0.5): Use LLM verifier to confirm

This reduces LLM usage while maintaining accuracy on edge cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .verifier import LLMVerifier

logger = logging.getLogger(__name__)


@dataclass
class NEDetectionResult:
    """Result of no-evidence detection."""

    is_no_evidence: bool
    confidence: float
    method: str  # 'gnn_confident', 'llm_verified', 'threshold'
    gnn_max_score: float
    llm_verification: Optional[Dict] = None


class HybridNEDetector:
    """Hybrid No-Evidence detector using GNN confidence + LLM fallback.

    This detector addresses the limitation of P1 NE Gate (AUROC=0.577) by:
    1. Using GNN scores as a fast confidence indicator
    2. Only calling LLM for uncertain cases (10-50% of queries)
    3. Providing interpretable confidence scores

    Example:
        detector = HybridNEDetector(llm_verifier)
        result = detector.detect(
            gnn_scores=[0.8, 0.3, 0.1],
            post_text="...",
            criterion_text="...",
            candidates=[...]
        )
        if result.is_no_evidence:
            return "No evidence found"
    """

    def __init__(
        self,
        llm_verifier: Optional["LLMVerifier"] = None,
        high_confidence_threshold: float = 0.5,
        low_confidence_threshold: float = 0.1,
        enable_llm_fallback: bool = True,
    ):
        """Initialize hybrid NE detector.

        Args:
            llm_verifier: LLM verifier for uncertain cases (lazy loaded if None)
            high_confidence_threshold: Score above which we're confident has evidence
            low_confidence_threshold: Score below which we're confident no evidence
            enable_llm_fallback: Whether to use LLM for uncertain cases
        """
        self._llm_verifier = llm_verifier
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.enable_llm_fallback = enable_llm_fallback

        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "gnn_confident_positive": 0,
            "gnn_confident_negative": 0,
            "llm_verified": 0,
        }

    @property
    def llm_verifier(self) -> "LLMVerifier":
        """Get or create LLM verifier (lazy loading)."""
        if self._llm_verifier is None:
            from .verifier import LLMVerifier
            logger.info("Lazy loading LLM verifier for NE detection...")
            self._llm_verifier = LLMVerifier()
        return self._llm_verifier

    def detect(
        self,
        gnn_scores: List[float],
        post_text: str,
        criterion_text: str,
        candidates: List[Dict],
        top_k_for_verification: int = 5,
    ) -> NEDetectionResult:
        """Detect if there's no evidence for the criterion.

        Args:
            gnn_scores: GNN-refined scores for each candidate (highest first)
            post_text: Full post text
            criterion_text: Criterion description
            candidates: List of candidate sentences with 'sentence' key
            top_k_for_verification: Number of top candidates to send to LLM

        Returns:
            NEDetectionResult with detection outcome
        """
        self._stats["total_queries"] += 1

        if not gnn_scores:
            return NEDetectionResult(
                is_no_evidence=True,
                confidence=1.0,
                method="empty_candidates",
                gnn_max_score=0.0,
            )

        max_score = max(gnn_scores)

        # Fast path: High confidence -> has evidence
        if max_score > self.high_confidence_threshold:
            self._stats["gnn_confident_positive"] += 1
            return NEDetectionResult(
                is_no_evidence=False,
                confidence=min(max_score, 0.99),  # Cap at 0.99
                method="gnn_confident",
                gnn_max_score=max_score,
            )

        # Fast path: Very low confidence -> likely no evidence
        if max_score < self.low_confidence_threshold:
            self._stats["gnn_confident_negative"] += 1
            return NEDetectionResult(
                is_no_evidence=True,
                confidence=1.0 - max_score,  # Higher confidence when score is lower
                method="gnn_confident",
                gnn_max_score=max_score,
            )

        # Uncertain: Use LLM if enabled
        if self.enable_llm_fallback:
            self._stats["llm_verified"] += 1
            return self._verify_with_llm(
                max_score=max_score,
                post_text=post_text,
                criterion_text=criterion_text,
                candidates=candidates,
                top_k=top_k_for_verification,
            )

        # LLM disabled: Use threshold-based decision
        is_no_evidence = max_score < (
            self.high_confidence_threshold + self.low_confidence_threshold
        ) / 2
        return NEDetectionResult(
            is_no_evidence=is_no_evidence,
            confidence=0.5 + abs(max_score - 0.3) * 0.5,  # Lower confidence for uncertain
            method="threshold",
            gnn_max_score=max_score,
        )

    def _verify_with_llm(
        self,
        max_score: float,
        post_text: str,
        criterion_text: str,
        candidates: List[Dict],
        top_k: int,
    ) -> NEDetectionResult:
        """Use LLM to verify evidence presence for uncertain cases.

        Args:
            max_score: Maximum GNN score
            post_text: Full post text
            criterion_text: Criterion description
            candidates: List of candidate sentences
            top_k: Number of candidates to verify

        Returns:
            NEDetectionResult with LLM verification
        """
        # Extract top-k sentences for verification
        evidence_sentences = [
            c.get("sentence", c.get("text", ""))
            for c in candidates[:top_k]
            if c.get("sentence") or c.get("text")
        ]

        if not evidence_sentences:
            return NEDetectionResult(
                is_no_evidence=True,
                confidence=0.9,
                method="no_valid_candidates",
                gnn_max_score=max_score,
            )

        try:
            verification = self.llm_verifier.verify(
                post_text=post_text,
                criterion_text=criterion_text,
                evidence_sentences=evidence_sentences,
                self_consistency_runs=3,
            )

            is_no_evidence = not verification.get("has_evidence", False)
            llm_confidence = verification.get("confidence", 0.5)

            # Combine GNN and LLM signals
            # If they agree, boost confidence; if they disagree, lower it
            gnn_suggests_no_evidence = max_score < 0.3
            if is_no_evidence == gnn_suggests_no_evidence:
                combined_confidence = min(llm_confidence + 0.1, 0.99)
            else:
                combined_confidence = max(llm_confidence - 0.1, 0.5)

            return NEDetectionResult(
                is_no_evidence=is_no_evidence,
                confidence=combined_confidence,
                method="llm_verified",
                gnn_max_score=max_score,
                llm_verification=verification,
            )

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}, falling back to threshold")
            # Fallback to threshold-based decision
            is_no_evidence = max_score < 0.3
            return NEDetectionResult(
                is_no_evidence=is_no_evidence,
                confidence=0.6,  # Lower confidence due to LLM failure
                method="llm_fallback_error",
                gnn_max_score=max_score,
            )

    def get_stats(self) -> Dict:
        """Get detection statistics.

        Returns:
            Dictionary with detection statistics
        """
        total = self._stats["total_queries"]
        if total == 0:
            return self._stats

        return {
            **self._stats,
            "gnn_confident_rate": (
                self._stats["gnn_confident_positive"] +
                self._stats["gnn_confident_negative"]
            ) / total,
            "llm_usage_rate": self._stats["llm_verified"] / total,
        }

    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self._stats = {
            "total_queries": 0,
            "gnn_confident_positive": 0,
            "gnn_confident_negative": 0,
            "llm_verified": 0,
        }
