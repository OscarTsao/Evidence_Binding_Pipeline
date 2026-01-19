"""Clinical Dynamic-K selection with per-state policies.

Implements state-specific K selection:
- NEG: K = 0 (no evidence extraction)
- UNCERTAIN: Conservative (higher K, higher gamma)
- POS: Standard extraction

All K selection is DYNAMIC (adapts to candidate count N).
"""

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from final_sc_review.clinical.config import ClinicalConfig, DynamicKStateConfig
from final_sc_review.clinical.three_state_gate import GateDecision
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class ClinicalDynamicK:
    """Dynamic-K selector with per-state policies.

    CRITICAL: K must be dynamic (adapt to N). Fixed K is prohibited.
    """

    def __init__(self, config: ClinicalConfig):
        self.config = config

    def select_k_for_state(
        self,
        state: str,
        scores: np.ndarray,
        n_candidates: int
    ) -> int:
        """Select K for a single query based on state.

        Args:
            state: "NEG", "UNCERTAIN", or "POS"
            scores: Reranker scores for candidates [n_candidates]
            n_candidates: Total number of candidates

        Returns:
            Selected K value
        """
        if state == GateDecision.NEG.value:
            return 0  # No evidence extraction

        # Get state config
        state_config = self.config.get_state_config(state)

        # Apply mass policy
        k = self._mass_policy(scores, state_config, n_candidates)

        return k

    def _mass_policy(
        self,
        scores: np.ndarray,
        state_config: DynamicKStateConfig,
        n_candidates: int
    ) -> int:
        """Mass-based K selection.

        Select smallest K such that cumsum(prob) >= gamma, clamped to [k_min, k_max].
        """
        # Convert scores to probabilities via softmax
        probs = self._softmax(scores, temperature=state_config.temperature)

        # Sort probabilities descending
        sorted_probs = np.sort(probs)[::-1]

        # Find smallest K such that cumsum >= gamma
        cumsum = np.cumsum(sorted_probs)
        k_mass = np.searchsorted(cumsum, state_config.gamma) + 1

        # Compute k_max
        k_max = state_config.compute_k_max(n_candidates)

        # Clamp
        k = np.clip(k_mass, state_config.k_min, k_max)

        return int(k)

    def _softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax probabilities with temperature."""
        scores_temp = scores / temperature
        exp_scores = np.exp(scores_temp - np.max(scores_temp))  # Numerical stability
        return exp_scores / exp_scores.sum()

    def select_k_batch(
        self,
        states: np.ndarray,
        scores_list: List[np.ndarray],
        n_candidates_list: List[int]
    ) -> np.ndarray:
        """Select K for a batch of queries.

        Args:
            states: State decisions [batch_size]
            scores_list: List of score arrays
            n_candidates_list: List of candidate counts

        Returns:
            Selected K values [batch_size]
        """
        batch_size = len(states)
        k_values = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            k_values[i] = self.select_k_for_state(
                states[i],
                scores_list[i],
                n_candidates_list[i]
            )

        return k_values

    def get_sanity_check_stats(
        self,
        states: np.ndarray,
        k_values: np.ndarray,
        n_candidates_list: List[int]
    ) -> dict:
        """Compute sanity check statistics for Dynamic-K.

        CRITICAL: This must show that:
        1. gamma changes actually affect K distribution
        2. K adapts to N (candidate count)
        3. State-specific policies are applied correctly
        """
        stats = {}

        # Overall statistics
        stats["mean_k"] = float(np.mean(k_values))
        stats["std_k"] = float(np.std(k_values))
        stats["median_k"] = float(np.median(k_values))
        stats["mean_n"] = float(np.mean(n_candidates_list))

        # Per-state statistics
        for state in [GateDecision.NEG.value, GateDecision.UNCERTAIN.value, GateDecision.POS.value]:
            mask = states == state
            if mask.sum() > 0:
                state_k = k_values[mask]
                state_n = np.array(n_candidates_list)[mask]

                stats[f"{state}_count"] = int(mask.sum())
                stats[f"{state}_mean_k"] = float(np.mean(state_k))
                stats[f"{state}_std_k"] = float(np.std(state_k))
                stats[f"{state}_mean_n"] = float(np.mean(state_n))

                # K distribution
                stats[f"{state}_k_min"] = int(np.min(state_k))
                stats[f"{state}_k_max"] = int(np.max(state_k))
                stats[f"{state}_k_percentiles"] = {
                    "25": int(np.percentile(state_k, 25)),
                    "50": int(np.percentile(state_k, 50)),
                    "75": int(np.percentile(state_k, 75)),
                }

                # Correlation with N
                if len(state_k) > 1:
                    correlation = np.corrcoef(state_k, state_n)[0, 1]
                    stats[f"{state}_k_n_correlation"] = float(correlation)

        return stats
