"""Three-state NE gate for clinical deployment.

Implements NEG/UNCERTAIN/POS classification with:
1. Nested threshold selection (tune split only)
2. Probability calibration
3. Clinical validity (high sensitivity, workload management)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix

from final_sc_review.clinical.config import ThresholdConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


class GateDecision(Enum):
    """Three-state gate decision."""
    NEG = "NEG"  # High confidence NO_EVIDENCE
    UNCERTAIN = "UNCERTAIN"  # Low confidence / ambiguous
    POS = "POS"  # High confidence HAS_EVIDENCE


@dataclass
class ThresholdSelectionResult:
    """Result of nested threshold selection."""
    tau_neg: float
    tau_pos: float

    # Screening tier metrics (NOT NEG = flagged for evidence search)
    screening_sensitivity: float  # On tune split
    screening_fpr: float
    screening_npv: float
    screening_fn_per_1000: float

    # Alert tier metrics (POS = high confidence alert)
    alert_precision: float  # On tune split
    alert_fpr: float
    alert_recall: float

    # Distribution on tune split
    neg_rate: float
    uncertain_rate: float
    pos_rate: float

    def to_dict(self) -> Dict:
        return {
            "tau_neg": float(self.tau_neg),
            "tau_pos": float(self.tau_pos),
            "screening": {
                "sensitivity": float(self.screening_sensitivity),
                "fpr": float(self.screening_fpr),
                "npv": float(self.screening_npv),
                "fn_per_1000": float(self.screening_fn_per_1000),
            },
            "alert": {
                "precision": float(self.alert_precision),
                "fpr": float(self.alert_fpr),
                "recall": float(self.alert_recall),
            },
            "distribution": {
                "neg_rate": float(self.neg_rate),
                "uncertain_rate": float(self.uncertain_rate),
                "pos_rate": float(self.pos_rate),
            },
        }


class ThreeStateGate:
    """Three-state gate for clinical deployment.

    Workflow:
    1. Fit calibration on TUNE split (nested within each fold)
    2. Select tau_neg and tau_pos on TUNE split
    3. Apply thresholds to TEST split

    CRITICAL: All threshold selection must use TUNE split only (no test leakage).
    """

    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.calibrator = None
        self.tau_neg = None
        self.tau_pos = None

    def fit_calibration(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        method: str = "isotonic"
    ):
        """Fit probability calibration on TUNE split.

        Args:
            probs: Raw probabilities from P4 model [n_samples]
            labels: Binary labels (1 = has_evidence) [n_samples]
            method: "isotonic" or "platt"
        """
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probs, labels)
        elif method == "platt":
            # Platt scaling: logistic regression on logits
            from scipy.special import logit
            epsilon = 1e-7
            logits = logit(np.clip(probs, epsilon, 1 - epsilon))
            self.calibrator = LogisticRegression()
            self.calibrator.fit(logits.reshape(-1, 1), labels)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        logger.info(f"Fitted {method} calibration on {len(probs)} tune samples")

    def calibrate_probs(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities.

        Args:
            probs: Raw probabilities [n_samples]

        Returns:
            Calibrated probabilities [n_samples]
        """
        if self.calibrator is None:
            logger.warning("No calibration fitted, returning raw probabilities")
            return probs

        if isinstance(self.calibrator, IsotonicRegression):
            return self.calibrator.predict(probs)
        else:  # Platt
            from scipy.special import logit
            epsilon = 1e-7
            logits = logit(np.clip(probs, epsilon, 1 - epsilon))
            return self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]

    def select_thresholds(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> ThresholdSelectionResult:
        """Select tau_neg and tau_pos on TUNE split.

        CRITICAL: This function must ONLY be called on TUNE split.

        Args:
            probs: Calibrated probabilities [n_samples]
            labels: Binary labels (1 = has_evidence) [n_samples]

        Returns:
            Selected thresholds and metrics on tune split
        """
        n_samples = len(probs)
        n_positive = labels.sum()
        n_negative = n_samples - n_positive

        # Step 1: Select tau_neg (screening tier - minimize FN)
        tau_neg = self._select_tau_neg(probs, labels, n_samples, n_positive)

        # Step 2: Select tau_pos (alert tier - minimize FP)
        tau_pos = self._select_tau_pos(probs, labels, n_samples, n_negative, tau_neg)

        # Store thresholds
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos

        # Compute metrics on tune split for reporting
        decisions = self.predict(probs)
        metrics = self._compute_threshold_metrics(decisions, labels, n_samples, n_positive)

        return metrics

    def _select_tau_neg(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_samples: int,
        n_positive: int
    ) -> float:
        """Select tau_neg to achieve target sensitivity.

        Strategy:
        - Choose tau_neg such that Sensitivity(NOT NEG) >= sensitivity_target
        - Among valid thresholds, maximize NPV or minimize FPR for NEG decisions
        """
        sensitivity_target = self.config.sensitivity_target

        # Generate candidate thresholds
        candidates = np.percentile(probs, np.linspace(0, 50, 101))  # Focus on lower end

        best_tau = None
        best_npv = -1

        for tau_cand in candidates:
            # Classify: NEG if p <= tau_cand
            neg_mask = probs <= tau_cand
            flagged_mask = ~neg_mask  # POS or UNCERTAIN

            # Sensitivity = Recall of "flagged" on true positives
            if n_positive > 0:
                sensitivity = labels[flagged_mask].sum() / n_positive
            else:
                sensitivity = 1.0

            if sensitivity < sensitivity_target:
                continue  # Does not meet sensitivity target

            # NPV = TN / (TN + FN)
            tn = (labels == 0)[neg_mask].sum()
            fn = labels[neg_mask].sum()
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            if npv > best_npv:
                best_npv = npv
                best_tau = tau_cand

        if best_tau is None:
            # Fallback: use lowest probability (never classify as NEG)
            best_tau = probs.min() - 0.01
            logger.warning(f"Could not meet sensitivity target {sensitivity_target}, using fallback tau_neg={best_tau:.4f}")
        else:
            logger.info(f"Selected tau_neg={best_tau:.4f} with NPV={best_npv:.4f}")

        return best_tau

    def _select_tau_pos(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_samples: int,
        n_negative: int,
        tau_neg: float
    ) -> float:
        """Select tau_pos to balance precision and recall for POS decisions.

        Strategy (FIXED to prevent tau_pos=1.0):
        - Choose tau_pos such that:
          1. Recall(POS) >= min_recall_pos (prevents tau too high)
          2. Precision(POS) >= min_precision_pos (prevents tau too low)
          3. FPR(POS) <= max_fpr_pos (controls false positive rate)
        - Among valid thresholds, maximize F1 score
        """
        max_fpr_pos = self.config.max_fpr_pos
        min_precision_pos = self.config.min_precision_pos
        min_recall_pos = self.config.min_recall_pos
        n_positive = labels.sum()

        # Generate candidate thresholds using grid search (more stable)
        # Focus on 0.5 to 0.95 range (tau_pos=1.0 is too extreme)
        grid = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        # Also add percentile-based candidates for adaptivity
        percentile_candidates = np.percentile(probs, np.linspace(60, 95, 20))
        candidates = np.unique(np.concatenate([grid, percentile_candidates]))
        candidates = candidates[candidates > tau_neg]

        best_tau = None
        best_f1 = -1
        valid_count = 0

        for tau_cand in candidates:
            # Classify: POS if p >= tau_cand
            pos_mask = probs >= tau_cand
            n_pos = pos_mask.sum()

            if n_pos == 0:
                continue  # No predictions - skip

            # Compute metrics
            tp = labels[pos_mask].sum()
            fp = (labels == 0)[pos_mask].sum()
            fn = n_positive - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / n_positive if n_positive > 0 else 0
            fpr = fp / n_negative if n_negative > 0 else 0

            # Check constraints
            if recall < min_recall_pos:
                continue  # Does not meet recall target (tau too high)
            if precision < min_precision_pos:
                continue  # Does not meet precision target (tau too low)
            if fpr > max_fpr_pos:
                continue  # Exceeds FPR target

            # Valid candidate - compute F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            valid_count += 1

            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau_cand
                best_precision = precision
                best_recall = recall

        if best_tau is None:
            # Fallback strategy: relax constraints progressively
            logger.warning(f"No threshold meets all constraints (min_recall={min_recall_pos}, min_precision={min_precision_pos}, max_fpr={max_fpr_pos})")
            logger.warning(f"Relaxing constraints to find best trade-off...")

            # Try with relaxed precision (keep recall constraint strict)
            for tau_cand in candidates:
                pos_mask = probs >= tau_cand
                n_pos = pos_mask.sum()
                if n_pos == 0:
                    continue

                tp = labels[pos_mask].sum()
                fp = (labels == 0)[pos_mask].sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / n_positive if n_positive > 0 else 0
                fpr = fp / n_negative if n_negative > 0 else 0

                # Only enforce recall constraint (most critical)
                if recall < min_recall_pos:
                    continue

                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau_cand
                    best_precision = precision
                    best_recall = recall

        if best_tau is None:
            # Final fallback: use median of high-prob queries (conservative)
            high_prob_queries = probs[probs > tau_neg]
            if len(high_prob_queries) > 0:
                best_tau = np.percentile(high_prob_queries, 70)
            else:
                best_tau = tau_neg + 0.1
            logger.warning(f"Using fallback tau_pos={best_tau:.4f} (no constraints satisfied)")
        else:
            logger.info(f"Selected tau_pos={best_tau:.4f} with Precision={best_precision:.4f}, Recall={best_recall:.4f}, F1={best_f1:.4f} ({valid_count} valid candidates)")

        return best_tau

    def _compute_threshold_metrics(
        self,
        decisions: np.ndarray,
        labels: np.ndarray,
        n_samples: int,
        n_positive: int
    ) -> ThresholdSelectionResult:
        """Compute metrics for selected thresholds on tune split."""
        neg_mask = decisions == GateDecision.NEG.value
        unc_mask = decisions == GateDecision.UNCERTAIN.value
        pos_mask = decisions == GateDecision.POS.value

        # Screening tier (NOT NEG = flagged)
        flagged_mask = ~neg_mask
        screening_tp = labels[flagged_mask].sum()
        screening_fn = labels[neg_mask].sum()
        screening_tn = (labels == 0)[neg_mask].sum()
        screening_fp = (labels == 0)[flagged_mask].sum()

        screening_sensitivity = screening_tp / n_positive if n_positive > 0 else 0
        screening_fpr = screening_fp / (n_samples - n_positive) if (n_samples - n_positive) > 0 else 0
        screening_npv = screening_tn / (screening_tn + screening_fn) if (screening_tn + screening_fn) > 0 else 0
        screening_fn_per_1000 = (screening_fn / n_samples) * 1000 if n_samples > 0 else 0

        # Alert tier (POS only)
        alert_tp = labels[pos_mask].sum()
        alert_fp = (labels == 0)[pos_mask].sum()
        alert_fn = labels[~pos_mask].sum()

        alert_precision = alert_tp / (alert_tp + alert_fp) if (alert_tp + alert_fp) > 0 else 0
        alert_recall = alert_tp / n_positive if n_positive > 0 else 0
        alert_fpr = alert_fp / (n_samples - n_positive) if (n_samples - n_positive) > 0 else 0

        # Distribution
        neg_rate = neg_mask.sum() / n_samples
        unc_rate = unc_mask.sum() / n_samples
        pos_rate = pos_mask.sum() / n_samples

        return ThresholdSelectionResult(
            tau_neg=self.tau_neg,
            tau_pos=self.tau_pos,
            screening_sensitivity=screening_sensitivity,
            screening_fpr=screening_fpr,
            screening_npv=screening_npv,
            screening_fn_per_1000=screening_fn_per_1000,
            alert_precision=alert_precision,
            alert_fpr=alert_fpr,
            alert_recall=alert_recall,
            neg_rate=neg_rate,
            uncertain_rate=unc_rate,
            pos_rate=pos_rate,
        )

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Predict gate decisions.

        Args:
            probs: Calibrated probabilities [n_samples]

        Returns:
            Decisions [n_samples] - string values "NEG"/"UNCERTAIN"/"POS"
        """
        if self.tau_neg is None or self.tau_pos is None:
            raise ValueError("Thresholds not selected. Call select_thresholds() first.")

        decisions = np.empty(len(probs), dtype=object)
        decisions[probs <= self.tau_neg] = GateDecision.NEG.value
        decisions[probs >= self.tau_pos] = GateDecision.POS.value
        decisions[(probs > self.tau_neg) & (probs < self.tau_pos)] = GateDecision.UNCERTAIN.value

        return decisions

    def get_thresholds(self) -> Tuple[float, float]:
        """Get selected thresholds.

        Returns:
            (tau_neg, tau_pos)
        """
        if self.tau_neg is None or self.tau_pos is None:
            raise ValueError("Thresholds not selected")
        return self.tau_neg, self.tau_pos
