"""Clinical deployment configuration.

Defines all hyperparameters for clinical high-recall deployment mode with
explicit documentation of clinical intent and research validity requirements.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import yaml
from pathlib import Path


@dataclass
class ThresholdConfig:
    """Threshold selection configuration for 3-state gate.

    Clinical Intent:
    - tau_neg: High sensitivity threshold (minimize false negatives)
    - tau_pos: High specificity threshold (minimize false positives/alert fatigue)
    - UNCERTAIN state: Routes to conservative evidence extraction

    Research Gold Standard:
    - Must be selected using TUNE split only (nested CV)
    - Must NOT use test fold for threshold selection
    - Must use calibrated probabilities
    """
    # Screening tier (NEG threshold)
    sensitivity_target: float = 0.99  # Target sensitivity for NOT NEG
    min_npv: float = 0.95  # Minimum negative predictive value

    # Alert tier (POS threshold)
    max_fpr_pos: float = 0.05  # Maximum FPR for POS decisions
    min_precision_pos: float = 0.60  # Minimum precision for POS (relaxed from 0.80)
    min_recall_pos: float = 0.50  # Minimum recall for POS (NEW - prevents tau_pos=1.0)

    # Calibration
    calibration_method: str = "isotonic"  # "isotonic" or "platt"

    # Computed thresholds (filled during nested CV)
    tau_neg: Optional[float] = None
    tau_pos: Optional[float] = None


@dataclass
class DynamicKStateConfig:
    """Dynamic-K configuration for a specific state (NEG/UNCERTAIN/POS).

    Clinical Intent:
    - NEG: K=0 (skip evidence extraction)
    - UNCERTAIN: Conservative extraction (higher K, higher gamma)
    - POS: Standard extraction

    Research Gold Standard:
    - K must be dynamic (adapt to variable candidate size N)
    - Must NOT use fixed K
    - Must implement sanity checks
    """
    k_min: int
    k_max1: int  # Absolute maximum
    k_max_ratio: float  # Relative to candidate count N

    # Mass policy parameters (for POS/UNCERTAIN)
    gamma: Optional[float] = None  # Cumulative mass threshold
    temperature: float = 1.0  # Softmax temperature

    def compute_k_max(self, n_candidates: int) -> int:
        """Compute effective k_max for given candidate count."""
        import math
        k_max_relative = math.ceil(self.k_max_ratio * n_candidates)
        return min(self.k_max1, k_max_relative)


@dataclass
class ClinicalConfig:
    """Complete configuration for clinical high-recall deployment.

    This configuration ensures:
    1. No data leakage (post-ID disjoint 5-fold CV)
    2. Nested threshold selection (tune split only)
    3. Reproducible evaluation
    4. Clinical validity (high recall, workload management)
    """
    # Threshold selection
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Dynamic-K per state
    neg_config: DynamicKStateConfig = field(
        default_factory=lambda: DynamicKStateConfig(
            k_min=0, k_max1=0, k_max_ratio=0.0, gamma=None
        )
    )

    uncertain_config: DynamicKStateConfig = field(
        default_factory=lambda: DynamicKStateConfig(
            k_min=5, k_max1=12, k_max_ratio=0.70, gamma=0.95, temperature=1.0
        )
    )

    pos_config: DynamicKStateConfig = field(
        default_factory=lambda: DynamicKStateConfig(
            k_min=3, k_max1=12, k_max_ratio=0.60, gamma=0.90, temperature=1.0
        )
    )

    # Cross-validation
    n_folds: int = 5
    random_seed: int = 42
    tune_ratio: float = 0.30  # Portion of train for threshold tuning

    # Evaluation
    k_values: list = field(default_factory=lambda: [1, 3, 5, 10, 20])
    fpr_targets: list = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.10])

    # Output
    output_dir: Optional[Path] = None
    save_per_query_predictions: bool = True
    save_visualizations: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def save(self, path: Path):
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> 'ClinicalConfig':
        """Load configuration from YAML file."""
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Reconstruct nested dataclasses
        threshold_config = ThresholdConfig(**data.get('threshold_config', {}))
        neg_config = DynamicKStateConfig(**data.get('neg_config', {}))
        uncertain_config = DynamicKStateConfig(**data.get('uncertain_config', {}))
        pos_config = DynamicKStateConfig(**data.get('pos_config', {}))

        # Remove nested configs from data
        for key in ['threshold_config', 'neg_config', 'uncertain_config', 'pos_config']:
            data.pop(key, None)

        return cls(
            threshold_config=threshold_config,
            neg_config=neg_config,
            uncertain_config=uncertain_config,
            pos_config=pos_config,
            **data
        )

    def get_state_config(self, state: str) -> DynamicKStateConfig:
        """Get Dynamic-K config for a specific state."""
        if state == "NEG":
            return self.neg_config
        elif state == "UNCERTAIN":
            return self.uncertain_config
        elif state == "POS":
            return self.pos_config
        else:
            raise ValueError(f"Unknown state: {state}")


# Default clinical configuration
DEFAULT_CLINICAL_CONFIG = ClinicalConfig()
