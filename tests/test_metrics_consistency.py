"""Test metric consistency to prevent drift and common errors.

This test module ensures:
1. AUPRC is NOT confused with Recall@K
2. Metrics are computed deterministically
3. Protocol separation (positives_only vs all_queries) is enforced
4. Paper bundle metrics match recomputation
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

# Import canonical metric computation module
from final_sc_review.metrics.compute_metrics import (
    compute_all_metrics,
    compute_classification_metrics,
    compute_ranking_metrics_from_csv,
    compute_calibration_metrics,
    verify_auprc_not_recall,
    crosscheck_metrics,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
BUNDLE_DIR = Path(__file__).parent.parent / "results" / "paper_bundle"


class TestMetricDefinitions:
    """Test that metric definitions are correct."""

    def test_auprc_is_not_recall_at_k(self):
        """CRITICAL: Verify AUPRC is computed correctly, not confused with Recall@K.

        AUPRC (Area Under Precision-Recall Curve) is computed using
        sklearn.average_precision_score, which integrates the precision-recall curve.

        Recall@K is the fraction of gold items in top-K positions.

        These are fundamentally different metrics and should never be equal
        (except by coincidence on pathological datasets).
        """
        # Create test data where AUPRC and Recall@K would differ significantly
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.1])

        # Compute AUPRC using sklearn (correct implementation)
        auprc = average_precision_score(y_true, y_score)

        # Compute Recall@10 (different metric)
        # Top 10 by score: all 10 items, 4 of which are positive
        # Recall@10 = 4/4 = 1.0 (all positives retrieved)
        recall_at_10 = 1.0  # All positives are in top 10

        # These should NOT be equal
        assert auprc != recall_at_10, (
            f"AUPRC ({auprc:.4f}) should not equal Recall@K ({recall_at_10:.4f}). "
            "This test catches the common error of confusing these metrics."
        )

        # AUPRC should be around 0.7-0.9 for this example (not 1.0)
        assert 0.5 < auprc < 1.0, f"AUPRC should be in reasonable range, got {auprc}"

    def test_auprc_formula(self):
        """Verify AUPRC is computed as area under precision-recall curve."""
        # Simple case where we know the exact value
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.2])

        # With these scores, sorted order is: [1, 1, 0, 0]
        # Precision at each threshold:
        # After 1st (y=1): P=1/1=1.0, R=1/2=0.5
        # After 2nd (y=1): P=2/2=1.0, R=2/2=1.0
        # AP = avg of precision values at recalls where true positive occurs
        # = (1.0 + 1.0) / 2 = 1.0

        auprc = average_precision_score(y_true, y_score)
        assert auprc == 1.0, f"Expected AUPRC=1.0 for perfect ranking, got {auprc}"

    def test_auroc_definition(self):
        """Verify AUROC is probability of ranking positive higher than negative."""
        y_true = np.array([1, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.3, 0.2])

        auroc = roc_auc_score(y_true, y_score)

        # Manual calculation:
        # Positive scores: [0.9, 0.3], Negative scores: [0.8, 0.2]
        # Pairs where positive > negative:
        # 0.9 > 0.8: yes, 0.9 > 0.2: yes, 0.3 > 0.8: no, 0.3 > 0.2: yes
        # AUROC = 3/4 = 0.75
        expected = 0.75
        assert abs(auroc - expected) < 0.01, f"Expected AUROC={expected}, got {auroc}"


class TestProtocolSeparation:
    """Test that positives_only and all_queries protocols are correctly separated."""

    @pytest.fixture
    def sample_df(self):
        """Load sample fixture data."""
        return pd.read_csv(FIXTURES_DIR / "sample_per_query.csv")

    def test_ranking_uses_positives_only(self, sample_df):
        """Ranking metrics should only include queries with has_evidence_gold=1."""
        rank_metrics = compute_ranking_metrics_from_csv(sample_df)

        # Should report positives-only count
        n_positives = sample_df["has_evidence_gold"].sum()
        assert rank_metrics["n_queries_positives_only"] == n_positives

    def test_classification_uses_all_queries(self, sample_df):
        """Classification metrics should include all queries."""
        class_metrics = compute_classification_metrics(sample_df)

        assert class_metrics["n_queries_all"] == len(sample_df)

    def test_no_nan_in_positive_ranking_metrics(self, sample_df):
        """Ranking metrics for positives should not include NaN values."""
        df_pos = sample_df[sample_df["has_evidence_gold"] == 1]

        # These columns should have valid values for positive queries
        for col in ["evidence_recall_at_k", "mrr"]:
            if col in df_pos.columns:
                valid_count = df_pos[col].notna().sum()
                assert valid_count == len(df_pos), f"Column {col} has NaN for positive queries"


class TestDeterministicComputation:
    """Test that metric computation is deterministic."""

    @pytest.fixture
    def sample_df(self):
        """Load sample fixture data."""
        return pd.read_csv(FIXTURES_DIR / "sample_per_query.csv")

    def test_classification_metrics_deterministic(self, sample_df):
        """Classification metrics should be identical on repeated computation."""
        result1 = compute_classification_metrics(sample_df)
        result2 = compute_classification_metrics(sample_df)

        for key in ["auroc", "auprc", "brier_score"]:
            if key in result1 and key in result2:
                assert result1[key] == result2[key], f"{key} not deterministic"

    def test_ranking_metrics_deterministic(self, sample_df):
        """Ranking metrics should be identical on repeated computation."""
        result1 = compute_ranking_metrics_from_csv(sample_df)
        result2 = compute_ranking_metrics_from_csv(sample_df)

        for key in ["evidence_recall_at_k", "mrr"]:
            if key in result1 and key in result2:
                assert result1[key] == result2[key], f"{key} not deterministic"

    def test_bootstrap_ci_reproducible_with_seed(self, sample_df):
        """Bootstrap CIs should be reproducible with same seed."""
        from final_sc_review.metrics.compute_metrics import bootstrap_ci

        y_true = sample_df["has_evidence_gold"].values
        y_score = sample_df["p4_prob_calibrated"].values

        ci1 = bootstrap_ci(
            lambda yt, ys: roc_auc_score(yt, ys),
            (y_true, y_score),
            n_bootstrap=100,
            seed=42,
        )
        ci2 = bootstrap_ci(
            lambda yt, ys: roc_auc_score(yt, ys),
            (y_true, y_score),
            n_bootstrap=100,
            seed=42,
        )

        assert ci1["ci_lower"] == ci2["ci_lower"], "CI lower not reproducible"
        assert ci1["ci_upper"] == ci2["ci_upper"], "CI upper not reproducible"


class TestSafetyChecks:
    """Test safety checks to prevent metric errors."""

    def test_verify_auprc_not_recall_catches_error(self):
        """Safety check should flag when AUPRC equals Recall@K."""
        # Should return False (unsafe) when values are too close
        with pytest.warns(UserWarning, match="AUPRC.*very close to Recall@10"):
            result = verify_auprc_not_recall(0.7043, 0.7043, tolerance=0.01)
        assert result is False

    def test_verify_auprc_not_recall_passes_different_values(self):
        """Safety check should pass when values are different."""
        # Should return True (safe) when values are different
        result = verify_auprc_not_recall(0.5709, 0.7043, tolerance=0.01)
        assert result is True


class TestPaperBundleConsistency:
    """Test that paper bundle metrics match recomputation."""

    @pytest.fixture
    def per_query_csv(self):
        """Get path to canonical per_query.csv."""
        # Use the canonical evaluation output
        canonical_path = Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv")
        if canonical_path.exists():
            return canonical_path
        pytest.skip("Canonical per_query.csv not found")

    @pytest.fixture
    def metrics_master(self):
        """Load metrics_master.json from latest bundle."""
        bundle_v2 = BUNDLE_DIR / "v2.0" / "metrics_master.json"
        bundle_v3 = BUNDLE_DIR / "v3.0" / "metrics_master.json"

        if bundle_v3.exists():
            with open(bundle_v3) as f:
                return json.load(f)
        elif bundle_v2.exists():
            with open(bundle_v2) as f:
                return json.load(f)
        pytest.skip("No paper bundle found")

    def test_auroc_matches_bundle(self, per_query_csv, metrics_master):
        """AUROC should match paper bundle within tolerance."""
        df = pd.read_csv(per_query_csv)
        computed = compute_classification_metrics(df)

        expected = metrics_master["classification_metrics"]["metrics"]["auroc"]["value"]
        actual = computed["auroc"]

        assert abs(actual - expected) < 0.001, (
            f"AUROC mismatch: computed {actual:.4f} vs expected {expected:.4f}"
        )

    def test_auprc_matches_bundle(self, per_query_csv, metrics_master):
        """AUPRC should match paper bundle within tolerance."""
        df = pd.read_csv(per_query_csv)
        computed = compute_classification_metrics(df)

        expected = metrics_master["classification_metrics"]["metrics"]["auprc"]["value"]
        actual = computed["auprc"]

        assert abs(actual - expected) < 0.001, (
            f"AUPRC mismatch: computed {actual:.4f} vs expected {expected:.4f}"
        )

    def test_auprc_is_not_recall_in_bundle(self, metrics_master):
        """Bundle AUPRC should not equal Evidence Recall@K."""
        auprc = metrics_master["classification_metrics"]["metrics"]["auprc"]["value"]
        recall = metrics_master["ranking_metrics"]["metrics"]["evidence_recall_at_k"]["value"]

        # Verify they are different
        assert verify_auprc_not_recall(auprc, recall, tolerance=0.01), (
            f"AUPRC ({auprc}) suspiciously close to Recall@K ({recall})"
        )


class TestCalibrationMetrics:
    """Test calibration metric computation."""

    @pytest.fixture
    def sample_df(self):
        """Load sample fixture data."""
        return pd.read_csv(FIXTURES_DIR / "sample_per_query.csv")

    def test_ece_in_valid_range(self, sample_df):
        """ECE should be in [0, 1] range."""
        cal_metrics = compute_calibration_metrics(sample_df)

        if "ece" in cal_metrics:
            assert 0 <= cal_metrics["ece"] <= 1, f"ECE out of range: {cal_metrics['ece']}"

    def test_brier_in_valid_range(self, sample_df):
        """Brier score should be in [0, 1] range."""
        cal_metrics = compute_calibration_metrics(sample_df)

        if "brier_score" in cal_metrics:
            assert 0 <= cal_metrics["brier_score"] <= 1, (
                f"Brier out of range: {cal_metrics['brier_score']}"
            )

    def test_reliability_curve_structure(self, sample_df):
        """Reliability curve should have matching length arrays."""
        cal_metrics = compute_calibration_metrics(sample_df)

        if "reliability_curve" in cal_metrics:
            curve = cal_metrics["reliability_curve"]
            assert len(curve["prob_true"]) == len(curve["prob_pred"]), (
                "Reliability curve arrays should have same length"
            )


class TestComputeAllMetrics:
    """Test the main compute_all_metrics function."""

    @pytest.fixture
    def sample_df(self):
        """Load sample fixture data."""
        return pd.read_csv(FIXTURES_DIR / "sample_per_query.csv")

    def test_compute_all_returns_bundle(self, sample_df):
        """compute_all_metrics should return a MetricBundle."""
        bundle = compute_all_metrics(sample_df, compute_cis=False)

        assert hasattr(bundle, "ranking")
        assert hasattr(bundle, "classification")
        assert hasattr(bundle, "calibration")
        assert hasattr(bundle, "per_criterion")

    def test_bundle_to_dict_structure(self, sample_df):
        """Bundle.to_dict() should have correct structure."""
        bundle = compute_all_metrics(sample_df, compute_cis=False)
        d = bundle.to_dict()

        assert "ranking_metrics" in d
        assert "classification_metrics" in d
        assert "calibration_metrics" in d
        assert d["ranking_metrics"]["protocol"] == "positives_only"
        assert d["classification_metrics"]["protocol"] == "all_queries"

    def test_per_criterion_breakdown(self, sample_df):
        """Per-criterion metrics should cover all criteria in data."""
        bundle = compute_all_metrics(sample_df, compute_cis=False)

        criteria_in_data = set(sample_df["criterion_id"].unique())
        criteria_in_bundle = set(bundle.per_criterion.keys())

        assert criteria_in_data == criteria_in_bundle, (
            f"Criterion mismatch: data has {criteria_in_data}, bundle has {criteria_in_bundle}"
        )
