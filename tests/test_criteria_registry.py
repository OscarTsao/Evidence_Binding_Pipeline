"""Test criteria registry consistency.

Ensures:
1. Every criterion in data/eval exists in registry
2. Registry has no duplicates
3. Paper tables include only intended criteria
4. A.10 is correctly identified as SPECIAL_CASE (per ReDSM5 taxonomy)
"""

import json
from pathlib import Path

import pytest
import yaml


CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


class TestCriteriaRegistry:
    """Test criteria registry integrity."""

    @pytest.fixture
    def registry(self):
        """Load criteria registry."""
        registry_path = CONFIGS_DIR / "criteria_registry.yaml"
        if not registry_path.exists():
            pytest.skip("Criteria registry not found")
        with open(registry_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def mdd_criteria_json(self):
        """Load MDD_Criteira.json."""
        json_path = DATA_DIR / "DSM5" / "MDD_Criteira.json"
        if not json_path.exists():
            pytest.skip("MDD_Criteira.json not found")
        with open(json_path) as f:
            return json.load(f)

    def test_registry_has_all_criteria(self, registry):
        """Registry should have criteria A.1 through A.10."""
        criteria = registry["criteria"]
        expected_ids = [f"A.{i}" for i in range(1, 11)]

        for crit_id in expected_ids:
            assert crit_id in criteria, f"Missing criterion {crit_id} in registry"

    def test_no_duplicate_criteria(self, registry):
        """Registry should not have duplicate criterion IDs."""
        criteria = registry["criteria"]
        # Since YAML uses dict keys, duplicates would be overwritten
        # This test ensures we have exactly 10 criteria
        assert len(criteria) == 10, f"Expected 10 criteria, got {len(criteria)}"

    def test_a10_is_special_case(self, registry):
        """A.10 should be correctly labeled as SPECIAL_CASE (per ReDSM5 taxonomy)."""
        a10 = registry["criteria"]["A.10"]

        assert "special" in a10["short_name"].lower(), (
            f"A.10 short_name should be 'SPECIAL_CASE', got: {a10['short_name']}"
        )

        assert a10.get("is_special_case", False), (
            "A.10 should have is_special_case=true"
        )

        assert "expert" in a10["full_description"].lower() or "special" in a10["full_description"].lower(), (
            "A.10 description should mention expert discrimination or special case"
        )

    def test_a9_is_suicidal_ideation(self, registry):
        """A.9 should be Suicidal Ideation (not A.10)."""
        a9 = registry["criteria"]["A.9"]

        assert "suicid" in a9["short_name"].lower(), (
            f"A.9 should be Suicidal Ideation, got: {a9['short_name']}"
        )

    def test_all_criteria_marked_in_paper(self, registry):
        """All criteria A.1-A.10 should be marked as in_paper=true."""
        for crit_id, crit in registry["criteria"].items():
            assert crit.get("in_paper", False), (
                f"Criterion {crit_id} should be in_paper=true"
            )

    def test_json_matches_registry(self, registry, mdd_criteria_json):
        """MDD_Criteira.json should be consistent with registry."""
        json_criteria = {c["id"]: c for c in mdd_criteria_json["criteria"]}

        for crit_id, reg_crit in registry["criteria"].items():
            assert crit_id in json_criteria, (
                f"Criterion {crit_id} in registry but not in JSON"
            )

            json_crit = json_criteria[crit_id]

            # Short names should match
            if "short_name" in json_crit:
                assert json_crit["short_name"] == reg_crit["short_name"], (
                    f"Short name mismatch for {crit_id}: "
                    f"JSON={json_crit['short_name']}, Registry={reg_crit['short_name']}"
                )

    def test_all_registry_have_required_fields(self, registry):
        """All criteria should have required fields."""
        required_fields = ["short_name", "full_description", "in_paper"]

        for crit_id, crit in registry["criteria"].items():
            for field in required_fields:
                assert field in crit, (
                    f"Criterion {crit_id} missing required field: {field}"
                )


class TestCriteriaInEvaluation:
    """Test that evaluation data matches registry."""

    @pytest.fixture
    def per_query_criteria(self):
        """Get unique criteria from per_query.csv."""
        per_query_path = Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv")
        if not per_query_path.exists():
            pytest.skip("per_query.csv not found")

        import pandas as pd
        df = pd.read_csv(per_query_path)
        return set(df["criterion_id"].unique())

    @pytest.fixture
    def registry(self):
        """Load criteria registry."""
        registry_path = CONFIGS_DIR / "criteria_registry.yaml"
        if not registry_path.exists():
            pytest.skip("Criteria registry not found")
        with open(registry_path) as f:
            return yaml.safe_load(f)

    def test_all_eval_criteria_in_registry(self, per_query_criteria, registry):
        """All criteria in evaluation data should exist in registry."""
        registry_criteria = set(registry["criteria"].keys())

        missing = per_query_criteria - registry_criteria
        assert not missing, (
            f"Criteria in evaluation but not in registry: {missing}"
        )

    def test_all_registry_criteria_in_eval(self, per_query_criteria, registry):
        """All registry criteria marked in_paper should be in evaluation data."""
        registry_in_paper = {
            k for k, v in registry["criteria"].items()
            if v.get("in_paper", False)
        }

        missing = registry_in_paper - per_query_criteria
        assert not missing, (
            f"Criteria in registry (in_paper=true) but not in evaluation: {missing}"
        )


class TestPaperBundleCriteria:
    """Test paper bundle criterion consistency."""

    @pytest.fixture
    def metrics_master(self):
        """Load metrics_master.json."""
        for version in ["v3.0", "v2.0"]:
            path = RESULTS_DIR / "paper_bundle" / version / "metrics_master.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        pytest.skip("No metrics_master.json found")

    @pytest.fixture
    def registry(self):
        """Load criteria registry."""
        registry_path = CONFIGS_DIR / "criteria_registry.yaml"
        with open(registry_path) as f:
            return yaml.safe_load(f)

    def test_per_criterion_matches_registry(self, metrics_master, registry):
        """Per-criterion metrics should match registry criteria."""
        bundle_criteria = set(metrics_master["per_criterion_performance"]["criteria"].keys())
        registry_criteria = set(registry["criteria"].keys())

        assert bundle_criteria == registry_criteria, (
            f"Bundle criteria {bundle_criteria} != Registry criteria {registry_criteria}"
        )

    def test_a10_is_special_case_in_bundle(self, metrics_master):
        """A.10 should be labeled as SPECIAL_CASE in paper bundle."""
        a10_info = metrics_master["per_criterion_performance"]["criteria"]["A.10"]

        assert "special" in a10_info["description"].lower(), (
            f"A.10 description should mention SPECIAL_CASE, got: {a10_info['description']}"
        )

    def test_a10_has_lowest_auroc(self, metrics_master):
        """A.10 (SPECIAL_CASE) should have lowest AUROC (hardest criterion)."""
        criteria = metrics_master["per_criterion_performance"]["criteria"]

        aurocs = {k: v["auroc"] for k, v in criteria.items()}
        min_crit = min(aurocs, key=aurocs.get)

        # A.10 should be among the lowest (allow some tolerance)
        assert aurocs["A.10"] == min(aurocs.values()) or aurocs["A.10"] < 0.70, (
            f"A.10 (SPECIAL_CASE) should have low AUROC (got {aurocs['A.10']:.2f}), "
            f"lowest is {min_crit} with {min(aurocs.values()):.2f}"
        )
