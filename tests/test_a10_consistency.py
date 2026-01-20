"""Test A.10 criterion definition consistency across codebase.

CRITICAL: Per ReDSM5 taxonomy, A.10 MUST be SPECIAL_CASE, NOT Duration.

The canonical definition is in src/final_sc_review/constants.py:
    CRITERION_TO_SYMPTOM = {
        ...
        "A.10": "SPECIAL_CASE",
    }

This test ensures all files that define A.10 use the correct definition.
"""

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).parent.parent


class TestA10Consistency:
    """Ensure A.10 = SPECIAL_CASE across all files."""

    def test_constants_is_canonical(self):
        """src/final_sc_review/constants.py should define A.10 = SPECIAL_CASE."""
        constants_path = REPO_ROOT / "src/final_sc_review/constants.py"
        assert constants_path.exists(), "constants.py not found"

        content = constants_path.read_text()

        # Check canonical definition
        assert '"A.10": "SPECIAL_CASE"' in content or "'A.10': 'SPECIAL_CASE'" in content, (
            "constants.py must define A.10 = SPECIAL_CASE"
        )

    def test_criteria_registry_yaml(self):
        """configs/criteria_registry.yaml should have A.10 = SPECIAL_CASE."""
        registry_path = REPO_ROOT / "configs/criteria_registry.yaml"
        if not registry_path.exists():
            pytest.skip("criteria_registry.yaml not found")

        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        a10 = registry["criteria"]["A.10"]

        assert "SPECIAL" in a10["short_name"].upper(), (
            f"A.10 short_name should be SPECIAL_CASE, got: {a10['short_name']}"
        )

        # Should NOT mention Duration
        assert "duration" not in a10["short_name"].lower(), (
            f"A.10 should NOT be Duration, got: {a10['short_name']}"
        )

    def test_mdd_criteria_json(self):
        """data/DSM5/MDD_Criteira.json should have A.10 = SPECIAL_CASE."""
        json_path = REPO_ROOT / "data/DSM5/MDD_Criteira.json"
        if not json_path.exists():
            pytest.skip("MDD_Criteira.json not found")

        with open(json_path) as f:
            data = json.load(f)

        a10_entries = [c for c in data["criteria"] if c["id"] == "A.10"]
        assert len(a10_entries) == 1, "Should have exactly one A.10 entry"

        a10 = a10_entries[0]

        assert "SPECIAL" in a10["short_name"].upper(), (
            f"A.10 short_name should be SPECIAL_CASE, got: {a10['short_name']}"
        )

        # Should NOT mention Duration
        assert "duration" not in a10["short_name"].lower(), (
            f"A.10 should NOT be Duration, got: {a10['short_name']}"
        )

    def test_no_duration_in_a10_definitions(self):
        """Grep for incorrect A.10 Duration definitions."""
        # Use grep to find any remaining incorrect definitions
        patterns_to_check = [
            r'"A\.10".*[Dd]uration',
            r"'A\.10'.*[Dd]uration",
            r"A\.10.*Duration \(2\+ weeks\)",
        ]

        excluded_dirs = [
            ".git",
            "__pycache__",
            "*.pyc",
            "node_modules",
            ".venv",
            "venv",
        ]

        issues_found = []

        for pattern in patterns_to_check:
            try:
                result = subprocess.run(
                    ["grep", "-rn", "--include=*.py", "--include=*.yaml",
                     "--include=*.yml", "--include=*.json", pattern, str(REPO_ROOT)],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    for line in result.stdout.strip().split("\n"):
                        # Exclude this test file itself
                        if "test_a10_consistency.py" not in line:
                            issues_found.append(line)
            except FileNotFoundError:
                pytest.skip("grep not available")

        if issues_found:
            msg = "Found incorrect A.10 = Duration definitions:\n" + "\n".join(issues_found)
            pytest.fail(msg)

    def test_error_analysis_criterion_names(self):
        """scripts/analysis/error_analysis.py should use SPECIAL_CASE for A.10."""
        script_path = REPO_ROOT / "scripts/analysis/error_analysis.py"
        if not script_path.exists():
            pytest.skip("error_analysis.py not found")

        content = script_path.read_text()

        # Find the CRITERION_NAMES dict
        # Look for A.10 value
        pattern = r'"A\.10"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, content)

        for match in matches:
            assert "SPECIAL" in match.upper() or "special" in match.lower(), (
                f"error_analysis.py A.10 should be SPECIAL_CASE, got: {match}"
            )
            assert "duration" not in match.lower(), (
                f"error_analysis.py A.10 should NOT be Duration, got: {match}"
            )

    def test_gnn_scripts_criteria_descriptions(self):
        """GNN rebuild scripts should use SPECIAL_CASE for A.10."""
        gnn_scripts = [
            REPO_ROOT / "scripts/gnn/rebuild_graph_cache.py",
            REPO_ROOT / "scripts/gnn/rebuild_graph_cache_phase2.py",
        ]

        for script_path in gnn_scripts:
            if not script_path.exists():
                continue

            content = script_path.read_text()

            # Find A.10 in criteria_descriptions
            pattern = r'"A\.10"\s*:\s*"([^"]+)"'
            matches = re.findall(pattern, content)

            for match in matches:
                assert "duration" not in match.lower(), (
                    f"{script_path.name} A.10 should NOT be Duration, got: {match}"
                )


class TestA10InPaperBundle:
    """Check A.10 in paper bundle metrics."""

    @pytest.fixture
    def metrics_master(self):
        """Load metrics_master.json if available."""
        for version in ["v3.0", "v2.0"]:
            path = REPO_ROOT / "results/paper_bundle" / version / "metrics_master.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        pytest.skip("No metrics_master.json found")

    def test_bundle_a10_description(self, metrics_master):
        """Paper bundle should describe A.10 as SPECIAL_CASE."""
        if "per_criterion_performance" not in metrics_master:
            pytest.skip("No per_criterion_performance in metrics")

        criteria = metrics_master["per_criterion_performance"].get("criteria", {})
        if "A.10" not in criteria:
            pytest.skip("No A.10 in criteria")

        a10_info = criteria["A.10"]
        description = a10_info.get("description", "").lower()

        # Should NOT mention Duration for A.10
        assert "duration" not in description or "special" in description, (
            f"A.10 description should reference SPECIAL_CASE, not Duration: {description}"
        )
