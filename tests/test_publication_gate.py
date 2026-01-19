"""Publication gate tests.

These tests verify the repository is in a publication-ready state:
- Required scripts exist
- Required docs exist
- No forbidden tracked paths
- Proper metric ranges
"""

import subprocess
from pathlib import Path

import pytest


# Repository root
REPO_ROOT = Path(__file__).parent.parent


class TestRequiredScriptsExist:
    """Verify all paper-critical scripts exist."""

    REQUIRED_SCRIPTS = [
        "scripts/run_paper_reproduce.sh",
        "scripts/build_groundtruth.py",
        "scripts/build_sentence_corpus.py",
        "scripts/eval_zoo_pipeline.py",
        "scripts/audit_splits.py",
        "scripts/encode_corpus.py",
        "scripts/verification/metric_crosscheck.py",
        "scripts/verification/audit_splits_and_leakage.py",
        "scripts/verification/generate_publication_plots.py",
        "scripts/reporting/package_paper_bundle.py",
        "scripts/ablation/run_ablation_suite.py",
        "scripts/clinical/run_clinical_high_recall_eval.py",
    ]

    @pytest.mark.parametrize("script_path", REQUIRED_SCRIPTS)
    def test_script_exists(self, script_path):
        """Each required script must exist."""
        full_path = REPO_ROOT / script_path
        assert full_path.exists(), f"Required script missing: {script_path}"


class TestRequiredDocsExist:
    """Verify all required documentation exists."""

    REQUIRED_DOCS = [
        "README.md",
        "CLAUDE.md",
        "docs/ENVIRONMENT_SETUP.md",
        "docs/final/PAPER_REPRODUCIBILITY.md",
        "docs/final/PAPER_COMMANDS.md",
        "docs/verification/COMPREHENSIVE_VERIFICATION_REPORT.md",
        "docs/cleanup/INVENTORY.md",
        "docs/cleanup/REMOVALS.md",
    ]

    @pytest.mark.parametrize("doc_path", REQUIRED_DOCS)
    def test_doc_exists(self, doc_path):
        """Each required doc must exist."""
        full_path = REPO_ROOT / doc_path
        assert full_path.exists(), f"Required doc missing: {doc_path}"


class TestNoForbiddenTrackedPaths:
    """Verify no generated/cached content is tracked in git."""

    FORBIDDEN_PATTERNS = [
        "outputs/",  # Generated outputs should not be tracked
        "data/cache/",  # Embedding caches should not be tracked
        "*.pkl",  # Pickle files should not be tracked
        "*.pt",  # PyTorch checkpoints should not be tracked
        "*.ckpt",  # Checkpoints should not be tracked
        "*.bin",  # Binary files should not be tracked
        "__pycache__/",  # Python cache should not be tracked
    ]

    def test_no_outputs_tracked(self):
        """outputs/ directory should not be tracked."""
        result = subprocess.run(
            ["git", "ls-files", "outputs/"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        tracked_outputs = result.stdout.strip()
        assert not tracked_outputs, f"Forbidden: outputs/ is tracked: {tracked_outputs[:200]}"

    def test_no_data_cache_tracked(self):
        """data/cache/ should not be tracked."""
        result = subprocess.run(
            ["git", "ls-files", "data/cache/"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        tracked_cache = result.stdout.strip()
        assert not tracked_cache, f"Forbidden: data/cache/ is tracked: {tracked_cache[:200]}"

    def test_no_pickle_files_tracked(self):
        """No .pkl files should be tracked."""
        result = subprocess.run(
            ["git", "ls-files", "*.pkl", "**/*.pkl"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        tracked_pkl = result.stdout.strip()
        assert not tracked_pkl, f"Forbidden: .pkl files tracked: {tracked_pkl[:200]}"

    def test_no_checkpoint_files_tracked(self):
        """No .pt/.ckpt/.bin files should be tracked."""
        result = subprocess.run(
            ["git", "ls-files", "*.pt", "*.ckpt", "*.bin"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        tracked_ckpt = result.stdout.strip()
        assert not tracked_ckpt, f"Forbidden: checkpoint files tracked: {tracked_ckpt[:200]}"


class TestMetricRangeAssertions:
    """Verify metric computation functions return valid ranges."""

    def test_recall_at_k_range(self):
        """Recall@K must be in [0, 1]."""
        from final_sc_review.metrics.ranking import recall_at_k

        # Test with known values
        gold = ["a", "b", "c"]
        ranked = ["a", "d", "b", "e", "c"]

        for k in [1, 3, 5, 10]:
            result = recall_at_k(gold, ranked, k)
            assert 0.0 <= result <= 1.0, f"recall@{k} out of range: {result}"

    def test_mrr_at_k_range(self):
        """MRR@K must be in [0, 1]."""
        from final_sc_review.metrics.ranking import mrr_at_k

        gold = ["a", "b"]
        ranked = ["x", "a", "y", "b"]

        for k in [1, 3, 5, 10]:
            result = mrr_at_k(gold, ranked, k)
            assert 0.0 <= result <= 1.0, f"mrr@{k} out of range: {result}"

    def test_ndcg_at_k_range(self):
        """nDCG@K must be in [0, 1]."""
        from final_sc_review.metrics.ranking import ndcg_at_k

        gold = ["a", "b", "c"]
        ranked = ["a", "x", "b", "y", "c"]

        for k in [1, 3, 5, 10]:
            result = ndcg_at_k(gold, ranked, k)
            assert 0.0 <= result <= 1.0, f"ndcg@{k} out of range: {result}"

    def test_empty_gold_handling(self):
        """Empty gold set should return 0, not raise error."""
        from final_sc_review.metrics.ranking import recall_at_k, mrr_at_k, ndcg_at_k

        gold = []
        ranked = ["a", "b", "c"]

        assert recall_at_k(gold, ranked, 5) == 0.0
        assert mrr_at_k(gold, ranked, 5) == 0.0
        assert ndcg_at_k(gold, ranked, 5) == 0.0


class TestNoBitwiseNegationBug:
    """Ensure no bitwise negation bugs (~labels instead of logical not).

    Note: ~mask on boolean tensors/arrays is CORRECT (e.g., ~gold_mask, ~y_true for FP calculation).
    This test only flags dangerous patterns where labels might not be boolean.
    """

    def test_no_dangerous_tilde_patterns(self):
        """Search for dangerous ~labels patterns (not on explicit boolean masks)."""
        import re

        # Only flag patterns that are likely bugs, not boolean mask inversions
        # ~gold_mask, ~y_true & y_pred are legitimate for boolean operations
        dangerous_patterns = [
            r"~\s*labels\s*[^\[]",  # ~labels not followed by index (likely int tensor)
            r"~\s*targets\s*[^\[]",  # ~targets not followed by index
            r"return\s+~",  # Returning bitwise negation (usually wrong)
        ]

        src_dir = REPO_ROOT / "src"
        issues = []

        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text()
            for pattern in dangerous_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Filter out known-good patterns
                    filtered = [m for m in matches if "mask" not in m.lower()]
                    if filtered:
                        issues.append(f"{py_file}: {filtered}")

        assert not issues, f"Potential bitwise negation bugs found: {issues}"


class TestGitignoreCompleteness:
    """Verify .gitignore covers required patterns."""

    REQUIRED_IGNORES = [
        "outputs/",
        "data/",
        "*.pkl",
        "*.pt",
        "*.bin",
        "__pycache__/",
        ".env",
    ]

    def test_gitignore_contains_required_patterns(self):
        """All required ignore patterns must be in .gitignore."""
        gitignore_path = REPO_ROOT / ".gitignore"
        assert gitignore_path.exists(), ".gitignore missing"

        content = gitignore_path.read_text()

        missing = []
        for pattern in self.REQUIRED_IGNORES:
            # Check for pattern or equivalent (with or without leading /)
            clean_pattern = pattern.strip("/")
            if clean_pattern not in content and f"/{clean_pattern}" not in content:
                missing.append(pattern)

        assert not missing, f"Missing patterns in .gitignore: {missing}"
