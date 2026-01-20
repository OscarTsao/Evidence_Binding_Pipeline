#!/usr/bin/env python3
"""Prepare repository for release.

This script:
1. Validates all publication requirements are met
2. Generates release notes
3. Creates release artifacts
4. Prepares for GitHub release and Zenodo archival

Usage:
    python scripts/release/prepare_release.py --version v3.0 --output release/
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent


def run_command(cmd: str, check: bool = True) -> Tuple[int, str]:
    """Run shell command and return exit code and output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO_ROOT)
    return result.returncode, result.stdout + result.stderr


def check_git_status() -> Dict:
    """Check git repository status."""
    logger.info("Checking git status...")
    
    code, output = run_command("git status --porcelain")
    uncommitted = [line for line in output.strip().split("\n") if line]
    
    code, branch = run_command("git branch --show-current")
    branch = branch.strip()
    
    code, commit = run_command("git rev-parse HEAD")
    commit = commit.strip()[:8]
    
    return {
        "branch": branch,
        "commit": commit,
        "uncommitted_changes": len(uncommitted),
        "clean": len(uncommitted) == 0,
    }


def run_tests() -> Dict:
    """Run test suite."""
    logger.info("Running test suite...")
    
    code, output = run_command("python -m pytest -q")
    
    passed = "passed" in output
    lines = output.strip().split("\n")
    summary = lines[-1] if lines else "Unknown"
    
    return {
        "passed": code == 0,
        "summary": summary,
        "exit_code": code,
    }


def verify_checksums(bundle_dir: Path) -> Dict:
    """Verify paper bundle checksums."""
    logger.info("Verifying checksums...")
    
    checksums_file = bundle_dir / "checksums.txt"
    if not checksums_file.exists():
        return {"passed": False, "error": "checksums.txt not found"}
    
    code, output = run_command(f"python scripts/verification/verify_checksums.py --bundle {bundle_dir}")
    
    return {
        "passed": "VERIFICATION PASSED" in output,
        "output": output.strip(),
    }


def check_required_files() -> Dict:
    """Check all required files exist."""
    logger.info("Checking required files...")
    
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "CLAUDE.md",
        "docs/ETHICS.md",
        "docs/DATA_AVAILABILITY.md",
        "docs/REPRODUCIBILITY.md",
        "docs/final/DATA_STATEMENT.md",
        "docs/final/PUBLICATION_READINESS_REPORT.md",
        "results/paper_bundle/v3.0/metrics_master.json",
        "results/paper_bundle/v3.0/checksums.txt",
    ]
    
    missing = []
    for f in required:
        if not (REPO_ROOT / f).exists():
            missing.append(f)
    
    return {
        "passed": len(missing) == 0,
        "missing": missing,
        "checked": len(required),
    }


def generate_release_notes(version: str, git_info: Dict) -> str:
    """Generate release notes."""
    logger.info("Generating release notes...")
    
    metrics_file = REPO_ROOT / "results/paper_bundle/v3.0/metrics_master.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    
    class_metrics = metrics.get("classification_metrics", {}).get("metrics", {})
    rank_metrics = metrics.get("ranking_metrics", {}).get("metrics", {})
    
    auroc = class_metrics.get("auroc", {}).get("value", "N/A")
    auprc = class_metrics.get("auprc", {}).get("value", "N/A")
    recall = rank_metrics.get("evidence_recall_at_k", {}).get("value", "N/A")
    mrr = rank_metrics.get("mrr", {}).get("value", "N/A")

    # Format metrics for display
    auroc_str = f"{auroc:.4f}" if isinstance(auroc, (int, float)) else str(auroc)
    auprc_str = f"{auprc:.4f}" if isinstance(auprc, (int, float)) else str(auprc)
    recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else str(recall)
    mrr_str = f"{mrr:.4f}" if isinstance(mrr, (int, float)) else str(mrr)

    notes = f"""# Release {version}

**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Commit:** {git_info['commit']}
**Branch:** {git_info['branch']}

---

## Overview

This release contains the complete Evidence Binding Pipeline for psychiatric symptom detection,
ready for academic publication.

## Key Metrics

| Metric | Value |
|--------|-------|
| AUROC | {auroc_str} |
| AUPRC | {auprc_str} |
| Evidence Recall@K | {recall_str} |
| MRR | {mrr_str} |

## What's Included

### Paper Bundle v3.0
- `metrics_master.json` - Single source of truth for all metrics
- `tables/` - Publication-ready tables
- `checksums.txt` - SHA256 verification

### Documentation
- Complete reproducibility instructions
- Data statement with ethics/IRB documentation
- Error analysis report

### Code
- Full pipeline implementation
- Baseline implementations (BM25, TF-IDF, E5, Contriever)
- Robustness and significance testing scripts
- 227+ automated tests

## Verification

```bash
# Verify checksums
python scripts/verification/verify_checksums.py --bundle results/paper_bundle/v3.0

# Verify metrics
python scripts/verification/metric_crosscheck.py --bundle results/paper_bundle/v3.0

# Run tests
pytest -q
```

## Citation

See `CITATION.cff` for citation information.

## License

MIT License - see `LICENSE` file.
"""
    
    return notes


def main():
    parser = argparse.ArgumentParser(description="Prepare release")
    parser.add_argument("--version", type=str, default="v3.0", help="Release version")
    parser.add_argument("--output", type=Path, default=Path("release"), help="Output directory")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info(f"PREPARING RELEASE {args.version}")
    logger.info("=" * 80)
    
    results = {}
    all_passed = True
    
    # 1. Check git status
    git_info = check_git_status()
    results["git"] = git_info
    if not git_info["clean"]:
        logger.warning(f"Repository has {git_info['uncommitted_changes']} uncommitted changes")
    
    # 2. Check required files
    files_check = check_required_files()
    results["files"] = files_check
    if not files_check["passed"]:
        logger.error(f"Missing required files: {files_check['missing']}")
        all_passed = False
    
    # 3. Run tests (unless skipped)
    if not args.skip_tests:
        test_results = run_tests()
        results["tests"] = test_results
        if not test_results["passed"]:
            logger.error(f"Tests failed: {test_results['summary']}")
            all_passed = False
    else:
        results["tests"] = {"skipped": True}
    
    # 4. Verify checksums
    bundle_dir = REPO_ROOT / "results/paper_bundle/v3.0"
    if bundle_dir.exists():
        checksum_results = verify_checksums(bundle_dir)
        results["checksums"] = checksum_results
        if not checksum_results["passed"]:
            logger.error("Checksum verification failed")
            all_passed = False
    else:
        results["checksums"] = {"passed": False, "error": "Bundle not found"}
        all_passed = False
    
    # 5. Generate release notes
    release_notes = generate_release_notes(args.version, git_info)
    results["release_notes"] = "Generated"
    
    # 6. Save results
    output_dir = REPO_ROOT / args.output / args.version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "RELEASE_NOTES.md", "w") as f:
        f.write(release_notes)
    
    with open(output_dir / "release_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RELEASE PREPARATION SUMMARY")
    print("=" * 60)
    print(f"\nVersion: {args.version}")
    print(f"Git commit: {git_info['commit']}")
    print(f"Clean repo: {'YES' if git_info['clean'] else 'NO'}")
    print(f"\nChecks:")
    print(f"  Required files: {'PASS' if files_check['passed'] else 'FAIL'}")
    print(f"  Tests: {'PASS' if results.get('tests', {}).get('passed', False) else 'SKIP/FAIL'}")
    print(f"  Checksums: {'PASS' if results.get('checksums', {}).get('passed', False) else 'FAIL'}")
    print(f"\nRelease notes: {output_dir / 'RELEASE_NOTES.md'}")
    print(f"\nOverall: {'READY FOR RELEASE' if all_passed else 'NOT READY - FIX ISSUES ABOVE'}")
    print("=" * 60)
    
    if all_passed:
        print(f"""
Next Steps for GitHub Release:

1. Create git tag:
   git tag -a {args.version} -m "Release {args.version}"
   git push origin {args.version}

2. Create GitHub release:
   gh release create {args.version} --title "Release {args.version}" --notes-file {output_dir / 'RELEASE_NOTES.md'}

3. For Zenodo archival:
   - Link repository to Zenodo
   - GitHub release will trigger DOI generation
   - Add DOI badge to README
""")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
