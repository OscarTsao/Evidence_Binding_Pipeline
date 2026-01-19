#!/usr/bin/env python3
"""Static Label Leakage Scan - VERSION A STEP 2B

This script performs static analysis to detect potential label leakage patterns in the codebase.

Label leakage occurs when:
1. Test labels are used during model training
2. Test data is used for feature engineering or preprocessing
3. Global statistics are computed on full dataset (train+test)
4. Data normalization uses test set statistics
5. Feature selection uses test labels
6. Threshold tuning on test set
7. Hyperparameter optimization on test set
8. Early stopping based on test metrics

Usage:
    python scripts/audit_label_leakage_static.py --src_dir src --output outputs/audit/label_leakage_static_report.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Suspicious patterns that may indicate label leakage
LEAKAGE_PATTERNS = [
    # Test data accessed during training
    {
        "name": "Test data in training",
        "pattern": r"(train|fit|learn).*test_(data|labels|df|split|set)",
        "severity": "HIGH",
        "description": "Test data may be accessed during training"
    },
    {
        "name": "Test labels in training",
        "pattern": r"(y_test|test_labels|test_y).*\.(fit|train|learn)",
        "severity": "HIGH",
        "description": "Test labels used in training"
    },

    # Global statistics computed on full dataset
    {
        "name": "Global mean/std computation",
        "pattern": r"(\.mean\(\)|\.std\(\)|\.min\(\)|\.max\(\)).*#.*all.*data",
        "severity": "MEDIUM",
        "description": "Statistics may be computed on full dataset"
    },
    {
        "name": "Normalization on full data",
        "pattern": r"(StandardScaler|MinMaxScaler|normalize).*fit.*\+",
        "severity": "MEDIUM",
        "description": "Normalization may use test data statistics"
    },

    # Feature selection leakage
    {
        "name": "Feature selection on full data",
        "pattern": r"(SelectKBest|VarianceThreshold|RFE).*fit.*\+",
        "severity": "HIGH",
        "description": "Feature selection may use test labels"
    },

    # Threshold tuning on test
    {
        "name": "Threshold tuning on test",
        "pattern": r"(threshold|tau).*=.*test.*\.",
        "severity": "HIGH",
        "description": "Threshold tuned on test set"
    },
    {
        "name": "Calibration on test",
        "pattern": r"(calibrat|platt|isotonic).*fit.*test",
        "severity": "HIGH",
        "description": "Calibration fitted on test set"
    },

    # HPO leakage
    {
        "name": "HPO on test split",
        "pattern": r"(optuna|hyperopt|GridSearch|RandomSearch).*test",
        "severity": "CRITICAL",
        "description": "Hyperparameter optimization on test split"
    },

    # Early stopping leakage
    {
        "name": "Early stopping on test",
        "pattern": r"(early_stop|patience).*test.*metric",
        "severity": "HIGH",
        "description": "Early stopping based on test metrics"
    },

    # Cross-validation leakage
    {
        "name": "Preprocessing inside CV",
        "pattern": r"(for.*fold|cross_val).*\n.*\n.*(fit_transform|normalize)",
        "severity": "MEDIUM",
        "description": "Preprocessing may leak across folds"
    },

    # Data augmentation leakage
    {
        "name": "Augmentation with test",
        "pattern": r"(augment|oversample|SMOTE).*\+.*test",
        "severity": "HIGH",
        "description": "Data augmentation may include test data"
    },
]

# Allowed patterns (exceptions to avoid false positives)
ALLOWED_PATTERNS = [
    r"#.*test.*only",  # Comments about test-only code
    r"if.*test.*:",  # Conditional test-only code
    r"assert.*test",  # Test assertions
    r"def test_",  # Test function definitions
    r"class Test",  # Test class definitions
    r"\"test\"",  # String literals
    r"'test'",  # String literals
]


def grep_pattern(pattern: str, directory: Path, file_extensions: List[str]) -> List[Tuple[str, int, str]]:
    """Search for pattern using grep.

    Returns:
        List of (file_path, line_number, line_content)
    """
    matches = []

    # Use git grep if in git repo, otherwise regular grep
    try:
        # Try git grep first (faster and respects .gitignore)
        cmd = ["git", "grep", "-n", "-E", "-i", pattern]
        result = subprocess.run(
            cmd,
            cwd=directory,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    file_path, line_num, content = parts[0], parts[1], parts[2]
                    matches.append((file_path, int(line_num), content.strip()))
    except:
        # Fallback to regular grep
        for ext in file_extensions:
            try:
                cmd = ["grep", "-n", "-r", "-E", "-i", "--include", f"*.{ext}", pattern, str(directory)]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if not line:
                            continue
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            file_path, line_num, content = parts[0], parts[1], parts[2]
                            matches.append((file_path, int(line_num), content.strip()))
            except:
                pass

    return matches


def is_allowed(line: str) -> bool:
    """Check if line matches allowed patterns (false positives)."""
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def scan_leakage_patterns(
    src_dir: Path,
    file_extensions: List[str] = ["py"],
) -> Dict[str, List]:
    """Scan codebase for label leakage patterns.

    Returns:
        Dictionary mapping pattern names to list of matches
    """
    results = defaultdict(list)

    for pattern_info in LEAKAGE_PATTERNS:
        pattern_name = pattern_info["name"]
        pattern = pattern_info["pattern"]
        severity = pattern_info["severity"]

        print(f"Scanning for: {pattern_name} ({severity})...")

        matches = grep_pattern(pattern, src_dir, file_extensions)

        # Filter out allowed patterns
        filtered_matches = []
        for file_path, line_num, content in matches:
            if not is_allowed(content):
                filtered_matches.append({
                    "file": file_path,
                    "line": line_num,
                    "content": content,
                    "severity": severity,
                    "description": pattern_info["description"],
                })

        if filtered_matches:
            results[pattern_name] = filtered_matches
            print(f"  Found {len(filtered_matches)} potential issues")
        else:
            print(f"  ✅ Clean")

    return results


def generate_report(results: Dict[str, List], output_path: Path):
    """Generate markdown report of label leakage scan."""

    total_issues = sum(len(matches) for matches in results.values())

    with open(output_path, 'w') as f:
        f.write("# Label Leakage Static Scan Report - VERSION A STEP 2B\n\n")
        f.write("**Date:** 2026-01-19\n")
        f.write("**Purpose:** Detect potential label leakage patterns via static analysis\n\n")
        f.write("---\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")

        if total_issues == 0:
            f.write("**Status:** ✅ **PASS** - No suspicious patterns detected\n\n")
        else:
            f.write(f"**Status:** ⚠️ **REVIEW REQUIRED** - {total_issues} potential issues found\n\n")

        # Count by severity
        severity_counts = defaultdict(int)
        for matches in results.values():
            for match in matches:
                severity_counts[match["severity"]] += 1

        f.write("**Issues by Severity:**\n")
        f.write(f"- CRITICAL: {severity_counts['CRITICAL']}\n")
        f.write(f"- HIGH: {severity_counts['HIGH']}\n")
        f.write(f"- MEDIUM: {severity_counts['MEDIUM']}\n")
        f.write(f"- LOW: {severity_counts['LOW']}\n\n")

        f.write("**Note:** These are potential issues identified by pattern matching. ")
        f.write("Manual review is required to confirm if they constitute actual leakage.\n\n")

        f.write("---\n\n")

        # Detailed findings
        f.write("## Detailed Findings\n\n")

        if total_issues == 0:
            f.write("No suspicious patterns detected. The codebase appears to follow ")
            f.write("proper train/test separation practices.\n\n")
        else:
            for pattern_name, matches in sorted(results.items()):
                severity = matches[0]["severity"]
                description = matches[0]["description"]

                f.write(f"### {pattern_name}\n\n")
                f.write(f"**Severity:** {severity}\n")
                f.write(f"**Description:** {description}\n")
                f.write(f"**Occurrences:** {len(matches)}\n\n")

                f.write("**Locations:**\n\n")
                for i, match in enumerate(matches[:10]):  # Show max 10 per pattern
                    f.write(f"{i+1}. `{match['file']}:{match['line']}`\n")
                    f.write(f"   ```python\n")
                    f.write(f"   {match['content']}\n")
                    f.write(f"   ```\n\n")

                if len(matches) > 10:
                    f.write(f"... and {len(matches) - 10} more occurrences\n\n")

        f.write("---\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        if total_issues == 0:
            f.write("✅ No action required. Continue with VERSION A evaluation.\n\n")
        else:
            f.write("**Manual Review Required:**\n\n")
            f.write("1. Review each flagged location to determine if it's actual leakage\n")
            f.write("2. For false positives, document why they're safe\n")
            f.write("3. For confirmed leakage, fix the code before proceeding\n")
            f.write("4. Add unit tests to prevent future leakage (STEP 2D)\n\n")

            f.write("**Common False Positives:**\n")
            f.write("- Test code (unit tests, evaluation scripts)\n")
            f.write("- Comments and docstrings\n")
            f.write("- Variable names containing 'test'\n")
            f.write("- Legitimate test-only evaluation code\n\n")

        f.write("---\n\n")
        f.write("**Generated by:** `scripts/audit_label_leakage_static.py`\n")
        f.write("**Scan method:** Static pattern matching (grep-based)\n")
        f.write("**File types:** Python (*.py)\n")


def main():
    parser = argparse.ArgumentParser(description="Static scan for label leakage patterns")
    parser.add_argument("--src_dir", type=str, default="src", help="Source directory to scan")
    parser.add_argument("--output", type=str, default="outputs/audit/label_leakage_static_report.md",
                       help="Output report path")
    parser.add_argument("--extensions", type=str, nargs="+", default=["py"],
                       help="File extensions to scan")

    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist")
        return 1

    print("="*60)
    print("LABEL LEAKAGE STATIC SCAN")
    print("="*60)
    print(f"Source directory: {src_dir}")
    print(f"File extensions: {args.extensions}")
    print()

    # Scan for leakage patterns
    results = scan_leakage_patterns(src_dir, args.extensions)

    # Generate report
    generate_report(results, output_path)

    # Print summary
    total_issues = sum(len(matches) for matches in results.values())

    print()
    print("="*60)
    print("SCAN COMPLETE")
    print("="*60)
    print(f"Total patterns scanned: {len(LEAKAGE_PATTERNS)}")
    print(f"Suspicious patterns found: {len(results)}")
    print(f"Total potential issues: {total_issues}")
    print()

    if total_issues == 0:
        print("✅ PASS - No suspicious patterns detected")
    else:
        print(f"⚠️  REVIEW REQUIRED - {total_issues} potential issues need manual review")

    print(f"\nFull report: {output_path}")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
