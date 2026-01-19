#!/usr/bin/env python3
"""Verify SHA256 checksums for paper bundle artifacts.

Usage:
    python scripts/verification/verify_checksums.py
    python scripts/verification/verify_checksums.py --checksums results/paper_bundle/v1.0/checksums.txt
"""

import argparse
import hashlib
import sys
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def parse_checksums_file(checksums_path: Path) -> dict:
    """Parse checksums.txt file.

    Expected format (sha256sum compatible):
    <hash>  <filepath>
    # comments are ignored
    """
    checksums = {}
    with open(checksums_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                hash_value, filepath = parts
                checksums[filepath] = hash_value
    return checksums


def verify_checksums(checksums_path: Path, base_dir: Path = None) -> tuple:
    """Verify all checksums in the file.

    Args:
        checksums_path: Path to checksums.txt
        base_dir: Base directory for resolving relative paths

    Returns:
        Tuple of (passed, failed, missing) counts and details
    """
    if base_dir is None:
        base_dir = checksums_path.parent

    checksums = parse_checksums_file(checksums_path)

    passed = []
    failed = []
    missing = []

    for filepath, expected_hash in checksums.items():
        full_path = base_dir / filepath
        if not full_path.exists():
            missing.append((filepath, "File not found"))
            continue

        actual_hash = compute_sha256(full_path)
        if actual_hash == expected_hash:
            passed.append(filepath)
        else:
            failed.append((filepath, expected_hash, actual_hash))

    return passed, failed, missing


def main():
    parser = argparse.ArgumentParser(description="Verify paper bundle checksums")
    parser.add_argument(
        "--checksums",
        type=Path,
        default=Path("results/paper_bundle/v1.0/checksums.txt"),
        help="Path to checksums.txt file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print details for each file"
    )

    args = parser.parse_args()

    if not args.checksums.exists():
        print(f"ERROR: Checksums file not found: {args.checksums}")
        return 1

    print(f"Verifying checksums from: {args.checksums}")
    print("-" * 60)

    passed, failed, missing = verify_checksums(args.checksums)

    # Print results
    if args.verbose:
        for filepath in passed:
            print(f"  PASS: {filepath}")

    for filepath, expected, actual in failed:
        print(f"  FAIL: {filepath}")
        print(f"        Expected: {expected}")
        print(f"        Actual:   {actual}")

    for filepath, reason in missing:
        print(f"  MISSING: {filepath} ({reason})")

    print("-" * 60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed, {len(missing)} missing")

    if failed or missing:
        print("\nVERIFICATION FAILED")
        return 1
    else:
        print("\nVERIFICATION PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
