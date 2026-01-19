#!/usr/bin/env python3
"""Package paper artifacts into a reproducible bundle.

This script collects evaluation results, figures, and tables
into a standardized paper_bundle directory for publication.

Usage:
    python scripts/reporting/package_paper_bundle.py \
        --results_dir outputs/final_eval \
        --output results/paper_bundle/v1.0

Outputs:
    results/paper_bundle/<version>/
    ├── MANIFEST.md           # File listing and checksums
    ├── summary.json          # Aggregated metrics
    ├── report.md             # Final evaluation report
    ├── figures/              # Publication-ready figures
    │   ├── roc_pr_curves.pdf
    │   ├── calibration_plot.pdf
    │   └── ...
    └── tables/               # Publication-ready tables
        ├── main_results.csv
        ├── ablation_results.csv
        └── ...
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def collect_figures(
    results_dir: Path,
    output_dir: Path,
    figure_patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """Collect figure files from results directory."""
    if figure_patterns is None:
        figure_patterns = ["*.png", "*.pdf", "*.svg"]

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    collected = []

    # Search common figure locations
    search_dirs = [
        results_dir,
        results_dir / "figures",
        results_dir / "plots",
        Path("docs/verification/figures"),
        Path("paper/figures"),
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in figure_patterns:
            for fig_path in search_dir.glob(pattern):
                dest_path = figures_dir / fig_path.name
                if not dest_path.exists():
                    shutil.copy2(fig_path, dest_path)
                    collected.append({
                        "name": fig_path.name,
                        "source": str(fig_path),
                        "size_bytes": dest_path.stat().st_size,
                        "sha256": compute_sha256(dest_path),
                    })

    return collected


def collect_tables(
    results_dir: Path,
    output_dir: Path,
    table_patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """Collect table files from results directory."""
    if table_patterns is None:
        table_patterns = ["*.csv", "*.json"]

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    collected = []

    # Search common table locations
    search_dirs = [
        results_dir,
        results_dir / "tables",
        results_dir / "metrics",
    ]

    # Key table files to look for
    key_tables = [
        "main_results.csv",
        "ablation_results.csv",
        "per_criterion_metrics.csv",
        "clinical_results.csv",
        "gnn_results.csv",
        "evaluation_summary.json",
        "summary.json",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in table_patterns:
            for table_path in search_dir.glob(pattern):
                # Prioritize key tables
                dest_name = table_path.name
                dest_path = tables_dir / dest_name
                if not dest_path.exists():
                    shutil.copy2(table_path, dest_path)
                    collected.append({
                        "name": dest_name,
                        "source": str(table_path),
                        "size_bytes": dest_path.stat().st_size,
                        "sha256": compute_sha256(dest_path),
                    })

    return collected


def create_manifest(
    output_dir: Path,
    figures: List[Dict],
    tables: List[Dict],
    version: str,
) -> None:
    """Create MANIFEST.md with file listing and checksums."""
    manifest_path = output_dir / "MANIFEST.md"

    with open(manifest_path, "w") as f:
        f.write(f"# Paper Bundle Manifest\n\n")
        f.write(f"**Version:** {version}\n")
        f.write(f"**Created:** {datetime.now().isoformat()}\n\n")
        f.write(f"## Figures\n\n")
        f.write("| File | Size | SHA256 | Manuscript Reference |\n")
        f.write("|------|------|--------|---------------------|\n")
        for fig in figures:
            size_kb = fig["size_bytes"] / 1024
            f.write(f"| {fig['name']} | {size_kb:.1f} KB | `{fig['sha256'][:12]}...` | TBD |\n")

        f.write(f"\n## Tables\n\n")
        f.write("| File | Size | SHA256 | Manuscript Reference |\n")
        f.write("|------|------|--------|---------------------|\n")
        for table in tables:
            size_kb = table["size_bytes"] / 1024
            f.write(f"| {table['name']} | {size_kb:.1f} KB | `{table['sha256'][:12]}...` | TBD |\n")

        f.write(f"\n## Verification\n\n")
        f.write("To verify file integrity:\n\n")
        f.write("```bash\n")
        f.write("sha256sum -c checksums.txt\n")
        f.write("```\n")

    # Also create checksums.txt
    checksums_path = output_dir / "checksums.txt"
    with open(checksums_path, "w") as f:
        for fig in figures:
            f.write(f"{fig['sha256']}  figures/{fig['name']}\n")
        for table in tables:
            f.write(f"{table['sha256']}  tables/{table['name']}\n")


def create_summary_json(
    output_dir: Path,
    figures: List[Dict],
    tables: List[Dict],
    version: str,
    results_dir: Optional[Path] = None,
) -> None:
    """Create summary.json with aggregated information."""
    summary = {
        "version": version,
        "created": datetime.now().isoformat(),
        "figures": {
            "count": len(figures),
            "files": [f["name"] for f in figures],
        },
        "tables": {
            "count": len(tables),
            "files": [t["name"] for t in tables],
        },
    }

    # Try to load existing evaluation summary
    if results_dir:
        for name in ["evaluation_summary.json", "summary.json"]:
            summary_path = results_dir / name
            if summary_path.exists():
                with open(summary_path) as f:
                    eval_summary = json.load(f)
                summary["evaluation"] = eval_summary
                break

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Package paper artifacts into a reproducible bundle"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/final_eval"),
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_bundle/v1.0"),
        help="Output directory for paper bundle",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Version string for this bundle",
    )
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Packaging paper bundle: {args.output}")
    print(f"Source results: {args.results_dir}")

    # Collect artifacts
    figures = collect_figures(args.results_dir, args.output)
    print(f"Collected {len(figures)} figures")

    tables = collect_tables(args.results_dir, args.output)
    print(f"Collected {len(tables)} tables")

    # Create manifest and summary
    create_manifest(args.output, figures, tables, args.version)
    create_summary_json(args.output, figures, tables, args.version, args.results_dir)

    # Copy existing report if available
    report_sources = [
        args.results_dir / "report.md",
        args.results_dir / "EVALUATION_REPORT.md",
        Path("docs/final/ACADEMIC_EVAL_REPORT.md"),
    ]
    for report_src in report_sources:
        if report_src.exists():
            shutil.copy2(report_src, args.output / "report.md")
            print(f"Copied report from {report_src}")
            break

    print(f"\nPaper bundle created at: {args.output}")
    print(f"See {args.output / 'MANIFEST.md'} for file listing")


if __name__ == "__main__":
    main()
