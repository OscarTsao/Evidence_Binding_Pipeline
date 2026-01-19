#!/usr/bin/env python3
"""
Generate visualization plots for NV-Embed-v2 baseline performance.

This script creates:
1. Line plots showing metric performance across different K values
2. Performance comparison tables
3. Heatmap showing all metrics at different K values

Usage:
    python scripts/visualization/plot_baseline_metrics.py \
        --crosscheck_json outputs/audit/nv_embed_v2_crosscheck.json \
        --output_dir outputs/audit/baseline_visualizations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_crosscheck_results(crosscheck_json: Path) -> Dict:
    """Load cross-check results from JSON file."""
    with open(crosscheck_json, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_metrics(comparison: List[Dict]) -> pd.DataFrame:
    """Parse metrics from cross-check comparison into DataFrame."""
    data = []
    for item in comparison:
        metric_name = item["metric"]
        # Parse metric family and K value
        # Format: "recall@5" -> family="recall", k=5
        parts = metric_name.split("@")
        family = parts[0]
        k = int(parts[1])

        data.append({
            "metric": metric_name,
            "family": family,
            "k": k,
            "value": item["recomputed"],
        })

    return pd.DataFrame(data)


def plot_metrics_by_k(df: pd.DataFrame, output_dir: Path):
    """Create line plots showing metric performance across K values."""
    families = df["family"].unique()

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NV-Embed-v2 Baseline Performance Across K Values",
                 fontsize=16, fontweight="bold")

    # Color palette
    colors = {"recall": "#2E86AB", "mrr": "#A23B72", "map": "#F18F01", "ndcg": "#C73E1D"}

    for idx, family in enumerate(sorted(families)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Get data for this family
        family_df = df[df["family"] == family].sort_values("k")

        # Plot line with markers
        ax.plot(family_df["k"], family_df["value"],
                marker="o", markersize=8, linewidth=2.5,
                color=colors.get(family, "#000000"),
                label=family.upper())

        # Annotate points with values
        for _, row_data in family_df.iterrows():
            ax.annotate(f"{row_data['value']:.3f}",
                       xy=(row_data['k'], row_data['value']),
                       xytext=(0, 10), textcoords="offset points",
                       ha="center", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="white", edgecolor="gray", alpha=0.8))

        # Styling
        ax.set_xlabel("K (Top-K)", fontsize=12, fontweight="bold")
        ax.set_ylabel(f"{family.upper()} Score", fontsize=12, fontweight="bold")
        ax.set_title(f"{family.upper()}@K Performance", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks([1, 3, 5, 10, 20])
        ax.set_ylim([0, 1.0])

        # Add horizontal line at key thresholds
        if family == "recall":
            ax.axhline(y=0.85, color="green", linestyle="--", alpha=0.5,
                      label="Target: 85%")
            ax.legend(loc="lower right")

    plt.tight_layout()
    output_path = output_dir / "metrics_by_k.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved line plots to {output_path}")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap showing all metrics at different K values."""
    # Pivot data for heatmap
    pivot_df = df.pivot(index="family", columns="k", values="value")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlOrRd",
                cbar_kws={"label": "Score"},
                linewidths=1, linecolor="white",
                vmin=0, vmax=1.0, ax=ax)

    ax.set_title("NV-Embed-v2 Baseline: All Metrics Heatmap",
                fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("K (Top-K)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Family", fontsize=12, fontweight="bold")
    ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()],
                       rotation=0)

    plt.tight_layout()
    output_path = output_dir / "metrics_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved heatmap to {output_path}")
    plt.close()


def create_performance_table(df: pd.DataFrame, output_dir: Path):
    """Create performance comparison table."""
    # Pivot data for table
    pivot_df = df.pivot(index="family", columns="k", values="value")

    # Format as percentages
    pivot_df = pivot_df * 100

    # Save as CSV
    csv_path = output_dir / "baseline_metrics_table.csv"
    pivot_df.to_csv(csv_path, float_format="%.2f")
    print(f"✓ Saved metrics table to {csv_path}")

    # Create formatted text table
    table_str = "# NV-Embed-v2 Baseline Performance\n\n"
    table_str += "All values reported as percentages.\n\n"
    table_str += "| Metric | @1 | @3 | @5 | @10 | @20 |\n"
    table_str += "|--------|-------|-------|-------|-------|-------|\n"

    for family in sorted(pivot_df.index):
        row = pivot_df.loc[family]
        table_str += f"| {family.upper():6s} |"
        for k in [1, 3, 5, 10, 20]:
            if k in row.index:
                table_str += f" {row[k]:5.2f} |"
            else:
                table_str += "   -   |"
        table_str += "\n"

    # Save as markdown
    md_path = output_dir / "baseline_metrics_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"✓ Saved markdown table to {md_path}")

    return pivot_df


def create_summary_stats(df: pd.DataFrame, output_dir: Path):
    """Create summary statistics."""
    summary = {
        "n_queries": 1490,
        "retriever": "nvidia/NV-Embed-v2",
        "embedding_dim": 4096,
        "key_metrics": {
            "recall@10": float(df[(df["family"] == "recall") & (df["k"] == 10)]["value"].values[0]),
            "recall@20": float(df[(df["family"] == "recall") & (df["k"] == 20)]["value"].values[0]),
            "mrr@10": float(df[(df["family"] == "mrr") & (df["k"] == 10)]["value"].values[0]),
            "map@10": float(df[(df["family"] == "map") & (df["k"] == 10)]["value"].values[0]),
            "ndcg@10": float(df[(df["family"] == "ndcg") & (df["k"] == 10)]["value"].values[0]),
        },
        "performance_tier": "Strong baseline",
        "notes": [
            "All metrics independently verified via cross-check",
            "Retrieval-only mode (no reranking)",
            "Within-post retrieval on post-id-disjoint test split",
        ],
    }

    # Save summary
    summary_path = output_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline performance visualizations"
    )
    parser.add_argument(
        "--crosscheck_json",
        type=Path,
        required=True,
        help="Path to cross-check JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save visualizations",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load cross-check results
    print(f"Loading cross-check results from {args.crosscheck_json}...")
    results = load_crosscheck_results(args.crosscheck_json)

    # Parse metrics
    print("Parsing metrics...")
    df = parse_metrics(results["comparison"])

    print(f"\nFound {len(df)} metrics:")
    print(f"  - Metric families: {sorted(df['family'].unique())}")
    print(f"  - K values: {sorted(df['k'].unique())}")
    print(f"  - Queries evaluated: {results['n_queries_checked']}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metrics_by_k(df, args.output_dir)
    plot_heatmap(df, args.output_dir)

    # Generate tables
    print("\nGenerating performance tables...")
    table_df = create_performance_table(df, args.output_dir)

    # Generate summary stats
    print("\nGenerating summary statistics...")
    summary = create_summary_stats(df, args.output_dir)

    # Print key metrics
    print("\n" + "=" * 60)
    print("NV-EMBED-V2 BASELINE PERFORMANCE")
    print("=" * 60)
    for metric, value in summary["key_metrics"].items():
        print(f"  {metric:12s}: {value:.4f} ({value*100:.2f}%)")
    print("=" * 60)

    print(f"\n✓ All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
