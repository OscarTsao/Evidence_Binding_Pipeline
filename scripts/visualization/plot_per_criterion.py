#!/usr/bin/env python3
"""
Generate visualizations for per-criterion analysis (STEP 6).

Creates:
1. Bar chart showing nDCG@10 for each criterion
2. Heatmap showing all metrics across criteria
3. Scatter plot showing difficulty vs. data availability
4. A.10 comparison chart

Usage:
    python scripts/visualization/plot_per_criterion.py \
        --analysis outputs/analysis/per_criterion_analysis.json \
        --output_dir outputs/analysis/visualizations
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_analysis(analysis_path: Path) -> dict:
    """Load analysis results from JSON."""
    with open(analysis_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_ndcg_by_criterion(data: dict, output_dir: Path):
    """Bar chart showing nDCG@10 for each criterion."""
    difficulty_ranking = data["difficulty_ranking"]

    criteria = [item["criterion"] for item in difficulty_ranking]
    ndcg_values = [item["ndcg@10"] for item in difficulty_ranking]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bars with color gradient (red = hard, green = easy)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(criteria)))

    bars = ax.bar(range(len(criteria)), ndcg_values, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, ndcg_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Styling
    ax.set_xlabel("Criterion", fontsize=13, fontweight='bold')
    ax.set_ylabel("nDCG@10", fontsize=13, fontweight='bold')
    ax.set_title("Per-Criterion Difficulty (NV-Embed-v2 Baseline)", fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(criteria)))
    ax.set_xticklabels(criteria, fontsize=11)
    ax.set_ylim([0, max(ndcg_values) * 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add average line
    avg_ndcg = np.mean(ndcg_values)
    ax.axhline(y=avg_ndcg, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_ndcg:.3f}')
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / "ndcg_by_criterion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_metrics_heatmap(data: dict, output_dir: Path):
    """Heatmap showing all metrics across criteria."""
    per_criterion = data["per_criterion_results"]

    # Extract data for heatmap
    criteria = sorted(per_criterion.keys())
    metrics = ["ndcg@10", "recall@10", "mrr@10", "map@10"]

    matrix = []
    for criterion in criteria:
        row = []
        for metric in metrics:
            val = per_criterion[criterion]["metrics"].get(metric, {}).get("mean", 0.0)
            row.append(val)
        matrix.append(row)

    df = pd.DataFrame(matrix, index=criteria, columns=[m.upper() for m in metrics])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=1, linecolor="white",
                vmin=0, vmax=df.max().max(),
                cbar_kws={"label": "Score"},
                ax=ax)

    ax.set_title("Per-Criterion Performance Heatmap", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Metric", fontsize=12, fontweight='bold')
    ax.set_ylabel("Criterion", fontsize=12, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    output_path = output_dir / "metrics_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_difficulty_vs_data(data: dict, output_dir: Path):
    """Scatter plot showing difficulty vs. data availability."""
    per_criterion = data["per_criterion_results"]

    criteria_list = []
    ndcg_values = []
    n_queries_list = []

    for criterion, crit_data in sorted(per_criterion.items()):
        criteria_list.append(criterion)
        ndcg_values.append(crit_data["metrics"].get("ndcg@10", {}).get("mean", 0.0))
        n_queries_list.append(crit_data["n_queries_with_positives"])

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by criterion
    colors = plt.cm.tab10(np.arange(len(criteria_list)))
    scatter = ax.scatter(n_queries_list, ndcg_values, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Add labels
    for i, criterion in enumerate(criteria_list):
        ax.annotate(criterion, (n_queries_list[i], ndcg_values[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # Styling
    ax.set_xlabel("Number of Queries with Evidence", fontsize=13, fontweight='bold')
    ax.set_ylabel("nDCG@10", fontsize=13, fontweight='bold')
    ax.set_title("Criterion Difficulty vs. Data Availability", fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add trend line
    z = np.polyfit(n_queries_list, ndcg_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(n_queries_list), max(n_queries_list), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "difficulty_vs_data.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_a10_comparison(data: dict, output_dir: Path):
    """Comparison chart for A.10 vs. others."""
    a10_analysis = data["a10_analysis"]
    differences = a10_analysis["differences"]

    metrics = ["ndcg@10", "recall@10", "mrr@10", "map@10"]
    a10_values = [differences[m]["a10"] for m in metrics]
    others_values = [differences[m]["others_avg"] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, a10_values, width, label='A.10', color='#2E86AB', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, others_values, width, label='Others (Avg)', color='#F18F01', edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Styling
    ax.set_xlabel("Metric", fontsize=13, fontweight='bold')
    ax.set_ylabel("Score", fontsize=13, fontweight='bold')
    ax.set_title("A.10 Performance vs. Other Criteria", fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax.set_ylim([0, max(max(a10_values), max(others_values)) * 1.15])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = output_dir / "a10_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-criterion visualizations (STEP 6)"
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        required=True,
        help="Path to per-criterion analysis JSON",
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

    # Load analysis
    print(f"Loading analysis from {args.analysis}")
    data = load_analysis(args.analysis)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_ndcg_by_criterion(data, args.output_dir)
    plot_metrics_heatmap(data, args.output_dir)
    plot_difficulty_vs_data(data, args.output_dir)
    plot_a10_comparison(data, args.output_dir)

    print(f"\n✓ All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
