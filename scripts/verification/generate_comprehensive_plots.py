#!/usr/bin/env python3
"""Generate comprehensive visualizations for VERSION A audit.

Creates all plots required for academic publication:
1. ROC + PR curves (overall + per criterion)
2. Calibration curves
3. Dynamic-K analysis
4. Ablation comparison
5. Per-criterion performance heatmap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_roc_pr_curves(
    per_query_csvs: List[Path],
    output_file: Path
):
    """Plot ROC and PR curves.

    Args:
        per_query_csvs: List of per-query prediction CSVs (one per fold)
        output_file: Output plot file
    """
    logger.info("Generating ROC + PR curves...")

    # Load all fold data
    dfs = [pd.read_csv(f) for f in per_query_csvs]
    df_all = pd.concat(dfs, ignore_index=True)

    probs = df_all['p4_prob_calibrated'].values
    labels = df_all['has_evidence_gold'].values

    # Compute curves
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(labels, probs)

    ax1.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate (Recall)')
    ax1.set_title('ROC Curve (NE Detection)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # PR curve
    auprc = average_precision_score(labels, probs)
    baseline = labels.mean()

    ax2.plot(recall, precision, linewidth=2, label=f'AUPRC = {auprc:.3f}')
    ax2.axhline(baseline, color='k', linestyle='--', linewidth=1,
                alpha=0.5, label=f'Baseline = {baseline:.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_file}")


def plot_calibration_curve(
    per_query_csvs: List[Path],
    output_file: Path
):
    """Plot calibration curve.

    Args:
        per_query_csvs: List of per-query prediction CSVs
        output_file: Output plot file
    """
    logger.info("Generating calibration curve...")

    # Load data
    dfs = [pd.read_csv(f) for f in per_query_csvs]
    df_all = pd.concat(dfs, ignore_index=True)

    probs = df_all['p4_prob_calibrated'].values
    labels = df_all['has_evidence_gold'].values

    # Compute calibration curve
    n_bins = 10
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy='uniform')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8,
            label='Model (isotonic calibration)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')

    # Compute ECE
    bin_totals = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
    nonzero = bin_totals > 0
    ece = np.sum(bin_totals[nonzero] * np.abs(prob_true[nonzero] - prob_pred[nonzero])) / len(probs)

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Fraction of Positives')
    ax.set_title(f'Calibration Curve (ECE = {ece:.4f})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_file}")


def plot_dynamic_k_analysis(
    per_query_csvs: List[Path],
    output_file: Path
):
    """Plot dynamic-K analysis.

    Args:
        per_query_csvs: List of per-query prediction CSVs
        output_file: Output plot file
    """
    logger.info("Generating dynamic-K analysis...")

    # Load data
    dfs = [pd.read_csv(f) for f in per_query_csvs]
    df_all = pd.concat(dfs, ignore_index=True)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. K histogram by state
    ax = axes[0, 0]
    for state in ['NEG', 'UNCERTAIN', 'POS']:
        state_data = df_all[df_all['state'] == state]
        if len(state_data) > 0:
            ax.hist(state_data['selected_k'], bins=20, alpha=0.6, label=state, edgecolor='black')

    ax.set_xlabel('Selected K')
    ax.set_ylabel('Count')
    ax.set_title('Dynamic-K Distribution by State')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. K vs N (candidate count) scatter
    ax = axes[0, 1]
    uncertain = df_all[df_all['state'] == 'UNCERTAIN']

    if len(uncertain) > 0:
        # Sample for visibility
        sample_size = min(1000, len(uncertain))
        sample = uncertain.sample(n=sample_size, random_state=42)

        ax.scatter(sample['n_candidates'], sample['selected_k'],
                   alpha=0.3, s=10, c='steelblue')

        # Add trend line
        z = np.polyfit(uncertain['n_candidates'], uncertain['selected_k'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(uncertain['n_candidates'].min(),
                              uncertain['n_candidates'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: K={z[0]:.2f}*N + {z[1]:.2f}')

    ax.set_xlabel('Candidate Count (N)')
    ax.set_ylabel('Selected K')
    ax.set_title('Dynamic-K vs Candidate Count (UNCERTAIN state)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mean K by state (box plot)
    ax = axes[1, 0]
    state_k_data = []
    state_labels = []

    for state in ['NEG', 'UNCERTAIN', 'POS']:
        state_data = df_all[df_all['state'] == state]['selected_k'].dropna()
        if len(state_data) > 0:
            state_k_data.append(state_data)
            state_labels.append(state)

    ax.boxplot(state_k_data, labels=state_labels)
    ax.set_ylabel('Selected K')
    ax.set_title('K Distribution by State (Box Plot)')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Evidence recall vs K (for queries with evidence)
    ax = axes[1, 1]
    with_evidence = df_all[df_all['has_evidence_gold'] == 1].copy()

    if len(with_evidence) > 0 and 'evidence_recall_at_k' in with_evidence.columns:
        # Group by K bins
        with_evidence['k_bin'] = pd.cut(with_evidence['selected_k'],
                                         bins=[0, 3, 5, 8, 12], labels=['1-3', '4-5', '6-8', '9-12'])

        k_recall = with_evidence.groupby('k_bin')['evidence_recall_at_k'].agg(['mean', 'std', 'count'])

        ax.bar(range(len(k_recall)), k_recall['mean'], yerr=k_recall['std'],
               capsize=5, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(k_recall)))
        ax.set_xticklabels(k_recall.index)
        ax.set_xlabel('K Range')
        ax.set_ylabel('Evidence Recall@K')
        ax.set_title('Evidence Recall vs Selected K')
        ax.grid(True, alpha=0.3, axis='y')

        # Add count annotations
        for i, count in enumerate(k_recall['count']):
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={int(count)}',
                    ha='center', va='top', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_file}")


def plot_per_criterion_performance(
    summary_file: Path,
    output_file: Path
):
    """Plot per-criterion performance heatmap.

    Args:
        summary_file: Path to summary.json with per_criterion_metrics
        output_file: Output plot file
    """
    logger.info("Generating per-criterion performance heatmap...")

    with open(summary_file) as f:
        summary = json.load(f)

    fold_results = summary.get("fold_results", [])

    # Collect per-criterion metrics across folds
    criteria = [f"A.{i}" for i in range(1, 11)]
    metrics = ["auroc", "auprc", "sensitivity_at_screening", "evidence_recall"]

    # Initialize matrix
    matrix = np.zeros((len(criteria), len(metrics)))

    for i, criterion in enumerate(criteria):
        # Average across folds
        fold_values = {metric: [] for metric in metrics}

        for fold in fold_results:
            per_crit = fold.get("per_criterion_metrics", {})
            crit_data = per_crit.get(criterion, {})

            for j, metric in enumerate(metrics):
                value = crit_data.get(metric)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    fold_values[metric].append(value)

        # Compute means
        for j, metric in enumerate(metrics):
            if len(fold_values[metric]) > 0:
                matrix[i, j] = np.mean(fold_values[metric])

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    metric_labels = ["AUROC", "AUPRC", "Sensitivity@Screening", "Evidence Recall"]

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(metric_labels)))
    ax.set_yticks(range(len(criteria)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(criteria)

    # Add values
    for i in range(len(criteria)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

    ax.set_title('Per-Criterion Performance (Averaged Across Folds)')
    plt.colorbar(im, ax=ax, label='Metric Value')

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_file}")


def plot_ablation_waterfall(
    ablation_file: Path,
    output_file: Path
):
    """Plot ablation waterfall chart.

    Args:
        ablation_file: Path to ablation_study.json
        output_file: Output plot file
    """
    logger.info("Generating ablation waterfall chart...")

    with open(ablation_file) as f:
        ablation_data = json.load(f)

    ablations = ablation_data.get("ablations", {})

    # Extract configs in order
    config_order = ["A0_baseline", "A1_p4_only", "A2_p4_calibrated",
                    "A3_with_gate", "A4_full_system"]

    names = []
    aurocs = []
    auprcs = []

    for config in config_order:
        if config in ablations:
            names.append(ablations[config]["name"])
            aurocs.append(ablations[config].get("auroc", 0))
            auprcs.append(ablations[config].get("auprc", 0))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    x = range(len(names))

    # AUROC waterfall
    ax1.bar(x, aurocs, color=['lightgray', 'steelblue', 'steelblue', 'darkgreen', 'darkgreen'],
            edgecolor='black', alpha=0.8)
    ax1.set_ylabel('AUROC')
    ax1.set_title('Ablation Study: AUROC by Configuration')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.4, 1.0])

    # Add value labels
    for i, (name, auroc) in enumerate(zip(names, aurocs)):
        ax1.text(i, auroc + 0.01, f'{auroc:.3f}', ha='center', va='bottom', fontsize=9)

    # Add delta annotations
    for i in range(1, len(aurocs)):
        delta = aurocs[i] - aurocs[i-1]
        y_pos = (aurocs[i] + aurocs[i-1]) / 2
        ax1.annotate('', xy=(i, aurocs[i]), xytext=(i-1, aurocs[i-1]),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
        ax1.text(i-0.5, y_pos, f'+{delta:.3f}', ha='center', va='bottom',
                 fontsize=8, color='red', weight='bold')

    # AUPRC waterfall
    ax2.bar(x, auprcs, color=['lightgray', 'steelblue', 'steelblue', 'darkgreen', 'darkgreen'],
            edgecolor='black', alpha=0.8)
    ax2.set_ylabel('AUPRC')
    ax2.set_title('Ablation Study: AUPRC by Configuration')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.0, 0.7])

    # Add value labels
    for i, (name, auprc) in enumerate(zip(names, auprcs)):
        ax2.text(i, auprc + 0.01, f'{auprc:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive plots")
    parser.add_argument(
        "--clinical_dir",
        type=Path,
        default=Path("outputs/clinical_high_recall/20260118_015913"),
        help="Clinical evaluation directory"
    )
    parser.add_argument(
        "--ablation_file",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/verification/ablation_study.json"),
        help="Ablation study JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/version_a_full_audit/plots"),
        help="Output directory for plots"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("COMPREHENSIVE PLOT GENERATION")
    logger.info("="*80)

    # Find per-query CSVs
    fold_csvs = sorted((args.clinical_dir / "fold_results").glob("fold_*_predictions.csv"))
    logger.info(f"\nFound {len(fold_csvs)} fold prediction CSVs")

    summary_file = args.clinical_dir / "summary.json"

    # Generate plots
    logger.info("\nGenerating plots...")
    logger.info("-" * 80)

    plot_roc_pr_curves(fold_csvs, args.output_dir / "roc_pr_curves.png")
    plot_calibration_curve(fold_csvs, args.output_dir / "calibration_curve.png")
    plot_dynamic_k_analysis(fold_csvs, args.output_dir / "dynamic_k_analysis.png")
    plot_per_criterion_performance(summary_file, args.output_dir / "per_criterion_heatmap.png")
    plot_ablation_waterfall(args.ablation_file, args.output_dir / "ablation_waterfall.png")

    logger.info("\n" + "="*80)
    logger.info("✅ ALL PLOTS GENERATED")
    logger.info("="*80)
    logger.info(f"\nOutputs saved to: {args.output_dir}")
    logger.info("  1. roc_pr_curves.png")
    logger.info("  2. calibration_curve.png")
    logger.info("  3. dynamic_k_analysis.png")
    logger.info("  4. per_criterion_heatmap.png")
    logger.info("  5. ablation_waterfall.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
