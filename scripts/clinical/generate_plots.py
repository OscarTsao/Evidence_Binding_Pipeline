#!/usr/bin/env python3
"""Generate visualization plots from clinical evaluation summary.

This script creates all visualization plots for clinical deployment review:
1. ROC/PR curves
2. Calibration plot
3. Tradeoff curves
4. Per-criterion analysis
5. Dynamic-K analysis

Usage:
    python scripts/clinical/generate_plots.py \\
        --summary outputs/clinical_high_recall/YYYYMMDD_HHMMSS/summary.json \\
        --output outputs/clinical_high_recall/YYYYMMDD_HHMMSS/curves
"""

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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 9


def load_summary(summary_path: Path) -> Dict:
    """Load summary.json file."""
    with open(summary_path) as f:
        return json.load(f)


def plot_roc_pr_curves(summary: Dict, output_file: Path):
    """Generate ROC and PR curves.

    This function would normally plot ROC/PR curves, but since we don't have
    the raw predictions (only aggregated metrics), we create a placeholder
    showing the aggregated AUROC/AUPRC values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract AUROC and AUPRC values
    auroc_values = []
    auprc_values = []

    for fold_result in summary['fold_results']:
        auroc = fold_result['test_metrics']['ne_gate']['auroc']
        auprc = fold_result['test_metrics']['ne_gate']['auprc']
        auroc_values.append(auroc)
        auprc_values.append(auprc)

    # ROC curve placeholder
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve\nAUROC = {np.mean(auroc_values):.4f} ± {np.std(auroc_values):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.6, 0.2, 'Note: Full ROC curve requires\nper-query probability data',
            fontsize=8, style='italic', alpha=0.6)

    # PR curve placeholder
    ax = axes[1]
    ax.axhline(y=np.mean(auroc_values), color='k', linestyle='--', alpha=0.3, label='Baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve\nAUPRC = {np.mean(auprc_values):.4f} ± {np.std(auprc_values):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.2, 'Note: Full PR curve requires\nper-query probability data',
            fontsize=8, style='italic', alpha=0.6)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC/PR curves to {output_file}")


def plot_calibration(summary: Dict, output_file: Path):
    """Generate calibration plot."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Extract ECE values
    ece_values = []
    for fold_result in summary['fold_results']:
        ece = fold_result['test_metrics']['ne_gate']['ece']
        ece_values.append(ece)

    # Diagonal for perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    # Add text with ECE
    mean_ece = np.mean(ece_values)
    std_ece = np.std(ece_values)

    ax.text(0.05, 0.95, f'ECE = {mean_ece:.4f} ± {std_ece:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Fraction')
    ax.set_title('Calibration Plot (Isotonic Calibration)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.2, 'Note: Full calibration curve requires\nper-query probability data',
            fontsize=8, style='italic', alpha=0.6, ha='center')

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {output_file}")


def plot_tradeoff_curves(summary: Dict, output_file: Path):
    """Generate tradeoff curves."""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)

    # Extract threshold values across folds
    tau_neg_vals = []
    tau_pos_vals = []
    screening_sens = []
    screening_fpr_vals = []
    alert_prec = []
    alert_rec = []
    neg_rates = []
    pos_rates = []

    for fold_result in summary['fold_results']:
        thresh = fold_result['threshold_selection']
        test = fold_result['test_metrics']['deployment']

        tau_neg_vals.append(thresh['tau_neg'])
        tau_pos_vals.append(thresh['tau_pos'])
        screening_sens.append(test['screening_sensitivity'])
        screening_fpr_vals.append(test['screening_fpr'])
        alert_prec.append(test['alert_precision'])
        alert_rec.append(test['alert_recall'])
        neg_rates.append(test['neg_rate'])
        pos_rates.append(test['pos_rate'])

    # Plot 1: Sensitivity vs FPR tradeoff
    ax1 = fig.add_subplot(gs[0])
    mean_tau_neg = np.mean(tau_neg_vals)
    ax1.axvline(x=mean_tau_neg, color='r', linestyle='--', alpha=0.5,
                label=f'τ_neg = {mean_tau_neg:.4f}')
    ax1.scatter(tau_neg_vals, screening_sens, alpha=0.6, s=100)
    ax1.set_xlabel('Threshold τ_neg')
    ax1.set_ylabel('Screening Sensitivity')
    ax1.set_title('Screening Tier: Sensitivity vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Workload distribution
    ax2 = fig.add_subplot(gs[1])
    fold_ids = list(range(len(neg_rates)))
    width = 0.25
    ax2.bar(np.array(fold_ids) - width, neg_rates, width, label='NEG', alpha=0.7)
    ax2.bar(fold_ids, [1 - n - p for n, p in zip(neg_rates, pos_rates)], width, label='UNCERTAIN', alpha=0.7)
    ax2.bar(np.array(fold_ids) + width, pos_rates, width, label='POS', alpha=0.7)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Rate')
    ax2.set_title('Workload Distribution by State')
    ax2.set_xticks(fold_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Alert precision vs volume
    ax3 = fig.add_subplot(gs[2])
    ax3.scatter(pos_rates, alert_prec, alpha=0.6, s=100)
    ax3.set_xlabel('Alert Volume (POS Rate)')
    ax3.set_ylabel('Alert Precision')
    ax3.set_title('Alert Tier: Precision vs Volume')
    ax3.grid(True, alpha=0.3)

    # Annotate current operating point
    mean_pos_rate = np.mean(pos_rates)
    mean_alert_prec = np.mean(alert_prec)
    ax3.scatter([mean_pos_rate], [mean_alert_prec], color='red', s=200, marker='*',
                label=f'Current: {mean_pos_rate:.1%} volume, {mean_alert_prec:.1%} precision')
    ax3.legend()

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved tradeoff curves to {output_file}")


def plot_per_criterion_analysis(summary: Dict, output_file: Path):
    """Generate per-criterion performance analysis."""
    # Check if per_criterion_metrics exists
    if 'fold_results' not in summary or len(summary['fold_results']) == 0:
        print("Warning: No fold results found, skipping per-criterion analysis")
        return

    first_fold = summary['fold_results'][0]
    if 'per_criterion_metrics' not in first_fold:
        print("Warning: No per-criterion metrics found, skipping per-criterion analysis")
        return

    # Aggregate per-criterion metrics across folds
    all_criteria = set()
    for fold_result in summary['fold_results']:
        all_criteria.update(fold_result['per_criterion_metrics'].keys())

    criteria_data = {crit: {'auroc': [], 'sensitivity': [], 'precision': [], 'n_queries': []}
                     for crit in all_criteria}

    for fold_result in summary['fold_results']:
        for crit, metrics in fold_result['per_criterion_metrics'].items():
            criteria_data[crit]['auroc'].append(metrics['auroc'])
            criteria_data[crit]['sensitivity'].append(metrics['sensitivity_at_screening'])
            criteria_data[crit]['precision'].append(metrics['precision_at_alert'])
            criteria_data[crit]['n_queries'].append(metrics['n_queries_total'])

    # Sort criteria by name
    sorted_criteria = sorted(all_criteria)

    # Compute means
    auroc_means = [np.nanmean(criteria_data[c]['auroc']) for c in sorted_criteria]
    sens_means = [np.nanmean(criteria_data[c]['sensitivity']) for c in sorted_criteria]
    prec_means = [np.nanmean(criteria_data[c]['precision']) for c in sorted_criteria]
    n_queries = [np.mean(criteria_data[c]['n_queries']) for c in sorted_criteria]

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: AUROC by criterion
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_criteria)))
    bars1 = ax1.barh(sorted_criteria, auroc_means, color=colors, alpha=0.7)
    ax1.set_xlabel('AUROC')
    ax1.set_title('AUROC by Criterion')
    ax1.set_xlim([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: Sensitivity @ screening
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.barh(sorted_criteria, sens_means, color=colors, alpha=0.7)
    ax2.set_xlabel('Sensitivity @ Screening')
    ax2.set_title('Screening Sensitivity by Criterion')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Precision @ alert
    ax3 = fig.add_subplot(gs[1, 0])
    bars3 = ax3.barh(sorted_criteria, prec_means, color=colors, alpha=0.7)
    ax3.set_xlabel('Precision @ Alert')
    ax3.set_title('Alert Precision by Criterion')
    ax3.set_xlim([0, 1])
    ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Query counts
    ax4 = fig.add_subplot(gs[1, 1])
    bars4 = ax4.barh(sorted_criteria, n_queries, color=colors, alpha=0.7)
    ax4.set_xlabel('Number of Queries')
    ax4.set_title('Query Count by Criterion')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved per-criterion analysis to {output_file}")


def plot_dynamic_k_analysis(summary: Dict, output_file: Path):
    """Generate dynamic-K analysis plots."""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Collect data across folds
    all_mean_k = []
    all_neg_mean_k = []
    all_pos_mean_k = []
    all_neg_count = []
    all_pos_count = []

    for fold_result in summary['fold_results']:
        dk = fold_result['dynamic_k_stats']
        all_mean_k.append(dk['mean_k'])
        all_neg_mean_k.append(dk['NEG_mean_k'])
        all_pos_mean_k.append(dk['POS_mean_k'])
        all_neg_count.append(dk['NEG_count'])
        all_pos_count.append(dk['POS_count'])

    # Plot 1: Mean K by fold
    ax1 = fig.add_subplot(gs[0, 0])
    fold_ids = list(range(len(all_mean_k)))
    ax1.bar(fold_ids, all_mean_k, alpha=0.7)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Mean K')
    ax1.set_title('Mean Selected K by Fold')
    ax1.set_xticks(fold_ids)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: K by state (NEG vs POS)
    ax2 = fig.add_subplot(gs[0, 1])
    width = 0.35
    ax2.bar(np.array(fold_ids) - width/2, all_neg_mean_k, width, label='NEG', alpha=0.7)
    ax2.bar(np.array(fold_ids) + width/2, all_pos_mean_k, width, label='POS', alpha=0.7)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Mean K')
    ax2.set_title('Mean K by State')
    ax2.set_xticks(fold_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: State distribution
    ax3 = fig.add_subplot(gs[1, 0])
    total_counts = [n + p for n, p in zip(all_neg_count, all_pos_count)]
    neg_frac = [n / t for n, t in zip(all_neg_count, total_counts)]
    pos_frac = [p / t for p, t in zip(all_pos_count, total_counts)]
    ax3.bar(fold_ids, neg_frac, label='NEG', alpha=0.7)
    ax3.bar(fold_ids, pos_frac, bottom=neg_frac, label='POS', alpha=0.7)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Fraction')
    ax3.set_title('State Distribution by Fold')
    ax3.set_xticks(fold_ids)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: K statistics summary
    ax4 = fig.add_subplot(gs[1, 1])
    summary_data = {
        'Overall Mean K': np.mean(all_mean_k),
        'NEG Mean K': np.mean(all_neg_mean_k),
        'POS Mean K': np.mean(all_pos_mean_k),
    }
    colors_summary = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax4.barh(list(summary_data.keys()), list(summary_data.values()),
             color=colors_summary, alpha=0.7)
    ax4.set_xlabel('Mean K')
    ax4.set_title('Dynamic-K Summary Statistics')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved dynamic-K analysis to {output_file}")


def generate_all_plots(summary_path: Path, output_dir: Path):
    """Generate all visualization plots."""
    print(f"\nGenerating visualization plots from {summary_path}")
    print(f"Output directory: {output_dir}\n")

    # Load summary
    summary = load_summary(summary_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_roc_pr_curves(summary, output_dir / "roc_pr_curves.png")
    plot_calibration(summary, output_dir / "calibration_plot.png")
    plot_tradeoff_curves(summary, output_dir / "tradeoff_curves.png")
    plot_per_criterion_analysis(summary, output_dir / "per_criterion_analysis.png")
    plot_dynamic_k_analysis(summary, output_dir / "dynamic_k_analysis.png")

    print(f"\n✓ All plots generated successfully!")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots from clinical evaluation summary")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for plots"
    )

    args = parser.parse_args()

    if not args.summary.exists():
        print(f"Error: Summary file not found: {args.summary}")
        sys.exit(1)

    generate_all_plots(args.summary, args.output)


if __name__ == "__main__":
    main()
