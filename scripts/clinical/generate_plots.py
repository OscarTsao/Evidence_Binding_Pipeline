#!/usr/bin/env python3
"""Generate visualization plots from clinical evaluation results.

This script creates visualization plots for clinical deployment review:
1. ROC/PR curves (from real per-query predictions)
2. Calibration plot (reliability diagram)
3. Tradeoff curves
4. Per-criterion analysis
5. Dynamic-K analysis

Usage:
    # From per_query.csv (recommended - generates real curves):
    python scripts/clinical/generate_plots.py \\
        --per_query outputs/final_research_eval/20260118_031312_complete/per_query.csv \\
        --output outputs/clinical/plots

    # From summary.json (legacy - shows aggregated metrics only):
    python scripts/clinical/generate_plots.py \\
        --summary outputs/clinical_high_recall/YYYYMMDD_HHMMSS/summary.json \\
        --output outputs/clinical_high_recall/YYYYMMDD_HHMMSS/curves
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Optional sklearn imports for curve computation
try:
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
        roc_auc_score, calibration_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 9


def load_per_query(per_query_path: Path) -> pd.DataFrame:
    """Load per_query.csv file."""
    return pd.read_csv(per_query_path)


def load_summary(summary_path: Path) -> Dict:
    """Load summary.json file."""
    with open(summary_path) as f:
        return json.load(f)


def plot_roc_pr_curves_from_predictions(
    df: pd.DataFrame,
    output_file: Path,
    prob_col: str = "p4_prob_calibrated",
    label_col: str = "has_evidence_gold",
):
    """Generate ROC and PR curves from actual per-query predictions.

    Uses sklearn to compute proper ROC and PR curves from raw predictions.
    """
    if not HAS_SKLEARN:
        print("Warning: sklearn not available, cannot generate real curves")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get predictions and labels
    y_true = df[label_col].values
    y_score = df[prob_col].values

    # Filter out any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]

    # Compute ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random', alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_roc[optimal_idx]
    ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, zorder=5,
               label=f'Optimal (threshold={optimal_threshold:.3f})')
    ax.legend(loc="lower right")

    # Compute PR curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.mean()  # Random baseline for PR

    # Plot PR curve
    ax = axes[1]
    ax.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUPRC = {ap:.4f})')
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2, label=f'Random ({baseline:.4f})', alpha=0.5)
    ax.fill_between(recall, precision, alpha=0.2, color='green')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add statistics
    n_pos = y_true.sum()
    n_total = len(y_true)
    fig.suptitle(f'Classification Performance (n={n_total:,}, positive rate={n_pos/n_total:.1%})', fontsize=12)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC/PR curves to {output_file}")

    return {"auroc": roc_auc, "auprc": ap, "optimal_threshold": optimal_threshold}


def plot_calibration_from_predictions(
    df: pd.DataFrame,
    output_file: Path,
    prob_col: str = "p4_prob_calibrated",
    label_col: str = "has_evidence_gold",
    n_bins: int = 10,
):
    """Generate calibration (reliability) plot from actual predictions."""
    if not HAS_SKLEARN:
        print("Warning: sklearn not available, cannot generate calibration plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get predictions and labels
    y_true = df[label_col].values
    y_prob = df[prob_col].values

    # Filter out any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    # Compute ECE (Expected Calibration Error)
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_weight = bin_counts / bin_counts.sum()
    ece = np.sum(bin_weight[:len(fraction_of_positives)] * np.abs(fraction_of_positives - mean_predicted_value))

    # Compute Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    # Plot 1: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', color='blue', lw=2,
            label=f'Calibrated model (ECE={ece:.4f})')
    ax.fill_between(mean_predicted_value, mean_predicted_value, fraction_of_positives,
                    alpha=0.2, color='blue')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Plot (Reliability Diagram)')
    ax.legend(loc='upper left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add ECE and Brier in text box
    textstr = f'ECE = {ece:.4f}\nBrier = {brier:.4f}'
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Histogram of predicted probabilities
    ax = axes[1]
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label='Negative', density=True, color='blue')
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label='Positive', density=True, color='orange')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Predicted Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot to {output_file}")

    return {"ece": ece, "brier": brier}


def plot_per_criterion_from_predictions(
    df: pd.DataFrame,
    output_file: Path,
    prob_col: str = "p4_prob_calibrated",
    label_col: str = "has_evidence_gold",
):
    """Generate per-criterion analysis from per-query predictions."""
    if not HAS_SKLEARN:
        print("Warning: sklearn not available for per-criterion analysis")
        return

    # Group by criterion
    criteria = sorted(df['criterion_id'].unique())

    aurocs = []
    auprcs = []
    n_queries = []
    positive_rates = []

    for crit in criteria:
        subset = df[df['criterion_id'] == crit]
        y_true = subset[label_col].values
        y_score = subset[prob_col].values

        # Filter NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_score))
        y_true = y_true[mask]
        y_score = y_score[mask]

        if len(y_true) > 0 and y_true.sum() > 0:
            aurocs.append(roc_auc_score(y_true, y_score))
            auprcs.append(average_precision_score(y_true, y_score))
        else:
            aurocs.append(np.nan)
            auprcs.append(np.nan)

        n_queries.append(len(subset))
        positive_rates.append(y_true.mean() if len(y_true) > 0 else 0)

    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, len(criteria)))

    # Plot 1: AUROC by criterion
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(criteria, aurocs, color=colors, alpha=0.7)
    ax1.axvline(x=np.nanmean(aurocs), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.nanmean(aurocs):.3f}')
    ax1.set_xlabel('AUROC')
    ax1.set_title('AUROC by Criterion')
    ax1.set_xlim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    # Plot 2: AUPRC by criterion
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.barh(criteria, auprcs, color=colors, alpha=0.7)
    ax2.axvline(x=np.nanmean(auprcs), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.nanmean(auprcs):.3f}')
    ax2.set_xlabel('AUPRC')
    ax2.set_title('AUPRC by Criterion')
    ax2.set_xlim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Positive rate by criterion
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.barh(criteria, positive_rates, color=colors, alpha=0.7)
    ax3.set_xlabel('Positive Rate')
    ax3.set_title('Positive Rate by Criterion')
    ax3.set_xlim([0, max(positive_rates) * 1.1 if positive_rates else 1])
    ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Query counts
    ax4 = fig.add_subplot(gs[1, 1])
    bars = ax4.barh(criteria, n_queries, color=colors, alpha=0.7)
    ax4.set_xlabel('Number of Queries')
    ax4.set_title('Query Count by Criterion')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved per-criterion analysis to {output_file}")


def plot_roc_pr_curves_from_summary(summary: Dict, output_file: Path):
    """Generate ROC and PR curves from aggregated summary (legacy).

    This function displays aggregated AUROC/AUPRC values when raw predictions
    are not available. For proper curves, use --per_query instead.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract AUROC and AUPRC values from folds
    auroc_values = []
    auprc_values = []

    for fold_result in summary.get('fold_results', []):
        ne_gate = fold_result.get('test_metrics', {}).get('ne_gate', {})
        auroc = ne_gate.get('auroc', 0)
        auprc = ne_gate.get('auprc', 0)
        auroc_values.append(auroc)
        auprc_values.append(auprc)

    mean_auroc = np.mean(auroc_values) if auroc_values else 0
    mean_auprc = np.mean(auprc_values) if auprc_values else 0

    # ROC placeholder
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve\nAUROC = {mean_auroc:.4f} ± {np.std(auroc_values):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.3, 'Aggregated metrics only.\nFor full curves, use:\n--per_query per_query.csv',
            fontsize=9, style='italic', alpha=0.7, ha='center',
            transform=ax.transAxes)

    # PR placeholder
    ax = axes[1]
    ax.axhline(y=mean_auprc, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve\nAUPRC = {mean_auprc:.4f} ± {np.std(auprc_values):.4f}')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.3, 'Aggregated metrics only.\nFor full curves, use:\n--per_query per_query.csv',
            fontsize=9, style='italic', alpha=0.7, ha='center',
            transform=ax.transAxes)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC/PR curves (aggregated) to {output_file}")


def plot_calibration_from_summary(summary: Dict, output_file: Path):
    """Generate calibration plot from aggregated summary (legacy)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Extract ECE values
    ece_values = []
    for fold_result in summary.get('fold_results', []):
        ne_gate = fold_result.get('test_metrics', {}).get('ne_gate', {})
        ece = ne_gate.get('ece', 0)
        ece_values.append(ece)

    mean_ece = np.mean(ece_values) if ece_values else 0
    std_ece = np.std(ece_values) if ece_values else 0

    # Diagonal for perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    ax.text(0.05, 0.95, f'ECE = {mean_ece:.4f} ± {std_ece:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Fraction')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.5, 0.3, 'Aggregated metrics only.\nFor full calibration curve,\nuse --per_query per_query.csv',
            fontsize=9, style='italic', alpha=0.7, ha='center',
            transform=ax.transAxes)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration plot (aggregated) to {output_file}")


def plot_tradeoff_curves(summary: Dict, output_file: Path):
    """Generate tradeoff curves from summary."""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3)

    # Extract threshold values across folds
    tau_neg_vals = []
    tau_pos_vals = []
    screening_sens = []
    alert_prec = []
    neg_rates = []
    pos_rates = []

    for fold_result in summary.get('fold_results', []):
        thresh = fold_result.get('threshold_selection', {})
        test = fold_result.get('test_metrics', {}).get('deployment', {})

        tau_neg_vals.append(thresh.get('tau_neg', 0))
        tau_pos_vals.append(thresh.get('tau_pos', 1))
        screening_sens.append(test.get('screening_sensitivity', 0))
        alert_prec.append(test.get('alert_precision', 0))
        neg_rates.append(test.get('neg_rate', 0))
        pos_rates.append(test.get('pos_rate', 0))

    if not tau_neg_vals:
        print("Warning: No fold results found for tradeoff curves")
        return

    # Plot 1: Sensitivity vs Threshold
    ax1 = fig.add_subplot(gs[0])
    mean_tau_neg = np.mean(tau_neg_vals)
    ax1.axvline(x=mean_tau_neg, color='r', linestyle='--', alpha=0.5,
                label=f'Mean τ_neg = {mean_tau_neg:.4f}')
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
    uncertain_rates = [1 - n - p for n, p in zip(neg_rates, pos_rates)]
    ax2.bar(np.array(fold_ids) - width, neg_rates, width, label='NEG', alpha=0.7)
    ax2.bar(fold_ids, uncertain_rates, width, label='UNCERTAIN', alpha=0.7)
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

    mean_pos_rate = np.mean(pos_rates)
    mean_alert_prec = np.mean(alert_prec)
    ax3.scatter([mean_pos_rate], [mean_alert_prec], color='red', s=200, marker='*',
                label=f'Mean: {mean_pos_rate:.1%} volume, {mean_alert_prec:.1%} precision')
    ax3.legend()

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved tradeoff curves to {output_file}")


def generate_plots_from_per_query(per_query_path: Path, output_dir: Path):
    """Generate all plots from per_query.csv (recommended approach)."""
    print(f"\nGenerating plots from per-query predictions: {per_query_path}")
    print(f"Output directory: {output_dir}\n")

    # Load data
    df = load_per_query(per_query_path)
    print(f"Loaded {len(df)} queries")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    metrics = {}

    roc_pr_metrics = plot_roc_pr_curves_from_predictions(
        df, output_dir / "roc_pr_curves.png"
    )
    if roc_pr_metrics:
        metrics.update(roc_pr_metrics)

    cal_metrics = plot_calibration_from_predictions(
        df, output_dir / "calibration_plot.png"
    )
    if cal_metrics:
        metrics.update(cal_metrics)

    plot_per_criterion_from_predictions(
        df, output_dir / "per_criterion_analysis.png"
    )

    # Save metrics summary
    if metrics:
        with open(output_dir / "plot_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"\n✓ All plots generated successfully!")
    print(f"  Output directory: {output_dir}")


def generate_plots_from_summary(summary_path: Path, output_dir: Path):
    """Generate plots from summary.json (legacy approach)."""
    print(f"\nGenerating plots from summary: {summary_path}")
    print(f"Output directory: {output_dir}")
    print("\nNote: For proper ROC/PR curves, use --per_query instead\n")

    # Load summary
    summary = load_summary(summary_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_roc_pr_curves_from_summary(summary, output_dir / "roc_pr_curves.png")
    plot_calibration_from_summary(summary, output_dir / "calibration_plot.png")
    plot_tradeoff_curves(summary, output_dir / "tradeoff_curves.png")

    print(f"\n✓ Plots generated (aggregated metrics only)")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from evaluation results"
    )
    parser.add_argument(
        "--per_query",
        type=Path,
        help="Path to per_query.csv file (recommended - generates real curves)"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Path to summary.json file (legacy - aggregated metrics only)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for plots"
    )

    args = parser.parse_args()

    if args.per_query:
        if not args.per_query.exists():
            print(f"Error: per_query file not found: {args.per_query}")
            sys.exit(1)
        generate_plots_from_per_query(args.per_query, args.output)
    elif args.summary:
        if not args.summary.exists():
            print(f"Error: Summary file not found: {args.summary}")
            sys.exit(1)
        generate_plots_from_summary(args.summary, args.output)
    else:
        print("Error: Must provide either --per_query or --summary")
        sys.exit(1)


if __name__ == "__main__":
    main()
