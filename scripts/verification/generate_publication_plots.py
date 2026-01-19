#!/usr/bin/env python3
"""Generate publication-quality visualizations for academic paper.

Creates comprehensive plots from per_query.csv for gold-standard evaluation.

Usage:
    python scripts/verification/generate_publication_plots.py \
        --per_query_csv outputs/final_research_eval/.../per_query.csv \
        --output_dir outputs/verification_recompute/publication_plots

Generates:
1. ROC curve with 95% CI band
2. Precision-Recall curve with baseline
3. Calibration reliability diagram
4. Confusion matrix heatmap
5. Per-criterion AUROC bar chart
6. Dynamic-K distribution analysis
7. Threshold sensitivity analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")


def plot_roc_curve_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "ROC Curve - Evidence Detection (P4 GNN)"
):
    """Plot ROC curve with 95% confidence interval using bootstrap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Compute main ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    # Bootstrap for confidence interval
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_aurocs = []

    tprs = []
    base_fpr = np.linspace(0, 1, 100)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        boot_auroc = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_aurocs.append(boot_auroc)

        # Compute ROC for this bootstrap sample
        boot_fpr, boot_tpr, _ = roc_curve(y_true[indices], y_prob[indices])
        tpr_interp = np.interp(base_fpr, boot_fpr, boot_tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)

    tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)

    # Plot CI band
    ax.fill_between(base_fpr, tpr_lower, tpr_upper, color='gray', alpha=0.2,
                     label='95% CI')

    # Plot main ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUROC = {auroc:.4f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUROC = 0.5)')

    # Compute 95% CI for AUROC
    auroc_lower = np.percentile(bootstrapped_aurocs, 2.5)
    auroc_upper = np.percentile(bootstrapped_aurocs, 97.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box with AUROC CI
    textstr = f'AUROC: {auroc:.4f}\n95% CI: [{auroc_lower:.4f}, {auroc_upper:.4f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.15, textstr, fontsize=11, bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved ROC curve to {output_path}")


def plot_pr_curve_with_baseline(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Precision-Recall Curve - Evidence Detection"
):
    """Plot PR curve with random baseline."""
    fig, ax = plt.subplots(figsize=(8, 7))

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Random baseline
    baseline = y_true.sum() / len(y_true)

    # Plot PR curve
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR Curve (AUPRC = {auprc:.4f})')

    # Plot baseline
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1,
               label=f'Random Baseline (AUPRC = {baseline:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box
    textstr = f'AUPRC: {auprc:.4f}\nBaseline: {baseline:.4f}\nLift: {auprc/baseline:.2f}×'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax.text(0.5, 0.85, textstr, fontsize=11, bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved PR curve to {output_path}")


def plot_calibration_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Calibration Reliability Diagram"
):
    """Plot calibration reliability diagram with ECE."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')

    # Compute ECE
    bin_counts = np.histogram(y_prob, bins=10, range=(0, 1))[0]
    total = len(y_prob)
    ece = np.sum(bin_counts / total * np.abs(prob_pred - prob_true))

    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, 's-', color='blue', markersize=8, lw=2,
            label=f'Model (ECE = {ece:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add ECE interpretation
    if ece < 0.05:
        interp = "Excellent"
    elif ece < 0.1:
        interp = "Good"
    elif ece < 0.15:
        interp = "Moderate"
    else:
        interp = "Poor"

    textstr = f'ECE: {ece:.4f}\nCalibration: {interp}'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax.text(0.6, 0.15, textstr, fontsize=11, bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved calibration diagram to {output_path}")


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix - Evidence Detection"
):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1, linecolor='black',
                xticklabels=['NEG', 'POS'], yticklabels=['NEG', 'POS'],
                ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    metrics_text = (
        f'Sensitivity: {sensitivity:.2%}\n'
        f'Specificity: {specificity:.2%}\n'
        f'Precision: {precision:.2%}\n'
        f'NPV: {npv:.2%}'
    )

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    ax.text(1.5, 0.5, metrics_text, fontsize=10, bbox=props,
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved confusion matrix to {output_path}")


def plot_per_criterion_auroc(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Per-Criterion AUROC - Evidence Detection"
):
    """Plot AUROC for each criterion."""
    criteria = sorted(df['criterion_id'].unique())

    aurocs = []
    criterion_labels = []
    n_positives = []

    for criterion in criteria:
        crit_df = df[df['criterion_id'] == criterion]
        y_true = crit_df['has_evidence_gold'].values
        y_prob = crit_df['p4_prob_calibrated'].values

        if y_true.sum() > 0:  # Only if there are positive examples
            auroc = roc_auc_score(y_true, y_prob)
            aurocs.append(auroc)
            criterion_labels.append(criterion)
            n_positives.append(y_true.sum())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color code by performance
    colors = ['green' if a >= 0.9 else 'orange' if a >= 0.85 else 'red' for a in aurocs]

    bars = ax.barh(criterion_labels, aurocs, color=colors, alpha=0.7, edgecolor='black')

    # Add AUROC values on bars
    for i, (bar, auroc, n_pos) in enumerate(zip(bars, aurocs, n_positives)):
        ax.text(auroc + 0.01, i, f'{auroc:.3f} (n={n_pos})',
                va='center', fontsize=9, fontweight='bold')

    # Add target line
    ax.axvline(x=0.85, color='red', linestyle='--', lw=1.5, label='Target (0.85)')
    ax.axvline(x=0.90, color='green', linestyle='--', lw=1.5, label='Excellent (0.90)')

    ax.set_xlim([0.75, 1.0])
    ax.set_xlabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Criterion', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved per-criterion AUROC to {output_path}")


def plot_dynamic_k_analysis(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Dynamic-K Selection Analysis"
):
    """Plot dynamic K distribution and analysis."""
    # Filter to queries with evidence
    df_evidence = df[df['has_evidence_gold'] == 1].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. K distribution histogram
    ax = axes[0, 0]
    selected_k = df_evidence['selected_k'].dropna()
    ax.hist(selected_k, bins=np.arange(selected_k.min()-0.5, selected_k.max()+1.5, 1),
            color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(selected_k.mean(), color='red', linestyle='--', lw=2,
               label=f'Mean = {selected_k.mean():.2f}')
    ax.axvline(selected_k.median(), color='green', linestyle='--', lw=2,
               label=f'Median = {selected_k.median():.1f}')
    ax.set_xlabel('Selected K', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('(A) K Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 2. K vs N (candidate count)
    ax = axes[0, 1]
    k_vals = df_evidence['selected_k'].dropna()
    n_vals = df_evidence['n_candidates'].dropna()
    ax.scatter(n_vals, k_vals, alpha=0.5, s=30, color='navy')

    # Add trend line
    z = np.polyfit(n_vals, k_vals, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(n_vals.min(), n_vals.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', lw=2, label=f'Trend: K = {z[0]:.2f}N + {z[1]:.2f}')

    ax.set_xlabel('N (Candidate Count)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Selected K', fontsize=11, fontweight='bold')
    ax.set_title('(B) K vs Candidate Count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Evidence Recall vs K
    ax = axes[1, 0]
    recall_data = df_evidence.groupby('selected_k')['evidence_recall_at_k'].mean()
    ax.plot(recall_data.index, recall_data.values, 'o-', color='green', lw=2, markersize=8)
    ax.set_xlabel('Selected K', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Evidence Recall', fontsize=11, fontweight='bold')
    ax.set_title('(C) Evidence Recall vs K', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # 4. K statistics table
    ax = axes[1, 1]
    ax.axis('off')

    stats_data = [
        ['Statistic', 'Value'],
        ['Mean K', f'{selected_k.mean():.2f}'],
        ['Median K', f'{selected_k.median():.1f}'],
        ['Mode K', f'{selected_k.mode().values[0]:.0f}'],
        ['Min K', f'{selected_k.min():.0f}'],
        ['Max K', f'{selected_k.max():.0f}'],
        ['P25', f'{selected_k.quantile(0.25):.1f}'],
        ['P75', f'{selected_k.quantile(0.75):.1f}'],
        ['P90', f'{selected_k.quantile(0.90):.1f}'],
        ['Std Dev', f'{selected_k.std():.2f}'],
    ]

    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(stats_data)):
        for j in range(2):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    ax.set_title('(D) K Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved dynamic-K analysis to {output_path}")


def plot_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "Threshold Sensitivity Analysis"
):
    """Plot how metrics vary with threshold."""
    thresholds = np.linspace(0, 1, 100)

    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        precisions.append(precision)
        f1_scores.append(f1)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(thresholds, sensitivities, label='Sensitivity (TPR)', lw=2, color='blue')
    ax.plot(thresholds, specificities, label='Specificity (TNR)', lw=2, color='green')
    ax.plot(thresholds, precisions, label='Precision (PPV)', lw=2, color='red')
    ax.plot(thresholds, f1_scores, label='F1 Score', lw=2, color='purple')

    # Mark optimal F1 threshold
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_f1_idx]
    ax.axvline(optimal_thresh, color='black', linestyle='--', lw=1.5,
               label=f'Optimal F1 Threshold ({optimal_thresh:.3f})')

    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved threshold sensitivity to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Publication-Quality Plots')
    parser.add_argument('--per_query_csv', type=str, required=True,
                        help='Path to per_query.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for plots')

    args = parser.parse_args()

    per_query_csv = Path(args.per_query_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("PUBLICATION-QUALITY VISUALIZATION GENERATOR")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading {per_query_csv}...")
    df = pd.read_csv(per_query_csv)
    print(f"  Loaded {len(df)} queries\n")

    # Extract data
    y_true = df['has_evidence_gold'].values
    y_prob = df['p4_prob_calibrated'].values

    # Use default threshold of 0.5 for binary predictions
    y_pred = (y_prob >= 0.5).astype(int)

    # Generate plots
    print("Generating plots...\n")

    plot_roc_curve_with_ci(y_true, y_prob, output_dir / '1_roc_curve_with_ci.png')
    plot_pr_curve_with_baseline(y_true, y_prob, output_dir / '2_pr_curve_with_baseline.png')
    plot_calibration_diagram(y_true, y_prob, output_dir / '3_calibration_diagram.png')
    plot_confusion_matrix_heatmap(y_true, y_pred, output_dir / '4_confusion_matrix.png')
    plot_per_criterion_auroc(df, output_dir / '5_per_criterion_auroc.png')
    plot_dynamic_k_analysis(df, output_dir / '6_dynamic_k_analysis.png')
    plot_threshold_sensitivity(y_true, y_prob, output_dir / '7_threshold_sensitivity.png')

    print(f"\n{'='*80}")
    print(f"✅ ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Total plots: 7")
    print(f"\nPlots created:")
    for i, name in enumerate([
        '1_roc_curve_with_ci.png',
        '2_pr_curve_with_baseline.png',
        '3_calibration_diagram.png',
        '4_confusion_matrix.png',
        '5_per_criterion_auroc.png',
        '6_dynamic_k_analysis.png',
        '7_threshold_sensitivity.png'
    ], 1):
        print(f"  {i}. {name}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
