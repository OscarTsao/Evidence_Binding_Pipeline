#!/usr/bin/env python3
"""Create stage-by-stage summary from existing clinical evaluation results.

This script provides a research-grade summary of the full pipeline performance,
showing performance at each conceptual stage.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_clinical_results(summary_file: Path) -> Dict:
    """Load clinical evaluation summary."""
    with open(summary_file) as f:
        return json.load(f)


def load_per_query_results(fold_results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all per-query results."""
    dfs = []
    for csv_file in sorted(fold_results_dir.glob("fold_*_predictions.csv")):
        df = pd.read_csv(csv_file)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def compute_stage_metrics(
    df: pd.DataFrame,
    summary: Dict
) -> Dict:
    """Compute metrics for each conceptual stage.

    Stages (conceptual - based on available data):
    - S_BASE: Baseline (prior = 9.34% positive rate)
    - S_P4_RAW: P4 GNN model (before calibration)
    - S_P4_CAL: P4 GNN model (after calibration)
    - S_GATE: 3-state clinical gate (deployment configuration)
    """

    results = {
        'meta': {
            'n_queries': len(df),
            'n_with_evidence': int(df['has_evidence_gold'].sum()),
            'baseline_positive_rate': float(df['has_evidence_gold'].mean())
        },
        'stages': {}
    }

    # S_BASE: Baseline (always predict prior)
    # This represents what happens with no model
    prior = df['has_evidence_gold'].mean()
    results['stages']['S_BASE_prior'] = {
        'description': 'Baseline: Always predict class prior (9.34%)',
        'auroc': 0.5,  # Random classifier
        'auprc': prior,  # Always predicting prior gives AUPRC = prior
        'accuracy': max(prior, 1 - prior)  # Majority class accuracy
    }

    # S_P4_RAW: P4 model before calibration
    labels = df['has_evidence_gold'].values
    probs_raw = df['p4_prob_raw'].values

    from sklearn.metrics import roc_auc_score, average_precision_score

    auroc_raw = roc_auc_score(labels, probs_raw)
    auprc_raw = average_precision_score(labels, probs_raw)

    results['stages']['S_P4_RAW'] = {
        'description': 'P4 GNN model (raw probabilities)',
        'auroc': float(auroc_raw),
        'auprc': float(auprc_raw)
    }

    # S_P4_CAL: P4 model after calibration
    probs_cal = df['p4_prob_calibrated'].values

    auroc_cal = roc_auc_score(labels, probs_cal)
    auprc_cal = average_precision_score(labels, probs_cal)

    # ECE (expected calibration error)
    def compute_ece(labels, probs, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            if in_bin.sum() > 0:
                prop_in_bin = in_bin.sum() / len(probs)
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    ece_raw = compute_ece(labels, probs_raw)
    ece_cal = compute_ece(labels, probs_cal)

    results['stages']['S_P4_CAL'] = {
        'description': 'P4 GNN model (calibrated probabilities)',
        'auroc': float(auroc_cal),
        'auprc': float(auprc_cal),
        'ece': float(ece_cal),
        'ece_improvement': float(ece_raw - ece_cal)
    }

    # S_GATE: Clinical 3-state gate (deployment)
    # From summary aggregated metrics
    results['stages']['S_GATE_deployment'] = {
        'description': '3-state clinical gate (NEG/UNCERTAIN/POS)',
        'auroc': summary['aggregated_metrics']['ne_gate.auroc']['mean'],
        'auprc': summary['aggregated_metrics']['ne_gate.auprc']['mean'],
        'screening_sensitivity': summary['aggregated_metrics']['deployment.screening_sensitivity']['mean'],
        'screening_fpr': summary['aggregated_metrics']['deployment.screening_fpr']['mean'],
        'alert_precision': summary['aggregated_metrics']['deployment.alert_precision']['mean'],
        'alert_recall': summary['aggregated_metrics']['deployment.alert_recall']['mean'],
        'neg_rate': summary['aggregated_metrics']['deployment.neg_rate']['mean'],
        'uncertain_rate': summary['aggregated_metrics']['deployment.uncertain_rate']['mean'],
        'pos_rate': summary['aggregated_metrics']['deployment.pos_rate']['mean']
    }

    # Evidence recall analysis (for queries with evidence)
    evidence_df = df[df['has_evidence_gold'] == 1].copy()

    if len(evidence_df) > 0:
        # Overall evidence recall
        evidence_recall = evidence_df['evidence_recall_at_k'].mean()
        evidence_precision = evidence_df['evidence_precision_at_k'].mean()
        mrr = evidence_df['mrr'].mean()

        results['evidence_retrieval'] = {
            'n_queries_with_evidence': len(evidence_df),
            'mean_evidence_recall': float(evidence_recall),
            'mean_evidence_precision': float(evidence_precision),
            'mean_mrr': float(mrr),
            'mean_selected_k': float(evidence_df['selected_k'].mean())
        }

        # By state
        for state in ['NEG', 'UNCERTAIN', 'POS']:
            state_df = evidence_df[evidence_df['state'] == state]
            if len(state_df) > 0:
                results['evidence_retrieval'][f'{state}_evidence_recall'] = float(
                    state_df['evidence_recall_at_k'].mean()
                )
                results['evidence_retrieval'][f'{state}_mean_k'] = float(
                    state_df['selected_k'].mean()
                )

    return results


def print_stage_summary(results: Dict):
    """Print formatted stage summary."""
    print("\n" + "="*80)
    print("STAGE-BY-STAGE PERFORMANCE SUMMARY")
    print("="*80 + "\n")

    print(f"Dataset: {results['meta']['n_queries']} queries, "
          f"{results['meta']['n_with_evidence']} with evidence "
          f"({results['meta']['baseline_positive_rate']:.2%})\n")

    print("CLASSIFICATION PERFORMANCE (NE Gate):")
    print("-" * 80)
    print(f"{'Stage':<25} {'AUROC':<12} {'AUPRC':<12} {'ECE':<12}")
    print("-" * 80)

    for stage_name, stage_data in results['stages'].items():
        auroc = stage_data.get('auroc', None)
        auprc = stage_data.get('auprc', None)
        ece = stage_data.get('ece', None)

        auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
        auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
        ece_str = f"{ece:.4f}" if ece is not None else "N/A"

        print(f"{stage_name:<25} {auroc_str:<12} {auprc_str:<12} {ece_str:<12}")
        print(f"  {stage_data['description']}")
        print()

    print("\nDEPLOYMENT METRICS (3-State Clinical Gate):")
    print("-" * 80)
    gate = results['stages']['S_GATE_deployment']
    print(f"Screening Sensitivity: {gate['screening_sensitivity']:.2%}")
    print(f"Screening FPR: {gate['screening_fpr']:.4f}")
    print(f"Alert Precision: {gate['alert_precision']:.2%}")
    print(f"Alert Recall: {gate['alert_recall']:.2%}")
    print(f"\nWorkload Distribution:")
    print(f"  NEG (skip): {gate['neg_rate']:.1%}")
    print(f"  UNCERTAIN (review): {gate['uncertain_rate']:.1%}")
    print(f"  POS (alert): {gate['pos_rate']:.1%}")

    if 'evidence_retrieval' in results:
        print("\n\nEVIDENCE RETRIEVAL PERFORMANCE:")
        print("-" * 80)
        ev = results['evidence_retrieval']
        print(f"Queries with evidence: {ev['n_queries_with_evidence']}")
        print(f"Mean Evidence Recall@K: {ev['mean_evidence_recall']:.2%}")
        print(f"Mean Evidence Precision@K: {ev['mean_evidence_precision']:.2%}")
        print(f"Mean MRR: {ev['mean_mrr']:.4f}")
        print(f"Mean Selected K: {ev['mean_selected_k']:.1f}")

        print("\nEvidence Recall by State:")
        for state in ['NEG', 'UNCERTAIN', 'POS']:
            key = f'{state}_evidence_recall'
            if key in ev:
                k_key = f'{state}_mean_k'
                print(f"  {state}: {ev[key]:.2%} (K={ev[k_key]:.1f})")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Create stage-by-stage summary"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json"
    )
    parser.add_argument(
        "--fold_results_dir",
        type=Path,
        required=True,
        help="Directory with fold CSV files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file"
    )

    args = parser.parse_args()

    print("Loading clinical evaluation results...")
    summary = load_clinical_results(args.summary)

    print("Loading per-query results...")
    df = load_per_query_results(args.fold_results_dir)

    print(f"Loaded {len(df)} query predictions\n")

    # Compute stage metrics
    results = compute_stage_metrics(df, summary)

    # Print summary
    print_stage_summary(results)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
