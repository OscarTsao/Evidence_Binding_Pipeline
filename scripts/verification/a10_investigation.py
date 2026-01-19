#!/usr/bin/env python3
"""Investigation of A.10 (Suicidal ideation) criterion performance.

This script analyzes why A.10 has lower performance compared to other
MDD criteria and provides recommendations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_summary(summary_file: Path) -> Dict:
    """Load summary JSON."""
    with open(summary_file) as f:
        return json.load(f)


def extract_per_criterion_metrics(summary: Dict) -> pd.DataFrame:
    """Extract per-criterion metrics from all folds."""

    rows = []
    for fold_result in summary['fold_results']:
        fold = fold_result['fold']

        if 'per_criterion_metrics' not in fold_result:
            continue

        for criterion, metrics in fold_result['per_criterion_metrics'].items():
            row = {
                'fold': fold,
                'criterion': criterion,
                **metrics
            }
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_a10_performance(df: pd.DataFrame) -> Dict:
    """Analyze A.10 performance and compare to other criteria."""

    results = {
        'a10_statistics': {},
        'comparison_to_others': {},
        'potential_causes': [],
        'recommendations': []
    }

    # A.10 statistics across folds
    a10_df = df[df['criterion'] == 'A.10']

    results['a10_statistics'] = {
        'mean_auroc': float(a10_df['auroc'].mean()),
        'std_auroc': float(a10_df['auroc'].std()),
        'min_auroc': float(a10_df['auroc'].min()),
        'max_auroc': float(a10_df['auroc'].max()),
        'mean_auprc': float(a10_df['auprc'].mean()),
        'mean_baseline_rate': float(a10_df['baseline_rate'].mean()),
        'mean_n_with_evidence': float(a10_df['n_queries_with_evidence'].mean()),
        'total_queries': int(a10_df['n_queries_total'].iloc[0])
    }

    # Compare to other criteria
    other_criteria = df[df['criterion'] != 'A.10']

    # Group by criterion and compute mean AUROC
    criterion_auroc = other_criteria.groupby('criterion')['auroc'].mean().sort_values(ascending=False)

    results['comparison_to_others'] = {
        'a10_rank': len(criterion_auroc) + 1,  # A.10 is last
        'total_criteria': len(criterion_auroc) + 1,
        'best_criterion': criterion_auroc.index[0],
        'best_auroc': float(criterion_auroc.iloc[0]),
        'median_auroc': float(other_criteria.groupby('criterion')['auroc'].mean().median()),
        'a10_gap_from_median': float(results['a10_statistics']['mean_auroc'] - other_criteria.groupby('criterion')['auroc'].mean().median())
    }

    # Analyze potential causes
    a10_stats = results['a10_statistics']

    # 1. Class imbalance
    if a10_stats['mean_baseline_rate'] < 0.10:
        results['potential_causes'].append({
            'cause': 'Severe class imbalance',
            'evidence': f"A.10 positive rate is {a10_stats['mean_baseline_rate']:.1%} (only ~{a10_stats['mean_n_with_evidence']:.0f} positive samples per fold)",
            'impact': 'Limited training data makes it hard to learn discriminative features'
        })

    # 2. Low sample count
    if a10_stats['mean_n_with_evidence'] < 20:
        results['potential_causes'].append({
            'cause': 'Very low sample count',
            'evidence': f"Only ~{a10_stats['mean_n_with_evidence']:.0f} queries with evidence per fold",
            'impact': 'Insufficient data for reliable model training and calibration'
        })

    # 3. Criterion specificity
    results['potential_causes'].append({
        'cause': 'Criterion semantic complexity',
        'evidence': 'A.10 (Suicidal ideation) may use indirect/euphemistic language',
        'impact': 'Harder to detect with embedding-based models compared to explicit mood symptoms'
    })

    # 4. Performance variance
    if a10_stats['std_auroc'] > 0.05:
        results['potential_causes'].append({
            'cause': 'High cross-fold variance',
            'evidence': f"AUROC std = {a10_stats['std_auroc']:.4f} (range: {a10_stats['min_auroc']:.4f} - {a10_stats['max_auroc']:.4f})",
            'impact': 'Performance is unstable across folds, suggesting dataset sampling issues'
        })

    # Recommendations
    results['recommendations'] = [
        {
            'priority': 'HIGH',
            'recommendation': 'Collect more A.10 training data',
            'rationale': f"Current sample size (~{a10_stats['mean_n_with_evidence']:.0f} per fold) is too small for robust learning",
            'action': 'Target collection of posts with suicidal ideation mentions'
        },
        {
            'priority': 'HIGH',
            'recommendation': 'Use class weighting or focal loss',
            'rationale': f"Severe class imbalance ({a10_stats['mean_baseline_rate']:.1%} positive rate)",
            'action': 'Apply stronger loss weighting for A.10 during GNN training'
        },
        {
            'priority': 'MEDIUM',
            'recommendation': 'Criterion-specific fine-tuning',
            'rationale': f"A.10 AUROC ({a10_stats['mean_auroc']:.4f}) is {abs(results['comparison_to_others']['a10_gap_from_median']):.4f} below median",
            'action': 'Train separate model head for A.10 or use curriculum learning'
        },
        {
            'priority': 'MEDIUM',
            'recommendation': 'Add lexicon-based features',
            'rationale': 'Suicidal ideation has known keywords/phrases',
            'action': 'Incorporate rule-based features for suicide-related terms'
        },
        {
            'priority': 'LOW',
            'recommendation': 'External data augmentation',
            'rationale': 'Limited in-domain data',
            'action': 'Consider transfer learning from larger suicide detection datasets'
        }
    ]

    return results


def print_investigation_report(results: Dict, df: pd.DataFrame):
    """Print formatted investigation report."""

    print("\n" + "="*80)
    print("A.10 (Suicidal Ideation) CRITERION PERFORMANCE INVESTIGATION")
    print("="*80 + "\n")

    # Statistics
    stats = results['a10_statistics']
    print("A.10 STATISTICS:")
    print("-" * 80)
    print(f"Mean AUROC: {stats['mean_auroc']:.4f} ± {stats['std_auroc']:.4f}")
    print(f"AUROC Range: {stats['min_auroc']:.4f} - {stats['max_auroc']:.4f}")
    print(f"Mean AUPRC: {stats['mean_auprc']:.4f}")
    print(f"Positive Rate: {stats['mean_baseline_rate']:.2%}")
    print(f"Avg Queries with Evidence: {stats['mean_n_with_evidence']:.0f} per fold")
    print(f"Total Queries: {stats['total_queries']}")

    # Comparison
    comp = results['comparison_to_others']
    print("\nCOMPARISON TO OTHER CRITERIA:")
    print("-" * 80)
    print(f"A.10 Rank: {comp['a10_rank']} of {comp['total_criteria']} (WORST)")
    print(f"Best Criterion: {comp['best_criterion']} (AUROC: {comp['best_auroc']:.4f})")
    print(f"Median AUROC: {comp['median_auroc']:.4f}")
    print(f"A.10 Gap from Median: {comp['a10_gap_from_median']:.4f} (worse)")

    # All criteria ranking
    print("\nALL CRITERIA RANKING:")
    print("-" * 80)
    criterion_auroc = df.groupby('criterion')['auroc'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    print(f"{'Rank':<6} {'Criterion':<12} {'Mean AUROC':<15} {'Std':<10}")
    print("-" * 80)

    for rank, (criterion, row) in enumerate(criterion_auroc.iterrows(), 1):
        print(f"{rank:<6} {criterion:<12} {row['mean']:.4f}           {row['std']:.4f}")

    # Potential causes
    print("\n" + "="*80)
    print("POTENTIAL CAUSES OF LOW A.10 PERFORMANCE")
    print("="*80 + "\n")

    for i, cause in enumerate(results['potential_causes'], 1):
        print(f"{i}. {cause['cause']}")
        print(f"   Evidence: {cause['evidence']}")
        print(f"   Impact: {cause['impact']}")
        print()

    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*80 + "\n")

    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. [{rec['priority']}] {rec['recommendation']}")
        print(f"   Rationale: {rec['rationale']}")
        print(f"   Action: {rec['action']}")
        print()

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Investigate A.10 criterion performance"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file"
    )

    args = parser.parse_args()

    print("Loading summary...")
    summary = load_summary(args.summary)

    print("Extracting per-criterion metrics...")
    df = extract_per_criterion_metrics(summary)

    print(f"Loaded metrics for {df['criterion'].nunique()} criteria across {df['fold'].nunique()} folds\n")

    # Analyze A.10
    results = analyze_a10_performance(df)

    # Print report
    print_investigation_report(results, df)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nInvestigation results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
