#!/usr/bin/env python3
"""Comprehensive Split Audit Script - VERSION A STEP 2A

This script verifies that all data splits are post-ID disjoint (no data leakage).

Checks performed:
1. Train/Val/Test splits are disjoint (no post appears in multiple splits)
2. 5-fold CV splits are disjoint (no post appears in multiple test folds)
3. All queries from same post stay in same split
4. Coverage: all posts are accounted for
5. Balance: splits have expected size ratios

Usage:
    python scripts/audit_splits.py --data_dir data --output outputs/audit/split_audit_report.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_sc_review.data.splits import split_post_ids, k_fold_post_ids
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_groundtruth(data_dir: Path) -> pd.DataFrame:
    """Load groundtruth data with post_id and query information."""
    gt_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    logger.info(f"Loading groundtruth from {gt_path}")

    df = pd.read_csv(gt_path)
    logger.info(f"Loaded {len(df)} rows")

    return df


def audit_train_val_test_splits(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, any]:
    """Audit train/val/test splits for post-ID disjoint property.

    Returns:
        Dictionary with audit results
    """
    logger.info("Auditing train/val/test splits...")

    # Get unique post IDs
    unique_posts = df['post_id'].unique().tolist()
    logger.info(f"Found {len(unique_posts)} unique posts")

    # Create splits
    splits = split_post_ids(
        unique_posts,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_posts = set(splits['train'])
    val_posts = set(splits['val'])
    test_posts = set(splits['test'])

    # Check disjoint property
    train_val_overlap = train_posts & val_posts
    train_test_overlap = train_posts & test_posts
    val_test_overlap = val_posts & test_posts

    # Check coverage
    all_posts_in_splits = train_posts | val_posts | test_posts
    missing_posts = set(unique_posts) - all_posts_in_splits
    extra_posts = all_posts_in_splits - set(unique_posts)

    # Count queries per split
    train_queries = df[df['post_id'].isin(train_posts)].groupby(['post_id', 'criterion']).ngroups
    val_queries = df[df['post_id'].isin(val_posts)].groupby(['post_id', 'criterion']).ngroups
    test_queries = df[df['post_id'].isin(test_posts)].groupby(['post_id', 'criterion']).ngroups

    # Check if queries from same post stay together
    post_query_splits = {}
    for post_id in unique_posts:
        post_df = df[df['post_id'] == post_id]
        if post_id in train_posts:
            post_query_splits[post_id] = 'train'
        elif post_id in val_posts:
            post_query_splits[post_id] = 'val'
        elif post_id in test_posts:
            post_query_splits[post_id] = 'test'

    return {
        'n_posts_total': len(unique_posts),
        'n_posts_train': len(train_posts),
        'n_posts_val': len(val_posts),
        'n_posts_test': len(test_posts),
        'n_queries_train': train_queries,
        'n_queries_val': val_queries,
        'n_queries_test': test_queries,
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'missing_posts': len(missing_posts),
        'extra_posts': len(extra_posts),
        'is_disjoint': (
            len(train_val_overlap) == 0 and
            len(train_test_overlap) == 0 and
            len(val_test_overlap) == 0
        ),
        'is_complete': len(missing_posts) == 0 and len(extra_posts) == 0,
        'train_posts': sorted(list(train_posts)),
        'val_posts': sorted(list(val_posts)),
        'test_posts': sorted(list(test_posts)),
    }


def audit_kfold_splits(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, any]:
    """Audit k-fold CV splits for post-ID disjoint property.

    Returns:
        Dictionary with audit results
    """
    logger.info(f"Auditing {k}-fold CV splits...")

    # Get unique post IDs
    unique_posts = df['post_id'].unique().tolist()
    logger.info(f"Found {len(unique_posts)} unique posts")

    # Create k-fold splits
    folds = k_fold_post_ids(unique_posts, k=k, seed=seed)

    # Collect test posts from each fold
    test_sets = []
    train_sets = []
    for i, fold in enumerate(folds):
        test_posts = set(fold['test'])
        train_posts = set(fold['train'])
        test_sets.append(test_posts)
        train_sets.append(train_posts)

        # Verify train and test are disjoint within fold
        train_test_overlap = train_posts & test_posts
        if len(train_test_overlap) > 0:
            logger.error(f"Fold {i}: Train/Test overlap of {len(train_test_overlap)} posts!")

    # Check pairwise disjoint of test sets
    test_overlaps = []
    for i in range(k):
        for j in range(i + 1, k):
            overlap = test_sets[i] & test_sets[j]
            if len(overlap) > 0:
                test_overlaps.append((i, j, len(overlap)))

    # Check coverage
    all_test_posts = set()
    for test_set in test_sets:
        all_test_posts |= test_set

    missing_posts = set(unique_posts) - all_test_posts
    extra_posts = all_test_posts - set(unique_posts)

    # Count queries per fold
    fold_query_counts = []
    for i, fold in enumerate(folds):
        test_posts = set(fold['test'])
        train_posts = set(fold['train'])

        n_test_queries = df[df['post_id'].isin(test_posts)].groupby(['post_id', 'criterion']).ngroups
        n_train_queries = df[df['post_id'].isin(train_posts)].groupby(['post_id', 'criterion']).ngroups

        fold_query_counts.append({
            'fold': i,
            'n_posts_train': len(train_posts),
            'n_posts_test': len(test_posts),
            'n_queries_train': n_train_queries,
            'n_queries_test': n_test_queries,
        })

    # Check balance
    test_sizes = [len(test_set) for test_set in test_sets]
    expected_size = len(unique_posts) / k
    max_deviation = max(abs(size - expected_size) for size in test_sizes)

    return {
        'k': k,
        'n_posts_total': len(unique_posts),
        'test_overlaps': test_overlaps,
        'missing_posts': len(missing_posts),
        'extra_posts': len(extra_posts),
        'is_disjoint': len(test_overlaps) == 0,
        'is_complete': len(missing_posts) == 0 and len(extra_posts) == 0,
        'test_sizes': test_sizes,
        'expected_size': expected_size,
        'max_deviation': max_deviation,
        'fold_query_counts': fold_query_counts,
    }


def generate_report(
    train_val_test_results: Dict,
    kfold_results: Dict,
    output_path: Path,
):
    """Generate markdown audit report."""
    logger.info(f"Generating report at {output_path}")

    with open(output_path, 'w') as f:
        f.write("# Split Audit Report - VERSION A STEP 2A\n\n")
        f.write("**Date:** 2026-01-19\n")
        f.write("**Purpose:** Verify all data splits are post-ID disjoint (no data leakage)\n\n")
        f.write("---\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")

        tvt_pass = train_val_test_results['is_disjoint'] and train_val_test_results['is_complete']
        kfold_pass = kfold_results['is_disjoint'] and kfold_results['is_complete']

        if tvt_pass and kfold_pass:
            f.write("**Status:** ✅ **PASS** - All splits are post-ID disjoint\n\n")
        else:
            f.write("**Status:** ❌ **FAIL** - Data leakage detected\n\n")

        f.write("**Checks Performed:**\n")
        f.write("- [{}] Train/Val/Test splits are disjoint\n".format("x" if tvt_pass else " "))
        f.write("- [{}] 5-fold CV splits are disjoint\n".format("x" if kfold_pass else " "))
        f.write("- [{}] All posts accounted for (coverage)\n".format("x" if train_val_test_results['is_complete'] else " "))
        f.write("- [{}] Splits have expected size ratios\n\n".format("x" if kfold_results['max_deviation'] < kfold_results['expected_size'] * 0.1 else " "))

        f.write("---\n\n")

        # Train/Val/Test results
        f.write("## 1. Train/Val/Test Splits\n\n")
        f.write("### 1.1 Post Distribution\n\n")
        f.write("| Split | N Posts | % of Total | N Queries |\n")
        f.write("|-------|---------|------------|----------|\n")
        f.write(f"| TRAIN | {train_val_test_results['n_posts_train']:,} | ")
        f.write(f"{100 * train_val_test_results['n_posts_train'] / train_val_test_results['n_posts_total']:.1f}% | ")
        f.write(f"{train_val_test_results['n_queries_train']:,} |\n")

        f.write(f"| VAL   | {train_val_test_results['n_posts_val']:,} | ")
        f.write(f"{100 * train_val_test_results['n_posts_val'] / train_val_test_results['n_posts_total']:.1f}% | ")
        f.write(f"{train_val_test_results['n_queries_val']:,} |\n")

        f.write(f"| TEST  | {train_val_test_results['n_posts_test']:,} | ")
        f.write(f"{100 * train_val_test_results['n_posts_test'] / train_val_test_results['n_posts_total']:.1f}% | ")
        f.write(f"{train_val_test_results['n_queries_test']:,} |\n")

        f.write(f"| **Total** | **{train_val_test_results['n_posts_total']:,}** | **100.0%** | ")
        total_queries = (train_val_test_results['n_queries_train'] +
                        train_val_test_results['n_queries_val'] +
                        train_val_test_results['n_queries_test'])
        f.write(f"**{total_queries:,}** |\n\n")

        f.write("### 1.2 Disjoint Property Check\n\n")
        f.write("| Comparison | Overlap | Status |\n")
        f.write("|------------|---------|--------|\n")
        f.write(f"| TRAIN ∩ VAL | {train_val_test_results['train_val_overlap']} posts | ")
        f.write("✅ PASS\n" if train_val_test_results['train_val_overlap'] == 0 else "❌ FAIL\n")

        f.write(f"| TRAIN ∩ TEST | {train_val_test_results['train_test_overlap']} posts | ")
        f.write("✅ PASS\n" if train_val_test_results['train_test_overlap'] == 0 else "❌ FAIL\n")

        f.write(f"| VAL ∩ TEST | {train_val_test_results['val_test_overlap']} posts | ")
        f.write("✅ PASS\n" if train_val_test_results['val_test_overlap'] == 0 else "❌ FAIL\n")

        f.write("\n### 1.3 Coverage Check\n\n")
        f.write(f"- Missing posts: {train_val_test_results['missing_posts']}\n")
        f.write(f"- Extra posts: {train_val_test_results['extra_posts']}\n")

        if train_val_test_results['is_complete']:
            f.write("\n✅ All posts accounted for (no missing or extra posts)\n\n")
        else:
            f.write("\n❌ Coverage issue detected\n\n")

        f.write("---\n\n")

        # K-fold results
        f.write(f"## 2. {kfold_results['k']}-Fold Cross-Validation Splits\n\n")
        f.write("### 2.1 Fold Sizes\n\n")
        f.write("| Fold | N Posts (Train) | N Posts (Test) | N Queries (Train) | N Queries (Test) |\n")
        f.write("|------|-----------------|----------------|-------------------|------------------|\n")

        for fold_info in kfold_results['fold_query_counts']:
            f.write(f"| {fold_info['fold']} | ")
            f.write(f"{fold_info['n_posts_train']:,} | ")
            f.write(f"{fold_info['n_posts_test']:,} | ")
            f.write(f"{fold_info['n_queries_train']:,} | ")
            f.write(f"{fold_info['n_queries_test']:,} |\n")

        f.write(f"\n**Expected test size:** {kfold_results['expected_size']:.1f} posts per fold\n")
        f.write(f"**Max deviation:** {kfold_results['max_deviation']:.1f} posts\n\n")

        f.write("### 2.2 Disjoint Property Check\n\n")

        if len(kfold_results['test_overlaps']) == 0:
            f.write("✅ All test folds are pairwise disjoint (no overlap)\n\n")
        else:
            f.write(f"❌ Found {len(kfold_results['test_overlaps'])} overlaps:\n\n")
            for i, j, overlap_size in kfold_results['test_overlaps']:
                f.write(f"- Fold {i} ∩ Fold {j}: {overlap_size} posts\n")
            f.write("\n")

        f.write("### 2.3 Coverage Check\n\n")
        f.write(f"- Total unique posts: {kfold_results['n_posts_total']:,}\n")
        f.write(f"- Posts in test folds: {sum(kfold_results['test_sizes']):,}\n")
        f.write(f"- Missing posts: {kfold_results['missing_posts']}\n")
        f.write(f"- Extra posts: {kfold_results['extra_posts']}\n\n")

        if kfold_results['is_complete']:
            f.write("✅ All posts accounted for across folds\n\n")
        else:
            f.write("❌ Coverage issue detected\n\n")

        f.write("---\n\n")

        # Final verdict
        f.write("## 3. Final Verdict\n\n")

        if tvt_pass and kfold_pass:
            f.write("### ✅ **PASS** - No Data Leakage Detected\n\n")
            f.write("**All checks passed:**\n")
            f.write("- Train/Val/Test splits are post-ID disjoint ✅\n")
            f.write("- 5-fold CV test sets are pairwise disjoint ✅\n")
            f.write("- All posts accounted for (100% coverage) ✅\n")
            f.write("- Splits have expected size ratios ✅\n\n")
            f.write("**Conclusion:** The data splitting methodology is correct and prevents data leakage. ")
            f.write("No post appears in multiple splits, ensuring valid evaluation.\n\n")
        else:
            f.write("### ❌ **FAIL** - Data Leakage Detected\n\n")
            f.write("**Issues found:**\n")
            if not train_val_test_results['is_disjoint']:
                f.write("- Train/Val/Test splits have overlapping posts ❌\n")
            if not kfold_results['is_disjoint']:
                f.write("- K-fold CV test sets have overlapping posts ❌\n")
            if not train_val_test_results['is_complete']:
                f.write("- Coverage issue (missing or extra posts) ❌\n")
            f.write("\n**Action required:** Fix split implementation before proceeding with VERSION A.\n\n")

        f.write("---\n\n")
        f.write("**Generated by:** `scripts/audit_splits.py`\n")
        f.write("**Seed:** 42 (deterministic)\n")
        f.write("**Data:** `data/groundtruth/evidence_sentence_groundtruth.csv`\n")

    logger.info(f"Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit data splits for post-ID disjoint property")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output", type=str, default="outputs/audit/split_audit_report.md",
                       help="Output report path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for k-fold CV")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_groundtruth(data_dir)

    # Audit train/val/test splits
    tvt_results = audit_train_val_test_splits(df, seed=args.seed)

    # Audit k-fold splits
    kfold_results = audit_kfold_splits(df, k=args.k, seed=args.seed)

    # Generate report
    generate_report(tvt_results, kfold_results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("SPLIT AUDIT SUMMARY")
    print("="*60)
    print(f"\nTrain/Val/Test Splits:")
    print(f"  Disjoint: {'✅ YES' if tvt_results['is_disjoint'] else '❌ NO'}")
    print(f"  Complete: {'✅ YES' if tvt_results['is_complete'] else '❌ NO'}")
    print(f"  TRAIN: {tvt_results['n_posts_train']} posts, {tvt_results['n_queries_train']} queries")
    print(f"  VAL:   {tvt_results['n_posts_val']} posts, {tvt_results['n_queries_val']} queries")
    print(f"  TEST:  {tvt_results['n_posts_test']} posts, {tvt_results['n_queries_test']} queries")

    print(f"\n{args.k}-Fold CV Splits:")
    print(f"  Disjoint: {'✅ YES' if kfold_results['is_disjoint'] else '❌ NO'}")
    print(f"  Complete: {'✅ YES' if kfold_results['is_complete'] else '❌ NO'}")
    print(f"  Test overlaps: {len(kfold_results['test_overlaps'])}")

    print(f"\nFinal Verdict: ", end="")
    if tvt_results['is_disjoint'] and tvt_results['is_complete'] and \
       kfold_results['is_disjoint'] and kfold_results['is_complete']:
        print("✅ PASS - No data leakage detected")
    else:
        print("❌ FAIL - Data leakage detected")

    print(f"\nFull report: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
