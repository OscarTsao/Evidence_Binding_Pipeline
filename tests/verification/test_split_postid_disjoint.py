"""
Comprehensive test to verify splits are Post-ID disjoint.

This test ensures no post appears in multiple splits (TRAIN/TUNE/TEST),
which would constitute data leakage.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.data.splits import create_stratified_splits


def test_splits_are_postid_disjoint():
    """Verify that all splits (TRAIN/TUNE/TEST) have disjoint post IDs."""

    # Create splits with known seed for reproducibility
    splits = create_stratified_splits(
        data_dir=Path("data"),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )

    train_posts = set(splits['TRAIN']['post_id'])
    tune_posts = set(splits['TUNE']['post_id'])
    test_posts = set(splits['TEST']['post_id'])

    # Check no overlap between TRAIN and TUNE
    train_tune_overlap = train_posts & tune_posts
    assert len(train_tune_overlap) == 0, (
        f"Found {len(train_tune_overlap)} posts in both TRAIN and TUNE: "
        f"{list(train_tune_overlap)[:5]}"
    )

    # Check no overlap between TRAIN and TEST
    train_test_overlap = train_posts & test_posts
    assert len(train_test_overlap) == 0, (
        f"Found {len(train_test_overlap)} posts in both TRAIN and TEST: "
        f"{list(train_test_overlap)[:5]}"
    )

    # Check no overlap between TUNE and TEST
    tune_test_overlap = tune_posts & test_posts
    assert len(tune_test_overlap) == 0, (
        f"Found {len(tune_test_overlap)} posts in both TUNE and TEST: "
        f"{list(tune_test_overlap)[:5]}"
    )

    # Verify all posts are accounted for
    all_posts_in_splits = train_posts | tune_posts | test_posts

    # All posts should be in exactly one split
    total_posts = len(train_posts) + len(tune_posts) + len(test_posts)
    assert total_posts == len(all_posts_in_splits), (
        f"Post count mismatch: {total_posts} posts in splits but "
        f"{len(all_posts_in_splits)} unique posts"
    )

    print(f"✅ Splits are Post-ID disjoint:")
    print(f"   TRAIN: {len(train_posts)} posts")
    print(f"   TUNE: {len(tune_posts)} posts")
    print(f"   TEST: {len(test_posts)} posts")
    print(f"   Total: {len(all_posts_in_splits)} unique posts")


def test_5fold_splits_are_postid_disjoint():
    """Verify that in 5-fold CV, each fold's test set is disjoint from others."""

    from final_sc_review.data.splits import create_kfold_splits

    folds = create_kfold_splits(
        data_dir=Path("data"),
        n_folds=5,
        seed=42
    )

    # Collect test posts from each fold
    test_sets = []
    for fold_id in range(5):
        test_posts = set(folds[fold_id]['TEST']['post_id'])
        test_sets.append(test_posts)

    # Check pairwise disjoint
    for i in range(5):
        for j in range(i + 1, 5):
            overlap = test_sets[i] & test_sets[j]
            assert len(overlap) == 0, (
                f"Found {len(overlap)} posts in both fold {i} and fold {j} test sets: "
                f"{list(overlap)[:5]}"
            )

    # Check union covers all posts
    all_test_posts = set()
    for test_set in test_sets:
        all_test_posts |= test_set

    # Each fold should have ~20% of data
    expected_size = len(all_test_posts) // 5
    for i, test_set in enumerate(test_sets):
        size_diff = abs(len(test_set) - expected_size)
        assert size_diff <= expected_size * 0.1, (
            f"Fold {i} test set size {len(test_set)} differs from expected {expected_size}"
        )

    print(f"✅ 5-fold CV splits are Post-ID disjoint:")
    for i, test_set in enumerate(test_sets):
        print(f"   Fold {i} TEST: {len(test_set)} posts")
    print(f"   Total unique: {len(all_test_posts)} posts")


if __name__ == '__main__':
    test_splits_are_postid_disjoint()
    test_5fold_splits_are_postid_disjoint()
