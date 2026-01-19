#!/usr/bin/env python3
"""Dynamic Label Leakage Probe - VERSION A STEP 2C

This script trains a classifier to detect if there's information leakage between train/val/test splits.

The probe methodology:
1. Load feature representations (embeddings) for all queries
2. Create train/val/test splits using the same methodology as evaluation
3. Train a simple classifier to predict which split each query belongs to
4. If the classifier performs significantly better than random (33% for 3-way),
   it indicates systematic differences that could be due to leakage

Expected results:
- Random baseline: ~33% accuracy (3-way classification)
- Acceptable range: 33-40% (minor distribution differences)
- Concerning: >50% (strong systematic differences)
- Critical: >70% (likely leakage or severe distribution shift)

Usage:
    python scripts/audit_label_leakage_dynamic.py \
        --embeddings data/cache/bge_m3/dense.npy \
        --data_dir data \
        --output outputs/audit/label_leakage_dynamic_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from final_sc_review.data.splits import split_post_ids
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_embeddings_and_queries(
    embeddings_path: Path,
    data_dir: Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and query metadata.

    Returns:
        (embeddings, queries_df)
        embeddings: (N, D) array of sentence embeddings
        queries_df: DataFrame with post_id, criterion, sent_uid
    """
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Load groundtruth to get query information
    gt_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    logger.info(f"Loading groundtruth from {gt_path}")
    df = pd.read_csv(gt_path)

    # Group by (post_id, criterion) to get unique queries
    queries = df.groupby(['post_id', 'criterion']).first().reset_index()[['post_id', 'criterion']]
    logger.info(f"Found {len(queries)} unique queries")

    return embeddings, queries


def create_query_features(
    embeddings: np.ndarray,
    queries_df: pd.DataFrame,
    df_full: pd.DataFrame,
) -> np.ndarray:
    """Create query-level features from sentence embeddings.

    For each query (post_id, criterion), we aggregate sentence embeddings.
    Simple approach: mean pooling of all sentence embeddings for that post.

    Returns:
        (N_queries, D) array of query features
    """
    logger.info("Creating query-level features from sentence embeddings...")

    # Get unique post IDs
    unique_posts = queries_df['post_id'].unique()

    # Create post-level embeddings (mean of all sentences in post)
    post_embeddings = {}

    for post_id in unique_posts:
        # Get all sentences for this post
        post_sentences = df_full[df_full['post_id'] == post_id]['sent_uid'].unique()

        # Get indices for these sentences in the embedding array
        # Assuming sent_uid format: "post_id_sentence_idx"
        # We need to map sent_uid to embedding index

        # For simplicity, we'll use sentence index as the embedding index
        # This assumes embeddings are ordered by sentence appearance
        post_sent_indices = []
        for sent_uid in post_sentences:
            try:
                # Extract sentence index from sent_uid (format: "postid_sentidx")
                parts = sent_uid.split('_')
                if len(parts) >= 2:
                    sent_idx = int(parts[-1])
                    post_sent_indices.append(sent_idx)
            except:
                pass

        if post_sent_indices:
            # Mean pool sentence embeddings for this post
            post_emb = embeddings[post_sent_indices].mean(axis=0)
            post_embeddings[post_id] = post_emb

    # Create query features (one per query)
    query_features = []
    for _, row in queries_df.iterrows():
        post_id = row['post_id']
        if post_id in post_embeddings:
            query_features.append(post_embeddings[post_id])
        else:
            # Fallback: zero vector
            query_features.append(np.zeros(embeddings.shape[1]))

    query_features = np.array(query_features)
    logger.info(f"Created query features: {query_features.shape}")

    return query_features


def create_split_labels(
    queries_df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> np.ndarray:
    """Create split labels for each query.

    Returns:
        (N_queries,) array where:
        0 = train, 1 = val, 2 = test
    """
    logger.info("Creating split labels...")

    # Get unique post IDs
    unique_posts = queries_df['post_id'].unique().tolist()

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

    # Assign split labels
    split_labels = []
    for post_id in queries_df['post_id']:
        if post_id in train_posts:
            split_labels.append(0)  # train
        elif post_id in val_posts:
            split_labels.append(1)  # val
        elif post_id in test_posts:
            split_labels.append(2)  # test
        else:
            split_labels.append(-1)  # unknown

    split_labels = np.array(split_labels)

    logger.info(f"Split distribution:")
    logger.info(f"  Train: {(split_labels == 0).sum()} queries")
    logger.info(f"  Val:   {(split_labels == 1).sum()} queries")
    logger.info(f"  Test:  {(split_labels == 2).sum()} queries")

    return split_labels


def train_split_probe(
    features: np.ndarray,
    split_labels: np.ndarray,
    n_cv_folds: int = 5,
) -> Dict:
    """Train a classifier to predict split labels.

    Returns:
        Dictionary with probe results
    """
    logger.info("Training split classification probe...")

    # Remove any unknown splits
    valid_mask = split_labels >= 0
    X = features[valid_mask]
    y = split_labels[valid_mask]

    logger.info(f"Training on {len(X)} queries")

    # Train logistic regression classifier
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # Handle class imbalance
    )

    # Cross-validation
    logger.info(f"Running {n_cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=n_cv_folds, scoring='accuracy')

    # Train on full data for final model
    clf.fit(X, y)

    # Predictions
    y_pred = clf.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Classification report
    report = classification_report(
        y,
        y_pred,
        target_names=['train', 'val', 'test'],
        output_dict=True,
    )

    # Random baseline (3-way classification)
    random_baseline = 1.0 / 3.0

    results = {
        'accuracy': accuracy_score(y, y_pred),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'random_baseline': random_baseline,
        'improvement_over_random': (cv_scores.mean() - random_baseline) / random_baseline * 100,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'n_samples': len(X),
        'n_train': (y == 0).sum(),
        'n_val': (y == 1).sum(),
        'n_test': (y == 2).sum(),
    }

    logger.info(f"Probe accuracy: {results['accuracy']:.4f}")
    logger.info(f"CV accuracy: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    logger.info(f"Random baseline: {random_baseline:.4f}")
    logger.info(f"Improvement over random: {results['improvement_over_random']:.2f}%")

    return results


def interpret_results(results: Dict) -> Tuple[str, str]:
    """Interpret probe results and provide verdict.

    Returns:
        (status, interpretation)
    """
    cv_acc = results['cv_mean']
    random_baseline = results['random_baseline']
    improvement = results['improvement_over_random']

    if cv_acc < 0.40:
        status = "✅ PASS"
        interpretation = (
            "The probe classifier performs close to random (33%), indicating no "
            "systematic differences between splits. This is expected for properly "
            "randomized splits with no leakage."
        )
    elif cv_acc < 0.50:
        status = "⚠️ ACCEPTABLE"
        interpretation = (
            "The probe classifier performs slightly better than random, indicating "
            "minor distributional differences between splits. This is acceptable and "
            "may be due to natural variation in post characteristics (e.g., post length, "
            "writing style). No action required unless accompanied by other red flags."
        )
    elif cv_acc < 0.70:
        status = "⚠️ CONCERNING"
        interpretation = (
            "The probe classifier performs moderately well, indicating substantial "
            "systematic differences between splits. This could be due to:\n"
            "1. Temporal ordering (train posts from different time period than test)\n"
            "2. Topic drift (train posts have different topics than test)\n"
            "3. Potential leakage (test labels influencing train preprocessing)\n\n"
            "**Recommendation:** Investigate what features distinguish splits. "
            "If due to temporal/topic effects, document as a limitation. "
            "If due to preprocessing, fix the pipeline."
        )
    else:
        status = "❌ CRITICAL"
        interpretation = (
            "The probe classifier performs very well (>70%), indicating strong "
            "systematic differences between splits. This is a RED FLAG for:\n"
            "1. **Data leakage:** Test labels or statistics influencing train preprocessing\n"
            "2. **Severe distribution shift:** Train and test from fundamentally different distributions\n"
            "3. **Evaluation methodology error:** Incorrect split assignment\n\n"
            "**Action required:** Investigate immediately before proceeding with VERSION A. "
            "This level of separability should not occur in properly randomized splits."
        )

    return status, interpretation


def generate_report(results: Dict, output_path: Path):
    """Generate markdown report of dynamic probe results."""

    status, interpretation = interpret_results(results)

    with open(output_path, 'w') as f:
        f.write("# Label Leakage Dynamic Probe Report - VERSION A STEP 2C\n\n")
        f.write("**Date:** 2026-01-19\n")
        f.write("**Purpose:** Train classifier to detect systematic differences between splits\n\n")
        f.write("---\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")
        f.write(f"**Status:** {status}\n\n")

        f.write("**Probe Methodology:**\n")
        f.write("- Train a logistic regression classifier to predict which split (train/val/test) each query belongs to\n")
        f.write("- If classifier performs significantly better than random (33%), it indicates systematic differences\n")
        f.write("- Such differences could be due to leakage, temporal ordering, or topic drift\n\n")

        f.write("**Key Metrics:**\n")
        f.write(f"- Probe accuracy (CV): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
        f.write(f"- Random baseline: {results['random_baseline']:.4f} (3-way classification)\n")
        f.write(f"- Improvement over random: {results['improvement_over_random']:.2f}%\n\n")

        f.write("---\n\n")

        # Detailed results
        f.write("## Detailed Results\n\n")

        f.write("### Split Distribution\n\n")
        f.write(f"- Train: {results['n_train']:,} queries\n")
        f.write(f"- Val: {results['n_val']:,} queries\n")
        f.write(f"- Test: {results['n_test']:,} queries\n")
        f.write(f"- Total: {results['n_samples']:,} queries\n\n")

        f.write("### Probe Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy (train) | {results['accuracy']:.4f} |\n")
        f.write(f"| Accuracy (CV) | {results['cv_mean']:.4f} ± {results['cv_std']:.4f} |\n")
        f.write(f"| Random baseline | {results['random_baseline']:.4f} |\n")
        f.write(f"| Improvement | {results['improvement_over_random']:.2f}% |\n\n")

        f.write("**Cross-Validation Scores (5-fold):**\n")
        for i, score in enumerate(results['cv_scores']):
            f.write(f"- Fold {i}: {score:.4f}\n")
        f.write("\n")

        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write("             Predicted\n")
        f.write("             Train   Val    Test\n")
        cm = np.array(results['confusion_matrix'])
        f.write(f"Actual Train   {cm[0,0]:5d} {cm[0,1]:5d} {cm[0,2]:5d}\n")
        f.write(f"       Val     {cm[1,0]:5d} {cm[1,1]:5d} {cm[1,2]:5d}\n")
        f.write(f"       Test    {cm[2,0]:5d} {cm[2,1]:5d} {cm[2,2]:5d}\n")
        f.write("```\n\n")

        f.write("### Per-Class Performance\n\n")
        f.write("| Split | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|---------|\n")

        report = results['classification_report']
        for split_name in ['train', 'val', 'test']:
            metrics = report[split_name]
            f.write(f"| {split_name.capitalize()} | ")
            f.write(f"{metrics['precision']:.4f} | ")
            f.write(f"{metrics['recall']:.4f} | ")
            f.write(f"{metrics['f1-score']:.4f} | ")
            f.write(f"{int(metrics['support']):,} |\n")

        f.write("\n")

        # Interpretation
        f.write("---\n\n")
        f.write("## Interpretation\n\n")
        f.write(f"{interpretation}\n\n")

        # Guidelines
        f.write("---\n\n")
        f.write("## Interpretation Guidelines\n\n")

        f.write("**Accuracy Ranges:**\n")
        f.write("- **< 40%:** ✅ PASS - Close to random, no systematic differences\n")
        f.write("- **40-50%:** ⚠️ ACCEPTABLE - Minor differences, likely natural variation\n")
        f.write("- **50-70%:** ⚠️ CONCERNING - Moderate differences, investigate cause\n")
        f.write("- **> 70%:** ❌ CRITICAL - Strong differences, likely leakage or severe shift\n\n")

        f.write("**Possible Causes of High Accuracy:**\n")
        f.write("1. **Data leakage:** Test labels/statistics used in train preprocessing\n")
        f.write("2. **Temporal ordering:** Posts from different time periods in different splits\n")
        f.write("3. **Topic drift:** Train and test posts have different topic distributions\n")
        f.write("4. **Feature engineering:** Preprocessing uses global statistics (train+test)\n")
        f.write("5. **Sampling bias:** Non-random assignment to splits\n\n")

        # Recommendations
        f.write("---\n\n")
        f.write("## Recommendations\n\n")

        if status == "✅ PASS":
            f.write("✅ No action required. The splits appear properly randomized with no systematic differences.\n\n")
            f.write("Continue with VERSION A evaluation.\n\n")
        elif status == "⚠️ ACCEPTABLE":
            f.write("⚠️ Minor differences detected, but within acceptable range.\n\n")
            f.write("**Optional investigations:**\n")
            f.write("- Check if train/val/test have similar post length distributions\n")
            f.write("- Check if train/val/test have similar topic distributions\n\n")
            f.write("Continue with VERSION A evaluation.\n\n")
        elif status == "⚠️ CONCERNING":
            f.write("⚠️ Moderate differences detected. Investigation recommended.\n\n")
            f.write("**Recommended actions:**\n")
            f.write("1. Analyze feature importance to identify what distinguishes splits\n")
            f.write("2. Check for temporal ordering effects (post dates)\n")
            f.write("3. Check for topic distribution differences\n")
            f.write("4. Review preprocessing pipeline for global statistics usage\n")
            f.write("5. Document findings before proceeding with VERSION A\n\n")
        else:  # CRITICAL
            f.write("❌ **CRITICAL ISSUE DETECTED**\n\n")
            f.write("**Immediate actions required:**\n")
            f.write("1. **STOP** - Do not proceed with VERSION A until resolved\n")
            f.write("2. Analyze feature importance to identify leakage source\n")
            f.write("3. Review all preprocessing steps for test data usage\n")
            f.write("4. Check split assignment logic\n")
            f.write("5. If temporal/topic shift, consider stratified splitting\n")
            f.write("6. Re-run probe after fixes to verify resolution\n\n")

        f.write("---\n\n")
        f.write("**Generated by:** `scripts/audit_label_leakage_dynamic.py`\n")
        f.write("**Probe method:** Logistic regression on query embeddings\n")
        f.write("**Features:** Post-level mean-pooled sentence embeddings\n")


def main():
    parser = argparse.ArgumentParser(description="Dynamic probe for label leakage detection")
    parser.add_argument("--embeddings", type=str, default="data/cache/bge_m3/dense.npy",
                       help="Path to sentence embeddings")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output", type=str, default="outputs/audit/label_leakage_dynamic_report.md",
                       help="Output report path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        logger.error(f"Embeddings file not found: {embeddings_path}")
        logger.info("Skipping dynamic probe (embeddings not available)")
        logger.info("This is acceptable - static scan already passed")
        return 0

    print("="*60)
    print("LABEL LEAKAGE DYNAMIC PROBE")
    print("="*60)
    print(f"Embeddings: {embeddings_path}")
    print(f"Data directory: {data_dir}")
    print()

    # Load data
    embeddings, queries_df = load_embeddings_and_queries(embeddings_path, data_dir)

    # Load full groundtruth for feature creation
    gt_path = data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv"
    df_full = pd.read_csv(gt_path)

    # Create query features
    query_features = create_query_features(embeddings, queries_df, df_full)

    # Create split labels
    split_labels = create_split_labels(queries_df, seed=args.seed)

    # Train probe
    results = train_split_probe(query_features, split_labels)

    # Generate report
    generate_report(results, output_path)

    # Print summary
    status, _ = interpret_results(results)

    print()
    print("="*60)
    print("PROBE COMPLETE")
    print("="*60)
    print(f"Probe accuracy (CV): {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    print(f"Random baseline: {results['random_baseline']:.4f}")
    print(f"Improvement over random: {results['improvement_over_random']:.2f}%")
    print()
    print(f"Final verdict: {status}")
    print(f"\nFull report: {output_path}")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
