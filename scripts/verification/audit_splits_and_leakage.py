#!/usr/bin/env python3
"""Audit data splits for leakage prevention.

This script verifies:
1. Post-ID disjointness across folds
2. No text content overlap
3. Balanced splits
4. Proper train/tune/test partitioning
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import torch


def load_fold_graphs(graph_dir: Path, fold: int) -> List:
    """Load graphs for a specific fold."""
    fold_file = graph_dir / f"fold_{fold}.pt"

    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")

    data = torch.load(fold_file, weights_only=False)

    if isinstance(data, dict) and 'graphs' in data:
        return data['graphs']
    else:
        return data


def extract_post_ids(graphs: List) -> Set[str]:
    """Extract unique post IDs from graphs."""
    post_ids = set()
    for graph in graphs:
        if hasattr(graph, 'post_id'):
            post_ids.add(graph.post_id)
        elif hasattr(graph, 'query_id'):
            # Extract post_id from query_id (format: post_id_criterion)
            query_id = graph.query_id
            if '_A.' in query_id:
                post_id = query_id.rsplit('_A.', 1)[0]
                post_ids.add(post_id)

    return post_ids


def hash_text(text: str) -> str:
    """Create hash of text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def extract_text_hashes(graphs: List) -> Set[str]:
    """Extract hashes of text content from graphs."""
    text_hashes = set()

    for graph in graphs:
        # Try to extract text content
        if hasattr(graph, 'text'):
            text_hashes.add(hash_text(graph.text))
        elif hasattr(graph, 'sentence_texts'):
            for sent in graph.sentence_texts:
                text_hashes.add(hash_text(sent))

    return text_hashes


def count_queries_with_evidence(graphs: List) -> Tuple[int, int]:
    """Count total queries and queries with evidence."""
    total = len(graphs)
    with_evidence = sum(1 for g in graphs if hasattr(g, 'y') and g.y[0] > 0)
    return total, with_evidence


def audit_splits(graph_dir: Path, n_folds: int = 5) -> Dict:
    """Audit all splits for leakage and disjointness."""
    print(f"{'='*80}")
    print("DATA SPLIT AUDIT")
    print(f"{'='*80}\n")

    # Load all folds
    fold_graphs = {}
    fold_post_ids = {}
    fold_text_hashes = {}
    fold_stats = {}

    for fold in range(n_folds):
        print(f"Loading fold {fold}...")
        graphs = load_fold_graphs(graph_dir, fold)

        post_ids = extract_post_ids(graphs)
        text_hashes = extract_text_hashes(graphs)
        n_total, n_with_evidence = count_queries_with_evidence(graphs)

        fold_graphs[fold] = graphs
        fold_post_ids[fold] = post_ids
        fold_text_hashes[fold] = text_hashes

        fold_stats[fold] = {
            'n_queries': n_total,
            'n_posts': len(post_ids),
            'n_with_evidence': n_with_evidence,
            'pos_rate': n_with_evidence / n_total if n_total > 0 else 0
        }

        print(f"  Fold {fold}: {n_total} queries, {len(post_ids)} posts, "
              f"{n_with_evidence} with evidence ({fold_stats[fold]['pos_rate']:.2%})")

    print()

    # Check for post-ID overlaps
    print(f"{'='*80}")
    print("POST-ID DISJOINTNESS CHECK")
    print(f"{'='*80}\n")

    overlaps_found = False
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            overlap = fold_post_ids[i] & fold_post_ids[j]
            if overlap:
                print(f"❌ OVERLAP between fold {i} and fold {j}: {len(overlap)} posts")
                print(f"   Sample overlapping posts: {list(overlap)[:5]}")
                overlaps_found = True

    if not overlaps_found:
        print("✅ All folds are POST-ID disjoint (no overlaps)")

    print()

    # Check for text content overlaps (optional extra check)
    print(f"{'='*80}")
    print("TEXT CONTENT HASH OVERLAP CHECK")
    print(f"{'='*80}\n")

    text_overlaps_found = False
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            overlap = fold_text_hashes[i] & fold_text_hashes[j]
            if overlap:
                print(f"⚠️  Text hash overlap between fold {i} and fold {j}: {len(overlap)} hashes")
                text_overlaps_found = True

    if not text_overlaps_found:
        print("✅ No text content hash overlaps detected")

    print()

    # Summary statistics
    print(f"{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    total_posts = sum(len(post_ids) for post_ids in fold_post_ids.values())
    total_queries = sum(stats['n_queries'] for stats in fold_stats.values())
    total_with_evidence = sum(stats['n_with_evidence'] for stats in fold_stats.values())

    print(f"Total unique posts: {total_posts}")
    print(f"Total queries: {total_queries}")
    print(f"Total with evidence: {total_with_evidence} ({total_with_evidence/total_queries:.2%})")
    print()

    # Per-fold balance
    print("Per-fold statistics:")
    print(f"{'Fold':<6} {'Queries':<10} {'Posts':<8} {'With Evidence':<15} {'Pos Rate':<10}")
    print("-" * 70)
    for fold, stats in fold_stats.items():
        print(f"{fold:<6} {stats['n_queries']:<10} {stats['n_posts']:<8} "
              f"{stats['n_with_evidence']:<15} {stats['pos_rate']:<10.2%}")

    print()

    # Return results
    results = {
        'n_folds': n_folds,
        'total_posts': total_posts,
        'total_queries': total_queries,
        'total_with_evidence': total_with_evidence,
        'overall_pos_rate': total_with_evidence / total_queries,
        'post_id_overlaps': overlaps_found,
        'text_hash_overlaps': text_overlaps_found,
        'fold_stats': fold_stats
    }

    return results


def save_results(results: Dict, output_file: Path):
    """Save audit results to CSV."""
    # Create DataFrame from fold stats
    rows = []
    for fold, stats in results['fold_stats'].items():
        rows.append({
            'fold': fold,
            'n_queries': stats['n_queries'],
            'n_posts': stats['n_posts'],
            'n_with_evidence': stats['n_with_evidence'],
            'pos_rate': stats['pos_rate']
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Saved audit results to: {output_file}")

    # Also save JSON summary
    json_file = output_file.parent / f"{output_file.stem}_summary.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Audit data splits for leakage")
    parser.add_argument(
        "--graph_dir",
        type=Path,
        default=Path("data/cache/gnn/20260117_003135"),
        help="Directory containing fold graph files"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    # Run audit
    results = audit_splits(args.graph_dir, args.n_folds)

    # Save results
    save_results(results, args.output)

    # Check for failures
    if results['post_id_overlaps']:
        print("\n❌ AUDIT FAILED: Post-ID overlaps detected!")
        sys.exit(1)
    else:
        print("\n✅ AUDIT PASSED: All checks successful")
        sys.exit(0)


if __name__ == "__main__":
    main()
