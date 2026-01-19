#!/usr/bin/env python3
"""Full pipeline stage-by-stage evaluation for research verification.

This script evaluates the complete evidence retrieval pipeline across all stages
with multiple K values to demonstrate incremental improvements.

Stages:
- S0: Baseline (random ranking)
- S1: BM25 retrieval
- S2: Dense retrieval (retriever zoo)
- S3: Sparse retrieval (if applicable)
- S4: ColBERT retrieval (if applicable)
- S5: Fusion (dense + sparse + ColBERT)
- S6: Reranking (final stage)
- S7: GNN P4 enhancement (if available)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.final_sc_review.data.io import load_groundtruth, load_sentence_corpus
from src.final_sc_review.data.splits import create_post_disjoint_splits
from src.final_sc_review.metrics.ranking import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    map_at_k,
)


def load_config(config_path: Path) -> Dict:
    """Load configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_ranking(
    relevance: np.ndarray,
    scores: np.ndarray,
    k_values: List[int]
) -> Dict:
    """Evaluate ranking at multiple K values.

    Args:
        relevance: Binary relevance labels [n_items]
        scores: Ranking scores [n_items]
        k_values: List of K values to evaluate

    Returns:
        Dictionary of metrics at each K
    """
    if len(relevance) == 0 or relevance.sum() == 0:
        return {f'recall@{k}': 0.0 for k in k_values}

    results = {}
    for k in k_values:
        results[f'recall@{k}'] = recall_at_k(relevance, scores, k)
        results[f'precision@{k}'] = precision_at_k(relevance, scores, k)
        results[f'hit_rate@{k}'] = hit_rate_at_k(relevance, scores, k)
        results[f'ndcg@{k}'] = ndcg_at_k(relevance, scores, k)

    # MRR and MAP don't depend on K
    results['mrr'] = mrr_at_k(relevance, scores, max(k_values))
    results['map'] = map_at_k(relevance, scores, max(k_values))

    return results


def stage_s0_random_baseline(
    queries: List[Dict],
    sentence_corpus: Dict,
    k_values: List[int]
) -> Dict:
    """S0: Random ranking baseline."""
    print("\nEvaluating S0: Random Baseline...")

    all_results = {f'recall@{k}': [] for k in k_values}
    all_results.update({f'precision@{k}': [] for k in k_values})
    all_results.update({f'hit_rate@{k}': [] for k in k_values})
    all_results.update({f'ndcg@{k}': [] for k in k_values})
    all_results['mrr'] = []
    all_results['map'] = []

    for query in tqdm(queries, desc="S0"):
        post_id = query['post_id']
        criterion = query['criterion']
        ground_truth_sids = set(query['evidence_sids'])

        # Get candidate sentences for this post
        post_sentences = sentence_corpus.get(post_id, [])
        if not post_sentences:
            continue

        # Create random scores
        n_sents = len(post_sentences)
        random_scores = np.random.rand(n_sents)

        # Create relevance vector
        relevance = np.array([
            1 if sent['sid'] in ground_truth_sids else 0
            for sent in post_sentences
        ])

        if relevance.sum() == 0:
            continue

        # Evaluate
        query_results = evaluate_ranking(relevance, random_scores, k_values)
        for metric, value in query_results.items():
            all_results[metric].append(value)

    # Aggregate
    aggregated = {
        metric: {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values))
        }
        for metric, values in all_results.items()
    }

    return aggregated


def stage_s1_bm25(
    queries: List[Dict],
    sentence_corpus: Dict,
    k_values: List[int]
) -> Dict:
    """S1: BM25 sparse retrieval."""
    print("\nEvaluating S1: BM25...")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("WARNING: rank_bm25 not installed. Skipping S1.")
        return {}

    all_results = {f'recall@{k}': [] for k in k_values}
    all_results.update({f'precision@{k}': [] for k in k_values})
    all_results.update({f'hit_rate@{k}': [] for k in k_values})
    all_results.update({f'ndcg@{k}': [] for k in k_values})
    all_results['mrr'] = []
    all_results['map'] = []

    # Group queries by post for efficiency
    queries_by_post = {}
    for query in queries:
        post_id = query['post_id']
        if post_id not in queries_by_post:
            queries_by_post[post_id] = []
        queries_by_post[post_id].append(query)

    for post_id, post_queries in tqdm(queries_by_post.items(), desc="S1"):
        # Get candidate sentences
        post_sentences = sentence_corpus.get(post_id, [])
        if not post_sentences:
            continue

        # Build BM25 index for this post
        tokenized_corpus = [sent['text'].lower().split() for sent in post_sentences]
        bm25 = BM25Okapi(tokenized_corpus)

        # Evaluate each query for this post
        for query in post_queries:
            criterion = query['criterion']
            ground_truth_sids = set(query['evidence_sids'])

            # BM25 scoring
            query_tokens = criterion.lower().split()
            bm25_scores = bm25.get_scores(query_tokens)

            # Create relevance vector
            relevance = np.array([
                1 if sent['sid'] in ground_truth_sids else 0
                for sent in post_sentences
            ])

            if relevance.sum() == 0:
                continue

            # Evaluate
            query_results = evaluate_ranking(relevance, bm25_scores, k_values)
            for metric, value in query_results.items():
                all_results[metric].append(value)

    # Aggregate
    aggregated = {
        metric: {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values))
        }
        for metric, values in all_results.items()
    }

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline stage-by-stage evaluation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config file path"
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
        help="Which split to evaluate"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50],
        help="K values for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/final_research_eval/stage_by_stage"),
        help="Output directory"
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Max queries to evaluate (for testing)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FULL PIPELINE STAGE-BY-STAGE EVALUATION")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"K values: {args.k_values}")
    print(f"Output: {args.output_dir}")
    print()

    # Load config
    config = load_config(args.config)

    # Load data
    print("Loading data...")
    groundtruth_df = load_groundtruth(
        Path(config['paths']['groundtruth'])
    )
    sentence_corpus = load_sentence_corpus(
        Path(config['paths']['sentence_corpus'])
    )

    # Create post-disjoint splits
    print("Creating post-disjoint splits...")
    splits = create_post_disjoint_splits(
        groundtruth_df,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        seed=config['split']['seed']
    )

    # Get queries for target split
    target_split_df = splits[args.split]
    print(f"Loaded {len(target_split_df)} queries from {args.split} split")

    # Convert to query list
    queries = []
    for _, row in target_split_df.iterrows():
        queries.append({
            'post_id': row['post_id'],
            'criterion': row['criterion'],
            'evidence_sids': [row['sid']]  # Single evidence sentence per row
        })

    # Group by (post_id, criterion) and aggregate evidence_sids
    from collections import defaultdict
    query_dict = defaultdict(lambda: {'post_id': None, 'criterion': None, 'evidence_sids': []})
    for q in queries:
        key = (q['post_id'], q['criterion'])
        query_dict[key]['post_id'] = q['post_id']
        query_dict[key]['criterion'] = q['criterion']
        query_dict[key]['evidence_sids'].extend(q['evidence_sids'])

    queries = list(query_dict.values())
    print(f"Aggregated to {len(queries)} unique (post, criterion) queries")

    # Limit queries if specified
    if args.max_queries:
        queries = queries[:args.max_queries]
        print(f"Limited to {len(queries)} queries for testing")

    # Organize sentence corpus by post_id
    corpus_by_post = {}
    for sent_uid, sent_data in sentence_corpus.items():
        post_id = sent_data['post_id']
        if post_id not in corpus_by_post:
            corpus_by_post[post_id] = []
        corpus_by_post[post_id].append(sent_data)

    print(f"Sentence corpus: {len(sentence_corpus)} sentences across {len(corpus_by_post)} posts")
    print()

    # Run stage-by-stage evaluation
    results = {
        'config': config,
        'split': args.split,
        'n_queries': len(queries),
        'k_values': args.k_values,
        'stages': {}
    }

    # S0: Random baseline
    results['stages']['S0_random'] = stage_s0_random_baseline(
        queries, corpus_by_post, args.k_values
    )

    # S1: BM25
    results['stages']['S1_bm25'] = stage_s1_bm25(
        queries, corpus_by_post, args.k_values
    )

    # TODO: Add S2-S7 (dense retrieval, fusion, reranking, GNN)
    # These would require loading the actual models

    # Save results
    output_file = args.output_dir / f"{args.split}_stage_by_stage.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    for stage_name, stage_results in results['stages'].items():
        print(f"{stage_name}:")
        for metric in ['recall@10', 'ndcg@10', 'mrr', 'map']:
            if metric in stage_results:
                mean = stage_results[metric]['mean']
                std = stage_results[metric]['std']
                print(f"  {metric}: {mean:.4f} ± {std:.4f}")
        print()

    print(f"Results saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
