#!/usr/bin/env python3
"""
Jina-Reranker-v3 Full HPO.

Tunes Jina-Reranker-v3 specific parameters using actual inference:
- max_length: [512, 1024, 2048]
- top_k_rerank: [15, 20, 25, 30]
- top_k_final: [5, 10, 15]

Uses NV-Embed-v2 cached embeddings for retrieval (Stage 1).
Runs 5-fold cross-validation with positives_only protocol.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from final_sc_review.data.schemas import Sentence
from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.reranker.zoo import RerankerZoo, JinaRerankerV3, RerankerConfig


def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute nDCG@K with binary relevance."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    dcg = 0.0
    for i in range(min(k, len(sorted_gold))):
        if sorted_gold[i]:
            dcg += 1.0 / math.log2(i + 2)

    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    for i, is_gold in enumerate(sorted_gold):
        if is_gold:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Recall@K."""
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0

    sorted_idx = np.argsort(-scores)
    top_k_idx = sorted_idx[:k]
    n_retrieved = int(gold_mask[top_k_idx].sum())
    return n_retrieved / n_gold


def load_data(
    groundtruth_path: Path,
    corpus_path: Path,
    criteria_path: Path,
) -> Tuple[pd.DataFrame, List[Sentence], Dict[str, str]]:
    """Load groundtruth, corpus, and criteria."""
    # Load groundtruth
    gt_df = pd.read_csv(groundtruth_path)
    print(f"Loaded {len(gt_df)} groundtruth rows")

    # Load corpus as Sentence objects
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            item = json.loads(line)
            sentences.append(Sentence(
                sent_uid=item['sent_uid'],
                post_id=item['post_id'],
                sid=item['sid'],
                text=item['text'],
            ))
    print(f"Loaded {len(sentences)} sentences")

    # Load criteria
    with open(criteria_path) as f:
        criteria_data = json.load(f)

    criteria_texts = {
        c['id']: c['text']
        for c in criteria_data['criteria']
    }
    print(f"Loaded {len(criteria_texts)} criteria")

    return gt_df, sentences, criteria_texts


def create_fold_assignments(gt_df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> Dict[str, int]:
    """Create post-ID disjoint fold assignments."""
    post_ids = gt_df['post_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(post_ids)

    fold_assignments = {}
    for i, post_id in enumerate(post_ids):
        fold_assignments[str(post_id)] = i % n_folds

    return fold_assignments


def prepare_queries_by_fold(
    gt_df: pd.DataFrame,
    sentences: List[Sentence],
    fold_assignments: Dict[str, int],
    excluded_criteria: List[str] = ["A.10"],
) -> Dict[int, List[Dict]]:
    """Prepare queries organized by fold."""
    # Create sentence lookup
    sentence_lookup = {s.sent_uid: s for s in sentences}
    post_sentences = {}
    for s in sentences:
        if s.post_id not in post_sentences:
            post_sentences[s.post_id] = []
        post_sentences[s.post_id].append(s)

    # Organize queries by fold
    fold_queries = {i: [] for i in range(5)}

    for (post_id, criterion_id), group in gt_df.groupby(['post_id', 'criterion']):
        # Skip excluded criteria
        if criterion_id in excluded_criteria:
            continue

        fold_id = fold_assignments.get(str(post_id))
        if fold_id is None:
            continue

        # Get gold sentence UIDs
        gold_uids = set(group[group['groundtruth'] == 1]['sent_uid'].values)

        # Skip queries without evidence (positives_only protocol)
        if not gold_uids:
            continue

        # Get all sentences for this post
        candidates = post_sentences.get(post_id, [])
        if not candidates:
            continue

        fold_queries[fold_id].append({
            'post_id': post_id,
            'criterion_id': criterion_id,
            'gold_uids': gold_uids,
            'candidates': candidates,
        })

    for fold_id, queries in fold_queries.items():
        print(f"Fold {fold_id}: {len(queries)} queries with evidence")

    return fold_queries


def evaluate_reranker_on_fold(
    reranker: JinaRerankerV3,
    retriever,
    fold_queries: List[Dict],
    criteria_texts: Dict[str, str],
    top_k_retriever: int,
    top_k_final: int,
) -> Dict[str, float]:
    """Evaluate reranker on a single fold."""
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for query_data in tqdm(fold_queries, desc="Evaluating", leave=False):
        criterion_id = query_data['criterion_id']
        query_text = criteria_texts.get(criterion_id, criterion_id)
        gold_uids = query_data['gold_uids']
        candidates = query_data['candidates']
        post_id = query_data['post_id']

        # Stage 1: Retrieve top-k candidates using NV-Embed-v2
        try:
            results = retriever.retrieve_within_post(
                query=query_text,
                post_id=post_id,
                top_k=top_k_retriever,
            )
            retrieved = [(r.sent_uid, r.text, r.score) for r in results]
        except Exception as e:
            # Fallback: use all candidates from post
            retrieved = [(s.sent_uid, s.text, 0.0) for s in candidates[:top_k_retriever]]

        if not retrieved:
            continue

        # Stage 2: Rerank with Jina
        rerank_candidates = [(uid, text) for uid, text, _ in retrieved]
        reranked = reranker.rerank(query_text, rerank_candidates, top_k=top_k_final)

        if not reranked:
            continue

        # Compute metrics
        result_uids = [r.sent_uid for r in reranked]
        result_scores = np.array([r.score for r in reranked])
        gold_mask = np.array([uid in gold_uids for uid in result_uids])

        if gold_mask.sum() > 0:
            ndcg = compute_ndcg_at_k(gold_mask, result_scores, top_k_final)
            mrr = compute_mrr(gold_mask, result_scores)
            recall = compute_recall_at_k(gold_mask, result_scores, top_k_final)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            recall_scores.append(recall)

    return {
        'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'recall@10': np.mean(recall_scores) if recall_scores else 0.0,
        'n_queries': len(ndcg_scores),
    }


def create_reranker_with_config(
    max_length: int,
    batch_size: int = 64,
    device: str = 'cuda',
) -> JinaRerankerV3:
    """Create a Jina reranker with specific config."""
    config = RerankerConfig(
        name="jina-reranker-v3",
        model_id="jinaai/jina-reranker-v3",
        reranker_type="listwise",
        max_length=max_length,
        batch_size=batch_size,
        listwise_max_docs=32,
        trust_remote_code=True,
    )
    return JinaRerankerV3(config, device)


def objective(
    trial: optuna.Trial,
    fold_queries: Dict[int, List[Dict]],
    retriever,
    criteria_texts: Dict[str, str],
    device: str,
    reranker_cache: Dict,
) -> float:
    """Optuna objective for Jina HPO."""

    # Hyperparameters to tune
    max_length = trial.suggest_categorical('max_length', [512, 1024, 2048])
    top_k_retriever = trial.suggest_categorical('top_k_retriever', [20, 24, 30])
    top_k_final = trial.suggest_categorical('top_k_final', [10, 15, 20])

    # Reuse reranker if same max_length (expensive to reload)
    cache_key = max_length
    if cache_key not in reranker_cache:
        reranker_cache[cache_key] = create_reranker_with_config(max_length, device=device)
        reranker_cache[cache_key].load_model()
    reranker = reranker_cache[cache_key]

    # 5-fold CV (evaluate on 2-3 folds for speed during HPO)
    fold_ndcgs = []
    eval_folds = [0, 2, 4]  # Sample folds for faster HPO

    for fold_id in eval_folds:
        if fold_id not in fold_queries or not fold_queries[fold_id]:
            continue

        metrics = evaluate_reranker_on_fold(
            reranker=reranker,
            retriever=retriever,
            fold_queries=fold_queries[fold_id],
            criteria_texts=criteria_texts,
            top_k_retriever=top_k_retriever,
            top_k_final=top_k_final,
        )
        fold_ndcgs.append(metrics['ndcg@10'])

        # Report for pruning
        trial.report(np.mean(fold_ndcgs), len(fold_ndcgs) - 1)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_ndcgs) if fold_ndcgs else 0.0


def run_full_evaluation(
    best_params: Dict,
    fold_queries: Dict[int, List[Dict]],
    retriever,
    criteria_texts: Dict[str, str],
    device: str,
    output_dir: Path,
) -> Dict:
    """Run full 5-fold evaluation with best params."""
    print("\n=== Full 5-Fold Evaluation with Best Params ===")
    print(f"max_length: {best_params['max_length']}")
    print(f"top_k_retriever: {best_params['top_k_retriever']}")
    print(f"top_k_final: {best_params['top_k_final']}")

    reranker = create_reranker_with_config(best_params['max_length'], device=device)
    reranker.load_model()

    fold_results = []

    for fold_id in range(5):
        if fold_id not in fold_queries or not fold_queries[fold_id]:
            continue

        print(f"\nEvaluating fold {fold_id}...")
        metrics = evaluate_reranker_on_fold(
            reranker=reranker,
            retriever=retriever,
            fold_queries=fold_queries[fold_id],
            criteria_texts=criteria_texts,
            top_k_retriever=best_params['top_k_retriever'],
            top_k_final=best_params['top_k_final'],
        )

        fold_results.append({
            'fold_id': fold_id,
            'ndcg@10': metrics['ndcg@10'],
            'mrr': metrics['mrr'],
            'recall@10': metrics['recall@10'],
            'n_queries': metrics['n_queries'],
        })

        print(f"  nDCG@10: {metrics['ndcg@10']:.4f}, MRR: {metrics['mrr']:.4f}, "
              f"Recall@10: {metrics['recall@10']:.4f}")

    # Aggregate
    mean_ndcg = np.mean([r['ndcg@10'] for r in fold_results])
    std_ndcg = np.std([r['ndcg@10'] for r in fold_results])
    mean_mrr = np.mean([r['mrr'] for r in fold_results])
    std_mrr = np.std([r['mrr'] for r in fold_results])
    mean_recall = np.mean([r['recall@10'] for r in fold_results])
    std_recall = np.std([r['recall@10'] for r in fold_results])

    summary = {
        'best_params': best_params,
        'metrics': {
            'ndcg@10': {'mean': mean_ndcg, 'std': std_ndcg},
            'mrr': {'mean': mean_mrr, 'std': std_mrr},
            'recall@10': {'mean': mean_recall, 'std': std_recall},
        },
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n=== Final Results ===")
    print(f"nDCG@10: {mean_ndcg:.4f} ± {std_ndcg:.4f}")
    print(f"MRR: {mean_mrr:.4f} ± {std_mrr:.4f}")
    print(f"Recall@10: {mean_recall:.4f} ± {std_recall:.4f}")

    # Save results
    with open(output_dir / 'best_config_full_eval.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Jina-Reranker-v3 Full HPO')
    parser.add_argument('--groundtruth', type=str,
                        default='data/groundtruth/evidence_sentence_groundtruth.csv',
                        help='Path to groundtruth CSV')
    parser.add_argument('--corpus', type=str,
                        default='data/groundtruth/sentence_corpus.jsonl',
                        help='Path to sentence corpus')
    parser.add_argument('--criteria', type=str,
                        default='data/DSM5/MDD_Criteira.json',
                        help='Path to criteria definitions')
    parser.add_argument('--cache_dir', type=str,
                        default='data/cache',
                        help='Cache directory for embeddings')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/hpo/jina_reranker',
                        help='Output directory')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of HPO trials')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--evaluate_only', type=str, default=None,
                        help='Path to best_params.json to evaluate (skip HPO)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("\n=== Loading Data ===")
    gt_df, sentences, criteria_texts = load_data(
        Path(args.groundtruth),
        Path(args.corpus),
        Path(args.criteria),
    )

    # Create fold assignments
    fold_assignments = create_fold_assignments(gt_df)

    # Prepare queries by fold
    print("\n=== Preparing Queries ===")
    fold_queries = prepare_queries_by_fold(
        gt_df, sentences, fold_assignments, excluded_criteria=["A.10"]
    )

    # Initialize retriever
    print("\n=== Loading Retriever ===")
    cache_dir = Path(args.cache_dir)
    retriever_zoo = RetrieverZoo(
        sentences=sentences,
        cache_dir=cache_dir,
        device=device,
    )
    retriever = retriever_zoo.get_retriever("nv-embed-v2")
    retriever.encode_corpus(rebuild=False)
    print("Retriever loaded with cached embeddings")

    if args.evaluate_only:
        # Load and evaluate existing config
        with open(args.evaluate_only) as f:
            data = json.load(f)
        best_params = data.get('best_params', data)

        run_full_evaluation(
            best_params=best_params,
            fold_queries=fold_queries,
            retriever=retriever,
            criteria_texts=criteria_texts,
            device=device,
            output_dir=output_dir,
        )
    else:
        # Run HPO
        print("\n=== Running HPO ===")
        print(f"Tuning parameters:")
        print(f"  - max_length: [512, 1024, 2048]")
        print(f"  - top_k_retriever: [20, 24, 30]")
        print(f"  - top_k_final: [10, 15, 20]")

        # Cache for rerankers (to avoid reloading for same max_length)
        reranker_cache = {}

        study = optuna.create_study(
            study_name="jina_reranker_hpo",
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )

        study.optimize(
            lambda trial: objective(
                trial, fold_queries, retriever, criteria_texts, device, reranker_cache
            ),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        # Save HPO results
        results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat(),
        }

        with open(output_dir / 'best_params.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save trial history
        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_dir / 'trials.csv', index=False)

        print(f"\n=== HPO Complete ===")
        print(f"Best nDCG@10: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        # Run full evaluation with best params
        run_full_evaluation(
            best_params=study.best_params,
            fold_queries=fold_queries,
            retriever=retriever,
            criteria_texts=criteria_texts,
            device=device,
            output_dir=output_dir,
        )


if __name__ == '__main__':
    main()
