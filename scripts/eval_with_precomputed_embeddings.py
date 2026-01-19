#!/usr/bin/env python3
"""Evaluate pipeline using pre-computed NV-Embed-v2 embeddings.

This script loads pre-computed corpus and query embeddings and performs
retrieval + reranking without needing to load NV-Embed-v2.

Usage:
    # First, encode corpus and queries in nv-embed-v2 env:
    conda activate nv-embed-v2
    python scripts/encode_corpus_nv_embed.py --config configs/default.yaml
    python scripts/encode_queries_nv_embed.py --config configs/default.yaml --split test
    conda deactivate

    # Then run evaluation in main env:
    conda activate llmhe
    python scripts/eval_with_precomputed_embeddings.py --config configs/default.yaml --split test
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import yaml
from tqdm import tqdm

from final_sc_review.data.io import load_criteria, load_groundtruth
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.retrieval_eval import evaluate_rankings
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_cached_corpus_embeddings(cache_dir: Path):
    """Load pre-computed corpus embeddings and metadata."""
    embeddings_file = cache_dir / "corpus_embeddings.npy"
    metadata_file = cache_dir / "corpus_metadata.pkl"

    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Corpus embeddings not found: {embeddings_file}\n"
            f"Please run: conda activate nv-embed-v2 && "
            f"python scripts/encode_corpus_nv_embed.py --config configs/default.yaml"
        )

    logger.info(f"Loading cached corpus embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"Loaded {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")

    # Create sent_uid -> index mapping
    sent_uid_to_idx = {uid: idx for idx, uid in enumerate(metadata['sent_uids'])}

    # Create post_id -> indices mapping
    post_id_to_indices = {}
    for idx, post_id in enumerate(metadata['post_ids']):
        post_id_to_indices.setdefault(post_id, []).append(idx)

    return embeddings, sent_uid_to_idx, post_id_to_indices, metadata


def load_cached_query_embeddings(cache_dir: Path, split: str):
    """Load pre-computed query embeddings."""
    query_embeddings_file = cache_dir / f"query_embeddings_{split}.pkl"

    if not query_embeddings_file.exists():
        raise FileNotFoundError(
            f"Query embeddings not found: {query_embeddings_file}\n"
            f"Please run: conda activate nv-embed-v2 && "
            f"python scripts/encode_queries_nv_embed.py --config configs/default.yaml --split {split}"
        )

    logger.info(f"Loading cached query embeddings from {query_embeddings_file}")
    with open(query_embeddings_file, 'rb') as f:
        query_embeddings = pickle.load(f)

    logger.info(f"Loaded {len(query_embeddings)} query embeddings")

    return query_embeddings


def retrieve_with_embeddings(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    post_indices: List[int],
    sent_uids: List[str],
    top_k: int = 24
):
    """Retrieve top-k sentences using pre-computed embeddings."""
    # Filter corpus to only this post's sentences
    post_embeddings = corpus_embeddings[post_indices]
    post_sent_uids = [sent_uids[idx] for idx in post_indices]

    # Compute cosine similarity (embeddings are already normalized)
    scores = np.dot(post_embeddings, query_embedding)

    # Get top-k indices
    top_k_local_indices = np.argsort(-scores)[:top_k]

    # Map back to sent_uids and scores
    results = [
        (post_sent_uids[idx], float(scores[idx]))
        for idx in top_k_local_indices
    ]

    return results


def rerank_with_jina(query_text: str, candidates: List[tuple], sent_uid_to_text: Dict[str, str], top_k: int = 10):
    """Rerank candidates with Jina-Reranker-v3."""
    from final_sc_review.reranker.jina_v3 import JinaV3Reranker

    # Initialize reranker
    reranker = JinaV3Reranker(
        model_name="jinaai/jina-reranker-v3",
        max_length=1024,
        device="cuda",
        dtype="auto",
        use_listwise=True,
    )

    # Prepare documents
    documents = [sent_uid_to_text[uid] for uid, _ in candidates]

    # Score pairs using JinaV3Reranker
    scores = reranker.score_pairs(query_text, documents)

    # Sort by scores (descending) and take top-k
    scored_results = [(candidates[i][0], scores[i]) for i in range(len(candidates))]
    scored_results.sort(key=lambda x: x[1], reverse=True)

    return scored_results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Evaluate with pre-computed NV-Embed-v2 embeddings")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--skip_no_positives", action="store_true")
    parser.add_argument("--save_per_query_rankings", action="store_true")
    parser.add_argument("--skip_reranking", action="store_true", help="Skip reranking step (retrieval only)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Load data
    groundtruth = load_groundtruth(Path(cfg['paths']['groundtruth']))
    criteria = load_criteria(Path(cfg['paths']['data_dir']) / "DSM5" / "MDD_Criteira.json")
    criteria_map = {c.criterion_id: c.text for c in criteria}

    # Load sentence corpus for sent_uid -> text mapping
    sent_uid_to_text = {}
    with open(cfg['paths']['sentence_corpus'], 'r') as f:
        for line in f:
            data = json.loads(line)
            sent_uid_to_text[data['sent_uid']] = data['text']

    # Split data
    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(
        post_ids,
        seed=cfg['split']['seed'],
        train_ratio=cfg['split']['train_ratio'],
        val_ratio=cfg['split']['val_ratio'],
        test_ratio=cfg['split']['test_ratio'],
    )
    eval_posts = set(splits[args.split])

    # Load pre-computed embeddings
    cache_dir = Path(cfg['paths']['cache_dir'])
    corpus_embeddings, sent_uid_to_idx, post_id_to_indices, corpus_metadata = load_cached_corpus_embeddings(cache_dir)
    query_embeddings = load_cached_query_embeddings(cache_dir, args.split)

    logger.info(f"Retriever: {corpus_metadata['model_id']} (cached embeddings)")
    logger.info(f"Reranker: {'jinaai/jina-reranker-v3' if not args.skip_reranking else 'None (retrieval only)'}")

    # Group groundtruth by (post_id, criterion)
    grouped: Dict[tuple, List] = {}
    for row in groundtruth:
        if row.post_id not in eval_posts:
            continue
        grouped.setdefault((row.post_id, row.criterion_id), []).append(row)

    # Run evaluation
    rankings_reranked = []
    total_queries = len(grouped)

    logger.info(f"Evaluating {total_queries} queries on {args.split} split...")

    for i, ((post_id, criterion_id), rows) in enumerate(tqdm(sorted(grouped.items()), desc="Processing queries")):
        query_text = criteria_map.get(criterion_id)
        if query_text is None:
            continue

        gold_uids = {r.sent_uid for r in rows if r.groundtruth == 1}

        # Skip if no positives and flag is set
        if args.skip_no_positives and not gold_uids:
            continue

        # Check if post has embeddings
        if post_id not in post_id_to_indices:
            logger.warning(f"Post {post_id} not found in cached embeddings, skipping")
            continue

        # Get query embedding
        if criterion_id not in query_embeddings:
            logger.warning(f"Criterion {criterion_id} not found in cached query embeddings, skipping")
            continue

        query_embedding = query_embeddings[criterion_id]

        # Retrieve with cached embeddings
        top_k_retriever = cfg['retriever'].get('top_k_retriever', 24)
        retrieval_results = retrieve_with_embeddings(
            query_embedding,
            corpus_embeddings,
            post_id_to_indices[post_id],
            corpus_metadata['sent_uids'],
            top_k=top_k_retriever
        )

        # Rerank (optional)
        if not args.skip_reranking:
            top_k_final = cfg['retriever'].get('top_k_final', 10)
            reranked_results = rerank_with_jina(query_text, retrieval_results, sent_uid_to_text, top_k=top_k_final)
            final_results = reranked_results
        else:
            final_results = retrieval_results

        # Extract rankings
        ranked_uids = [r[0] for r in final_results]

        rankings_reranked.append({
            "query_id": f"{post_id}_{criterion_id}",
            "ranked_uids": ranked_uids,
            "gold_uids": gold_uids,
        })

    # Compute metrics
    logger.info("Computing metrics...")
    metrics_reranked = evaluate_rankings(rankings_reranked, ks=args.ks)

    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({args.split} split)")
    print(f"Retriever: {corpus_metadata['model_id']} (pre-computed embeddings)")
    print(f"Reranker: {'jinaai/jina-reranker-v3' if not args.skip_reranking else 'None'}")
    print("=" * 60)
    print(f"\nQueries evaluated: {len(rankings_reranked)}")
    print("\nResults:")
    for metric, value in sorted(metrics_reranked.items()):
        print(f"  {metric}: {value:.4f}")

    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/eval_nv_embed")
        output_dir.mkdir(parents=True, exist_ok=True)
        stage = "reranked" if not args.skip_reranking else "retrieval_only"
        output_path = output_dir / f"eval_{args.split}_{stage}_{timestamp}.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "split": args.split,
        "retriever": corpus_metadata['model_id'],
        "retriever_mode": "pre_computed_embeddings",
        "reranker": "jinaai/jina-reranker-v3" if not args.skip_reranking else None,
        "n_queries": len(rankings_reranked),
        "metrics": metrics_reranked,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Save per-query rankings if requested
    if args.save_per_query_rankings:
        rankings_path = Path(output_path).parent / f"{Path(output_path).stem}_per_query_rankings.jsonl"
        with open(rankings_path, "w", encoding="utf-8") as f:
            for ranking in rankings_reranked:
                ranking_serializable = {
                    "query_id": ranking["query_id"],
                    "ranked_uids": ranking["ranked_uids"],
                    "gold_uids": sorted(list(ranking["gold_uids"])),
                }
                f.write(json.dumps(ranking_serializable) + "\n")
        logger.info(f"Per-query rankings saved to {rankings_path}")


if __name__ == "__main__":
    main()
