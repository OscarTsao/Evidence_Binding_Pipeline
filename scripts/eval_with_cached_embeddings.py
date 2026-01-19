#!/usr/bin/env python3
"""Evaluate pipeline using cached NV-Embed-v2 embeddings.

This script loads pre-computed embeddings from the nv-embed-v2 environment
and runs retrieval + reranking in the main environment.

Usage:
    # First, encode corpus in nv-embed-v2 env:
    conda activate nv-embed-v2
    python scripts/encode_corpus_nv_embed.py --config configs/default.yaml
    conda deactivate

    # Then run evaluation in main env:
    conda activate llmhe
    python scripts/eval_with_cached_embeddings.py --config configs/default.yaml --split test
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


def load_cached_embeddings(cache_dir: Path):
    """Load pre-computed embeddings and metadata."""
    embeddings_file = cache_dir / "corpus_embeddings.npy"
    metadata_file = cache_dir / "corpus_metadata.pkl"

    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_file}\n"
            f"Please run: conda activate nv-embed-v2 && "
            f"python scripts/encode_corpus_nv_embed.py --config configs/default.yaml"
        )

    logger.info(f"Loading cached embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"Loaded {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
    logger.info(f"Model: {metadata['model_id']}")

    # Create sent_uid -> index mapping
    sent_uid_to_idx = {uid: idx for idx, uid in enumerate(metadata['sent_uids'])}

    # Create post_id -> indices mapping
    post_id_to_indices = {}
    for idx, post_id in enumerate(metadata['post_ids']):
        post_id_to_indices.setdefault(post_id, []).append(idx)

    return embeddings, sent_uid_to_idx, post_id_to_indices, metadata


def encode_query_with_nv_embed(query_text: str, model_id="nvidia/NV-Embed-v2"):
    """Encode a single query with NV-Embed-v2 using the model's encode() method."""
    import torch
    from transformers import AutoModel

    # Load model (will be slow on first call, but cached after)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Add instruction prefix for queries
    query_instruction = "Instruct: Given a criterion, retrieve sentences that provide direct supporting evidence.\nQuery: "

    with torch.no_grad():
        query_emb = model.encode(
            [query_text],
            instruction=query_instruction,
            max_length=512,
        )
        query_emb = query_emb.cpu().numpy()[0]

        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        return query_emb


def retrieve_with_cached_embeddings(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    post_indices: List[int],
    sent_uids: List[str],
    top_k: int = 24
):
    """Retrieve top-k sentences from post using cached embeddings."""
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

    # Rerank
    reranked = reranker.rerank(query_text, documents, top_k=top_k)

    # Map back to sent_uids
    results = [
        (candidates[result['index']][0], result['score'])
        for result in reranked
    ]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate with cached NV-Embed-v2 embeddings")
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
            sent_uid_to_text[data['sent_uid']] = data['text']  # Field is 'text' in JSONL

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

    # Load cached embeddings
    cache_dir = Path(cfg['paths']['cache_dir'])
    corpus_embeddings, sent_uid_to_idx, post_id_to_indices, metadata = load_cached_embeddings(cache_dir)

    logger.info(f"Retriever: {metadata['model_id']}")
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

    # Load query encoder (will be slow on first query, but necessary)
    logger.info("Loading NV-Embed-v2 for query encoding...")
    # We'll encode queries on-the-fly (model cached after first call)

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

        # Encode query
        query_embedding = encode_query_with_nv_embed(query_text, model_id=metadata['model_id'])

        # Retrieve with cached embeddings
        top_k_retriever = cfg['retriever'].get('top_k_retriever', 24)
        retrieval_results = retrieve_with_cached_embeddings(
            query_embedding,
            corpus_embeddings,
            post_id_to_indices[post_id],
            metadata['sent_uids'],
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
    print(f"Retriever: {metadata['model_id']} (cached embeddings)")
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
        "retriever": metadata['model_id'],
        "retriever_mode": "cached_embeddings",
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
