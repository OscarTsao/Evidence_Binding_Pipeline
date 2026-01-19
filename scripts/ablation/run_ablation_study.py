#!/usr/bin/env python3
"""
Run systematic ablation studies on DEV split.

Supports:
- Study 1: Retriever comparison
- Study 2: Reranker ablation
- Study 3: Top-K retrieval pool ablation
- Study 4: BGE-M3 modality ablation
- Study 5: Fusion method ablation
- Study 6: Reranker input size ablation

Usage:
    # Study 1: Retriever comparison
    python scripts/ablation/run_ablation_study.py \
        --study retriever_comparison \
        --output outputs/ablation/study_1_retriever_comparison.json

    # Study 2: Reranker ablation
    python scripts/ablation/run_ablation_study.py \
        --study reranker_ablation \
        --retriever nv-embed-v2 \
        --output outputs/ablation/study_2_reranker_ablation.json

    # Study 3: Top-K ablation
    python scripts/ablation/run_ablation_study.py \
        --study topk_ablation \
        --retriever nv-embed-v2 \
        --output outputs/ablation/study_3_topk_ablation.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.retrieval_eval import evaluate_rankings
from final_sc_review.pipeline.zoo_pipeline import load_zoo_pipeline_from_config, ZooPipeline
from final_sc_review.reranker.zoo import RerankerZoo
from final_sc_review.retriever.zoo import RetrieverZoo
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dev_queries(groundtruth_rows, criteria_map, eval_posts):
    """Get queries for DEV split."""
    # Group by (post_id, criterion)
    grouped = {}
    for row in groundtruth_rows:
        if row.post_id not in eval_posts:
            continue
        key = (row.post_id, row.criterion_id)
        grouped.setdefault(key, []).append(row)

    # Create query dict
    queries = {}
    for (post_id, criterion_id), rows in grouped.items():
        query_id = f"{post_id}_{criterion_id}"
        gold_uids = {r.sent_uid for r in rows if r.groundtruth == 1}

        queries[query_id] = {
            "post_id": post_id,
            "criterion": criterion_id,
            "criterion_text": criteria_map.get(criterion_id, criterion_id),
            "gold_uids": gold_uids,
        }

    return queries


def evaluate_config(
    config_name: str,
    retriever_name: str,
    reranker_name: Optional[str],
    top_k_retriever: int,
    top_k_rerank: Optional[int],
    top_k_final: int,
    queries: Dict,
    sentences: List,
    cache_dir: Path,
    device: str = "cuda",
    **kwargs,
) -> Dict:
    """Evaluate a single configuration."""
    logger.info(f"Evaluating config: {config_name}")
    logger.info(f"  Retriever: {retriever_name}")
    logger.info(f"  Reranker: {reranker_name or 'None'}")
    logger.info(f"  top_k_retriever: {top_k_retriever}")
    logger.info(f"  top_k_rerank: {top_k_rerank or 'N/A'}")
    logger.info(f"  top_k_final: {top_k_final}")

    # Initialize pipeline
    start_time = time.time()

    # Import config class
    from final_sc_review.pipeline.zoo_pipeline import ZooPipelineConfig

    # Create config
    pipeline_config = ZooPipelineConfig(
        retriever_name=retriever_name,
        reranker_name=reranker_name,
        top_k_retriever=top_k_retriever,
        top_k_final=top_k_final,
        device=device,
    )

    # Create pipeline
    pipeline = ZooPipeline(
        sentences=sentences,
        cache_dir=cache_dir,
        config=pipeline_config,
    )

    # Encode corpus (cached if already done)
    pipeline.retriever.encode_corpus(rebuild=False)
    encoding_time = time.time() - start_time

    # Run evaluation
    eval_start = time.time()
    all_rankings = []
    queries_evaluated = 0

    for query_id, query_data in queries.items():
        if len(query_data["gold_uids"]) == 0:
            continue  # Skip queries with no positives

        # Retrieve
        results = pipeline.retrieve(
            query=query_data["criterion_text"],
            post_id=query_data["post_id"],
            top_k=top_k_final,
        )

        # Extract rankings (results are tuples: (sent_uid, text, score))
        ranked_uids = [r[0] for r in results]
        all_rankings.append({
            "query_id": query_id,
            "ranked_uids": ranked_uids,
            "gold_uids": list(query_data["gold_uids"]),
        })
        queries_evaluated += 1

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    # Compute metrics
    ks = [1, 3, 5, 10, 20]
    metrics = evaluate_rankings(all_rankings, ks=ks)

    # Add runtime metrics
    metrics["runtime"] = {
        "encoding_time_sec": encoding_time,
        "evaluation_time_sec": eval_time,
        "total_time_sec": total_time,
        "queries_evaluated": queries_evaluated,
        "queries_per_sec": queries_evaluated / eval_time if eval_time > 0 else 0,
    }

    # Add configuration info
    result = {
        "config_name": config_name,
        "retriever": retriever_name,
        "reranker": reranker_name,
        "top_k_retriever": top_k_retriever,
        "top_k_rerank": top_k_rerank,
        "top_k_final": top_k_final,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Add any extra kwargs
    for key, value in kwargs.items():
        if key not in result:
            result[key] = value

    logger.info(f"  nDCG@10: {metrics.get('ndcg@10', 0.0):.4f}")
    logger.info(f"  Recall@10: {metrics.get('recall@10', 0.0):.4f}")
    logger.info(f"  Runtime: {total_time:.1f}s ({metrics['runtime']['queries_per_sec']:.1f} q/s)")

    return result


def run_retriever_comparison(
    queries: Dict,
    sentences: List,
    cache_dir: Path,
    output_path: Path,
    device: str = "cuda",
):
    """Study 1: Retriever comparison."""
    logger.info("=" * 60)
    logger.info("STUDY 1: Retriever Comparison")
    logger.info("=" * 60)

    retrievers = [
        "nv-embed-v2",           # SOTA (baseline from STEP 4)
        "qwen3-embed-4b",        # SOTA
        "gte-qwen2-7b",          # SOTA
        "bge-m3",                # Hybrid baseline
        "e5-mistral-7b",         # SOTA
        "bge-large-en-v1.5",     # Strong baseline
        "e5-large-v2",           # Baseline
    ]

    results = []
    for retriever_name in retrievers:
        try:
            result = evaluate_config(
                config_name=f"retriever_{retriever_name}",
                retriever_name=retriever_name,
                reranker_name=None,  # Retrieval-only
                top_k_retriever=24,  # From HPO
                top_k_rerank=None,
                top_k_final=10,
                queries=queries,
                sentences=sentences,
                cache_dir=cache_dir,
                device=device,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {retriever_name}: {e}")
            continue

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "study": "retriever_comparison",
            "timestamp": datetime.now().isoformat(),
            "n_configs": len(results),
            "results": results,
        }, f, indent=2)

    logger.info(f"✓ Study 1 complete. Results saved to {output_path}")
    return results


def run_reranker_ablation(
    retriever_name: str,
    queries: Dict,
    sentences: List,
    cache_dir: Path,
    output_path: Path,
    device: str = "cuda",
):
    """Study 2: Reranker ablation."""
    logger.info("=" * 60)
    logger.info("STUDY 2: Reranker Ablation")
    logger.info(f"Retriever: {retriever_name}")
    logger.info("=" * 60)

    rerankers = [
        None,                    # Baseline (retrieval-only)
        "jina-reranker-v3",      # Best from HPO
        "mxbai-rerank-base-v2",  # Fast strong baseline
        "bge-reranker-v2-m3",    # Strong baseline
        "ms-marco-minilm",       # Fast baseline
    ]

    results = []
    for reranker_name in rerankers:
        try:
            config_name = f"reranker_{reranker_name if reranker_name else 'none'}"
            result = evaluate_config(
                config_name=config_name,
                retriever_name=retriever_name,
                reranker_name=reranker_name,
                top_k_retriever=24,
                top_k_rerank=10 if reranker_name else None,
                top_k_final=10,
                queries=queries,
                sentences=sentences,
                cache_dir=cache_dir,
                device=device,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate reranker {reranker_name}: {e}")
            continue

    # Compute gains
    baseline_ndcg = results[0]["metrics"].get("ndcg@10", 0.0) if results else 0.0
    for i, result in enumerate(results):
        if i == 0:
            continue  # Skip baseline
        ndcg = result["metrics"].get("ndcg@10", 0.0)
        result["reranking_gain"] = {
            "absolute": ndcg - baseline_ndcg,
            "relative": (ndcg - baseline_ndcg) / baseline_ndcg if baseline_ndcg > 0 else 0.0,
        }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "study": "reranker_ablation",
            "retriever": retriever_name,
            "timestamp": datetime.now().isoformat(),
            "n_configs": len(results),
            "baseline_ndcg@10": baseline_ndcg,
            "results": results,
        }, f, indent=2)

    logger.info(f"✓ Study 2 complete. Results saved to {output_path}")
    return results


def run_topk_ablation(
    retriever_name: str,
    queries: Dict,
    sentences: List,
    cache_dir: Path,
    output_path: Path,
    device: str = "cuda",
):
    """Study 3: Top-K retrieval pool ablation."""
    logger.info("=" * 60)
    logger.info("STUDY 3: Top-K Ablation")
    logger.info(f"Retriever: {retriever_name}")
    logger.info("=" * 60)

    top_k_values = [10, 20, 24, 30, 50, 100]

    results = []
    for top_k in top_k_values:
        try:
            result = evaluate_config(
                config_name=f"topk_{top_k}",
                retriever_name=retriever_name,
                reranker_name=None,  # Retrieval-only
                top_k_retriever=top_k,
                top_k_rerank=None,
                top_k_final=10,
                queries=queries,
                sentences=sentences,
                cache_dir=cache_dir,
                device=device,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate top_k={top_k}: {e}")
            continue

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "study": "topk_ablation",
            "retriever": retriever_name,
            "timestamp": datetime.now().isoformat(),
            "n_configs": len(results),
            "results": results,
        }, f, indent=2)

    logger.info(f"✓ Study 3 complete. Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic ablation studies on DEV split"
    )
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        choices=[
            "retriever_comparison",
            "reranker_ablation",
            "topk_ablation",
            "all",
        ],
        help="Which ablation study to run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="nv-embed-v2",
        help="Retriever to use (for studies 2, 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Load data
    logger.info("Loading sentence corpus...")
    corpus_path = Path(config["paths"]["sentence_corpus"])
    sentences = load_sentence_corpus(corpus_path)
    logger.info(f"  Loaded {len(sentences)} sentences")

    logger.info("Loading groundtruth...")
    gt_path = Path(config["paths"]["groundtruth"])
    groundtruth_rows = load_groundtruth(gt_path)
    logger.info(f"  Loaded {len(groundtruth_rows)} groundtruth rows")

    logger.info("Loading criteria...")
    criteria_path = Path(config["paths"]["data_dir"]) / "DSM5" / "MDD_Criteira.json"
    criteria = load_criteria(criteria_path)
    criteria_map = {c.criterion_id: c.text for c in criteria}
    logger.info(f"  Loaded {len(criteria)} criteria")

    # Create splits
    logger.info("Creating post-ID-disjoint splits...")
    split_config = config["split"]
    post_ids = sorted({row.post_id for row in groundtruth_rows})
    splits = split_post_ids(
        post_ids,
        seed=args.seed,
        train_ratio=split_config.get("train_ratio", 0.6),
        val_ratio=split_config.get("val_ratio", 0.2),
        test_ratio=split_config.get("test_ratio", 0.2),
    )
    eval_posts = set(splits["val"])  # DEV split
    logger.info(f"  DEV split has {len(eval_posts)} posts")

    # Get DEV (validation) queries
    logger.info("Extracting DEV queries...")
    dev_queries = get_dev_queries(groundtruth_rows, criteria_map, eval_posts)
    logger.info(f"  Found {len(dev_queries)} DEV queries")

    # Filter to queries with positives
    dev_queries = {k: v for k, v in dev_queries.items() if len(v["gold_uids"]) > 0}
    logger.info(f"  {len(dev_queries)} queries with positive examples")

    # Set up cache directory
    cache_dir = Path(config["paths"]["cache_dir"]) / "ablation"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Run study
    if args.study == "retriever_comparison":
        run_retriever_comparison(
            queries=dev_queries,
            sentences=sentences,
            cache_dir=cache_dir,
            output_path=args.output,
            device=args.device,
        )
    elif args.study == "reranker_ablation":
        run_reranker_ablation(
            retriever_name=args.retriever,
            queries=dev_queries,
            sentences=sentences,
            cache_dir=cache_dir,
            output_path=args.output,
            device=args.device,
        )
    elif args.study == "topk_ablation":
        run_topk_ablation(
            retriever_name=args.retriever,
            queries=dev_queries,
            sentences=sentences,
            cache_dir=cache_dir,
            output_path=args.output,
            device=args.device,
        )
    elif args.study == "all":
        logger.info("Running all ablation studies...")

        # Study 1
        run_retriever_comparison(
            queries=dev_queries,
            sentences=sentences,
            cache_dir=cache_dir,
            output_path=args.output.parent / "study_1_retriever_comparison.json",
            device=args.device,
        )

        # Study 2
        run_reranker_ablation(
            retriever_name=args.retriever,
            queries=dev_queries,
            sentences=sentences,
            cache_dir=cache_dir,
            output_path=args.output.parent / "study_2_reranker_ablation.json",
            device=args.device,
        )

        # Study 3
        run_topk_ablation(
            retriever_name=args.retriever,
            queries=dev_queries,
            sentences=sentences,
            cache_dir=cache_dir,
            output_path=args.output.parent / "study_3_topk_ablation.json",
            device=args.device,
        )

        logger.info(f"✓ All studies complete. Results saved to {args.output.parent}/")

    logger.info("=" * 60)
    logger.info("ABLATION STUDY COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
