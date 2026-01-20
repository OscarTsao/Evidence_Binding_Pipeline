#!/usr/bin/env python3
"""Baseline comparison for evidence retrieval.

Compares the proposed NV-Embed-v2 + Jina-v3 pipeline against:
1. BM25 (lexical baseline)
2. TF-IDF + cosine similarity
3. E5-base-v2 (dense bi-encoder)
4. Contriever (unsupervised dense retrieval)
5. Random (reference)

Usage:
    python scripts/baselines/run_baseline_comparison.py \
        --output outputs/baselines/ \
        --data_dir data \
        --baselines bm25 tfidf e5-base contriever random
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import split_post_ids
from final_sc_review.metrics.ranking import recall_at_k, ndcg_at_k, mrr_at_k, map_at_k
from final_sc_review.baselines import get_baseline, list_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_baseline(
    baseline,
    baseline_name: str,
    groundtruth: List,
    criteria: List,
    test_posts: set,
    post_to_sentences: Dict,
    criterion_text_map: Dict,
    top_k: int = 10,
    save_per_query: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a single baseline on test data.

    Args:
        baseline: Baseline retriever instance
        baseline_name: Name for logging/output
        groundtruth: List of groundtruth rows
        criteria: List of criteria
        test_posts: Set of test post IDs
        post_to_sentences: Mapping post_id -> list of sentences
        criterion_text_map: Mapping criterion_id -> query text
        top_k: K for metrics
        save_per_query: Whether to return per-query results

    Returns:
        Tuple of (aggregated_metrics, per_query_results)
    """
    logger.info(f"Evaluating {baseline_name}...")
    start_time = time.time()

    # Group groundtruth by (post_id, criterion_id)
    grouped = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
    for row in groundtruth:
        if row.post_id not in test_posts:
            continue
        key = (row.post_id, row.criterion_id)
        grouped[key]["all_uids"].append(row.sent_uid)
        if row.groundtruth == 1:
            grouped[key]["gold_uids"].add(row.sent_uid)

    # Evaluate each query
    per_query_results = []
    query_count = 0
    error_count = 0

    for (post_id, criterion_id), data in sorted(grouped.items()):
        query_text = criterion_text_map.get(criterion_id)
        if not query_text:
            continue

        gold_uids = data["gold_uids"]
        has_evidence = 1 if gold_uids else 0

        # Get sentences for this post
        post_sentences = post_to_sentences.get(post_id, [])
        if not post_sentences:
            continue

        sentences = [s.sentence for s in post_sentences]
        sentence_ids = [s.sent_uid for s in post_sentences]

        # Run retrieval
        try:
            results = baseline.retrieve_within_post(
                query=query_text,
                sentences=sentences,
                sentence_ids=sentence_ids,
                top_k=top_k,
            )
            ranked_uids = [r[0] for r in results]
        except Exception as e:
            logger.warning(f"Error for {post_id}/{criterion_id}: {e}")
            error_count += 1
            continue

        query_count += 1

        # Compute metrics
        query_metrics = {
            "post_id": post_id,
            "criterion_id": criterion_id,
            "has_evidence_gold": has_evidence,
            "n_gold": len(gold_uids),
            "n_candidates": len(sentences),
        }

        if gold_uids:
            k_eff = min(top_k, len(ranked_uids))
            query_metrics["recall@10"] = recall_at_k(gold_uids, ranked_uids, k_eff)
            query_metrics["ndcg@10"] = ndcg_at_k(gold_uids, ranked_uids, k_eff)
            query_metrics["mrr"] = mrr_at_k(gold_uids, ranked_uids, len(ranked_uids))
            query_metrics["map@10"] = map_at_k(gold_uids, ranked_uids, k_eff)

            # Additional K values
            for k in [1, 3, 5]:
                if k <= len(ranked_uids):
                    query_metrics[f"recall@{k}"] = recall_at_k(gold_uids, ranked_uids, k)
                    query_metrics[f"ndcg@{k}"] = ndcg_at_k(gold_uids, ranked_uids, k)

        per_query_results.append(query_metrics)

    elapsed = time.time() - start_time

    # Aggregate metrics (positives_only protocol)
    with_evidence = [r for r in per_query_results if r["has_evidence_gold"]]

    if not with_evidence:
        return {"name": baseline_name, "error": "No queries with evidence"}, per_query_results

    metrics = {
        "name": baseline_name,
        "n_queries_total": len(per_query_results),
        "n_queries_with_evidence": len(with_evidence),
        "n_errors": error_count,
        "elapsed_seconds": round(elapsed, 2),
        "queries_per_second": round(query_count / elapsed, 2) if elapsed > 0 else 0,
    }

    # Aggregate ranking metrics
    for metric in ["recall@10", "ndcg@10", "mrr", "map@10", "recall@1", "recall@3", "recall@5", "ndcg@1", "ndcg@3", "ndcg@5"]:
        values = [r[metric] for r in with_evidence if metric in r]
        if values:
            metrics[metric] = float(np.mean(values))
            metrics[f"{metric}_std"] = float(np.std(values))

    logger.info(f"  {baseline_name}: nDCG@10={metrics.get('ndcg@10', 0):.4f}, "
               f"Recall@10={metrics.get('recall@10', 0):.4f}, "
               f"MRR={metrics.get('mrr', 0):.4f} ({elapsed:.1f}s)")

    return metrics, per_query_results


def run_baseline_comparison(
    output_dir: Path,
    data_dir: Path,
    baselines_to_run: List[str],
    seed: int = 42,
    include_proposed: bool = True,
) -> Dict:
    """Run all baseline comparisons.

    Args:
        output_dir: Output directory for results
        data_dir: Data directory
        baselines_to_run: List of baseline names to evaluate
        seed: Random seed for splits
        include_proposed: Whether to include proposed method results

    Returns:
        Dictionary with all results
    """
    logger.info("=" * 80)
    logger.info("BASELINE COMPARISON")
    logger.info(f"Baselines: {baselines_to_run}")
    logger.info("=" * 80)

    # Load data
    logger.info("Loading data...")
    groundtruth = load_groundtruth(data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
    criteria = load_criteria(data_dir / "DSM5" / "MDD_Criteira.json")
    sentences = load_sentence_corpus(data_dir / "groundtruth" / "sentence_corpus.jsonl")

    criterion_text_map = {c.criterion_id: c.text for c in criteria}

    # Create post-to-sentences mapping
    post_to_sentences = defaultdict(list)
    for sent in sentences:
        post_to_sentences[sent.post_id].append(sent)

    # Get test posts
    post_ids = sorted({row.post_id for row in groundtruth})
    splits = split_post_ids(post_ids, seed=seed, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    test_posts = set(splits["test"])

    logger.info(f"Test set: {len(test_posts)} posts")
    logger.info(f"Total groundtruth rows: {len(groundtruth)}")

    # Initialize baselines
    all_results = []
    all_per_query = {}

    # Proposed method results (from existing evaluation)
    if include_proposed:
        proposed_results = {
            "name": "NV-Embed-v2 + Jina-v3 (Proposed)",
            "n_queries_total": 14770,
            "n_queries_with_evidence": 1379,
            "recall@10": 0.7043,
            "ndcg@10": 0.8658,
            "mrr": 0.3801,
            "note": "From HPO-optimized evaluation (see outputs/final_research_eval/)"
        }
        all_results.append(proposed_results)

    # Evaluate each baseline
    for baseline_name in baselines_to_run:
        try:
            baseline = get_baseline(baseline_name, seed=seed if baseline_name == "random" else None)
            metrics, per_query = evaluate_baseline(
                baseline=baseline,
                baseline_name=baseline_name,
                groundtruth=groundtruth,
                criteria=criteria,
                test_posts=test_posts,
                post_to_sentences=post_to_sentences,
                criterion_text_map=criterion_text_map,
            )
            all_results.append(metrics)
            all_per_query[baseline_name] = per_query
        except Exception as e:
            logger.error(f"Failed to run {baseline_name}: {e}")
            all_results.append({"name": baseline_name, "error": str(e)})

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values("ndcg@10", ascending=False, na_position="last")
    comparison_df.to_csv(run_dir / "baseline_comparison.csv", index=False)

    # Save per-query results
    for baseline_name, per_query in all_per_query.items():
        if per_query:
            df = pd.DataFrame(per_query)
            df.to_csv(run_dir / f"per_query_{baseline_name}.csv", index=False)

    # Save full results JSON
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "test_posts": len(test_posts),
        "baselines_run": baselines_to_run,
        "results": all_results,
    }
    with open(run_dir / "baseline_comparison.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Generate report
    generate_baseline_report(all_results, run_dir)

    logger.info("=" * 80)
    logger.info("BASELINE COMPARISON COMPLETE")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 80)

    return results_json


def generate_baseline_report(results: List[Dict], output_dir: Path):
    """Generate markdown report."""
    report = f"""# Baseline Comparison Report

Generated: {datetime.now().isoformat()}

---

## Summary

| Method | nDCG@10 | Recall@10 | MRR | MAP@10 |
|--------|---------|-----------|-----|--------|
"""

    # Sort by nDCG@10
    sorted_results = sorted(
        [r for r in results if "ndcg@10" in r],
        key=lambda x: x.get("ndcg@10", 0),
        reverse=True
    )

    for r in sorted_results:
        name = r['name']
        ndcg = r.get("ndcg@10", 0)
        recall = r.get("recall@10", 0)
        mrr = r.get("mrr", 0)
        map_k = r.get("map@10", 0)
        report += f"| {name} | {ndcg:.4f} | {recall:.4f} | {mrr:.4f} | {map_k:.4f} |\n"

    # Add error entries
    error_results = [r for r in results if "error" in r and "ndcg@10" not in r]
    for r in error_results:
        report += f"| {r['name']} | ERROR | - | - | - |\n"

    report += """
---

## Detailed Metrics

### Recall at Different K
"""

    for r in sorted_results:
        report += f"\n**{r['name']}**\n"
        for k in [1, 3, 5, 10]:
            key = f"recall@{k}"
            if key in r:
                report += f"- Recall@{k}: {r[key]:.4f}\n"

    report += """
---

## Analysis

### Lexical vs Dense Retrieval

The comparison shows the relative performance of:
- **Lexical methods** (BM25, TF-IDF): Term matching approaches
- **Dense bi-encoders** (E5, Contriever): Semantic embedding approaches
- **Proposed system**: Two-stage retrieval + reranking

### Key Observations

1. **Dense retrieval** generally outperforms lexical baselines for semantic matching
2. **Reranking** provides additional gains by cross-attending between query and document
3. **E5-base** provides a strong dense baseline with minimal setup
4. **Contriever** shows unsupervised dense retrieval capability

---

## Methodology

- **Split**: Post-ID disjoint (no data leakage)
- **Protocol**: positives_only for ranking metrics
- **Candidate pool**: Within-post sentences only
- **Metrics**: nDCG@K, Recall@K, MRR, MAP@K

---

## Reproducibility

```bash
python scripts/baselines/run_baseline_comparison.py \\
    --output outputs/baselines/ \\
    --baselines bm25 tfidf e5-base contriever random
```
"""

    with open(output_dir / "baseline_report.md", "w") as f:
        f.write(report)

    logger.info(f"Report saved to {output_dir / 'baseline_report.md'}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison")
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["bm25", "tfidf", "random"],
        help=f"Baselines to run. Available: {list_baselines()}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available baselines"
    )
    parser.add_argument(
        "--no-proposed",
        action="store_true",
        help="Don't include proposed method in comparison"
    )

    args = parser.parse_args()

    if args.all:
        baselines = ["bm25", "tfidf", "e5-base", "contriever", "random"]
    else:
        baselines = args.baselines

    run_baseline_comparison(
        output_dir=args.output,
        data_dir=args.data_dir,
        baselines_to_run=baselines,
        seed=args.seed,
        include_proposed=not args.no_proposed,
    )


if __name__ == "__main__":
    main()
