#!/usr/bin/env python3
"""Analyze LLM Reranker impact on DEV split (v2 - with proper LLM score application).

This script evaluates the LLM reranker results on the DEV split by:
1. Loading baseline P3 Graph Reranker scores
2. Loading LLM reranker results and mapping back to candidates
3. Creating hybrid scores: LLM reranking for top-10, P3 for rest
4. Computing ranking metrics for both
5. Generating comparison report

Usage:
    python scripts/llm_integration/analyze_dev_reranker_v2.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import ndcg_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_sentence_corpus():
    """Load sentence corpus for text -> sent_uid mapping."""
    corpus_file = Path("data/groundtruth/sentence_corpus.jsonl")
    sent_uid_to_text = {}
    text_to_sent_uid = {}

    with open(corpus_file) as f:
        for line in f:
            sent = json.loads(line)
            sent_uid = sent["sent_uid"]
            text = sent["text"]
            sent_uid_to_text[sent_uid] = text
            # Use first 80 chars as key (matches truncation in LLM results)
            text_key = text[:80]
            text_to_sent_uid[text_key] = sent_uid

    logger.info(f"Loaded {len(sent_uid_to_text)} sentences from corpus")
    return sent_uid_to_text, text_to_sent_uid


def load_graphs(cache_dir: Path, n_samples: int = 885):
    """Load graphs from GNN cache."""
    cache_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()])
    if not cache_dirs:
        raise ValueError(f"No graph cache found in {cache_dir}")

    latest_cache = cache_dirs[-1]
    logger.info(f"Loading graphs from {latest_cache}")

    fold_file = latest_cache / "fold_0.pt"
    data = torch.load(fold_file, weights_only=False)

    if isinstance(data, dict) and 'graphs' in data:
        graphs = data['graphs']
    else:
        graphs = data

    logger.info(f"Loaded {len(graphs)} graphs from fold 0")
    return graphs[:n_samples]


def load_llm_reranker_results(results_file: Path):
    """Load LLM reranker results."""
    with open(results_file) as f:
        results = json.load(f)
    logger.info(f"Loaded {len(results)} LLM reranker results")
    return results


def apply_llm_reranking(graph, llm_result, sent_uid_to_text):
    """Apply LLM reranking to graph scores.

    Strategy:
    - LLM reranked top-5 sentences get scores based on their LLM ranking
    - Remaining candidates keep their P3 scores (scaled down)

    Args:
        graph: PyG graph with candidate_uids and reranker_scores
        llm_result: LLM reranker result with reranked_top5 and llm_scores_top5
        sent_uid_to_text: Mapping from sent_uid to text

    Returns:
        Updated scores array
    """
    if not llm_result['success']:
        # Fallback to P3 scores
        return graph.reranker_scores.numpy() if hasattr(graph.reranker_scores, 'numpy') else graph.reranker_scores

    # Get baseline P3 scores
    p3_scores = graph.reranker_scores.numpy() if hasattr(graph.reranker_scores, 'numpy') else graph.reranker_scores
    candidate_uids = graph.candidate_uids

    # Create mapping from candidate index to sent_uid
    idx_to_uid = {i: uid for i, uid in enumerate(candidate_uids)}

    # Create mapping from sent_uid to candidate index
    uid_to_idx = {uid: i for i, uid in enumerate(candidate_uids)}

    # Get LLM reranked sentences (truncated to 80 chars)
    llm_top5_texts = llm_result['reranked_top5']
    llm_top5_scores = llm_result['llm_scores_top5']

    # Find which candidates match the LLM top-5
    llm_matched_indices = []
    for llm_text in llm_top5_texts:
        # Try to find matching candidate by text prefix
        for idx, uid in idx_to_uid.items():
            candidate_text = sent_uid_to_text.get(uid, "")[:80]
            if candidate_text == llm_text:
                llm_matched_indices.append(idx)
                break

    # Create new scores
    new_scores = p3_scores.copy()

    # Apply LLM scores to matched top-5 candidates
    # Scale LLM scores to be higher than all P3 scores
    if llm_matched_indices:
        max_p3_score = np.max(p3_scores)
        score_range = max_p3_score - np.min(p3_scores)

        for i, idx in enumerate(llm_matched_indices[:len(llm_top5_scores)]):
            # LLM scores are already normalized [0, 1] with rank-based decay
            # Scale them to be above P3 scores
            llm_score = llm_top5_scores[i]
            new_scores[idx] = max_p3_score + (1.0 - llm_score) * score_range * 0.1 + llm_score * score_range

    return new_scores


def compute_ranking_metrics(graphs, llm_results=None, sent_uid_to_text=None):
    """Compute ranking metrics.

    Args:
        graphs: List of PyG graphs
        llm_results: Optional LLM reranker results. If None, uses baseline P3 scores.
        sent_uid_to_text: Mapping from sent_uid to text (needed for LLM reranking)

    Returns:
        Dictionary with metrics
    """
    mrr_list = []
    recall_at_5 = []
    recall_at_10 = []
    ndcg_at_5 = []
    ndcg_at_10 = []

    applied_llm = 0

    for i, graph in enumerate(graphs):
        # Get ground truth labels
        gold_labels = graph.node_labels.numpy() if hasattr(graph.node_labels, 'numpy') else graph.node_labels

        if llm_results and i < len(llm_results) and sent_uid_to_text:
            # Apply LLM reranking
            scores = apply_llm_reranking(graph, llm_results[i], sent_uid_to_text)
            if llm_results[i]['success']:
                applied_llm += 1
        else:
            # Use baseline P3 scores
            scores = graph.reranker_scores.numpy() if hasattr(graph.reranker_scores, 'numpy') else graph.reranker_scores

        # Compute metrics
        sorted_indices = np.argsort(scores)[::-1]

        # MRR
        gold_ranks = []
        for idx, label in enumerate(gold_labels):
            if label == 1:
                rank = np.where(sorted_indices == idx)[0]
                if len(rank) > 0:
                    gold_ranks.append(rank[0] + 1)

        if gold_ranks:
            mrr_list.append(1.0 / min(gold_ranks))

        # Recall@K
        top_5_indices = sorted_indices[:5]
        top_10_indices = sorted_indices[:10]

        if sum(gold_labels) > 0:
            recall_at_5.append(sum(gold_labels[top_5_indices]) / sum(gold_labels))
            recall_at_10.append(sum(gold_labels[top_10_indices]) / sum(gold_labels))

        # nDCG@K
        try:
            ndcg_at_5.append(ndcg_score([gold_labels], [scores], k=5))
            ndcg_at_10.append(ndcg_score([gold_labels], [scores], k=10))
        except:
            pass

    logger.info(f"Applied LLM reranking to {applied_llm}/{len(graphs)} queries")

    return {
        'mrr': np.mean(mrr_list) if mrr_list else 0.0,
        'recall@5': np.mean(recall_at_5) if recall_at_5 else 0.0,
        'recall@10': np.mean(recall_at_10) if recall_at_10 else 0.0,
        'ndcg@5': np.mean(ndcg_at_5) if ndcg_at_5 else 0.0,
        'ndcg@10': np.mean(ndcg_at_10) if ndcg_at_10 else 0.0,
        'n_queries': len(graphs),
        'n_with_gold': sum(1 for g in graphs if sum(g.node_labels.numpy() if hasattr(g.node_labels, 'numpy') else g.node_labels) > 0),
        'n_llm_applied': applied_llm,
    }


def generate_report(baseline_metrics, llm_metrics, output_file: Path):
    """Generate comparison report."""
    report = []
    report.append("=" * 80)
    report.append(" LLM RERANKER EVALUATION - DEV SPLIT ANALYSIS")
    report.append("=" * 80)
    report.append("")
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Queries: {baseline_metrics['n_queries']}")
    report.append(f"Queries with Evidence: {baseline_metrics['n_with_gold']}")
    report.append(f"LLM Reranking Applied: {llm_metrics.get('n_llm_applied', 0)}/{llm_metrics['n_queries']}")
    report.append("")
    report.append("-" * 80)
    report.append("RANKING METRICS COMPARISON")
    report.append("-" * 80)
    report.append("")

    metrics_names = ['mrr', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']

    report.append(f"{'Metric':<20} {'Baseline (P3)':<20} {'+ LLM Reranker':<20} {'Delta':<15}")
    report.append("-" * 80)

    for metric in metrics_names:
        baseline_val = baseline_metrics[metric]
        llm_val = llm_metrics[metric]
        delta = llm_val - baseline_val
        delta_pct = (delta / baseline_val * 100) if baseline_val > 0 else 0

        delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
        report.append(f"{metric:<20} {baseline_val:<20.4f} {llm_val:<20.4f} {delta_str:<15}")

    report.append("")
    report.append("-" * 80)
    report.append("INTERPRETATION")
    report.append("-" * 80)
    report.append("")

    # Determine if improvement is significant
    mrr_delta = llm_metrics['mrr'] - baseline_metrics['mrr']
    ndcg10_delta = llm_metrics['ndcg@10'] - baseline_metrics['ndcg@10']
    recall5_delta = llm_metrics['recall@5'] - baseline_metrics['recall@5']

    if mrr_delta > 0.02 or ndcg10_delta > 0.02 or recall5_delta > 0.02:
        report.append("✅ SIGNIFICANT IMPROVEMENT")
        report.append(f"   LLM reranker shows meaningful gains: MRR {mrr_delta:+.4f}, ")
        report.append(f"   nDCG@10 {ndcg10_delta:+.4f}, Recall@5 {recall5_delta:+.4f}")
        report.append("")
        report.append("   Recommendation: Pursue full evaluation with verifier.")
        report.append("   Consider enabling billing for complete 5-fold CV.")
    elif mrr_delta > 0.005 or ndcg10_delta > 0.005 or recall5_delta > 0.005:
        report.append("🟡 MODEST IMPROVEMENT")
        report.append(f"   LLM reranker shows small gains: MRR {mrr_delta:+.4f}, ")
        report.append(f"   nDCG@10 {ndcg10_delta:+.4f}, Recall@5 {recall5_delta:+.4f}")
        report.append("")
        report.append("   Recommendation: Consider cost/benefit trade-off.")
        report.append("   V7 baseline may be sufficient for deployment.")
    else:
        report.append("❌ MARGINAL/NO IMPROVEMENT")
        report.append(f"   LLM reranker does not show clear improvement.")
        report.append(f"   Deltas: MRR {mrr_delta:+.4f}, nDCG@10 {ndcg10_delta:+.4f}, Recall@5 {recall5_delta:+.4f}")
        report.append("")
        report.append("   Recommendation: Deploy V7 baseline without LLM.")
        report.append("   Current P3 Graph Reranker is performing well.")

    report.append("")
    report.append("-" * 80)
    report.append("NOTES")
    report.append("-" * 80)
    report.append("")
    report.append("- Baseline: P3 Graph Reranker scores only")
    report.append("- LLM Reranker: Hybrid scores (LLM for top-K, P3 for rest)")
    report.append("- Evaluation split: DEV (30% of fold 0)")
    report.append("- LLM Model: gemini-2.5-flash")
    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(output_file, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {output_file}")

    return report_text


def main():
    logger.info("=" * 60)
    logger.info("LLM Reranker DEV Split Analysis (v2)")
    logger.info("=" * 60)

    # Setup paths
    graph_cache_dir = Path("data/cache/gnn")
    llm_results_file = Path("outputs/llm_dev_eval/20260118_010109/reranker_results.json")
    output_dir = Path("outputs/llm_dev_eval/20260118_010109")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sentence corpus for text mapping
    logger.info("Loading sentence corpus...")
    sent_uid_to_text, text_to_sent_uid = load_sentence_corpus()

    # Load data
    logger.info("Loading graphs...")
    graphs = load_graphs(graph_cache_dir, n_samples=885)

    logger.info("Loading LLM reranker results...")
    llm_results = load_llm_reranker_results(llm_results_file)

    # Compute baseline metrics
    logger.info("Computing baseline metrics (P3 Graph Reranker only)...")
    baseline_metrics = compute_ranking_metrics(graphs, llm_results=None)

    logger.info("Baseline metrics:")
    for k, v in baseline_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Compute LLM metrics
    logger.info("Computing LLM reranker metrics...")
    llm_metrics = compute_ranking_metrics(graphs, llm_results=llm_results, sent_uid_to_text=sent_uid_to_text)

    logger.info("LLM reranker metrics:")
    for k, v in llm_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Generate report
    logger.info("Generating comparison report...")
    report_file = output_dir / "reranker_analysis_v2.txt"
    report_text = generate_report(baseline_metrics, llm_metrics, report_file)

    print()
    print(report_text)

    logger.info("=" * 60)
    logger.info("Analysis Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
