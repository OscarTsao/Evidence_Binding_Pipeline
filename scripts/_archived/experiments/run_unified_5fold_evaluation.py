#!/usr/bin/env python3
"""Unified 5-Fold Evaluation for Baseline vs GNN Comparison.

This script provides a consistent evaluation protocol for comparing:
1. Baseline: NV-Embed-v2 + Jina-Reranker-v3 (no GNN)
2. GNN: NV-Embed-v2 + Jina-Reranker-v3 + P3 Graph Reranker

Both are evaluated with identical:
- Candidate pool size (top_k_final=10, matching HPO best config)
- Evaluation protocol (positives_only for ranking, all_queries for classification)
- Data splits (5-fold CV on full dataset, final TEST evaluation)
- Metric computation (same functions)

Usage:
    python scripts/experiments/run_unified_5fold_evaluation.py \
        --output_dir outputs/unified_evaluation

Author: Evidence Binding Pipeline
Date: 2026-01-21
"""

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metric Computation (Single Source of Truth)
# ============================================================================

def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute nDCG@K with binary relevance.

    Args:
        gold_mask: Binary array indicating gold items
        scores: Score array for ranking
        k: Cutoff position

    Returns:
        nDCG@K value in [0, 1]
    """
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    # DCG
    dcg = 0.0
    for i in range(min(k, len(sorted_gold))):
        if sorted_gold[i]:
            dcg += 1.0 / math.log2(i + 2)

    # IDCG - based on gold items IN THE CANDIDATE POOL
    # This is the fair comparison since we can only rank what we retrieve
    n_gold = int(gold_mask.sum())
    if n_gold == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_gold)))

    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Recall@K."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    n_gold = gold_mask.sum()
    if n_gold == 0:
        return 0.0

    return sorted_gold[:k].sum() / n_gold


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) == 0:
        return 0.0

    return 1.0 / (gold_positions[0] + 1)


def compute_all_ranking_metrics(
    gold_mask: np.ndarray,
    scores: np.ndarray,
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Compute all ranking metrics for a single query."""
    metrics = {}

    # nDCG@K
    for k in ks:
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(gold_mask, scores, k)

    # Recall@K
    for k in ks:
        metrics[f"recall@{k}"] = compute_recall_at_k(gold_mask, scores, k)

    # MRR
    metrics["mrr"] = compute_mrr(gold_mask, scores)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across queries with mean and std."""
    if not all_metrics:
        return {}

    agg = {}
    keys = all_metrics[0].keys()

    for key in keys:
        values = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "n": len(values),
        }

    return agg


# ============================================================================
# Data Loading
# ============================================================================

def load_groundtruth(groundtruth_path: Path) -> pd.DataFrame:
    """Load groundtruth labels."""
    df = pd.read_csv(groundtruth_path)
    logger.info(f"Loaded groundtruth: {len(df)} rows")
    return df


def load_sentence_corpus(corpus_path: Path) -> Dict[str, Dict]:
    """Load sentence corpus."""
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            item = json.loads(line)
            corpus[item["sent_uid"]] = item
    logger.info(f"Loaded corpus: {len(corpus)} sentences")
    return corpus


def create_5fold_splits(
    groundtruth_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create 5-fold splits based on post_id (post-ID disjoint).

    Returns:
        DataFrame with fold_id column added
    """
    np.random.seed(seed)

    # Get unique post IDs
    post_ids = groundtruth_df["post_id"].unique()
    np.random.shuffle(post_ids)

    # Assign folds
    fold_assignments = {}
    fold_size = len(post_ids) // n_folds

    for i, pid in enumerate(post_ids):
        fold_id = min(i // fold_size, n_folds - 1)
        fold_assignments[pid] = fold_id

    # Add fold_id to dataframe
    df = groundtruth_df.copy()
    df["fold_id"] = df["post_id"].map(fold_assignments)

    # Log fold sizes
    for fold in range(n_folds):
        fold_posts = df[df["fold_id"] == fold]["post_id"].nunique()
        fold_queries = len(df[df["fold_id"] == fold])
        logger.info(f"Fold {fold}: {fold_posts} posts, {fold_queries} queries")

    return df


def build_query_df(groundtruth_df: pd.DataFrame) -> pd.DataFrame:
    """Build query dataframe with gold UIDs per query."""
    # Group by (post_id, criterion) to create queries
    queries = []

    for (post_id, criterion), group in groundtruth_df.groupby(["post_id", "criterion"]):
        gold_uids = group[group["groundtruth"] == 1]["sent_uid"].tolist()
        has_evidence = len(gold_uids) > 0
        fold_id = group["fold_id"].iloc[0]

        queries.append({
            "query_id": f"{post_id}_{criterion}",
            "post_id": post_id,
            "criterion_id": criterion,
            "gold_uids": gold_uids,
            "has_evidence": has_evidence,
            "n_gold": len(gold_uids),
            "fold_id": fold_id,
        })

    query_df = pd.DataFrame(queries)
    logger.info(f"Built {len(query_df)} queries")
    logger.info(f"  Queries with evidence: {query_df['has_evidence'].sum()}")

    return query_df


# ============================================================================
# Retrieval and Reranking
# ============================================================================

def run_retrieval_and_reranking(
    query_df: pd.DataFrame,
    corpus: Dict[str, Dict],
    groundtruth_df: pd.DataFrame,
    retriever_name: str = "nv-embed-v2",
    reranker_name: str = "jina-reranker-v3",
    top_k_retriever: int = 24,
    top_k_final: int = 10,
    cache_dir: Path = Path("data/cache"),
    device: str = "cuda",
) -> pd.DataFrame:
    """Run retrieval and reranking pipeline.

    Returns:
        DataFrame with candidate_uids, candidate_scores, and evaluation data
    """
    from final_sc_review.retriever.zoo import RetrieverZoo
    from final_sc_review.reranker.zoo import RerankerZoo
    from final_sc_review.data.schemas import Sentence

    # Prepare sentences for retriever (must be Sentence dataclass instances)
    sentences = []
    for uid, item in corpus.items():
        sentences.append(Sentence(
            sent_uid=uid,
            post_id=item["post_id"],
            sid=item["sid"],
            text=item["text"],
        ))

    # Initialize retriever
    logger.info(f"Initializing retriever: {retriever_name}")
    retriever_zoo = RetrieverZoo(
        sentences=sentences,
        cache_dir=cache_dir,
        device=device,
    )
    retriever = retriever_zoo.get_retriever(retriever_name)
    retriever.encode_corpus(rebuild=False)

    # Initialize reranker
    logger.info(f"Initializing reranker: {reranker_name}")
    reranker_zoo = RerankerZoo(device=device)
    reranker = reranker_zoo.get_reranker(reranker_name)

    # Criteria descriptions for query construction
    criteria_descriptions = {
        "A.1": "Depressed mood most of the day, nearly every day",
        "A.2": "Markedly diminished interest or pleasure in activities",
        "A.3": "Significant weight loss or gain, or change in appetite",
        "A.4": "Insomnia or hypersomnia nearly every day",
        "A.5": "Psychomotor agitation or retardation",
        "A.6": "Fatigue or loss of energy nearly every day",
        "A.7": "Feelings of worthlessness or excessive guilt",
        "A.8": "Diminished ability to think or concentrate",
        "A.9": "Recurrent thoughts of death or suicidal ideation",
        "A.10": "SPECIAL_CASE: expert discrimination cases",
    }

    results = []

    for _, query_row in tqdm(query_df.iterrows(), total=len(query_df), desc="Processing queries"):
        post_id = query_row["post_id"]
        criterion_id = query_row["criterion_id"]
        query_id = query_row["query_id"]
        gold_uids = set(query_row["gold_uids"])

        # Create query text
        criterion_desc = criteria_descriptions.get(criterion_id, criterion_id)
        query_text = f"Evidence for criterion {criterion_id}: {criterion_desc}"

        # Retrieve candidates within this post
        retrieval_results = retriever.retrieve_within_post(
            query=query_text,
            post_id=post_id,
            top_k=top_k_retriever,
        )

        if not retrieval_results:
            logger.warning(f"No candidates for {query_id}")
            continue

        candidate_uids = [r.sent_uid for r in retrieval_results]
        candidate_texts = [r.text for r in retrieval_results]

        # Rerank
        rerank_scores = reranker.score_pairs(query_text, candidate_texts)

        # Sort by reranker scores and take top_k_final
        sorted_indices = np.argsort(-np.array(rerank_scores))[:top_k_final]

        final_uids = [candidate_uids[i] for i in sorted_indices]
        final_scores = np.array([rerank_scores[i] for i in sorted_indices])

        # Get gold labels for these candidates
        gold_mask = np.array([1 if uid in gold_uids else 0 for uid in final_uids])

        results.append({
            "query_id": query_id,
            "post_id": post_id,
            "criterion_id": criterion_id,
            "candidate_uids": final_uids,
            "reranker_scores": final_scores.tolist(),
            "gold_mask": gold_mask.tolist(),
            "has_evidence": query_row["has_evidence"],
            "n_gold_in_pool": int(gold_mask.sum()),
            "n_gold_total": query_row["n_gold"],
            "fold_id": query_row["fold_id"],
        })

    result_df = pd.DataFrame(results)
    logger.info(f"Processed {len(result_df)} queries")

    return result_df


# ============================================================================
# GNN Training and Inference
# ============================================================================

def build_gnn_graphs(
    result_df: pd.DataFrame,
    corpus: Dict[str, Dict],
    embedding_cache: np.ndarray,
    uid_to_idx: Dict[str, int],
    knn_k: int = 5,
    knn_threshold: float = 0.7,
) -> List:
    """Build PyG graphs for GNN training/inference."""
    import torch
    from torch_geometric.data import Data

    graphs = []

    for _, row in tqdm(result_df.iterrows(), total=len(result_df), desc="Building graphs"):
        candidate_uids = row["candidate_uids"]
        reranker_scores = np.array(row["reranker_scores"])
        gold_mask = np.array(row["gold_mask"])

        n_nodes = len(candidate_uids)
        if n_nodes == 0:
            continue

        # Get embeddings
        embeddings = []
        for uid in candidate_uids:
            if uid in uid_to_idx:
                embeddings.append(embedding_cache[uid_to_idx[uid]])
            else:
                embeddings.append(np.zeros(embedding_cache.shape[1]))
        embeddings = np.stack(embeddings)

        # Build node features: [embedding, reranker_score, rank_percentile]
        ranks = np.argsort(np.argsort(-reranker_scores))
        rank_percentile = ranks / max(n_nodes - 1, 1)

        node_features = np.concatenate([
            embeddings,
            reranker_scores.reshape(-1, 1),
            rank_percentile.reshape(-1, 1),
        ], axis=1)

        # Build semantic kNN edges
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = embeddings / norms
        sim_matrix = normed @ normed.T

        edges_src, edges_dst = [], []
        for i in range(n_nodes):
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf
            top_k_idx = np.argsort(-sims)[:knn_k]
            for j in top_k_idx:
                if sims[j] >= knn_threshold:
                    edges_src.append(i)
                    edges_dst.append(j)

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)

        # Create PyG Data
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            reranker_scores=torch.tensor(reranker_scores, dtype=torch.float32),
            node_labels=torch.tensor(gold_mask, dtype=torch.float32),
            y=torch.tensor([int(row["has_evidence"])], dtype=torch.float32),
        )
        data.query_id = row["query_id"]
        data.criterion_id = row["criterion_id"]
        data.fold_id = row["fold_id"]
        data.candidate_uids = candidate_uids

        graphs.append(data)

    logger.info(f"Built {len(graphs)} graphs")
    return graphs


def train_gnn_fold(
    train_graphs: List,
    val_graphs: List,
    input_dim: int,
    config: Dict,
    device: str = "cuda",
) -> Tuple["torch.nn.Module", Dict]:
    """Train GNN on a single fold."""
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch_geometric.loader import DataLoader

    from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss

    # Initialize model
    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
        alpha_init=config.get("alpha_init", 0.7),
        learn_alpha=config.get("learn_alpha", True),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )

    loss_fn = GraphRerankerLoss(
        alpha_rank=config.get("alpha_rank", 1.0),
        alpha_align=config.get("alpha_align", 0.5),
        alpha_reg=config.get("alpha_reg", 0.1),
        margin=config.get("margin", 0.1),
    )

    train_loader = DataLoader(train_graphs, batch_size=config.get("batch_size", 32), shuffle=True)

    best_val_ndcg = -1
    best_model_state = None
    patience_counter = 0
    max_patience = config.get("patience", 10)

    for epoch in range(config.get("max_epochs", 30)):
        # Training
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss, _ = loss_fn(refined, batch.reranker_scores, batch.node_labels, batch.batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        val_metrics = evaluate_gnn(model, val_graphs, device)
        val_ndcg = val_metrics.get("ndcg@10", {}).get("mean", 0)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {"best_val_ndcg": best_val_ndcg}


def evaluate_gnn(
    model: "torch.nn.Module",
    graphs: List,
    device: str = "cuda",
    use_refined: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate GNN on graphs."""
    import torch

    model.eval()
    all_metrics = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy()

            if not gold_mask.any():
                continue

            if use_refined:
                refined = model(g.x, g.edge_index, g.reranker_scores)
                scores = refined.cpu().numpy()
            else:
                scores = g.reranker_scores.cpu().numpy()

            metrics = compute_all_ranking_metrics(gold_mask, scores)
            all_metrics.append(metrics)

    return aggregate_metrics(all_metrics)


def evaluate_baseline(result_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluate baseline (reranker only) on result dataframe."""
    all_metrics = []

    for _, row in result_df.iterrows():
        gold_mask = np.array(row["gold_mask"])
        scores = np.array(row["reranker_scores"])

        if not gold_mask.any():
            continue

        metrics = compute_all_ranking_metrics(gold_mask, scores)
        all_metrics.append(metrics)

    return aggregate_metrics(all_metrics)


# ============================================================================
# Main Evaluation
# ============================================================================

def run_5fold_evaluation(
    output_dir: Path,
    groundtruth_path: Path,
    corpus_path: Path,
    retriever_name: str = "nv-embed-v2",
    reranker_name: str = "jina-reranker-v3",
    top_k_retriever: int = 24,
    top_k_final: int = 10,
    n_folds: int = 5,
    seed: int = 42,
    device: str = "cuda",
    exclude_a10: bool = True,
) -> Dict:
    """Run full 5-fold cross-validation evaluation.

    Returns:
        Dictionary with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    logger.info("Loading data...")
    groundtruth_df = load_groundtruth(groundtruth_path)
    corpus = load_sentence_corpus(corpus_path)

    # Create 5-fold splits
    logger.info("Creating 5-fold splits...")
    groundtruth_df = create_5fold_splits(groundtruth_df, n_folds=n_folds, seed=seed)

    # Build query dataframe
    query_df = build_query_df(groundtruth_df)

    # Exclude A.10 if specified
    if exclude_a10:
        original_count = len(query_df)
        query_df = query_df[query_df["criterion_id"] != "A.10"]
        logger.info(f"Excluded A.10: {original_count} -> {len(query_df)} queries")

    # Run retrieval and reranking
    logger.info("Running retrieval and reranking...")
    result_df = run_retrieval_and_reranking(
        query_df=query_df,
        corpus=corpus,
        groundtruth_df=groundtruth_df,
        retriever_name=retriever_name,
        reranker_name=reranker_name,
        top_k_retriever=top_k_retriever,
        top_k_final=top_k_final,
        device=device,
    )

    # Save intermediate results
    result_df.to_pickle(output_dir / "retrieval_results.pkl")

    # Build embedding cache for GNN
    logger.info("Building embedding cache...")
    cache_dir = Path("data/cache")
    embedding_path = cache_dir / "retriever_zoo" / retriever_name / "embeddings.npy"
    uid_mapping_path = cache_dir / "retriever_zoo" / retriever_name / "uid_to_idx.json"

    embedding_cache = np.load(embedding_path)
    with open(uid_mapping_path) as f:
        uid_to_idx = json.load(f)

    logger.info(f"Loaded embeddings: {embedding_cache.shape}")

    # Build graphs
    logger.info("Building GNN graphs...")
    graphs = build_gnn_graphs(
        result_df=result_df,
        corpus=corpus,
        embedding_cache=embedding_cache,
        uid_to_idx=uid_to_idx,
    )

    # GNN config
    gnn_config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "max_epochs": 30,
        "patience": 10,
        "alpha_init": 0.7,
        "learn_alpha": True,
        "alpha_rank": 1.0,
        "alpha_align": 0.5,
        "alpha_reg": 0.1,
        "margin": 0.1,
    }

    input_dim = graphs[0].x.shape[1] if graphs else 0

    # 5-fold cross-validation
    fold_results = []

    for fold_id in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_id}")
        logger.info(f"{'='*60}")

        # Split graphs
        train_graphs = [g for g in graphs if g.fold_id != fold_id]
        val_graphs = [g for g in graphs if g.fold_id == fold_id]

        # Split result_df for baseline evaluation
        train_df = result_df[result_df["fold_id"] != fold_id]
        val_df = result_df[result_df["fold_id"] == fold_id]

        logger.info(f"Train: {len(train_graphs)} graphs, Val: {len(val_graphs)} graphs")

        # Evaluate baseline
        logger.info("Evaluating baseline (Jina-v3 only)...")
        baseline_metrics = evaluate_baseline(val_df)

        # Train and evaluate GNN
        logger.info("Training GNN...")
        model, train_info = train_gnn_fold(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            input_dim=input_dim,
            config=gnn_config,
            device=device,
        )

        logger.info("Evaluating GNN...")
        gnn_metrics = evaluate_gnn(model, val_graphs, device, use_refined=True)

        # Compute improvement
        improvement = {}
        for key in baseline_metrics:
            if key in gnn_metrics:
                base_val = baseline_metrics[key]["mean"]
                gnn_val = gnn_metrics[key]["mean"]
                improvement[key] = gnn_val - base_val

        fold_result = {
            "fold_id": fold_id,
            "n_train": len(train_graphs),
            "n_val": len(val_graphs),
            "baseline": baseline_metrics,
            "gnn": gnn_metrics,
            "improvement": improvement,
        }
        fold_results.append(fold_result)

        # Log key metrics
        logger.info(f"Baseline nDCG@10: {baseline_metrics.get('ndcg@10', {}).get('mean', 0):.4f}")
        logger.info(f"GNN nDCG@10: {gnn_metrics.get('ndcg@10', {}).get('mean', 0):.4f}")
        logger.info(f"Improvement: {improvement.get('ndcg@10', 0):.4f}")

    # Aggregate across folds
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATED RESULTS (5-Fold CV)")
    logger.info(f"{'='*60}")

    aggregated = {
        "baseline": {},
        "gnn": {},
        "improvement": {},
    }

    for key in fold_results[0]["baseline"]:
        baseline_means = [fr["baseline"][key]["mean"] for fr in fold_results]
        gnn_means = [fr["gnn"][key]["mean"] for fr in fold_results]

        aggregated["baseline"][key] = {
            "mean": np.mean(baseline_means),
            "std": np.std(baseline_means),
        }
        aggregated["gnn"][key] = {
            "mean": np.mean(gnn_means),
            "std": np.std(gnn_means),
        }
        aggregated["improvement"][key] = {
            "mean": np.mean(gnn_means) - np.mean(baseline_means),
            "std": np.std([g - b for g, b in zip(gnn_means, baseline_means)]),
        }

    # Log aggregated results
    for key in ["ndcg@10", "mrr", "recall@10"]:
        if key in aggregated["baseline"]:
            b = aggregated["baseline"][key]
            g = aggregated["gnn"][key]
            i = aggregated["improvement"][key]
            logger.info(f"{key}:")
            logger.info(f"  Baseline: {b['mean']:.4f} ± {b['std']:.4f}")
            logger.info(f"  GNN:      {g['mean']:.4f} ± {g['std']:.4f}")
            logger.info(f"  Δ:        {i['mean']:+.4f} ± {i['std']:.4f}")

    # Build final results
    results = {
        "config": {
            "retriever": retriever_name,
            "reranker": reranker_name,
            "top_k_retriever": top_k_retriever,
            "top_k_final": top_k_final,
            "n_folds": n_folds,
            "seed": seed,
            "exclude_a10": exclude_a10,
            "gnn_config": gnn_config,
        },
        "fold_results": fold_results,
        "aggregated": aggregated,
        "timestamp": timestamp,
    }

    # Save results
    results_path = output_dir / f"unified_5fold_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    logger.info(f"Saved results to {results_path}")

    # Save summary
    summary_path = output_dir / "latest_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified 5-Fold Evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/unified_evaluation")
    parser.add_argument("--groundtruth", type=str, default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--corpus", type=str, default="data/groundtruth/sentence_corpus.jsonl")
    parser.add_argument("--retriever", type=str, default="nv-embed-v2")
    parser.add_argument("--reranker", type=str, default="jina-reranker-v3")
    parser.add_argument("--top_k_retriever", type=int, default=24)
    parser.add_argument("--top_k_final", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include_a10", action="store_true", help="Include A.10 (not recommended)")

    args = parser.parse_args()

    results = run_5fold_evaluation(
        output_dir=Path(args.output_dir),
        groundtruth_path=Path(args.groundtruth),
        corpus_path=Path(args.corpus),
        retriever_name=args.retriever,
        reranker_name=args.reranker,
        top_k_retriever=args.top_k_retriever,
        top_k_final=args.top_k_final,
        n_folds=args.n_folds,
        seed=args.seed,
        device=args.device,
        exclude_a10=not args.include_a10,
    )

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Baseline nDCG@10: {results['aggregated']['baseline']['ndcg@10']['mean']:.4f} ± {results['aggregated']['baseline']['ndcg@10']['std']:.4f}")
    print(f"GNN nDCG@10:      {results['aggregated']['gnn']['ndcg@10']['mean']:.4f} ± {results['aggregated']['gnn']['ndcg@10']['std']:.4f}")
    print(f"Improvement:      {results['aggregated']['improvement']['ndcg@10']['mean']:+.4f}")


if __name__ == "__main__":
    main()
