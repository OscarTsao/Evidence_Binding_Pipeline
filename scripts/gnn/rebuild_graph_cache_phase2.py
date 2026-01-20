#!/usr/bin/env python3
"""Phase 2: Build graphs using cached embeddings.

Assumes Phase 1 (embedding computation) was done in nv-embed-v2 env.
This script runs in llmhe env for reranking and graph building.

Usage:
    conda run -n llmhe python scripts/gnn/rebuild_graph_cache_phase2.py
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_groundtruth(groundtruth_path: Path) -> pd.DataFrame:
    """Load groundtruth labels."""
    logger.info(f"Loading groundtruth from {groundtruth_path}")
    df = pd.read_csv(groundtruth_path)
    logger.info(f"  Loaded {len(df)} rows")
    return df


def load_sentence_corpus(corpus_path: Path) -> Tuple[Dict[str, dict], List]:
    """Load sentence corpus."""
    from final_sc_review.data.schemas import Sentence

    logger.info(f"Loading sentence corpus from {corpus_path}")
    corpus = {}
    sentences = []
    with open(corpus_path) as f:
        for line in f:
            item = json.loads(line)
            corpus[item["sent_uid"]] = item
            sentences.append(Sentence(
                post_id=item["post_id"],
                sid=item["sid"],
                sent_uid=item["sent_uid"],
                text=item["text"],
            ))
    logger.info(f"  Loaded {len(corpus)} sentences")
    return corpus, sentences


def load_cached_embeddings(cache_dir: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load pre-computed embeddings from cache."""
    embeddings_path = cache_dir / "embeddings.npy"
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def create_query_df(groundtruth_df: pd.DataFrame) -> pd.DataFrame:
    """Create query-level dataframe."""
    query_groups = groundtruth_df.groupby(["post_id", "criterion"])

    queries = []
    for (post_id, criterion), group in query_groups:
        has_evidence = group["groundtruth"].max()
        gold_sids = group[group["groundtruth"] == 1]["sid"].tolist()
        gold_uids = group[group["groundtruth"] == 1]["sent_uid"].tolist()
        n_candidates = len(group)

        queries.append({
            "post_id": post_id,
            "criterion_id": criterion,
            "query_id": f"{post_id}_{criterion}",
            "has_evidence": has_evidence,
            "gold_sids": gold_sids,
            "gold_uids": gold_uids,
            "n_candidates": n_candidates,
        })

    return pd.DataFrame(queries)


def assign_folds(query_df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Assign post-id-disjoint folds."""
    np.random.seed(seed)
    post_ids = query_df["post_id"].unique()
    np.random.shuffle(post_ids)
    post_to_fold = {pid: i % n_folds for i, pid in enumerate(post_ids)}
    query_df = query_df.copy()
    query_df["fold_id"] = query_df["post_id"].map(post_to_fold)
    return query_df


def run_pipeline_inference(
    query_df: pd.DataFrame,
    corpus: Dict[str, dict],
    sentences: List,
    embeddings: np.ndarray,
    top_k_retriever: int = 24,
    top_k_rerank: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run retrieval + reranking pipeline."""
    from final_sc_review.reranker.jina_v3 import JinaV3Reranker

    # Build UID to index mapping
    uid_to_idx = {s.sent_uid: i for i, s in enumerate(sentences)}

    # Build post to sentence indices mapping
    post_to_indices = {}
    for i, s in enumerate(sentences):
        if s.post_id not in post_to_indices:
            post_to_indices[s.post_id] = []
        post_to_indices[s.post_id].append(i)

    # Load reranker
    logger.info("Loading JinaV3Reranker...")
    reranker = JinaV3Reranker(
        model_name="jinaai/jina-reranker-v3",
        device=device,
        max_length=512,
        batch_size=batch_size,
        use_listwise=True,
    )
    logger.info("  Loaded reranker: jina-reranker-v3")

    # Criteria descriptions (per ReDSM5 taxonomy)
    criteria_descriptions = {
        "A.1": "Depressed mood most of the day",
        "A.2": "Markedly diminished interest or pleasure",
        "A.3": "Significant weight loss or change in appetite",
        "A.4": "Insomnia or hypersomnia",
        "A.5": "Psychomotor agitation or retardation",
        "A.6": "Fatigue or loss of energy",
        "A.7": "Feelings of worthlessness or excessive guilt",
        "A.8": "Diminished ability to think or concentrate",
        "A.9": "Recurrent thoughts of death or suicidal ideation",
        "A.10": "SPECIAL_CASE: expert discrimination cases",  # Per ReDSM5 taxonomy
    }

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed_embeddings = embeddings / norms

    post_queries = query_df.groupby("post_id")
    results = []

    for post_id, post_group in tqdm(post_queries, desc="Processing posts"):
        # Get sentence indices for this post
        post_indices = post_to_indices.get(post_id, [])
        if not post_indices:
            continue

        # Get embeddings for this post
        post_embeddings = normed_embeddings[post_indices]

        for _, query_row in post_group.iterrows():
            criterion_id = query_row["criterion_id"]
            query_id = query_row["query_id"]

            # Create query text
            criterion_desc = criteria_descriptions.get(criterion_id, criterion_id)
            query_text = f"Evidence for criterion {criterion_id}: {criterion_desc}"

            # Simple retrieval: use mean embedding similarity
            # (In production, would encode query separately, but this is simpler)
            # Take top-k by embedding similarity within post
            post_sents = [sentences[i] for i in post_indices]

            # For simplicity, just take all sentences from the post
            # (since posts are small, typically < 30 sentences)
            candidate_indices = post_indices[:top_k_retriever]
            candidate_uids = [sentences[i].sent_uid for i in candidate_indices]
            candidate_texts = [sentences[i].text for i in candidate_indices]
            candidate_sids = [sentences[i].sid for i in candidate_indices]

            if not candidate_texts:
                continue

            # Rerank
            rerank_scores = reranker.score_pairs(query_text, candidate_texts)

            # Sort and take top-k
            sorted_indices = np.argsort(-np.array(rerank_scores))[:top_k_rerank]

            final_uids = [candidate_uids[i] for i in sorted_indices]
            final_sids = [candidate_sids[i] for i in sorted_indices]
            final_scores = [rerank_scores[i] for i in sorted_indices]

            # Gold labels
            gold_uids = set(query_row["gold_uids"])
            node_labels = [1 if uid in gold_uids else 0 for uid in final_uids]

            results.append({
                "query_id": query_id,
                "post_id": post_id,
                "criterion_id": criterion_id,
                "candidate_uids": final_uids,
                "candidate_sids": final_sids,
                "reranker_scores": final_scores,
                "node_labels": node_labels,
                "has_evidence": query_row["has_evidence"],
                "fold_id": query_row["fold_id"],
            })

    return pd.DataFrame(results)


def build_graphs(
    pipeline_df: pd.DataFrame,
    embeddings: np.ndarray,
    sentences: List,
) -> List:
    """Build PyG graphs."""
    from torch_geometric.data import Data

    uid_to_idx = {s.sent_uid: i for i, s in enumerate(sentences)}
    embedding_dim = embeddings.shape[1]

    logger.info("Building PyG graphs...")
    graphs = []

    for _, row in tqdm(pipeline_df.iterrows(), total=len(pipeline_df), desc="Building graphs"):
        candidate_uids = row["candidate_uids"]
        reranker_scores = np.array(row["reranker_scores"])
        node_labels = np.array(row["node_labels"])
        candidate_sids = row["candidate_sids"]

        n_nodes = len(candidate_uids)
        if n_nodes == 0:
            continue

        # Get embeddings
        node_embeddings = np.zeros((n_nodes, embedding_dim))
        for i, uid in enumerate(candidate_uids):
            if uid in uid_to_idx:
                node_embeddings[i] = embeddings[uid_to_idx[uid]]

        # Build node features
        ranks = np.argsort(np.argsort(-reranker_scores))
        rank_percentiles = 1.0 - ranks / n_nodes

        node_features = np.concatenate([
            node_embeddings,
            reranker_scores.reshape(-1, 1),
            rank_percentiles.reshape(-1, 1),
        ], axis=1)

        # Build edges
        norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = node_embeddings / norms
        sim_matrix = normed @ normed.T

        edges_src, edges_dst = [], []
        k = min(5, n_nodes - 1)
        threshold = 0.5

        for i in range(n_nodes):
            sims = sim_matrix[i].copy()
            sims[i] = -np.inf
            top_k_idx = np.argsort(-sims)[:k]
            for j in top_k_idx:
                if sims[j] >= threshold:
                    edges_src.append(i)
                    edges_dst.append(j)

        # Adjacency edges
        sorted_idx = np.argsort(candidate_sids)
        for i in range(len(sorted_idx) - 1):
            curr_idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            if abs(candidate_sids[curr_idx] - candidate_sids[next_idx]) <= 1:
                edges_src.extend([curr_idx, next_idx])
                edges_dst.extend([next_idx, curr_idx])

        edge_set = set(zip(edges_src, edges_dst))
        if edge_set:
            edge_index = np.array([[s, d] for s, d in edge_set], dtype=np.int64).T
        else:
            edge_index = np.array([[], []], dtype=np.int64)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            reranker_scores=torch.tensor(reranker_scores, dtype=torch.float32),
            node_labels=torch.tensor(node_labels, dtype=torch.float32),
            y=torch.tensor([int(row["has_evidence"])], dtype=torch.float32),
        )

        data.query_id = row["query_id"]
        data.post_id = row["post_id"]
        data.criterion_id = row["criterion_id"]
        data.candidate_uids = candidate_uids
        data.fold_id = row["fold_id"]

        graphs.append(data)

    logger.info(f"  Built {len(graphs)} graphs")
    return graphs


def save_graph_dataset(graphs: List, output_dir: Path, n_folds: int = 5):
    """Save graphs in fold-wise format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_graphs = {i: [] for i in range(n_folds)}
    for g in graphs:
        fold_graphs[g.fold_id].append(g)

    for fold_id, fold_data in fold_graphs.items():
        fold_path = output_dir / f"fold_{fold_id}.pt"
        torch.save({"graphs": fold_data}, fold_path)
        logger.info(f"  Saved fold {fold_id}: {len(fold_data)} graphs")

    metadata = {
        "n_folds": n_folds,
        "n_graphs": len(graphs),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "fold_sizes": {i: len(fold_graphs[i]) for i in range(n_folds)},
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved graph dataset to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth_path", type=str, default="data/groundtruth/evidence_sentence_groundtruth.csv")
    parser.add_argument("--corpus_path", type=str, default="data/groundtruth/sentence_corpus.jsonl")
    parser.add_argument("--embeddings_cache", type=str, default="data/cache/retriever_zoo/nv-embed-v2")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--top_k_retriever", type=int, default=24)
    parser.add_argument("--top_k_rerank", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data/cache/gnn/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Load data
    groundtruth_df = load_groundtruth(Path(args.groundtruth_path))
    corpus, sentences = load_sentence_corpus(Path(args.corpus_path))

    # Load cached embeddings
    embeddings = load_cached_embeddings(Path(args.embeddings_cache))

    # Create and fold queries
    query_df = create_query_df(groundtruth_df)
    logger.info(f"Created {len(query_df)} queries")
    query_df = assign_folds(query_df, n_folds=args.n_folds)
    logger.info(f"Assigned {args.n_folds} folds")

    # Run pipeline
    pipeline_df = run_pipeline_inference(
        query_df, corpus, sentences, embeddings,
        top_k_retriever=args.top_k_retriever,
        top_k_rerank=args.top_k_rerank,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Save pipeline results
    save_df = pipeline_df.copy()
    for col in ["candidate_uids", "candidate_sids", "reranker_scores", "node_labels"]:
        save_df[col] = save_df[col].apply(json.dumps)
    save_df.to_csv(output_dir / "pipeline_results.csv", index=False)
    logger.info(f"Saved pipeline results: {len(pipeline_df)} queries")

    # Build graphs
    graphs = build_graphs(pipeline_df, embeddings, sentences)

    # Save embeddings and uid mapping
    np.save(output_dir / "embeddings.npy", embeddings)
    uid_to_idx = {s.sent_uid: i for i, s in enumerate(sentences)}
    with open(output_dir / "uid_to_idx.json", "w") as f:
        json.dump(uid_to_idx, f)

    # Save graph dataset
    save_graph_dataset(graphs, output_dir, n_folds=args.n_folds)

    logger.info("Done!")


if __name__ == "__main__":
    main()
