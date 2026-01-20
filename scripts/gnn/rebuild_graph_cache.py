#!/usr/bin/env python3
"""Rebuild GNN graph cache from scratch.

This script:
1. Runs the retrieval+reranking pipeline on all queries
2. Builds sentence embeddings using nv-embed-v2
3. Constructs PyG graphs for each query
4. Saves fold-wise graph datasets for GNN training/evaluation

Usage:
    python scripts/gnn/rebuild_graph_cache.py --output_dir data/cache/gnn/rebuild
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Setup logging
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
    """Load sentence corpus as dict and list of Sentence objects."""
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


def get_post_sentences(corpus: Dict[str, dict], post_id: str) -> List[dict]:
    """Get all sentences for a post, sorted by sid."""
    sentences = [v for v in corpus.values() if v["post_id"] == post_id]
    return sorted(sentences, key=lambda x: x["sid"])


def create_query_df(groundtruth_df: pd.DataFrame) -> pd.DataFrame:
    """Create query-level dataframe with has_evidence labels."""
    # Group by post_id, criterion to get query-level data
    query_groups = groundtruth_df.groupby(["post_id", "criterion"])

    queries = []
    for (post_id, criterion), group in query_groups:
        has_evidence = group["groundtruth"].max()  # 1 if any sentence is evidence
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

    # Get unique post_ids
    post_ids = query_df["post_id"].unique()
    np.random.shuffle(post_ids)

    # Assign folds to posts
    post_to_fold = {}
    for i, post_id in enumerate(post_ids):
        post_to_fold[post_id] = i % n_folds

    # Assign folds to queries
    query_df = query_df.copy()
    query_df["fold_id"] = query_df["post_id"].map(post_to_fold)

    return query_df


def run_pipeline_inference(
    query_df: pd.DataFrame,
    corpus: Dict[str, dict],
    sentences: List,
    retriever_name: str = "nv-embed-v2",
    reranker_name: str = "jina-reranker-v3",
    top_k_retriever: int = 24,
    top_k_rerank: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
    cache_dir: Path = Path("data/cache"),
) -> pd.DataFrame:
    """Run retrieval + reranking pipeline on all queries.

    Returns DataFrame with candidate_uids and reranker_scores for each query.
    """
    from final_sc_review.retriever.zoo import RetrieverZoo
    from final_sc_review.reranker.jina_v3 import JinaV3Reranker

    logger.info("Initializing retriever and reranker...")

    # Initialize RetrieverZoo with sentences
    retriever_zoo = RetrieverZoo(
        sentences=sentences,
        cache_dir=cache_dir,
        device=device,
    )

    # Get the specific retriever
    retriever = retriever_zoo.get_retriever(retriever_name)
    logger.info(f"  Loaded retriever: {retriever_name}")

    # Encode corpus
    logger.info("  Encoding corpus...")
    retriever.encode_corpus(rebuild=False)

    # Use JinaV3Reranker with jina-reranker-v3 (SOTA listwise reranker)
    logger.info(f"  Loading JinaV3Reranker...")
    reranker = JinaV3Reranker(
        model_name="jinaai/jina-reranker-v3",
        device=device,
        max_length=512,
        batch_size=batch_size,
        use_listwise=True,  # v3 supports listwise reranking
    )
    logger.info(f"  Loaded reranker: jina-reranker-v3")

    # Load criteria descriptions for better query formation (per ReDSM5 taxonomy)
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

    # Group queries by post_id for efficiency
    post_queries = query_df.groupby("post_id")

    results = []

    for post_id, post_group in tqdm(post_queries, desc="Processing posts"):
        # Process each query for this post
        for _, query_row in post_group.iterrows():
            criterion_id = query_row["criterion_id"]
            query_id = query_row["query_id"]

            # Create query text with criterion description
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
            candidate_sids = [corpus[uid]["sid"] for uid in candidate_uids]

            # Rerank using JinaV3Reranker API
            rerank_scores = reranker.score_pairs(query_text, candidate_texts)

            # Sort by reranker scores and take top_k_rerank
            sorted_indices = np.argsort(-np.array(rerank_scores))[:top_k_rerank]

            final_uids = [candidate_uids[i] for i in sorted_indices]
            final_sids = [candidate_sids[i] for i in sorted_indices]
            final_scores = [rerank_scores[i] for i in sorted_indices]

            # Get gold labels for these candidates
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


def build_sentence_embeddings(
    sentences: List,
    retriever_name: str = "nv-embed-v2",
    batch_size: int = 64,
    device: str = "cuda",
    cache_dir: Path = Path("data/cache"),
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build embeddings for all sentences using the retriever's cached embeddings."""
    from final_sc_review.retriever.zoo import RetrieverZoo

    logger.info("Building sentence embeddings...")

    # Initialize RetrieverZoo with sentences
    retriever_zoo = RetrieverZoo(
        sentences=sentences,
        cache_dir=cache_dir,
        device=device,
    )

    # Get the retriever
    retriever = retriever_zoo.get_retriever(retriever_name)
    logger.info(f"  Using retriever: {retriever_name}")

    # Encode corpus (this will use cache if available)
    logger.info("  Encoding corpus...")
    retriever.encode_corpus(rebuild=False)

    # Get embeddings from the retriever's cache
    # The retriever stores embeddings in self.corpus_embeddings after encoding
    if hasattr(retriever, 'corpus_embeddings') and retriever.corpus_embeddings is not None:
        embeddings = retriever.corpus_embeddings
    elif hasattr(retriever, '_embeddings') and retriever._embeddings is not None:
        embeddings = retriever._embeddings
    else:
        # Fallback: load from cache file
        cache_path = cache_dir / "retriever_zoo" / retriever_name / "embeddings.npy"
        if cache_path.exists():
            embeddings = np.load(cache_path)
        else:
            raise RuntimeError(f"Could not find embeddings for {retriever_name}")

    # Build uid to index mapping
    uid_to_idx = {s.sent_uid: i for i, s in enumerate(sentences)}

    logger.info(f"  Embeddings shape: {embeddings.shape}")

    return embeddings, uid_to_idx


def build_graphs(
    pipeline_df: pd.DataFrame,
    embeddings: np.ndarray,
    uid_to_idx: Dict[str, int],
    embedding_dim: int = 1024,
) -> List:
    """Build PyG graphs from pipeline results."""
    from torch_geometric.data import Data

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

        # Get embeddings for candidates
        node_embeddings = np.zeros((n_nodes, embedding_dim))
        for i, uid in enumerate(candidate_uids):
            if uid in uid_to_idx:
                node_embeddings[i] = embeddings[uid_to_idx[uid]]

        # Build node features: [embedding, score, rank_percentile]
        ranks = np.argsort(np.argsort(-reranker_scores))
        rank_percentiles = 1.0 - ranks / n_nodes

        # Concatenate features
        node_features = np.concatenate([
            node_embeddings,
            reranker_scores.reshape(-1, 1),
            rank_percentiles.reshape(-1, 1),
        ], axis=1)

        # Build edges: semantic kNN + adjacency
        # Semantic kNN
        norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normed = node_embeddings / norms
        sim_matrix = normed @ normed.T

        edges_src = []
        edges_dst = []
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

        # Remove duplicates
        edge_set = set(zip(edges_src, edges_dst))
        if edge_set:
            edge_index = np.array([[s, d] for s, d in edge_set], dtype=np.int64).T
        else:
            edge_index = np.array([[], []], dtype=np.int64)

        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            reranker_scores=torch.tensor(reranker_scores, dtype=torch.float32),
            node_labels=torch.tensor(node_labels, dtype=torch.float32),
            y=torch.tensor([int(row["has_evidence"])], dtype=torch.float32),
        )

        # Store metadata
        data.query_id = row["query_id"]
        data.post_id = row["post_id"]
        data.criterion_id = row["criterion_id"]
        data.candidate_uids = candidate_uids
        data.fold_id = row["fold_id"]

        graphs.append(data)

    logger.info(f"  Built {len(graphs)} graphs")
    return graphs


def save_graph_dataset(
    graphs: List,
    output_dir: Path,
    n_folds: int = 5,
):
    """Save graphs in fold-wise format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split by fold
    fold_graphs = {i: [] for i in range(n_folds)}
    for g in graphs:
        fold_graphs[g.fold_id].append(g)

    # Save each fold
    for fold_id, fold_data in fold_graphs.items():
        fold_path = output_dir / f"fold_{fold_id}.pt"
        torch.save({"graphs": fold_data}, fold_path)
        logger.info(f"  Saved fold {fold_id}: {len(fold_data)} graphs to {fold_path}")

    # Save metadata
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
    parser = argparse.ArgumentParser(description="Rebuild GNN graph cache")
    parser.add_argument(
        "--groundtruth_path",
        type=str,
        default="data/groundtruth/evidence_sentence_groundtruth.csv",
        help="Path to groundtruth CSV",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/groundtruth/sentence_corpus.jsonl",
        help="Path to sentence corpus JSONL",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for graph cache",
    )
    parser.add_argument(
        "--retriever_name",
        type=str,
        default="nv-embed-v2",
        help="Retriever name (from zoo)",
    )
    parser.add_argument(
        "--reranker_name",
        type=str,
        default="jina-reranker-v3",
        help="Reranker name (from zoo)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    parser.add_argument(
        "--top_k_retriever",
        type=int,
        default=24,
        help="Top-k for retrieval",
    )
    parser.add_argument(
        "--top_k_rerank",
        type=int,
        default=20,
        help="Top-k after reranking",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Skip pipeline inference (load from cache)",
    )
    parser.add_argument(
        "--pipeline_cache",
        type=str,
        default=None,
        help="Path to cached pipeline results CSV",
    )
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"data/cache/gnn/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load data
    groundtruth_df = load_groundtruth(Path(args.groundtruth_path))
    corpus, sentences = load_sentence_corpus(Path(args.corpus_path))

    # Create query dataframe
    query_df = create_query_df(groundtruth_df)
    logger.info(f"Created {len(query_df)} queries")

    # Assign folds
    query_df = assign_folds(query_df, n_folds=args.n_folds)
    logger.info(f"Assigned {args.n_folds} folds")

    # Cache directory for retriever/reranker
    cache_dir = Path("data/cache")

    # Run or load pipeline results
    if args.skip_pipeline and args.pipeline_cache:
        logger.info(f"Loading pipeline cache from {args.pipeline_cache}")
        pipeline_df = pd.read_csv(args.pipeline_cache)
        # Parse JSON columns
        for col in ["candidate_uids", "candidate_sids", "reranker_scores", "node_labels"]:
            if col in pipeline_df.columns:
                pipeline_df[col] = pipeline_df[col].apply(json.loads)
    else:
        logger.info("Running pipeline inference...")
        pipeline_df = run_pipeline_inference(
            query_df,
            corpus,
            sentences,
            retriever_name=args.retriever_name,
            reranker_name=args.reranker_name,
            top_k_retriever=args.top_k_retriever,
            top_k_rerank=args.top_k_rerank,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=cache_dir,
        )

        # Save pipeline results
        pipeline_cache_path = output_dir / "pipeline_results.csv"
        # Convert lists to JSON for CSV storage
        save_df = pipeline_df.copy()
        for col in ["candidate_uids", "candidate_sids", "reranker_scores", "node_labels"]:
            save_df[col] = save_df[col].apply(json.dumps)
        save_df.to_csv(pipeline_cache_path, index=False)
        logger.info(f"Saved pipeline results to {pipeline_cache_path}")

    logger.info(f"Pipeline results: {len(pipeline_df)} queries")

    # Build sentence embeddings
    embeddings, uid_to_idx = build_sentence_embeddings(
        sentences,
        retriever_name=args.retriever_name,
        batch_size=args.batch_size * 2,
        device=args.device,
        cache_dir=cache_dir,
    )

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)
    with open(output_dir / "uid_to_idx.json", "w") as f:
        json.dump(uid_to_idx, f)
    logger.info(f"Saved embeddings to {output_dir}")

    # Build graphs
    graphs = build_graphs(
        pipeline_df,
        embeddings,
        uid_to_idx,
        embedding_dim=embeddings.shape[1],
    )

    # Save graph dataset
    save_graph_dataset(graphs, output_dir, n_folds=args.n_folds)

    logger.info("Done!")
    logger.info(f"Graph cache saved to: {output_dir}")


if __name__ == "__main__":
    main()
