#!/usr/bin/env python3
"""Feature Engineering Ablation Experiments.

Tests different node feature combinations to find optimal settings:
- Embedding only vs embedding + score features
- Different score-derived features (gaps, stats)
- Positional encoding

Uses quick evaluation (single fold) for parameter search.

Usage:
    python scripts/experiments/feature_engineering_ablation.py
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss
from final_sc_review.gnn.config import GNNType, GNNModelConfig
from final_sc_review.constants import EXCLUDED_CRITERIA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureConfig:
    """Configuration for node features."""

    def __init__(
        self,
        name: str,
        use_embedding: bool = True,
        use_reranker_score: bool = True,
        use_rank_percentile: bool = True,
        use_score_gaps: bool = False,
        use_score_stats: bool = False,
        use_position_encoding: bool = False,
    ):
        self.name = name
        self.use_embedding = use_embedding
        self.use_reranker_score = use_reranker_score
        self.use_rank_percentile = use_rank_percentile
        self.use_score_gaps = use_score_gaps
        self.use_score_stats = use_score_stats
        self.use_position_encoding = use_position_encoding


def compute_position_encoding(sids: List[int], max_len: int = 100, dim: int = 16) -> np.ndarray:
    """Compute sinusoidal position encoding for sentence positions."""
    n = len(sids)
    pe = np.zeros((n, dim))

    for i, sid in enumerate(sids):
        position = sid / max_len  # Normalize to [0, 1]
        for j in range(dim // 2):
            freq = 10000 ** (2 * j / dim)
            pe[i, 2 * j] = np.sin(position * freq)
            pe[i, 2 * j + 1] = np.cos(position * freq)

    return pe


def rebuild_features(
    original_graph: Data,
    embeddings: np.ndarray,
    uid_to_idx: Dict[str, int],
    feature_config: FeatureConfig,
    embedding_dim: int = 4096,
) -> Data:
    """Rebuild graph with different feature configuration."""
    candidate_uids = original_graph.candidate_uids
    candidate_sids = [int(uid.split("_")[-1]) for uid in candidate_uids]
    reranker_scores = original_graph.reranker_scores.numpy()
    n = len(candidate_uids)

    # Get embeddings
    node_embeddings = np.zeros((n, embedding_dim))
    for i, uid in enumerate(candidate_uids):
        if uid in uid_to_idx:
            node_embeddings[i] = embeddings[uid_to_idx[uid]]

    # Compute ranks
    ranks = np.argsort(np.argsort(-reranker_scores))

    # Build feature list
    features_list = []

    # 1. Embeddings
    if feature_config.use_embedding:
        features_list.append(node_embeddings)

    # 2. Reranker scores (normalized)
    if feature_config.use_reranker_score:
        score_min = reranker_scores.min()
        score_max = reranker_scores.max()
        if score_max > score_min:
            norm_scores = (reranker_scores - score_min) / (score_max - score_min)
        else:
            norm_scores = np.ones(n) * 0.5
        features_list.append(norm_scores.reshape(-1, 1))

    # 3. Rank percentile
    if feature_config.use_rank_percentile:
        rank_pct = ranks / max(n - 1, 1)
        features_list.append(rank_pct.reshape(-1, 1))

    # 4. Score gaps
    if feature_config.use_score_gaps:
        sorted_idx = np.argsort(-reranker_scores)
        sorted_scores = reranker_scores[sorted_idx]

        gaps_prev = np.zeros(n)
        gaps_next = np.zeros(n)

        for i, idx in enumerate(sorted_idx):
            if i > 0:
                gaps_prev[idx] = sorted_scores[i - 1] - sorted_scores[i]
            if i < n - 1:
                gaps_next[idx] = sorted_scores[i] - sorted_scores[i + 1]

        features_list.append(gaps_prev.reshape(-1, 1))
        features_list.append(gaps_next.reshape(-1, 1))

    # 5. Score statistics
    if feature_config.use_score_stats:
        mean_score = reranker_scores.mean()
        std_score = reranker_scores.std() if n > 1 else 1.0
        median_score = np.median(reranker_scores)
        score_min = reranker_scores.min()
        score_max = reranker_scores.max()

        zscore = (reranker_scores - mean_score) / (std_score + 1e-8)
        minmax = (reranker_scores - score_min) / (score_max - score_min + 1e-8)
        above_mean = (reranker_scores > mean_score).astype(np.float32)
        below_median = (reranker_scores < median_score).astype(np.float32)

        features_list.append(zscore.reshape(-1, 1))
        features_list.append(minmax.reshape(-1, 1))
        features_list.append(above_mean.reshape(-1, 1))
        features_list.append(below_median.reshape(-1, 1))

    # 6. Position encoding
    if feature_config.use_position_encoding:
        pe = compute_position_encoding(candidate_sids, max_len=100, dim=16)
        features_list.append(pe)

    # Concatenate
    node_features = np.hstack(features_list).astype(np.float32)

    # Create new graph
    new_graph = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=original_graph.edge_index.clone(),
        reranker_scores=original_graph.reranker_scores.clone(),
        node_labels=original_graph.node_labels.clone(),
        y=original_graph.y.clone(),
    )
    new_graph.query_id = original_graph.query_id
    new_graph.post_id = original_graph.post_id
    new_graph.criterion_id = original_graph.criterion_id
    new_graph.candidate_uids = original_graph.candidate_uids
    new_graph.fold_id = original_graph.fold_id

    return new_graph


def load_data(
    graph_dir: Path,
    exclude_criteria: List[str],
) -> Tuple[Dict[int, List], np.ndarray, Dict[str, int]]:
    """Load graphs, embeddings, and UID mapping."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        if exclude_criteria:
            graphs = [g for g in graphs
                     if getattr(g, 'criterion_id', None) not in exclude_criteria]

        fold_graphs[fold_id] = graphs

    embeddings = np.load(graph_dir / "embeddings.npy")
    with open(graph_dir / "uid_to_idx.json") as f:
        uid_to_idx = json.load(f)

    return fold_graphs, embeddings, uid_to_idx


def compute_ranking_metrics(gold_mask: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute ranking metrics."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    metrics = {}

    # MRR
    gold_positions = np.where(sorted_gold)[0]
    mrr = 1.0 / (gold_positions[0] + 1) if len(gold_positions) > 0 else 0.0
    metrics["mrr"] = mrr

    # nDCG@K
    for k in [5, 10]:
        dcg = 0.0
        for i in range(min(k, len(sorted_gold))):
            if sorted_gold[i]:
                dcg += 1.0 / np.log2(i + 2)
        n_gold = int(gold_mask.sum())
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_gold)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    return metrics


def evaluate_graphs(
    model: nn.Module,
    graphs: List[Data],
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate model on graphs."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy() > 0.5

            if not gold_mask.any():
                continue

            refined = model(g.x, g.edge_index, g.reranker_scores)
            scores = refined.cpu().numpy()

            metrics = compute_ranking_metrics(gold_mask, scores)
            all_metrics.append(metrics)

    if not all_metrics:
        return {}

    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[key] = np.mean(values)

    return agg


def quick_train_evaluate(
    train_graphs: List[Data],
    val_graphs: List[Data],
    config: Dict,
    device: str = "cuda",
    max_epochs: int = 15,
) -> float:
    """Quick training for parameter search."""
    input_dim = train_graphs[0].x.shape[1]

    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0

    gnn_config = GNNModelConfig(
        gnn_type=GNNType.SAGE,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        layer_norm=False,
        residual=True,
    )
    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        alpha_init=config["alpha_init"],
        learn_alpha=True,
        config=gnn_config,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = GraphRerankerLoss(margin=config["margin"])

    train_loader = DataLoader(train_pos, batch_size=32, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss, _ = loss_fn(refined, batch.reranker_scores, batch.node_labels, batch.batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    metrics = evaluate_graphs(model, val_pos, device)
    return metrics.get("ndcg@10", 0.0)


def run_ablation(
    fold_graphs: Dict[int, List[Data]],
    embeddings: np.ndarray,
    uid_to_idx: Dict[str, int],
    output_dir: Path,
    device: str = "cuda",
):
    """Run feature engineering ablation."""
    train_config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 0.0001,
        "weight_decay": 1e-5,
        "alpha_init": 0.7,
        "margin": 0.1,
    }

    # Feature configurations to test
    feature_configs = [
        FeatureConfig("baseline_emb+score+rank", True, True, True, False, False, False),
        FeatureConfig("emb_only", True, False, False, False, False, False),
        FeatureConfig("emb+score", True, True, False, False, False, False),
        FeatureConfig("emb+rank", True, False, True, False, False, False),
        FeatureConfig("full_features", True, True, True, True, True, False),
        FeatureConfig("with_gaps", True, True, True, True, False, False),
        FeatureConfig("with_stats", True, True, True, False, True, False),
        FeatureConfig("with_position", True, True, True, False, False, True),
        FeatureConfig("kitchen_sink", True, True, True, True, True, True),
    ]

    results = []

    # Use fold 0 for validation
    quick_val_fold = 0
    quick_train_graphs = []
    for fid, graphs in fold_graphs.items():
        if fid != quick_val_fold:
            quick_train_graphs.extend(graphs)
    quick_val_graphs = fold_graphs[quick_val_fold]

    embedding_dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Quick search: {len(quick_train_graphs)} train, {len(quick_val_graphs)} val")

    for config_idx, feat_config in enumerate(feature_configs):
        logger.info(f"\n[{config_idx + 1}/{len(feature_configs)}] Testing: {feat_config.name}")

        # Rebuild graphs with these features
        rebuilt_train = []
        for g in tqdm(quick_train_graphs, desc="Rebuilding train", leave=False):
            new_g = rebuild_features(g, embeddings, uid_to_idx, feat_config, embedding_dim)
            rebuilt_train.append(new_g)

        rebuilt_val = []
        for g in tqdm(quick_val_graphs, desc="Rebuilding val", leave=False):
            new_g = rebuild_features(g, embeddings, uid_to_idx, feat_config, embedding_dim)
            rebuilt_val.append(new_g)

        # Get feature dimension
        feat_dim = rebuilt_train[0].x.shape[1]

        # Train and evaluate
        start_time = time.time()
        ndcg10 = quick_train_evaluate(
            rebuilt_train, rebuilt_val,
            train_config, device, max_epochs=15
        )
        elapsed = time.time() - start_time

        result = {
            "config_name": feat_config.name,
            "use_embedding": feat_config.use_embedding,
            "use_reranker_score": feat_config.use_reranker_score,
            "use_rank_percentile": feat_config.use_rank_percentile,
            "use_score_gaps": feat_config.use_score_gaps,
            "use_score_stats": feat_config.use_score_stats,
            "use_position_encoding": feat_config.use_position_encoding,
            "feature_dim": feat_dim,
            "ndcg@10": ndcg10,
            "time_seconds": elapsed,
        }
        results.append(result)

        logger.info(f"  nDCG@10: {ndcg10:.4f}, feat_dim: {feat_dim}")

    # Sort by nDCG@10
    results.sort(key=lambda x: -x["ndcg@10"])

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "feature_engineering_ablation.csv", index=False)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING ABLATION RESULTS")
    logger.info("=" * 70)

    for i, r in enumerate(results):
        logger.info(
            f"{i+1}. {r['config_name']}: nDCG@10={r['ndcg@10']:.4f}, "
            f"dim={r['feature_dim']}"
        )

    best = results[0]
    logger.info(f"\nBest configuration: {best['config_name']}")
    logger.info(f"  Feature dimension: {best['feature_dim']}")
    logger.info(f"  nDCG@10: {best['ndcg@10']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"outputs/experiments/feature_engineering/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    graph_dir = Path(args.graph_dir)
    fold_graphs, embeddings, uid_to_idx = load_data(
        graph_dir, exclude_criteria=EXCLUDED_CRITERIA
    )

    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")
    logger.info(f"Embeddings shape: {embeddings.shape}")

    results = run_ablation(
        fold_graphs, embeddings, uid_to_idx,
        output_dir, args.device
    )

    summary = {
        "timestamp": timestamp,
        "n_configs_tested": len(results),
        "best_config": results[0] if results else None,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
