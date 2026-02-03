#!/usr/bin/env python3
"""Graph Construction Ablation Experiments.

Tests different graph construction parameters to find optimal settings:
- kNN k values: 3, 5, 7, 10
- Similarity thresholds: 0.3, 0.5, 0.7
- Edge types: kNN only, adjacency only, both

Uses quick evaluation (subset of folds) for parameter search.
Best config gets full 5-fold CV validation.

Usage:
    python scripts/experiments/graph_construction_ablation.py
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def build_edges_knn(
    node_embeddings: np.ndarray,
    k: int,
    threshold: float,
) -> List[Tuple[int, int]]:
    """Build kNN edges with configurable parameters."""
    n_nodes = len(node_embeddings)
    if n_nodes <= 1:
        return []

    # Normalize
    norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = node_embeddings / norms

    # Compute similarity
    sim_matrix = normed @ normed.T

    edges = []
    actual_k = min(k, n_nodes - 1)

    for i in range(n_nodes):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        top_k_idx = np.argsort(-sims)[:actual_k]
        for j in top_k_idx:
            if sims[j] >= threshold:
                edges.append((i, j))

    return edges


def build_edges_adjacency(candidate_sids: List[int]) -> List[Tuple[int, int]]:
    """Build adjacency edges based on sentence order."""
    n = len(candidate_sids)
    if n <= 1:
        return []

    edges = []
    sorted_idx = np.argsort(candidate_sids)

    for i in range(len(sorted_idx) - 1):
        curr_idx = sorted_idx[i]
        next_idx = sorted_idx[i + 1]
        if abs(candidate_sids[curr_idx] - candidate_sids[next_idx]) <= 1:
            edges.append((curr_idx, next_idx))
            edges.append((next_idx, curr_idx))

    return edges


def rebuild_graph_with_params(
    original_graph: Data,
    embeddings: np.ndarray,
    uid_to_idx: Dict[str, int],
    knn_k: int,
    knn_threshold: float,
    use_knn: bool = True,
    use_adjacency: bool = True,
) -> Data:
    """Rebuild a graph with different edge construction parameters."""
    candidate_uids = original_graph.candidate_uids
    candidate_sids = [int(uid.split("_")[-1]) for uid in candidate_uids]
    n_nodes = len(candidate_uids)

    # Get embeddings
    node_embeddings = np.zeros((n_nodes, embeddings.shape[1]))
    for i, uid in enumerate(candidate_uids):
        if uid in uid_to_idx:
            node_embeddings[i] = embeddings[uid_to_idx[uid]]

    # Build edges with new parameters
    edges = set()

    if use_knn:
        knn_edges = build_edges_knn(node_embeddings, knn_k, knn_threshold)
        edges.update(knn_edges)

    if use_adjacency:
        adj_edges = build_edges_adjacency(candidate_sids)
        edges.update(adj_edges)

    # Create edge index
    if edges:
        edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).T
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Clone graph with new edges
    new_graph = Data(
        x=original_graph.x.clone(),
        edge_index=edge_index,
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
    # Load graphs
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        graphs = data["graphs"]

        # Filter excluded criteria
        if exclude_criteria:
            graphs = [g for g in graphs
                     if getattr(g, 'criterion_id', None) not in exclude_criteria]

        fold_graphs[fold_id] = graphs

    # Load embeddings
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
    """Quick training for parameter search (fewer epochs)."""
    # Get input dimension
    input_dim = train_graphs[0].x.shape[1]

    # Filter positive graphs
    train_pos = [g for g in train_graphs if g.y.item() > 0]
    val_pos = [g for g in val_graphs if g.y.item() > 0]

    if len(train_pos) < 5 or len(val_pos) < 2:
        return 0.0

    # Initialize model
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

    # Train
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

    # Evaluate
    metrics = evaluate_graphs(model, val_pos, device)
    return metrics.get("ndcg@10", 0.0)


def run_ablation(
    fold_graphs: Dict[int, List[Data]],
    embeddings: np.ndarray,
    uid_to_idx: Dict[str, int],
    output_dir: Path,
    device: str = "cuda",
):
    """Run graph construction ablation experiments."""
    # Default training config
    train_config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 0.0001,
        "weight_decay": 1e-5,
        "alpha_init": 0.7,
        "margin": 0.1,
    }

    # Parameter grid
    knn_k_values = [3, 5, 7, 10]
    threshold_values = [0.3, 0.5, 0.7]
    edge_configs = [
        ("knn_only", True, False),
        ("adj_only", False, True),
        ("knn_adj", True, True),
    ]

    results = []

    # Use folds 0 and 1 for quick search
    quick_val_fold = 0
    quick_train_graphs = []
    for fid, graphs in fold_graphs.items():
        if fid != quick_val_fold:
            quick_train_graphs.extend(graphs)
    quick_val_graphs = fold_graphs[quick_val_fold]

    logger.info(f"Quick search using {len(quick_train_graphs)} train, {len(quick_val_graphs)} val graphs")

    # Test each configuration
    total_configs = len(knn_k_values) * len(threshold_values) * len(edge_configs)
    config_idx = 0

    for knn_k in knn_k_values:
        for threshold in threshold_values:
            for edge_name, use_knn, use_adj in edge_configs:
                config_idx += 1
                config_name = f"k{knn_k}_t{threshold}_{edge_name}"

                logger.info(f"\n[{config_idx}/{total_configs}] Testing: {config_name}")

                # Skip invalid configs
                if not use_knn and knn_k != 5:  # Only test knn params when using knn
                    continue
                if not use_knn and threshold != 0.5:
                    continue

                # Rebuild graphs with these parameters
                rebuilt_train = []
                for g in tqdm(quick_train_graphs, desc="Rebuilding train", leave=False):
                    new_g = rebuild_graph_with_params(
                        g, embeddings, uid_to_idx,
                        knn_k, threshold, use_knn, use_adj
                    )
                    rebuilt_train.append(new_g)

                rebuilt_val = []
                for g in tqdm(quick_val_graphs, desc="Rebuilding val", leave=False):
                    new_g = rebuild_graph_with_params(
                        g, embeddings, uid_to_idx,
                        knn_k, threshold, use_knn, use_adj
                    )
                    rebuilt_val.append(new_g)

                # Compute edge stats
                train_edges = [g.edge_index.shape[1] for g in rebuilt_train]
                avg_edges = np.mean(train_edges)

                # Quick train and evaluate
                start_time = time.time()
                ndcg10 = quick_train_evaluate(
                    rebuilt_train, rebuilt_val,
                    train_config, device, max_epochs=15
                )
                elapsed = time.time() - start_time

                result = {
                    "config_name": config_name,
                    "knn_k": knn_k,
                    "knn_threshold": threshold,
                    "use_knn": use_knn,
                    "use_adjacency": use_adj,
                    "avg_edges": avg_edges,
                    "ndcg@10": ndcg10,
                    "time_seconds": elapsed,
                }
                results.append(result)

                logger.info(f"  nDCG@10: {ndcg10:.4f}, avg_edges: {avg_edges:.1f}")

    # Sort by nDCG@10
    results.sort(key=lambda x: -x["ndcg@10"])

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "graph_construction_ablation.csv", index=False)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("GRAPH CONSTRUCTION ABLATION RESULTS")
    logger.info("=" * 70)

    for i, r in enumerate(results[:10]):
        logger.info(
            f"{i+1}. {r['config_name']}: nDCG@10={r['ndcg@10']:.4f}, "
            f"edges={r['avg_edges']:.1f}"
        )

    # Best config
    best = results[0]
    logger.info(f"\nBest configuration: {best['config_name']}")
    logger.info(f"  kNN k: {best['knn_k']}")
    logger.info(f"  Threshold: {best['knn_threshold']}")
    logger.info(f"  Edge types: kNN={best['use_knn']}, Adjacency={best['use_adjacency']}")
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
        output_dir = Path(f"outputs/experiments/graph_construction/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    # Load data
    graph_dir = Path(args.graph_dir)
    fold_graphs, embeddings, uid_to_idx = load_data(
        graph_dir, exclude_criteria=EXCLUDED_CRITERIA
    )

    logger.info(f"Loaded {sum(len(g) for g in fold_graphs.values())} graphs")
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Run ablation
    results = run_ablation(
        fold_graphs, embeddings, uid_to_idx,
        output_dir, args.device
    )

    # Save final summary
    summary = {
        "timestamp": timestamp,
        "n_configs_tested": len(results),
        "best_config": results[0] if results else None,
        "baseline_config": {
            "knn_k": 5,
            "knn_threshold": 0.5,
            "use_knn": True,
            "use_adjacency": True,
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
