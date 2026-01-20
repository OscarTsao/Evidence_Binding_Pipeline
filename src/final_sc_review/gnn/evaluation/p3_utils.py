"""Shared utilities for P3 Graph Reranker evaluation.

This module consolidates common functions used across P3 scripts:
- Model loading
- Inference
- Ranking metric computation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def load_p3_model(
    checkpoint_path: Path,
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: str = "cuda",
):
    """Load P3 Graph Reranker model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        input_dim: Input feature dimension (embedding_dim + extra_features)
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        device: Device to load model on

    Returns:
        Loaded GraphRerankerGNN model in eval mode
    """
    from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both wrapped and raw state_dict formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
        hidden_dim = config.get("hidden_dim", hidden_dim)
        num_layers = config.get("num_layers", num_layers)
    else:
        state_dict = checkpoint

    model = GraphRerankerGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        alpha_init=0.7,
        learn_alpha=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def run_p3_inference(
    model,
    graphs: List,
    device: str = "cuda",
    show_progress: bool = True,
) -> List[np.ndarray]:
    """Run P3 inference to get refined scores.

    Args:
        model: Loaded P3 model
        graphs: List of PyG Data objects
        device: Device for inference
        show_progress: Whether to show progress bar

    Returns:
        List of refined score arrays, one per graph
    """
    refined_scores_list = []

    iterator = tqdm(graphs, desc="P3 inference") if show_progress else graphs

    with torch.no_grad():
        for g in iterator:
            g = g.to(device)
            original_scores = g.reranker_scores.to(device)
            refined_scores = model(g.x, g.edge_index, original_scores)
            refined_scores_list.append(refined_scores.cpu().numpy())

    return refined_scores_list


def compute_ranking_metrics(
    gold_mask: np.ndarray,
    scores: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10, 20],
) -> Dict[str, float]:
    """Compute ranking metrics for a single query.

    Args:
        gold_mask: Boolean array indicating gold evidence sentences
        scores: Predicted scores for each candidate
        k_values: K values for Recall@K and nDCG@K

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}

    # Sort by scores (descending)
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    # Recall@K
    n_gold = gold_mask.sum()
    for k in k_values:
        if n_gold > 0:
            recall_k = sorted_gold[:k].sum() / n_gold
        else:
            recall_k = 0.0
        metrics[f"recall@{k}"] = recall_k

    # MRR
    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) > 0:
        mrr = 1.0 / (gold_positions[0] + 1)
    else:
        mrr = 0.0
    metrics["mrr"] = mrr

    # nDCG@K
    for k in k_values:
        dcg = 0.0
        for i in range(min(k, len(sorted_gold))):
            if sorted_gold[i]:
                dcg += 1.0 / np.log2(i + 2)

        # Ideal DCG
        n_gold_int = int(gold_mask.sum())
        idcg = 0.0
        for i in range(min(k, n_gold_int)):
            idcg += 1.0 / np.log2(i + 2)

        if idcg > 0:
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        metrics[f"ndcg@{k}"] = ndcg

    return metrics


def evaluate_graphs(
    graphs: List,
    refined_scores_list: Optional[List[np.ndarray]] = None,
    use_refined: bool = False,
    k_values: List[int] = [1, 3, 5, 10, 20],
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate ranking metrics on a set of graphs.

    Args:
        graphs: List of PyG Data objects
        refined_scores_list: Optional refined scores from P3 inference
        use_refined: Whether to use refined scores (vs reranker_scores)
        k_values: K values for metrics

    Returns:
        (aggregated_metrics, per_query_metrics)
    """
    all_metrics = []

    for i, g in enumerate(graphs):
        # Ensure tensors are on CPU
        node_labels = g.node_labels.cpu() if g.node_labels.is_cuda else g.node_labels
        gold_mask = node_labels.numpy() > 0.5

        if not gold_mask.any():
            continue

        if use_refined and refined_scores_list is not None:
            scores = refined_scores_list[i]
        else:
            reranker_scores = g.reranker_scores.cpu() if g.reranker_scores.is_cuda else g.reranker_scores
            scores = reranker_scores.numpy()

        metrics = compute_ranking_metrics(gold_mask, scores, k_values)
        metrics["query_id"] = getattr(g, "query_id", f"query_{i}")
        metrics["post_id"] = getattr(g, "post_id", None)
        metrics["criterion_id"] = getattr(g, "criterion_id", None)
        metrics["n_gold"] = int(gold_mask.sum())
        metrics["n_candidates"] = len(gold_mask)

        all_metrics.append(metrics)

    if not all_metrics:
        return {}, []

    # Aggregate
    agg = {}
    metric_keys = [k for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], (int, float))]
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        agg[key] = np.mean(values)
        agg[f"{key}_std"] = np.std(values)

    agg["n_queries"] = len(all_metrics)

    return agg, all_metrics


def load_graph_dataset(graph_dir: Path) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache directory.

    Args:
        graph_dir: Path to graph cache directory

    Returns:
        (fold_graphs dict, metadata dict)
    """
    import json

    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        fold_graphs[fold_id] = data["graphs"]

    return fold_graphs, metadata


def ensure_cpu(graph):
    """Ensure all tensors in a PyG graph are on CPU.

    Args:
        graph: PyG Data object

    Returns:
        Graph with all tensors on CPU
    """
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph
