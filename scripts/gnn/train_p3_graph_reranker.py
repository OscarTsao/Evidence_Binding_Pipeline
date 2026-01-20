#!/usr/bin/env python3
"""Train P3 Graph Reranker GNN with cross-validation.

Trains on rebuilt graph cache with NV-Embed-v2 embeddings + Jina-v3 reranker scores.

Usage:
    conda run -n llmhe python scripts/gnn/train_p3_graph_reranker.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/gnn_research/p3_retrained
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_graph_dataset(graph_dir: Path) -> Tuple[Dict[int, List], Dict]:
    """Load graph dataset from cache."""
    metadata_path = graph_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    n_folds = metadata["n_folds"]
    fold_graphs = {}

    for fold_id in range(n_folds):
        fold_path = graph_dir / f"fold_{fold_id}.pt"
        data = torch.load(fold_path, weights_only=False)
        fold_graphs[fold_id] = data["graphs"]
        logger.info(f"Loaded fold {fold_id}: {len(data['graphs'])} graphs")

    return fold_graphs, metadata


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
    for k in [1, 3, 5, 10, 20]:
        dcg = 0.0
        for i in range(min(k, len(sorted_gold))):
            if sorted_gold[i]:
                dcg += 1.0 / np.log2(i + 2)
        n_gold = int(gold_mask.sum())
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, n_gold)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    # Recall@K
    n_gold = gold_mask.sum()
    for k in [1, 3, 5, 10, 20]:
        recall = sorted_gold[:k].sum() / n_gold if n_gold > 0 else 0.0
        metrics[f"recall@{k}"] = recall

    return metrics


def ensure_cpu(graph):
    """Ensure all tensors in graph are on CPU."""
    for key in graph.keys():
        if isinstance(graph[key], torch.Tensor):
            graph[key] = graph[key].cpu()
    return graph


def evaluate_graphs(
    model: nn.Module,
    graphs: List,
    device: str = "cuda",
    use_refined: bool = True,
) -> Dict[str, float]:
    """Evaluate model on graphs."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for g in graphs:
            # Ensure graph is on CPU first, then move to device
            g = ensure_cpu(g)
            g = g.to(device)
            gold_mask = g.node_labels.cpu().numpy() > 0.5

            if not gold_mask.any():
                continue

            if use_refined:
                refined = model(g.x, g.edge_index, g.reranker_scores)
                scores = refined.cpu().numpy()
            else:
                scores = g.reranker_scores.cpu().numpy()

            metrics = compute_ranking_metrics(gold_mask, scores)
            all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # Aggregate metrics
    agg = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        agg[key] = np.mean(values)
        agg[f"{key}_std"] = np.std(values)

    agg["n_queries"] = len(all_metrics)
    return agg


def train_fold(
    model: nn.Module,
    train_graphs: List,
    val_graphs: List,
    config: Dict,
    device: str = "cuda",
) -> Tuple[Dict, int, float]:
    """Train model on one fold."""
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    loss_fn = GraphRerankerLoss(
        alpha_rank=config["alpha_rank"],
        alpha_align=config["alpha_align"],
        alpha_reg=config["alpha_reg"],
        margin=config["margin"],
    )

    # Only train on positive graphs (with gold evidence)
    # Ensure all graphs are on CPU before DataLoader
    train_pos = [ensure_cpu(g) for g in train_graphs if g.y.item() > 0]
    val_pos = [ensure_cpu(g) for g in val_graphs if g.y.item() > 0]

    logger.info(f"  Training graphs: {len(train_pos)}")
    logger.info(f"  Validation graphs: {len(val_pos)}")

    train_loader = DataLoader(train_pos, batch_size=config["batch_size"], shuffle=True)

    best_improvement = -float("inf")
    best_epoch = 0
    patience_counter = 0

    # Baseline metrics
    original_metrics = evaluate_graphs(model, val_pos, device, use_refined=False)
    original_ndcg10 = original_metrics.get("ndcg@10", 0.0)

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            refined = model(batch.x, batch.edge_index, batch.reranker_scores, batch.batch)
            loss, components = loss_fn(
                refined,
                batch.reranker_scores,
                batch.node_labels,
                batch.batch,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate
        refined_metrics = evaluate_graphs(model, val_pos, device, use_refined=True)
        refined_ndcg10 = refined_metrics.get("ndcg@10", 0.0)
        improvement = refined_ndcg10 - original_ndcg10

        if improvement > best_improvement:
            best_improvement = improvement
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, "
                f"ndcg@10={refined_ndcg10:.4f} (+{improvement:.4f}), "
                f"alpha={model.alpha.item():.3f}"
            )

        if patience_counter >= config["patience"]:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    model.load_state_dict(best_state)

    return best_state, best_epoch, best_improvement


def run_cv_training(
    fold_graphs: Dict[int, List],
    config: Dict,
    output_dir: Path,
    device: str = "cuda",
) -> Dict:
    """Run cross-validation training."""
    n_folds = len(fold_graphs)
    results = []

    # Get input dimension from first graph
    sample_graph = fold_graphs[0][0]
    input_dim = sample_graph.x.shape[1]
    logger.info(f"Input dimension: {input_dim}")

    for fold_id in range(n_folds):
        logger.info(f"\n=== Fold {fold_id} ===")

        # Split: val=fold_id, train=others
        val_graphs = fold_graphs[fold_id]
        train_graphs = []
        for fid, graphs in fold_graphs.items():
            if fid != fold_id:
                train_graphs.extend(graphs)

        # Initialize model
        model = GraphRerankerGNN(
            input_dim=input_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            alpha_init=config["alpha_init"],
            learn_alpha=config["learn_alpha"],
        )

        # Train
        best_state, best_epoch, best_improvement = train_fold(
            model, train_graphs, val_graphs, config, device
        )

        # Save checkpoint
        ckpt_path = output_dir / f"fold_{fold_id}_best.pt"
        torch.save(best_state, ckpt_path)
        logger.info(f"  Saved checkpoint: {ckpt_path}")

        # Evaluate
        model.load_state_dict(best_state)
        val_pos = [g for g in val_graphs if g.y.item() > 0]

        original_metrics = evaluate_graphs(model, val_pos, device, use_refined=False)
        refined_metrics = evaluate_graphs(model, val_pos, device, use_refined=True)

        fold_result = {
            "fold_id": fold_id,
            "best_epoch": best_epoch,
            "best_improvement": best_improvement,
            "final_alpha": model.alpha.item(),
            "original_metrics": original_metrics,
            "refined_metrics": refined_metrics,
        }
        results.append(fold_result)

        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  Original nDCG@10: {original_metrics['ndcg@10']:.4f}")
        logger.info(f"  Refined nDCG@10: {refined_metrics['ndcg@10']:.4f}")
        logger.info(f"  Improvement: +{best_improvement:.4f}")

    return results


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate CV results."""
    metrics_to_agg = ["mrr", "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "recall@1", "recall@3", "recall@5", "recall@10"]

    original = {}
    refined = {}
    improvement = {}

    for metric in metrics_to_agg:
        orig_values = [r["original_metrics"].get(metric, 0) for r in results]
        ref_values = [r["refined_metrics"].get(metric, 0) for r in results]

        original[metric] = f"{np.mean(orig_values):.4f} +/- {np.std(orig_values):.4f}"
        refined[metric] = f"{np.mean(ref_values):.4f} +/- {np.std(ref_values):.4f}"
        improvement[metric] = f"+{np.mean(np.array(ref_values) - np.array(orig_values)):.4f} +/- {np.std(np.array(ref_values) - np.array(orig_values)):.4f}"

    return {
        "original": original,
        "refined": refined,
        "improvement": improvement,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # Training hyperparameters (from original P3 training)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--alpha_init", type=float, default=0.7)
    parser.add_argument("--alpha_rank", type=float, default=1.0)
    parser.add_argument("--alpha_align", type=float, default=0.5)
    parser.add_argument("--alpha_reg", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.1)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"outputs/gnn_research/p3_retrained/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")

    config = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "alpha_init": args.alpha_init,
        "learn_alpha": True,
        "alpha_rank": args.alpha_rank,
        "alpha_align": args.alpha_align,
        "alpha_reg": args.alpha_reg,
        "margin": args.margin,
    }

    # Load data
    graph_dir = Path(args.graph_dir)
    fold_graphs, metadata = load_graph_dataset(graph_dir)

    # Train
    results = run_cv_training(fold_graphs, config, output_dir, args.device)

    # Aggregate
    aggregated = aggregate_results(results)

    # Save results
    cv_results = {
        "config": config,
        "fold_results": results,
        "aggregated": aggregated,
        "timestamp": timestamp,
    }

    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("P3 GRAPH RERANKER TRAINING SUMMARY")
    logger.info("=" * 60)

    logger.info("\nOriginal (baseline):")
    for metric, value in aggregated["original"].items():
        logger.info(f"  {metric}: {value}")

    logger.info("\nRefined (with P3):")
    for metric, value in aggregated["refined"].items():
        logger.info(f"  {metric}: {value}")

    logger.info("\nImprovement:")
    for metric, value in aggregated["improvement"].items():
        logger.info(f"  {metric}: {value}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
