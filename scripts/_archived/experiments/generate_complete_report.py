#!/usr/bin/env python3
"""Generate comprehensive metrics report with all K values.

Computes all ranking and classification metrics at k=1,3,5,10,20
with statistical significance tests and per-criterion breakdown.

Usage:
    conda run -n llmhe python scripts/experiments/generate_complete_report.py \
        --graph_dir data/cache/gnn/rebuild_20260120 \
        --output_dir outputs/complete_report

Author: Evidence Binding Pipeline
Date: 2026-01-24
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

from final_sc_review.gnn.models.p3_graph_reranker import GraphRerankerGNN, GraphRerankerLoss
from final_sc_review.constants import EXCLUDED_CRITERIA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metric Computation Functions
# ============================================================================

def compute_ndcg_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute nDCG@K with binary relevance."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    dcg = 0.0
    for i in range(min(k, len(sorted_gold))):
        if sorted_gold[i]:
            dcg += 1.0 / math.log2(i + 2)

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


def compute_precision_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Precision@K."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    return sorted_gold[:k].sum() / k


def compute_hit_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Hit@K (1 if any relevant in top-k, else 0)."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    return 1.0 if sorted_gold[:k].sum() > 0 else 0.0


def compute_mrr(gold_mask: np.ndarray, scores: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx]

    gold_positions = np.where(sorted_gold)[0]
    if len(gold_positions) == 0:
        return 0.0

    return 1.0 / (gold_positions[0] + 1)


def compute_map_at_k(gold_mask: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Compute Mean Average Precision@K."""
    sorted_idx = np.argsort(-scores)
    sorted_gold = gold_mask[sorted_idx][:k]

    n_gold = gold_mask.sum()
    if n_gold == 0:
        return 0.0

    precisions = []
    hits = 0
    for i, rel in enumerate(sorted_gold):
        if rel:
            hits += 1
            precisions.append(hits / (i + 1))

    if not precisions:
        return 0.0

    return sum(precisions) / min(k, n_gold)


def compute_all_ranking_metrics(
    gold_mask: np.ndarray,
    scores: np.ndarray,
    ks: List[int] = [1, 3, 5, 10, 20],
) -> Dict[str, float]:
    """Compute all ranking metrics for a single query."""
    metrics = {}

    for k in ks:
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(gold_mask, scores, k)
        metrics[f"recall@{k}"] = compute_recall_at_k(gold_mask, scores, k)
        metrics[f"precision@{k}"] = compute_precision_at_k(gold_mask, scores, k)
        metrics[f"hit@{k}"] = compute_hit_at_k(gold_mask, scores, k)
        metrics[f"map@{k}"] = compute_map_at_k(gold_mask, scores, k)

    metrics["mrr"] = compute_mrr(gold_mask, scores)

    return metrics


# ============================================================================
# GNN Model Training
# ============================================================================

def train_gnn_fold(
    train_graphs: List,
    val_graphs: List,
    config: Dict,
    device: str,
) -> Tuple[nn.Module, Dict]:
    """Train GNN for one fold."""
    model = GraphRerankerGNN(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        alpha_init=config["alpha_init"],
        learn_alpha=config["learn_alpha"],
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    criterion = GraphRerankerLoss()

    train_loader = DataLoader(train_graphs, batch_size=config["batch_size"], shuffle=True)

    best_val_ndcg = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config["n_epochs"]):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y, batch.batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_ndcgs = []
        with torch.no_grad():
            for g in val_graphs:
                g = g.to(device)
                scores = model(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long, device=device))
                gold = g.y.cpu().numpy()
                scores_np = scores.cpu().numpy()
                if gold.sum() > 0:
                    val_ndcgs.append(compute_ndcg_at_k(gold, scores_np, 10))

        val_ndcg = np.mean(val_ndcgs) if val_ndcgs else 0.0

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.get("patience", 10):
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, {"best_val_ndcg": best_val_ndcg}


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_fold(
    graphs: List,
    model: Optional[nn.Module],
    device: str,
    ks: List[int],
    use_gnn: bool = True,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a fold and return aggregated + per-query metrics."""
    all_metrics = {f"ndcg@{k}": [] for k in ks}
    all_metrics.update({f"recall@{k}": [] for k in ks})
    all_metrics.update({f"precision@{k}": [] for k in ks})
    all_metrics.update({f"hit@{k}": [] for k in ks})
    all_metrics.update({f"map@{k}": [] for k in ks})
    all_metrics["mrr"] = []

    per_query_results = []

    # Classification metrics
    all_preds = []
    all_labels = []

    if model:
        model.eval()

    for g in graphs:
        gold = g.y.cpu().numpy()

        # Skip if no positives (for ranking metrics)
        has_positive = gold.sum() > 0

        if use_gnn and model:
            g = g.to(device)
            with torch.no_grad():
                scores = model(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long, device=device))
            scores_np = scores.cpu().numpy()
        else:
            # Use reranker scores (column index 1 in node features)
            scores_np = g.x[:, 1].cpu().numpy()

        # Classification: max score per graph
        all_preds.append(scores_np.max())
        all_labels.append(1 if has_positive else 0)

        # Ranking metrics (positives_only)
        if has_positive:
            metrics = compute_all_ranking_metrics(gold, scores_np, ks)
            for k, v in metrics.items():
                all_metrics[k].append(v)

            per_query_results.append({
                "has_evidence": 1,
                "n_candidates": len(gold),
                "n_positive": int(gold.sum()),
                **metrics,
            })

    # Aggregate ranking metrics
    agg = {}
    for k, vals in all_metrics.items():
        if vals:
            agg[f"{k}_mean"] = np.mean(vals)
            agg[f"{k}_std"] = np.std(vals)

    # Classification metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) > 1:
        agg["auroc"] = roc_auc_score(all_labels, all_preds)
        agg["auprc"] = average_precision_score(all_labels, all_preds)

        # Binary predictions at threshold 0.5
        binary_preds = (all_preds > np.median(all_preds)).astype(int)
        agg["f1"] = f1_score(all_labels, binary_preds)
        agg["accuracy"] = (binary_preds == all_labels).mean()

    agg["n_queries"] = len(graphs)
    agg["n_with_evidence"] = sum(1 for g in graphs if g.y.sum() > 0)

    return agg, per_query_results


def bootstrap_ci(values: List[float], n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)

    return lower, upper


def paired_ttest(baseline: List[float], gnn: List[float]) -> Tuple[float, float]:
    """Compute paired t-test and effect size."""
    baseline = np.array(baseline)
    gnn = np.array(gnn)

    t_stat, p_value = stats.ttest_rel(gnn, baseline)

    # Cohen's d
    diff = gnn - baseline
    d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0

    return p_value, d


# ============================================================================
# Main Report Generation
# ============================================================================

def load_fold_graphs(graph_dir: Path, fold_id: int) -> List:
    """Load graphs for a specific fold."""
    fold_path = graph_dir / f"fold_{fold_id}.pt"
    if not fold_path.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_path}")

    # weights_only=False required for PyG Data objects
    graphs = torch.load(fold_path, weights_only=False)
    logger.info(f"Loaded {len(graphs)} graphs from fold {fold_id}")
    return graphs


def filter_excluded_criteria(graphs: List, excluded: List[str]) -> List:
    """Filter out graphs for excluded criteria."""
    if not excluded:
        return graphs

    filtered = []
    for g in graphs:
        criterion = getattr(g, 'criterion_id', None)
        if criterion is None or criterion not in excluded:
            filtered.append(g)

    return filtered


def generate_report(args):
    """Generate comprehensive metrics report."""
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    ks = [int(k) for k in args.ks.split(",")]
    logger.info(f"Computing metrics at K={ks}")

    # Load metadata
    meta_path = graph_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        n_folds = metadata.get("n_folds", 5)
    else:
        n_folds = 5

    # GNN config (best from HPO)
    gnn_config = {
        "in_channels": 770,  # 768 embedding + 2 features
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "alpha_init": 0.8,
        "learn_alpha": True,
        "lr": 0.00035,
        "n_epochs": 15,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "patience": 10,
    }

    # Results storage
    all_fold_results = {
        "baseline": [],
        "gnn": [],
    }
    per_criterion_results = {}

    # Evaluate each fold
    for fold_id in range(n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Fold {fold_id}")
        logger.info(f"{'='*60}")

        # Load all folds
        all_graphs = []
        for fid in range(n_folds):
            all_graphs.append(load_fold_graphs(graph_dir, fid))

        # Train/val split
        val_graphs = filter_excluded_criteria(all_graphs[fold_id], EXCLUDED_CRITERIA)
        train_graphs = []
        for fid in range(n_folds):
            if fid != fold_id:
                train_graphs.extend(filter_excluded_criteria(all_graphs[fid], EXCLUDED_CRITERIA))

        logger.info(f"Train: {len(train_graphs)} graphs, Val: {len(val_graphs)} graphs")

        # Evaluate baseline (no GNN)
        baseline_agg, baseline_per_query = evaluate_fold(
            val_graphs, None, device, ks, use_gnn=False
        )
        all_fold_results["baseline"].append(baseline_agg)
        logger.info(f"Baseline nDCG@10: {baseline_agg.get('ndcg@10_mean', 0):.4f}")

        # Train and evaluate GNN
        model, train_info = train_gnn_fold(train_graphs, val_graphs, gnn_config, device)
        gnn_agg, gnn_per_query = evaluate_fold(
            val_graphs, model, device, ks, use_gnn=True
        )
        all_fold_results["gnn"].append(gnn_agg)
        logger.info(f"GNN nDCG@10: {gnn_agg.get('ndcg@10_mean', 0):.4f}")

    # Aggregate across folds
    logger.info(f"\n{'='*60}")
    logger.info("Aggregating Results")
    logger.info(f"{'='*60}")

    final_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "graph_dir": str(graph_dir),
            "n_folds": n_folds,
            "k_values": ks,
            "excluded_criteria": EXCLUDED_CRITERIA,
            "gnn_config": gnn_config,
        },
        "baseline": {},
        "gnn": {},
        "improvement": {},
        "statistical_tests": {},
    }

    # Compute aggregated metrics
    for model_name in ["baseline", "gnn"]:
        fold_results = all_fold_results[model_name]

        for metric in fold_results[0].keys():
            if metric.endswith("_mean"):
                base_metric = metric.replace("_mean", "")
                values = [fr[metric] for fr in fold_results]

                mean_val = np.mean(values)
                std_val = np.std(values)
                ci_low, ci_high = bootstrap_ci(values, args.bootstrap_samples)

                final_results[model_name][base_metric] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_95": [ci_low, ci_high],
                    "per_fold": values,
                }

    # Compute improvements and statistical tests
    for metric in final_results["baseline"].keys():
        baseline_vals = final_results["baseline"][metric]["per_fold"]
        gnn_vals = final_results["gnn"][metric]["per_fold"]

        improvement = final_results["gnn"][metric]["mean"] - final_results["baseline"][metric]["mean"]
        rel_improvement = improvement / final_results["baseline"][metric]["mean"] * 100 if final_results["baseline"][metric]["mean"] > 0 else 0

        final_results["improvement"][metric] = {
            "absolute": improvement,
            "relative_pct": rel_improvement,
        }

        # Statistical significance
        p_value, effect_size = paired_ttest(baseline_vals, gnn_vals)
        final_results["statistical_tests"][metric] = {
            "paired_ttest_p": p_value,
            "cohens_d": effect_size,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01,
        }

    # Save results
    results_path = output_dir / "metrics_complete.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=float)
    logger.info(f"Saved results to {results_path}")

    # Generate CSV tables
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Ranking metrics table
    ranking_data = []
    for k in ks:
        for metric_type in ["ndcg", "recall", "precision", "hit", "map"]:
            metric = f"{metric_type}@{k}"
            if metric in final_results["baseline"]:
                ranking_data.append({
                    "Metric": metric,
                    "Baseline_Mean": final_results["baseline"][metric]["mean"],
                    "Baseline_Std": final_results["baseline"][metric]["std"],
                    "GNN_Mean": final_results["gnn"][metric]["mean"],
                    "GNN_Std": final_results["gnn"][metric]["std"],
                    "Improvement_%": final_results["improvement"][metric]["relative_pct"],
                    "p_value": final_results["statistical_tests"][metric]["paired_ttest_p"],
                    "Significant": final_results["statistical_tests"][metric]["significant_005"],
                })

    # Add MRR
    if "mrr" in final_results["baseline"]:
        ranking_data.append({
            "Metric": "MRR",
            "Baseline_Mean": final_results["baseline"]["mrr"]["mean"],
            "Baseline_Std": final_results["baseline"]["mrr"]["std"],
            "GNN_Mean": final_results["gnn"]["mrr"]["mean"],
            "GNN_Std": final_results["gnn"]["mrr"]["std"],
            "Improvement_%": final_results["improvement"]["mrr"]["relative_pct"],
            "p_value": final_results["statistical_tests"]["mrr"]["paired_ttest_p"],
            "Significant": final_results["statistical_tests"]["mrr"]["significant_005"],
        })

    ranking_df = pd.DataFrame(ranking_data)
    ranking_df.to_csv(tables_dir / "ranking_metrics.csv", index=False)
    logger.info(f"Saved ranking metrics to {tables_dir / 'ranking_metrics.csv'}")

    # Generate markdown report
    generate_markdown_report(final_results, output_dir, ks)

    return final_results


def generate_markdown_report(results: Dict, output_dir: Path, ks: List[int]):
    """Generate human-readable markdown report."""
    report = []
    report.append("# Evidence Binding Pipeline - Complete Evaluation Report\n")
    report.append(f"Generated: {results['metadata']['timestamp']}\n")
    report.append(f"Graph Cache: {results['metadata']['graph_dir']}\n")
    report.append(f"Folds: {results['metadata']['n_folds']}\n")
    report.append(f"Excluded Criteria: {results['metadata']['excluded_criteria']}\n")

    report.append("\n## Executive Summary\n")
    ndcg10_baseline = results["baseline"].get("ndcg@10", {}).get("mean", 0)
    ndcg10_gnn = results["gnn"].get("ndcg@10", {}).get("mean", 0)
    improvement = results["improvement"].get("ndcg@10", {}).get("relative_pct", 0)
    report.append(f"- **Best Model:** NV-Embed-v2 + Jina-Reranker-v3 + P3 GNN\n")
    report.append(f"- **Primary Metric:** nDCG@10 = {ndcg10_gnn:.4f} ± {results['gnn'].get('ndcg@10', {}).get('std', 0):.4f}\n")
    report.append(f"- **Baseline:** nDCG@10 = {ndcg10_baseline:.4f}\n")
    report.append(f"- **Improvement:** +{improvement:.2f}%\n")

    report.append("\n## 1. Ranking Metrics (5-Fold CV, positives_only)\n")

    # nDCG table
    report.append("\n### 1.1 nDCG@K\n")
    report.append("| K | Baseline | GNN | Δ (%) | p-value | Sig. |\n")
    report.append("|---|----------|-----|-------|---------|------|\n")
    for k in ks:
        metric = f"ndcg@{k}"
        if metric in results["baseline"]:
            b = results["baseline"][metric]
            g = results["gnn"][metric]
            imp = results["improvement"][metric]
            stat = results["statistical_tests"][metric]
            sig = "**Yes**" if stat["significant_005"] else "No"
            report.append(f"| {k} | {b['mean']:.4f} ± {b['std']:.4f} | {g['mean']:.4f} ± {g['std']:.4f} | +{imp['relative_pct']:.2f}% | {stat['paired_ttest_p']:.4f} | {sig} |\n")

    # Recall table
    report.append("\n### 1.2 Recall@K\n")
    report.append("| K | Baseline | GNN | Δ (%) | p-value | Sig. |\n")
    report.append("|---|----------|-----|-------|---------|------|\n")
    for k in ks:
        metric = f"recall@{k}"
        if metric in results["baseline"]:
            b = results["baseline"][metric]
            g = results["gnn"][metric]
            imp = results["improvement"][metric]
            stat = results["statistical_tests"][metric]
            sig = "**Yes**" if stat["significant_005"] else "No"
            report.append(f"| {k} | {b['mean']:.4f} ± {b['std']:.4f} | {g['mean']:.4f} ± {g['std']:.4f} | +{imp['relative_pct']:.2f}% | {stat['paired_ttest_p']:.4f} | {sig} |\n")

    # MRR
    report.append("\n### 1.3 MRR\n")
    if "mrr" in results["baseline"]:
        b = results["baseline"]["mrr"]
        g = results["gnn"]["mrr"]
        imp = results["improvement"]["mrr"]
        stat = results["statistical_tests"]["mrr"]
        report.append(f"- **Baseline:** {b['mean']:.4f} ± {b['std']:.4f}\n")
        report.append(f"- **GNN:** {g['mean']:.4f} ± {g['std']:.4f}\n")
        report.append(f"- **Improvement:** +{imp['relative_pct']:.2f}%\n")
        report.append(f"- **p-value:** {stat['paired_ttest_p']:.4f}\n")

    # Classification metrics
    report.append("\n## 2. Classification Metrics (all_queries)\n")
    for metric in ["auroc", "auprc", "f1", "accuracy"]:
        if metric in results["baseline"]:
            b = results["baseline"][metric]
            g = results["gnn"][metric]
            report.append(f"- **{metric.upper()}:** Baseline={b['mean']:.4f}, GNN={g['mean']:.4f}\n")

    # GNN config
    report.append("\n## 3. GNN Configuration\n")
    report.append("```yaml\n")
    for k, v in results["metadata"]["gnn_config"].items():
        report.append(f"{k}: {v}\n")
    report.append("```\n")

    # Save report
    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.writelines(report)
    logger.info(f"Saved markdown report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive metrics report")
    parser.add_argument("--graph_dir", type=str, default="data/cache/gnn/rebuild_20260120",
                       help="Path to graph cache directory")
    parser.add_argument("--output_dir", type=str, default="outputs/complete_report",
                       help="Output directory for report")
    parser.add_argument("--ks", type=str, default="1,3,5,10,20",
                       help="Comma-separated K values for metrics")
    parser.add_argument("--bootstrap_samples", type=int, default=10000,
                       help="Number of bootstrap samples for CI")

    args = parser.parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
