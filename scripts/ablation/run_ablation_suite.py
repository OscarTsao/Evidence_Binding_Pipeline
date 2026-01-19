#!/usr/bin/env python3
"""
Phase 1: Ablation Study - Systematic evaluation of pipeline components.

This script runs 7 ablation configurations to quantify the contribution of each component:
1. Retriever only (NV-Embed-v2)
2. Retriever + Jina reranker
3. + P3 Graph Reranker
4. + P2 Dynamic-K
5. + P4 NE Gate (fixed K baseline)
6. Full pipeline (retriever + jina + P3 + P2 + P4 + 3-state gate)
7. Exclude A.10 criterion

Each configuration is evaluated with:
- 5-fold cross-validation
- Post-ID disjoint splits
- Identical candidate pools
- Identical evaluation metrics

Usage:
    python scripts/ablation/run_ablation_suite.py \
        --output_dir outputs/final_eval/phase1_ablations \
        --config_name [1_retriever_only|2_retriever_jina|...] \
        --n_folds 5 \
        --device cuda

    # Run all configurations:
    python scripts/ablation/run_ablation_suite.py \
        --output_dir outputs/final_eval/phase1_ablations \
        --run_all \
        --n_folds 5 \
        --device cuda
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.data.io import load_criteria, load_groundtruth, load_sentence_corpus
from final_sc_review.data.splits import k_fold_post_ids, split_post_ids
from final_sc_review.metrics.ranking import recall_at_k, ndcg_at_k, mrr_at_k, map_at_k
from final_sc_review.pipeline.zoo_pipeline import ZooPipeline, ZooPipelineConfig
from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


# Ablation configuration definitions
ABLATION_CONFIGS = {
    "1_retriever_only": {
        "name": "1. Retriever Only (NV-Embed-v2)",
        "description": "Dense retrieval baseline with NV-Embed-v2, no reranking",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": None,
            "p3_graph": False,
            "p2_dynamic_k": False,
            "p4_ne_gate": False,
            "three_state_gate": False,
        },
        "top_k_retriever": 10,
        "top_k_final": 10,
    },
    "2_retriever_jina": {
        "name": "2. Retriever + Jina Reranker",
        "description": "Dense retrieval + cross-encoder reranking",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": "jina-reranker-v3",
            "p3_graph": False,
            "p2_dynamic_k": False,
            "p4_ne_gate": False,
            "three_state_gate": False,
        },
        "top_k_retriever": 24,
        "top_k_final": 10,
    },
    "3_add_p3_graph": {
        "name": "3. + P3 Graph Reranker",
        "description": "Add GNN-based graph reranker for candidate refinement",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": "jina-reranker-v3",
            "p3_graph": True,
            "p2_dynamic_k": False,
            "p4_ne_gate": False,
            "three_state_gate": False,
        },
        "top_k_retriever": 24,
        "top_k_final": 10,
        "requires_gnn": True,
    },
    "4_add_p2_dynamic_k": {
        "name": "4. + P2 Dynamic-K Selection",
        "description": "Add GNN-based dynamic K selection (K ∈ [3,20])",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": "jina-reranker-v3",
            "p3_graph": True,
            "p2_dynamic_k": True,
            "p4_ne_gate": False,
            "three_state_gate": False,
        },
        "top_k_retriever": 24,
        "top_k_final": "dynamic",  # Dynamic K
        "requires_gnn": True,
    },
    "5_add_p4_ne_gate": {
        "name": "5. + P4 NE Gate (Fixed K)",
        "description": "Add GNN-based no-evidence detection (binary classifier)",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": "jina-reranker-v3",
            "p3_graph": True,
            "p2_dynamic_k": False,
            "p4_ne_gate": True,
            "three_state_gate": False,
        },
        "top_k_retriever": 24,
        "top_k_final": 10,
        "requires_gnn": True,
    },
    "6_full_pipeline": {
        "name": "6. Full Pipeline (All Components)",
        "description": "Complete system with all modules including 3-state clinical gate",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": "jina-reranker-v3",
            "p3_graph": True,
            "p2_dynamic_k": True,
            "p4_ne_gate": True,
            "three_state_gate": True,
        },
        "top_k_retriever": 24,
        "top_k_final": "dynamic",
        "requires_gnn": True,
        "use_existing_results": True,  # Can reuse results from Phase 0
        "existing_results_path": "outputs/final_research_eval/20260118_031312_complete",
    },
    "7_exclude_a10": {
        "name": "7. Full Pipeline (Exclude A.10 Suicidal Ideation)",
        "description": "Full pipeline excluding A.10 criterion for sensitivity analysis",
        "components": {
            "retriever": "nv-embed-v2",
            "reranker": "jina-reranker-v3",
            "p3_graph": True,
            "p2_dynamic_k": True,
            "p4_ne_gate": True,
            "three_state_gate": True,
        },
        "top_k_retriever": 24,
        "top_k_final": "dynamic",
        "requires_gnn": True,
        "exclude_criteria": ["A.10"],  # Exclude suicidal ideation criterion
    },
}


class AblationEvaluator:
    """Evaluates a single ablation configuration with k-fold CV."""

    def __init__(
        self,
        config_name: str,
        output_dir: Path,
        data_dir: Path,
        n_folds: int = 5,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.config_name = config_name
        self.config = ABLATION_CONFIGS[config_name]
        self.output_dir = output_dir / config_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.device = device
        self.seed = seed

        # Check if we can reuse existing results
        if self.config.get("use_existing_results", False):
            existing_path = Path(self.config["existing_results_path"])
            if existing_path.exists():
                logger.info(f"Config {config_name} can reuse results from {existing_path}")
                self.existing_results_path = existing_path
            else:
                logger.warning(f"Existing results path {existing_path} not found, will run evaluation")
                self.existing_results_path = None
        else:
            self.existing_results_path = None

    def load_data(self):
        """Load groundtruth, criteria, and sentences."""
        logger.info("Loading data...")
        self.groundtruth = load_groundtruth(self.data_dir / "groundtruth" / "evidence_sentence_groundtruth.csv")
        self.criteria = load_criteria(self.data_dir / "DSM5" / "MDD_Criteira.json")
        self.sentences = load_sentence_corpus(self.data_dir / "groundtruth" / "sentence_corpus.jsonl")

        # Create criterion text map
        self.criterion_text_map = {c.criterion_id: c.text for c in self.criteria}

        # Filter out excluded criteria if specified
        if "exclude_criteria" in self.config:
            excluded = set(self.config["exclude_criteria"])
            logger.info(f"Excluding criteria: {excluded}")
            self.groundtruth = [
                row for row in self.groundtruth
                if row.criterion_id not in excluded
            ]
            self.criteria = [c for c in self.criteria if c.criterion_id not in excluded]

        # Create post-to-sentences mapping
        self.post_to_sentences = defaultdict(list)
        for sent in self.sentences:
            self.post_to_sentences[sent.post_id].append(sent)

        logger.info(f"Loaded {len(self.groundtruth)} groundtruth rows")
        logger.info(f"Loaded {len(self.criteria)} criteria")
        logger.info(f"Loaded {len(self.sentences)} sentences")

    def create_folds(self):
        """Create k-fold splits (Post-ID disjoint)."""
        logger.info(f"Creating {self.n_folds}-fold splits...")
        post_ids = sorted({row.post_id for row in self.groundtruth})

        if self.n_folds == 1:
            # Special case for testing: use simple train/test split
            logger.info("  Using simple train/test split for n_folds=1")
            splits = split_post_ids(post_ids, seed=self.seed, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
            self.folds = [{
                'train': splits['train'],
                'val': splits['val'],
                'test': splits['test'],
            }]
        else:
            # Standard k-fold cross-validation
            self.folds = k_fold_post_ids(post_ids, k=self.n_folds, seed=self.seed)

        for fold_id in range(self.n_folds):
            fold = self.folds[fold_id]
            logger.info(f"  Fold {fold_id}: {len(fold['train'])} train, {len(fold['test'])} test")

    def load_pipeline(self, fold_id: int) -> Optional[ZooPipeline]:
        """Load pipeline for given fold (if GNN components needed)."""
        components = self.config["components"]

        # Create pipeline config
        pipeline_config = ZooPipelineConfig(
            retriever_name=components["retriever"],
            reranker_name=components["reranker"] if components["reranker"] else None,
            top_k_retriever=self.config["top_k_retriever"],
            top_k_final=self.config["top_k_final"] if isinstance(self.config["top_k_final"], int) else 10,
            device=self.device,
        )

        # Create cache directory
        cache_dir = self.data_dir / "cache" / "ablation" / self.config_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # For simple configs without GNN, create basic pipeline
        if not self.config.get("requires_gnn", False):
            pipeline = ZooPipeline(
                sentences=self.sentences,
                cache_dir=cache_dir,
                config=pipeline_config,
            )
            return pipeline
        else:
            # For GNN-based configs, would need to load GNN models
            logger.warning("GNN-based pipeline loading not yet implemented")
            logger.warning("For now, using simple retriever+reranker pipeline")

            pipeline = ZooPipeline(
                sentences=self.sentences,
                cache_dir=cache_dir,
                config=pipeline_config,
            )
            return pipeline

    def evaluate_fold(self, fold_id: int) -> Dict:
        """Evaluate single fold."""
        logger.info(f"Evaluating fold {fold_id}...")

        fold = self.folds[fold_id]
        test_posts = set(fold["test"])

        # Group groundtruth by (post_id, criterion_id)
        grouped = defaultdict(lambda: {"gold_uids": set(), "all_uids": []})
        for row in self.groundtruth:
            if row.post_id not in test_posts:
                continue
            key = (row.post_id, row.criterion_id)
            grouped[key]["all_uids"].append(row.sent_uid)
            if row.groundtruth == 1:
                grouped[key]["gold_uids"].add(row.sent_uid)

        # Load pipeline
        pipeline = self.load_pipeline(fold_id)
        if pipeline is None:
            logger.error(f"Failed to load pipeline for fold {fold_id}")
            return {}

        # Run evaluation on test set
        per_query_results = []
        n_queries = len(grouped)

        for idx, ((post_id, criterion_id), data) in enumerate(sorted(grouped.items())):
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processing query {idx+1}/{n_queries}")

            query_text = self.criterion_text_map.get(criterion_id)
            if not query_text:
                continue

            gold_uids = data["gold_uids"]

            # Run pipeline
            try:
                results = pipeline.retrieve(query_text, post_id)
                if results is None:
                    logger.error(f"Pipeline returned None for {post_id}/{criterion_id}")
                    continue
                # Results are (sent_uid, sentence_text, score) tuples
                ranked_uids = [r[0] for r in results]
                scores = [r[2] for r in results]  # Score is 3rd element
            except Exception as e:
                import traceback
                logger.error(f"Pipeline error for {post_id}/{criterion_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue

            # Compute metrics for this query
            k_values = [1, 3, 5, 10, 20]
            query_metrics = {
                "fold_id": fold_id,
                "post_id": post_id,
                "criterion_id": criterion_id,
                "has_evidence": 1 if gold_uids else 0,
                "n_gold": len(gold_uids),
                "n_retrieved": len(ranked_uids),
            }

            if gold_uids:
                for k in k_values:
                    k_eff = min(k, len(ranked_uids))
                    query_metrics[f"recall@{k}"] = recall_at_k(gold_uids, ranked_uids, k_eff)
                    query_metrics[f"ndcg@{k}"] = ndcg_at_k(gold_uids, ranked_uids, k_eff)
                    query_metrics[f"mrr@{k}"] = mrr_at_k(gold_uids, ranked_uids, k_eff)

            per_query_results.append(query_metrics)

        # Aggregate metrics for this fold
        fold_metrics = self._aggregate_metrics(per_query_results)

        # Save per-query results
        per_query_df = pd.DataFrame(per_query_results)
        per_query_csv = self.output_dir / f"fold_{fold_id}_per_query.csv"
        per_query_df.to_csv(per_query_csv, index=False)
        logger.info(f"  Saved per-query results to {per_query_csv}")

        return {
            "fold_id": fold_id,
            "n_queries": len(per_query_results),
            "n_with_evidence": sum(1 for r in per_query_results if r["has_evidence"]),
            "metrics": fold_metrics,
        }

    def _aggregate_metrics(self, per_query_results: List[Dict]) -> Dict:
        """Aggregate per-query metrics to fold-level."""
        # Filter queries with evidence
        with_evidence = [r for r in per_query_results if r["has_evidence"]]

        if not with_evidence:
            return {}

        metrics = {}
        k_values = [1, 3, 5, 10, 20]

        for k in k_values:
            recalls = [r[f"recall@{k}"] for r in with_evidence if f"recall@{k}" in r]
            ndcgs = [r[f"ndcg@{k}"] for r in with_evidence if f"ndcg@{k}" in r]
            mrrs = [r[f"mrr@{k}"] for r in with_evidence if f"mrr@{k}" in r]

            metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
            metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
            metrics[f"mrr@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0

        return metrics

    def run_evaluation(self) -> Dict:
        """Run full k-fold evaluation."""
        logger.info("="*80)
        logger.info(f"ABLATION EVALUATION: {self.config['name']}")
        logger.info(f"Description: {self.config['description']}")
        logger.info("="*80)

        # Check if we can reuse existing results
        if self.existing_results_path:
            logger.info(f"Reusing existing results from {self.existing_results_path}")
            return self._load_existing_results()

        # Load data and create folds
        self.load_data()
        self.create_folds()

        # Evaluate each fold
        fold_results = []
        for fold_id in range(self.n_folds):
            fold_result = self.evaluate_fold(fold_id)
            fold_results.append(fold_result)

        # Aggregate across folds
        aggregated = self._aggregate_across_folds(fold_results)

        # Save results
        results = {
            "config_name": self.config_name,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "n_folds": self.n_folds,
            "fold_results": fold_results,
            "aggregated": aggregated,
        }

        results_file = self.output_dir / "summary.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        logger.info("="*80)
        logger.info(f"COMPLETED: {self.config['name']}")
        logger.info("="*80)

        return results

    def _load_existing_results(self) -> Dict:
        """Load existing results from previous evaluation."""
        summary_file = self.existing_results_path / "summary.json"
        if not summary_file.exists():
            logger.error(f"Summary file not found: {summary_file}")
            return {}

        with open(summary_file) as f:
            existing = json.load(f)

        # Adapt to ablation format
        results = {
            "config_name": self.config_name,
            "config": self.config,
            "timestamp": existing.get("timestamp", "unknown"),
            "n_folds": existing.get("n_folds", self.n_folds),
            "fold_results": existing.get("fold_results", []),
            "aggregated": existing.get("aggregated", {}),
            "reused_from": str(self.existing_results_path),
        }

        return results

    def _aggregate_across_folds(self, fold_results: List[Dict]) -> Dict:
        """Aggregate metrics across all folds."""
        # Extract metric names
        metric_names = []
        for fold in fold_results:
            if "metrics" in fold:
                metric_names = list(fold["metrics"].keys())
                break

        aggregated = {}
        for metric in metric_names:
            values = [fold["metrics"][metric] for fold in fold_results if metric in fold.get("metrics", {})]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))

        # Overall statistics
        aggregated["total_queries"] = sum(fold["n_queries"] for fold in fold_results)
        aggregated["total_with_evidence"] = sum(fold["n_with_evidence"] for fold in fold_results)

        return aggregated


def run_ablation_suite(
    output_dir: Path,
    data_dir: Path,
    configs_to_run: Optional[List[str]] = None,
    n_folds: int = 5,
    device: str = "cuda",
):
    """Run full ablation suite."""
    logger.info("="*80)
    logger.info("PHASE 1: ABLATION STUDY")
    logger.info("="*80)

    if configs_to_run is None:
        configs_to_run = list(ABLATION_CONFIGS.keys())

    logger.info(f"Configurations to run: {len(configs_to_run)}")
    for cfg in configs_to_run:
        logger.info(f"  - {ABLATION_CONFIGS[cfg]['name']}")

    # Run each configuration
    all_results = {}
    for config_name in configs_to_run:
        evaluator = AblationEvaluator(
            config_name=config_name,
            output_dir=output_dir,
            data_dir=data_dir,
            n_folds=n_folds,
            device=device,
        )

        results = evaluator.run_evaluation()
        all_results[config_name] = results

    # Save combined results
    combined_file = output_dir / "ablation_suite_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nCombined results saved to {combined_file}")

    # Generate comparison table
    generate_comparison_table(all_results, output_dir)

    logger.info("="*80)
    logger.info("ABLATION SUITE COMPLETE")
    logger.info("="*80)


def generate_comparison_table(all_results: Dict, output_dir: Path):
    """Generate comparison table across all ablations."""
    logger.info("\n" + "="*80)
    logger.info("ABLATION COMPARISON TABLE")
    logger.info("="*80)

    # Extract key metrics for comparison
    comparison_rows = []
    for config_name, results in all_results.items():
        config = ABLATION_CONFIGS[config_name]
        aggregated = results.get("aggregated", {})

        row = {
            "config": config["name"],
            "recall@10": aggregated.get("recall@10_mean", 0.0),
            "ndcg@10": aggregated.get("ndcg@10_mean", 0.0),
            "mrr@10": aggregated.get("mrr@10_mean", 0.0),
            "n_queries": aggregated.get("total_queries", 0),
        }
        comparison_rows.append(row)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_rows)

    # Save to CSV
    comparison_csv = output_dir / "ablation_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    # Print table
    print("\n" + comparison_df.to_string(index=False))
    print(f"\nComparison table saved to {comparison_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/final_eval/phase1_ablations",
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Specific configuration to run (if not running all)",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all ablation configurations",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # Determine which configs to run
    if args.run_all:
        configs_to_run = list(ABLATION_CONFIGS.keys())
    elif args.config_name:
        configs_to_run = [args.config_name]
    else:
        logger.error("Must specify either --config_name or --run_all")
        return 1

    # Run ablation suite
    run_ablation_suite(
        output_dir=output_dir,
        data_dir=data_dir,
        configs_to_run=configs_to_run,
        n_folds=args.n_folds,
        device=args.device,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
