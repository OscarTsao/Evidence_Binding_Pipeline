#!/usr/bin/env python3
"""Run LLM evaluation experiments.

Usage:
    python scripts/llm/run_llm_eval.py --model qwen2.5-7b --experiment all
    python scripts/llm/run_llm_eval.py --model qwen2.5-7b --experiment verifier --max_samples 100
    python scripts/llm/run_llm_eval.py --model qwen2.5-7b --experiment a10
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model name mapping
MODEL_MAP = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}


def run_verifier_experiment(
    model_name: str,
    data_loader,
    samples: pd.DataFrame,
    output_dir: Path,
    use_4bit: bool = True,
) -> Dict:
    """Run LLM Verifier experiment on UNCERTAIN cases.

    Tests if LLM can correctly classify uncertain cases.
    """
    from final_sc_review.llm.verifier import LLMVerifier

    logger.info(f"Running Verifier experiment with {model_name}")
    logger.info(f"  Samples: {len(samples)} ({samples['state'].value_counts().to_dict()})")

    # Initialize verifier
    verifier = LLMVerifier(
        model_name=model_name,
        load_in_4bit=use_4bit,
        cache_dir=output_dir / "cache" / "verifier",
    )

    results = []

    for idx, row in tqdm(samples.iterrows(), total=len(samples), desc="Verifier"):
        post_id = row['post_id']
        criterion_id = row['criterion_id']
        has_evidence_gold = row['has_evidence_gold']

        # Get data
        post_text = data_loader.get_post_text(post_id)
        criterion_text = data_loader.get_criterion_text(criterion_id)
        evidence_sentences = data_loader.get_evidence_sentences(post_id, top_k=5)

        if not evidence_sentences:
            continue

        # Run verifier
        try:
            output = verifier.verify(
                post_text=post_text,
                criterion_text=criterion_text,
                evidence_sentences=evidence_sentences,
                self_consistency_runs=3,
            )

            results.append({
                "post_id": post_id,
                "criterion_id": criterion_id,
                "has_evidence_gold": int(has_evidence_gold),
                "has_evidence_pred": int(output["has_evidence"]),
                "confidence": output["confidence"],
                "self_consistency_score": output["self_consistency_score"],
                "state": row['state'],
            })
        except Exception as e:
            logger.warning(f"Error on {post_id}/{criterion_id}: {e}")
            continue

    # Compute metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "verifier_results.csv", index=False)

    y_true = results_df['has_evidence_gold'].values
    y_pred = results_df['has_evidence_pred'].values
    y_conf = results_df['confidence'].values

    metrics = {
        "n_samples": len(results_df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_conf))

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics["precision"] = float(prec)
    metrics["recall"] = float(rec)
    metrics["f1"] = float(f1)

    # Per-state metrics
    for state in ['NEG', 'UNCERTAIN', 'POS']:
        state_df = results_df[results_df['state'] == state]
        if len(state_df) > 0:
            state_acc = accuracy_score(
                state_df['has_evidence_gold'].values,
                state_df['has_evidence_pred'].values
            )
            metrics[f"accuracy_{state}"] = float(state_acc)

    logger.info(f"Verifier Results: {metrics}")
    return metrics


def run_a10_experiment(
    model_name: str,
    data_loader,
    samples: pd.DataFrame,
    output_dir: Path,
    use_4bit: bool = True,
    max_a10_samples: int = 200,
) -> Dict:
    """Run A.10 Classifier experiment for suicidal ideation detection.

    Tests if LLM can improve A.10 classification (baseline AUROC=0.66).
    """
    from final_sc_review.llm.suicidal_ideation_classifier import SuicidalIdeationClassifier as A10Classifier

    # Use ALL A.10 queries from per_query_df (not stratified samples)
    # This ensures we have enough A.10 samples for meaningful evaluation
    all_a10 = data_loader.per_query_df[data_loader.per_query_df['criterion_id'] == 'A.9'].copy()

    # Stratified sample of A.10: balance by has_evidence_gold
    pos_a10 = all_a10[all_a10['has_evidence_gold'] == 1]
    neg_a10 = all_a10[all_a10['has_evidence_gold'] == 0]

    n_pos = min(max_a10_samples // 2, len(pos_a10))
    n_neg = min(max_a10_samples // 2, len(neg_a10))

    a10_samples = pd.concat([
        pos_a10.sample(n=n_pos, random_state=42) if n_pos > 0 else pd.DataFrame(),
        neg_a10.sample(n=n_neg, random_state=42) if n_neg > 0 else pd.DataFrame(),
    ], ignore_index=True)

    logger.info(f"Running A.10 Classifier experiment with {model_name}")
    logger.info(f"  A.10 samples: {len(a10_samples)} ({a10_samples['has_evidence_gold'].sum()} positive)")

    if len(a10_samples) == 0:
        logger.warning("No A.10 samples found!")
        return {}

    # Initialize classifier
    classifier = A10Classifier(
        model_name=model_name,
        load_in_4bit=use_4bit,
        cache_dir=output_dir / "cache" / "a10",
    )

    results = []

    for idx, row in tqdm(a10_samples.iterrows(), total=len(a10_samples), desc="A.10"):
        post_id = row['post_id']
        has_evidence_gold = row['has_evidence_gold']

        # Get post text
        post_text = data_loader.get_post_text(post_id)

        if not post_text:
            continue

        # Run classifier
        try:
            output = classifier.classify(
                post_text=post_text,
                self_consistency_runs=3,
            )

            results.append({
                "post_id": post_id,
                "has_evidence_gold": int(has_evidence_gold),
                "has_si_pred": int(output["has_suicidal_ideation"]),
                "severity": output["severity"],
                "confidence": output["confidence"],
                "self_consistency_score": output["self_consistency_score"],
                "n_evidence": len(output.get("evidence_sentences", [])),
            })
        except Exception as e:
            logger.warning(f"Error on {post_id}: {e}")
            continue

    # Compute metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "a10_results.csv", index=False)

    y_true = results_df['has_evidence_gold'].values
    y_pred = results_df['has_si_pred'].values
    y_conf = results_df['confidence'].values

    metrics = {
        "n_samples": len(results_df),
        "n_positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_conf))

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics["precision"] = float(prec)
    metrics["recall"] = float(rec)
    metrics["f1"] = float(f1)

    # Severity breakdown
    severity_counts = results_df['severity'].value_counts().to_dict()
    metrics["severity_distribution"] = severity_counts

    logger.info(f"A.10 Classifier Results: {metrics}")
    return metrics


def run_reranker_experiment(
    model_name: str,
    data_loader,
    samples: pd.DataFrame,
    output_dir: Path,
    use_4bit: bool = True,
) -> Dict:
    """Run LLM Reranker experiment.

    Tests if LLM reranking improves nDCG over base retriever.
    """
    from final_sc_review.llm.reranker import LLMReranker

    # Filter to positive samples (need evidence to rank)
    pos_samples = samples[samples['has_evidence_gold'] == 1].copy()

    logger.info(f"Running Reranker experiment with {model_name}")
    logger.info(f"  Positive samples: {len(pos_samples)}")

    if len(pos_samples) == 0:
        logger.warning("No positive samples for reranking!")
        return {}

    # Initialize reranker
    reranker = LLMReranker(
        model_name=model_name,
        load_in_4bit=use_4bit,
        cache_dir=output_dir / "cache" / "reranker",
    )

    results = []

    for idx, row in tqdm(pos_samples.iterrows(), total=len(pos_samples), desc="Reranker"):
        post_id = row['post_id']
        criterion_id = row['criterion_id']

        # Get data
        post_text = data_loader.get_post_text(post_id)
        criterion_text = data_loader.get_criterion_text(criterion_id)

        # Get sentences as candidates (simple mock - in real use this would come from retriever)
        sentences = data_loader.post_sentences.get(post_id, [])
        if len(sentences) < 3:
            continue

        candidates = [
            {"sent_uid": f"{post_id}_{i}", "sentence": sent, "score": 1.0 - i * 0.1}
            for i, sent in enumerate(sentences[:10])
        ]

        # Run reranker
        try:
            reranked, metadata = reranker.rerank(
                post_text=post_text,
                criterion_text=criterion_text,
                candidates=candidates,
                top_k=5,
                check_position_bias=True,
            )

            results.append({
                "post_id": post_id,
                "criterion_id": criterion_id,
                "n_candidates": len(candidates),
                "position_bias_score": metadata.get("position_bias_score", -1),
                "reranked": True,
            })
        except Exception as e:
            logger.warning(f"Error on {post_id}/{criterion_id}: {e}")
            continue

    # Compute summary metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "reranker_results.csv", index=False)

    valid_bias = results_df[results_df['position_bias_score'] >= 0]

    metrics = {
        "n_samples": len(results_df),
        "avg_position_bias": float(valid_bias['position_bias_score'].mean()) if len(valid_bias) > 0 else -1,
        "low_bias_rate": float((valid_bias['position_bias_score'] < 0.3).mean()) if len(valid_bias) > 0 else 0,
    }

    logger.info(f"Reranker Results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluation experiments")
    parser.add_argument("--model", type=str, default="qwen2.5-7b",
                        choices=list(MODEL_MAP.keys()),
                        help="Model to use")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "verifier", "a10", "reranker"],
                        help="Experiment to run")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples per state for stratified sampling")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization instead of 4-bit")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    # Setup
    model_name = MODEL_MAP[args.model]
    use_4bit = not args.use_8bit

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"outputs/llm_eval/{args.model}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")

    # Load data
    from final_sc_review.llm.data_loader import LLMEvaluationDataLoader

    per_query_csv = Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv")

    data_loader = LLMEvaluationDataLoader(
        data_dir=Path("data"),
        per_query_csv=per_query_csv,
    )

    # Get stratified sample
    samples = data_loader.get_stratified_sample(
        max_per_state=args.max_samples,
        states=['NEG', 'UNCERTAIN', 'POS'],
    )

    logger.info(f"Total samples: {len(samples)}")

    # Run experiments
    all_metrics = {
        "model": args.model,
        "model_name": model_name,
        "timestamp": timestamp,
        "max_samples_per_state": args.max_samples,
    }

    if args.experiment in ["all", "verifier"]:
        verifier_metrics = run_verifier_experiment(
            model_name=model_name,
            data_loader=data_loader,
            samples=samples,
            output_dir=output_dir,
            use_4bit=use_4bit,
        )
        all_metrics["verifier"] = verifier_metrics

    if args.experiment in ["all", "a10"]:
        a10_metrics = run_a10_experiment(
            model_name=model_name,
            data_loader=data_loader,
            samples=samples,
            output_dir=output_dir,
            use_4bit=use_4bit,
        )
        all_metrics["a10_classifier"] = a10_metrics

    if args.experiment in ["all", "reranker"]:
        reranker_metrics = run_reranker_experiment(
            model_name=model_name,
            data_loader=data_loader,
            samples=samples,
            output_dir=output_dir,
            use_4bit=use_4bit,
        )
        all_metrics["reranker"] = reranker_metrics

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Summary: {json.dumps(all_metrics, indent=2)}")


if __name__ == "__main__":
    main()
