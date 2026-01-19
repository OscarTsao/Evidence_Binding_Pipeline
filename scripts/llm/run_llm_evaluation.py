#!/usr/bin/env python3
"""Run LLM integration evaluation on stratified subset.

This script evaluates three LLM modules:
1. LLM Reranker (listwise on top-10)
2. LLM Verifier (for UNCERTAIN cases)
3. A.10-specific LLM Classifier

With bias controls:
- Position bias checking (forward vs reverse)
- Self-consistency checking (multiple runs)
- Conservative safety thresholds for clinical use
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.final_sc_review.llm import LLMReranker, LLMVerifier, A10Classifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_data(
    per_query_csv: Path,
    stratify_by_state: bool = True,
    max_samples_per_state: int = 100,
) -> pd.DataFrame:
    """Load stratified subset for LLM evaluation.
    
    Args:
        per_query_csv: Path to per_query.csv from main evaluation
        stratify_by_state: Whether to stratify by NEG/UNCERTAIN/POS
        max_samples_per_state: Max samples per state
        
    Returns:
        Stratified DataFrame
    """
    logger.info(f"Loading data from {per_query_csv}")
    df = pd.read_csv(per_query_csv)
    
    if stratify_by_state:
        # Sample from each state
        samples = []
        for state in ['NEG', 'UNCERTAIN', 'POS']:
            state_df = df[df['state'] == state]
            n = min(max_samples_per_state, len(state_df))
            state_sample = state_df.sample(n=n, random_state=42)
            samples.append(state_sample)
            logger.info(f"  {state}: {n} samples")
        
        result = pd.concat(samples, ignore_index=True)
    else:
        # Random sample
        n = min(max_samples_per_state * 3, len(df))
        result = df.sample(n=n, random_state=42)
    
    logger.info(f"Total samples: {len(result)}")
    return result


def evaluate_llm_reranker(
    reranker: LLMReranker,
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """Evaluate LLM reranker module.
    
    Args:
        reranker: LLM reranker instance
        eval_df: Evaluation DataFrame
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    logger.info("Evaluating LLM Reranker...")
    
    results = []
    position_bias_scores = []
    
    # Group by post to get full context
    for (post_id, criterion_id), group in tqdm(eval_df.groupby(['post_id', 'criterion_id'])):
        # Skip if no post text available (would need to load from data)
        # For now, use a placeholder
        post_text = f"Post ID: {post_id}"  # TODO: Load actual post text
        criterion_text = f"Criterion {criterion_id}"  # TODO: Load actual criterion
        
        # Create mock candidates (would come from retrieval stage)
        candidates = [
            {"sent_uid": f"{post_id}_sent_{i}", "sentence": f"Sentence {i}", "score": 0.8}
            for i in range(5)
        ]
        
        # Rerank
        reranked, metadata = reranker.rerank(
            post_text=post_text,
            criterion_text=criterion_text,
            candidates=candidates,
            top_k=5,
            check_position_bias=True
        )
        
        results.append({
            "post_id": post_id,
            "criterion_id": criterion_id,
            "original_order": metadata["original_order"],
            "reranked_order": metadata["reranked_order"],
            "rationale": metadata["rationale"],
            "position_bias_score": metadata.get("position_bias_score", -1)
        })
        
        if metadata.get("position_bias_score", -1) >= 0:
            position_bias_scores.append(metadata["position_bias_score"])
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "llm_reranker_results.csv", index=False)
    
    # Compute metrics
    metrics = {
        "n_samples": len(results),
        "mean_position_bias": float(np.mean(position_bias_scores)) if position_bias_scores else -1,
        "std_position_bias": float(np.std(position_bias_scores)) if position_bias_scores else -1,
    }
    
    logger.info(f"  Position bias: {metrics['mean_position_bias']:.3f} ± {metrics['std_position_bias']:.3f}")
    
    return metrics


def evaluate_llm_verifier(
    verifier: LLMVerifier,
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """Evaluate LLM verifier on UNCERTAIN cases.
    
    Args:
        verifier: LLM verifier instance
        eval_df: Evaluation DataFrame (should be UNCERTAIN cases)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    logger.info("Evaluating LLM Verifier on UNCERTAIN cases...")
    
    # Filter to UNCERTAIN state only
    uncertain_df = eval_df[eval_df['state'] == 'UNCERTAIN'].copy()
    logger.info(f"  {len(uncertain_df)} UNCERTAIN cases")
    
    if len(uncertain_df) == 0:
        return {"n_samples": 0}
    
    results = []
    
    for idx, row in tqdm(uncertain_df.iterrows(), total=len(uncertain_df)):
        # Load context (placeholder)
        post_text = f"Post ID: {row['post_id']}"
        criterion_text = f"Criterion {row['criterion_id']}"
        evidence_sentences = [f"Evidence sentence {i}" for i in range(3)]
        
        # Verify
        verification = verifier.verify(
            post_text=post_text,
            criterion_text=criterion_text,
            evidence_sentences=evidence_sentences,
            self_consistency_runs=3
        )
        
        results.append({
            "post_id": row["post_id"],
            "criterion_id": row["criterion_id"],
            "has_evidence_gold": row["has_evidence_gold"],
            "p4_prob_calibrated": row["p4_prob_calibrated"],
            "llm_has_evidence": verification["has_evidence"],
            "llm_confidence": verification["confidence"],
            "self_consistency_score": verification["self_consistency_score"],
            "rationale": verification["rationale"],
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "llm_verifier_results.csv", index=False)
    
    # Compute metrics
    agreement = (results_df["llm_has_evidence"] == results_df["has_evidence_gold"]).mean()
    mean_confidence = results_df["llm_confidence"].mean()
    mean_consistency = results_df["self_consistency_score"].mean()
    
    metrics = {
        "n_samples": len(results),
        "agreement_with_gold": float(agreement),
        "mean_confidence": float(mean_confidence),
        "mean_self_consistency": float(mean_consistency),
    }
    
    logger.info(f"  Agreement: {metrics['agreement_with_gold']:.3f}")
    logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
    logger.info(f"  Self-consistency: {metrics['mean_self_consistency']:.3f}")
    
    return metrics


def evaluate_a10_classifier(
    classifier: A10Classifier,
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> Dict:
    """Evaluate A.10-specific LLM classifier.
    
    Args:
        classifier: A.10 classifier instance
        eval_df: Evaluation DataFrame (A.10 criterion only)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    logger.info("Evaluating A.10-specific LLM Classifier...")
    
    # Filter to A.10 criterion only
    a10_df = eval_df[eval_df['criterion_id'] == 'A.10'].copy()
    logger.info(f"  {len(a10_df)} A.10 cases")
    
    if len(a10_df) == 0:
        return {"n_samples": 0}
    
    results = []
    
    for idx, row in tqdm(a10_df.iterrows(), total=len(a10_df)):
        # Load post text (placeholder)
        post_text = f"Post ID: {row['post_id']}"
        
        # Classify
        classification = classifier.classify(
            post_text=post_text,
            self_consistency_runs=3
        )
        
        results.append({
            "post_id": row["post_id"],
            "has_evidence_gold": row["has_evidence_gold"],
            "p4_prob_calibrated": row["p4_prob_calibrated"],
            "llm_has_suicidal_ideation": classification["has_suicidal_ideation"],
            "llm_severity": classification["severity"],
            "llm_confidence": classification["confidence"],
            "self_consistency_score": classification["self_consistency_score"],
            "evidence_sentences": "; ".join(classification["evidence_sentences"]),
            "rationale": classification["rationale"],
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "llm_a10_classifier_results.csv", index=False)
    
    # Compute metrics
    agreement = (results_df["llm_has_suicidal_ideation"] == results_df["has_evidence_gold"]).mean()
    mean_confidence = results_df["llm_confidence"].mean()
    mean_consistency = results_df["self_consistency_score"].mean()
    
    # Severity distribution
    severity_dist = results_df["llm_severity"].value_counts().to_dict()
    
    metrics = {
        "n_samples": len(results),
        "agreement_with_gold": float(agreement),
        "mean_confidence": float(mean_confidence),
        "mean_self_consistency": float(mean_consistency),
        "severity_distribution": severity_dist,
    }
    
    logger.info(f"  Agreement: {metrics['agreement_with_gold']:.3f}")
    logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
    logger.info(f"  Severity distribution: {severity_dist}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM integration modules on stratified subset"
    )
    parser.add_argument(
        "--per_query_csv",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Path to per_query.csv from main evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for LLM evaluation results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="LLM model name"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max samples per state for evaluation"
    )
    parser.add_argument(
        "--skip_reranker",
        action="store_true",
        help="Skip reranker evaluation"
    )
    parser.add_argument(
        "--skip_verifier",
        action="store_true",
        help="Skip verifier evaluation"
    )
    parser.add_argument(
        "--skip_a10",
        action="store_true",
        help="Skip A.10 classifier evaluation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    config["per_query_csv"] = str(args.per_query_csv)
    config["output_dir"] = str(args.output_dir)
    
    with open(args.output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load evaluation data
    eval_df = load_evaluation_data(
        args.per_query_csv,
        stratify_by_state=True,
        max_samples_per_state=args.max_samples
    )
    
    # Initialize models
    logger.info(f"Initializing LLM: {args.model_name}")
    model_kwargs = {
        "model_name": args.model_name,
        "cache_dir": args.output_dir / "cache",
        "load_in_4bit": args.load_in_4bit,
    }
    
    # Run evaluations
    all_metrics = {}
    
    if not args.skip_reranker:
        reranker = LLMReranker(**model_kwargs)
        reranker_metrics = evaluate_llm_reranker(reranker, eval_df, args.output_dir)
        all_metrics["llm_reranker"] = reranker_metrics
        del reranker  # Free memory
    
    if not args.skip_verifier:
        verifier = LLMVerifier(**model_kwargs)
        verifier_metrics = evaluate_llm_verifier(verifier, eval_df, args.output_dir)
        all_metrics["llm_verifier"] = verifier_metrics
        del verifier
    
    if not args.skip_a10:
        a10_classifier = A10Classifier(**model_kwargs)
        a10_metrics = evaluate_a10_classifier(a10_classifier, eval_df, args.output_dir)
        all_metrics["llm_a10_classifier"] = a10_metrics
        del a10_classifier
    
    # Save summary
    summary = {
        "config": config,
        "metrics": all_metrics,
    }
    
    with open(args.output_dir / "llm_evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nLLM evaluation complete. Results saved to {args.output_dir}")
    logger.info(f"\nSummary:")
    for module, metrics in all_metrics.items():
        logger.info(f"  {module}:")
        for k, v in metrics.items():
            logger.info(f"    {k}: {v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
