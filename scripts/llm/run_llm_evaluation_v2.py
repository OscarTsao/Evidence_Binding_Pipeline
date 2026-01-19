#!/usr/bin/env python3
"""Run LLM integration evaluation with real data (Phase 1).

This script evaluates three LLM modules on actual posts and criteria:
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
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.final_sc_review.llm import LLMReranker, LLMVerifier, A10Classifier
from src.final_sc_review.llm.data_loader import LLMEvaluationDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_llm_reranker(
    reranker: LLMReranker,
    eval_df: pd.DataFrame,
    data_loader: LLMEvaluationDataLoader,
    output_dir: Path,
    max_samples: int = 50,
) -> Dict:
    """Evaluate LLM reranker module with real data.
    
    Args:
        reranker: LLM reranker instance
        eval_df: Evaluation DataFrame
        data_loader: Data loader for posts/criteria
        output_dir: Output directory
        max_samples: Max samples to evaluate
        
    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating LLM Reranker (max {max_samples} samples)...")
    
    results = []
    position_bias_scores = []
    
    # Sample queries
    sample_df = eval_df.sample(n=min(max_samples, len(eval_df)), random_state=42)
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="LLM Reranker"):
        post_id = row['post_id']
        criterion_id = row['criterion_id']
        
        # Get real data
        query_data = data_loader.get_query_data(post_id, criterion_id)
        post_text = query_data["post_text"]
        criterion_text = query_data["criterion_text"]
        
        # Get evidence sentences as candidates
        evidence_sentences = data_loader.get_evidence_sentences(post_id, top_k=10)
        
        if len(evidence_sentences) < 2:
            continue
        
        # Create candidates
        candidates = [
            {"sent_uid": f"{post_id}_sent_{i}", "sentence": sent, "score": 0.8}
            for i, sent in enumerate(evidence_sentences)
        ]
        
        try:
            # Rerank
            reranked, metadata = reranker.rerank(
                post_text=post_text,
                criterion_text=criterion_text,
                candidates=candidates,
                top_k=5,
                check_position_bias=(idx < 10)  # Only check bias for first 10 samples
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
                
        except Exception as e:
            logger.error(f"Error reranking {post_id}/{criterion_id}: {e}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "llm_reranker_results.csv", index=False)
    
    # Compute metrics
    metrics = {
        "n_samples": len(results),
        "mean_position_bias": float(np.mean(position_bias_scores)) if position_bias_scores else -1,
        "std_position_bias": float(np.std(position_bias_scores)) if position_bias_scores else -1,
        "n_bias_checks": len(position_bias_scores),
    }
    
    logger.info(f"  Evaluated: {metrics['n_samples']} queries")
    if position_bias_scores:
        logger.info(f"  Position bias: {metrics['mean_position_bias']:.3f} ± {metrics['std_position_bias']:.3f}")
    
    return metrics


def evaluate_llm_verifier(
    verifier: LLMVerifier,
    eval_df: pd.DataFrame,
    data_loader: LLMEvaluationDataLoader,
    output_dir: Path,
    max_samples: int = 50,
) -> Dict:
    """Evaluate LLM verifier on UNCERTAIN cases with real data.
    
    Args:
        verifier: LLM verifier instance
        eval_df: Evaluation DataFrame
        data_loader: Data loader
        output_dir: Output directory
        max_samples: Max samples to evaluate
        
    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating LLM Verifier on UNCERTAIN cases (max {max_samples} samples)...")
    
    # Filter to UNCERTAIN state only
    uncertain_df = eval_df[eval_df['state'] == 'UNCERTAIN'].copy()
    logger.info(f"  {len(uncertain_df)} total UNCERTAIN cases")
    
    if len(uncertain_df) == 0:
        return {"n_samples": 0}
    
    # Sample
    sample_df = uncertain_df.sample(n=min(max_samples, len(uncertain_df)), random_state=42)
    
    results = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="LLM Verifier"):
        post_id = row["post_id"]
        criterion_id = row["criterion_id"]
        
        # Get real data
        query_data = data_loader.get_query_data(post_id, criterion_id)
        post_text = query_data["post_text"]
        criterion_text = query_data["criterion_text"]
        evidence_sentences = data_loader.get_evidence_sentences(post_id, top_k=5)
        
        if not evidence_sentences:
            continue
        
        try:
            # Verify
            verification = verifier.verify(
                post_text=post_text,
                criterion_text=criterion_text,
                evidence_sentences=evidence_sentences,
                self_consistency_runs=3
            )
            
            results.append({
                "post_id": post_id,
                "criterion_id": criterion_id,
                "has_evidence_gold": row["has_evidence_gold"],
                "p4_prob_calibrated": row["p4_prob_calibrated"],
                "llm_has_evidence": verification["has_evidence"],
                "llm_confidence": verification["confidence"],
                "self_consistency_score": verification["self_consistency_score"],
                "n_runs": verification["n_runs"],
                "rationale": verification["rationale"],
            })
            
        except Exception as e:
            logger.error(f"Error verifying {post_id}/{criterion_id}: {e}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "llm_verifier_results.csv", index=False)
    
    # Compute metrics
    if results:
        results_df = pd.DataFrame(results)
        agreement = (results_df["llm_has_evidence"] == results_df["has_evidence_gold"]).mean()
        mean_confidence = results_df["llm_confidence"].mean()
        mean_consistency = results_df["self_consistency_score"].mean()
        
        metrics = {
            "n_samples": len(results),
            "agreement_with_gold": float(agreement),
            "mean_confidence": float(mean_confidence),
            "mean_self_consistency": float(mean_consistency),
        }
    else:
        metrics = {"n_samples": 0}
    
    if metrics.get("n_samples", 0) > 0:
        logger.info(f"  Evaluated: {metrics['n_samples']} UNCERTAIN queries")
        logger.info(f"  Agreement: {metrics['agreement_with_gold']:.3f}")
        logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
        logger.info(f"  Self-consistency: {metrics['mean_self_consistency']:.3f}")
    
    return metrics


def evaluate_a10_classifier(
    classifier: A10Classifier,
    eval_df: pd.DataFrame,
    data_loader: LLMEvaluationDataLoader,
    output_dir: Path,
    max_samples: int = 30,
) -> Dict:
    """Evaluate A.10-specific LLM classifier with real data.
    
    Args:
        classifier: A.10 classifier instance
        eval_df: Evaluation DataFrame
        data_loader: Data loader
        output_dir: Output directory
        max_samples: Max samples to evaluate
        
    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating A.10-specific LLM Classifier (max {max_samples} samples)...")
    
    # Note: A.9 is the actual suicidal ideation criterion
    # Filter to A.9 criterion
    a9_df = eval_df[eval_df['criterion_id'] == 'A.9'].copy()
    logger.info(f"  {len(a9_df)} total A.9 (suicidal ideation) cases")
    
    if len(a9_df) == 0:
        # Try A.10 if A.9 not found
        a9_df = eval_df[eval_df['criterion_id'] == 'A.10'].copy()
        logger.info(f"  {len(a9_df)} total A.10 cases")
    
    if len(a9_df) == 0:
        return {"n_samples": 0}
    
    # Sample
    sample_df = a9_df.sample(n=min(max_samples, len(a9_df)), random_state=42)
    
    results = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="A.10 Classifier"):
        post_id = row["post_id"]
        
        # Get real data
        post_text = data_loader.get_post_text(post_id)
        
        if not post_text or len(post_text) < 10:
            continue
        
        try:
            # Classify
            classification = classifier.classify(
                post_text=post_text,
                self_consistency_runs=3
            )
            
            results.append({
                "post_id": post_id,
                "has_evidence_gold": row["has_evidence_gold"],
                "p4_prob_calibrated": row["p4_prob_calibrated"],
                "llm_has_suicidal_ideation": classification["has_suicidal_ideation"],
                "llm_severity": classification["severity"],
                "llm_confidence": classification["confidence"],
                "self_consistency_score": classification["self_consistency_score"],
                "n_runs": classification["n_runs"],
                "evidence_sentences": "; ".join(classification["evidence_sentences"]),
                "rationale": classification["rationale"],
            })
            
        except Exception as e:
            logger.error(f"Error classifying {post_id}: {e}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "llm_a10_classifier_results.csv", index=False)
    
    # Compute metrics
    if results:
        results_df = pd.DataFrame(results)
        agreement = (results_df["llm_has_suicidal_ideation"] == results_df["has_evidence_gold"]).mean()
        mean_confidence = results_df["llm_confidence"].mean()
        mean_consistency = results_df["self_consistency_score"].mean()
        severity_dist = results_df["llm_severity"].value_counts().to_dict()
        
        metrics = {
            "n_samples": len(results),
            "agreement_with_gold": float(agreement),
            "mean_confidence": float(mean_confidence),
            "mean_self_consistency": float(mean_consistency),
            "severity_distribution": severity_dist,
        }
    else:
        metrics = {"n_samples": 0}
    
    if metrics.get("n_samples", 0) > 0:
        logger.info(f"  Evaluated: {metrics['n_samples']} A.9/A.10 queries")
        logger.info(f"  Agreement: {metrics['agreement_with_gold']:.3f}")
        logger.info(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
        logger.info(f"  Severity distribution: {metrics['severity_distribution']}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Evaluate LLM integration with real data"
    )
    parser.add_argument(
        "--per_query_csv",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Path to per_query.csv from main evaluation"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Data directory with posts and criteria"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for Phase 1 results"
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
        "--max_samples_reranker",
        type=int,
        default=50,
        help="Max samples for reranker evaluation"
    )
    parser.add_argument(
        "--max_samples_verifier",
        type=int,
        default=50,
        help="Max samples for verifier evaluation"
    )
    parser.add_argument(
        "--max_samples_a10",
        type=int,
        default=30,
        help="Max samples for A.10 classifier"
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
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (test data loading only, no model loading)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    config["phase"] = "Phase 1: Local Model Evaluation"
    config["per_query_csv"] = str(args.per_query_csv)
    config["data_dir"] = str(args.data_dir)
    config["output_dir"] = str(args.output_dir)
    
    with open(args.output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    logger.info("="*80)
    logger.info("PHASE 1: LLM INTEGRATION EVALUATION - LOCAL MODEL")
    logger.info("="*80)
    
    data_loader = LLMEvaluationDataLoader(
        data_dir=args.data_dir,
        per_query_csv=args.per_query_csv
    )
    
    eval_df = data_loader.get_stratified_sample(
        max_per_state=100,
        states=['NEG', 'UNCERTAIN', 'POS']
    )
    
    if args.dry_run:
        logger.info("\n=== DRY RUN MODE ===")
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Total queries: {len(eval_df)}")
        logger.info(f"  Unique posts: {eval_df['post_id'].nunique()}")
        logger.info(f"  Unique criteria: {eval_df['criterion_id'].nunique()}")
        logger.info(f"  State distribution:")
        for state, count in eval_df['state'].value_counts().items():
            logger.info(f"    {state}: {count}")
        
        # Test data loading for a few samples
        logger.info(f"\nTesting data loading on 3 samples:")
        for idx, row in eval_df.head(3).iterrows():
            query_data = data_loader.get_query_data(row['post_id'], row['criterion_id'])
            logger.info(f"\n  Post {row['post_id']}, Criterion {row['criterion_id']}:")
            logger.info(f"    Post length: {len(query_data['post_text'])} chars")
            logger.info(f"    Criterion: {query_data['criterion_text'][:80]}...")
            logger.info(f"    Post preview: {query_data['post_text'][:100]}...")
        
        logger.info("\nDry run complete. Set --dry_run=False to run actual evaluation.")
        return 0
    
    # Initialize models
    logger.info(f"\nInitializing LLM: {args.model_name}")
    logger.info(f"  4-bit quantization: {args.load_in_4bit}")
    
    model_kwargs = {
        "model_name": args.model_name,
        "cache_dir": args.output_dir / "cache",
        "load_in_4bit": args.load_in_4bit,
    }
    
    # Run evaluations
    all_metrics = {}
    
    if not args.skip_reranker:
        logger.info("\n" + "="*80)
        reranker = LLMReranker(**model_kwargs)
        reranker_metrics = evaluate_llm_reranker(
            reranker, eval_df, data_loader, args.output_dir,
            max_samples=args.max_samples_reranker
        )
        all_metrics["llm_reranker"] = reranker_metrics
        del reranker  # Free memory
    
    if not args.skip_verifier:
        logger.info("\n" + "="*80)
        verifier = LLMVerifier(**model_kwargs)
        verifier_metrics = evaluate_llm_verifier(
            verifier, eval_df, data_loader, args.output_dir,
            max_samples=args.max_samples_verifier
        )
        all_metrics["llm_verifier"] = verifier_metrics
        del verifier
    
    if not args.skip_a10:
        logger.info("\n" + "="*80)
        a10_classifier = A10Classifier(**model_kwargs)
        a10_metrics = evaluate_a10_classifier(
            a10_classifier, eval_df, data_loader, args.output_dir,
            max_samples=args.max_samples_a10
        )
        all_metrics["llm_a10_classifier"] = a10_metrics
        del a10_classifier
    
    # Save summary
    summary = {
        "phase": "Phase 1: Local Model Evaluation",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": all_metrics,
    }
    
    with open(args.output_dir / "phase1_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 1 EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info(f"\nSummary:")
    for module, metrics in all_metrics.items():
        logger.info(f"\n  {module}:")
        for k, v in metrics.items():
            if isinstance(v, dict):
                logger.info(f"    {k}:")
                for k2, v2 in v.items():
                    logger.info(f"      {k2}: {v2}")
            else:
                logger.info(f"    {k}: {v}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
