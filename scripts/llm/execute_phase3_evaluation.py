#!/usr/bin/env python3
"""Execute Phase 3: Gemini API Evaluation

This script runs the actual Phase 3 evaluation with real data and Gemini API.

Prerequisites:
- GEMINI_API_KEY environment variable must be set
- Post text mapping must exist (created by preparation script)

Usage:
    export GEMINI_API_KEY="your-api-key-here"
    python scripts/llm/execute_phase3_evaluation.py \\
        --mode subsample \\
        --n_queries 100
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.llm.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DSM-5 criterion descriptions
CRITERION_DESCRIPTIONS = {
    'A.1': 'Depressed mood most of the day, nearly every day',
    'A.2': 'Markedly diminished interest or pleasure in all, or almost all, activities',
    'A.3': 'Significant weight loss or weight gain, or decrease or increase in appetite',
    'A.4': 'Insomnia or hypersomnia nearly every day',
    'A.5': 'Psychomotor agitation or retardation nearly every day (observable by others)',
    'A.6': 'Fatigue or loss of energy nearly every day',
    'A.7': 'Feelings of worthlessness or excessive or inappropriate guilt',
    'A.8': 'Diminished ability to think or concentrate, or indecisiveness',
    'A.9': 'Recurrent thoughts of death (not just fear of dying)',
    'A.10': 'Recurrent suicidal ideation, suicide attempt, or specific plan for committing suicide'
}


def create_verification_prompt(criterion_id: str, post_text: str) -> str:
    """Create prompt for evidence verification."""
    criterion_desc = CRITERION_DESCRIPTIONS.get(criterion_id, criterion_id)

    prompt = f"""You are a clinical expert evaluating evidence for DSM-5 Major Depressive Disorder criteria.

Criterion: {criterion_desc}

Does the following text provide evidence for this criterion?

Text: {post_text}

Respond with ONLY a JSON object:
{{
    "has_evidence": true or false,
    "confidence": 0.0 to 1.0,
    "rationale": "brief clinical explanation"
}}"""

    return prompt


def evaluate_with_gemini(
    gemini_client: GeminiClient,
    queries_df: pd.DataFrame,
    post_texts: Dict[str, str],
    output_dir: Path
) -> List[Dict]:
    """Evaluate queries with Gemini API."""
    results = []
    errors = []

    logger.info(f"Evaluating {len(queries_df)} queries with Gemini...")

    for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Gemini evaluation"):
        post_id = row['post_id']
        criterion_id = row['criterion_id']

        # Get post text
        post_text = post_texts.get(post_id, '')
        if not post_text:
            logger.warning(f"No text for post {post_id}, skipping")
            continue

        # Truncate very long posts (>2000 words)
        words = post_text.split()
        if len(words) > 2000:
            post_text = ' '.join(words[:2000]) + '...'

        # Create prompt
        prompt = create_verification_prompt(criterion_id, post_text)

        # Call API
        start_time = time.time()
        try:
            response = gemini_client.generate_json(prompt)
            latency = time.time() - start_time

            has_evidence = response.get('has_evidence', False)
            confidence = response.get('confidence', 0.5)
            rationale = response.get('rationale', '')

            results.append({
                'post_id': post_id,
                'criterion_id': criterion_id,
                'has_evidence_gold': int(row['has_evidence_gold']),
                'has_evidence_pred': int(has_evidence),
                'confidence': float(confidence),
                'latency': float(latency),
                'rationale': rationale,
                'model': 'gemini-2.0-flash',
                'p4_prob_calibrated': float(row['p4_prob_calibrated']),
                'state': row['state']
            })

        except Exception as e:
            logger.error(f"Error on {post_id}, {criterion_id}: {e}")
            errors.append({
                'post_id': post_id,
                'criterion_id': criterion_id,
                'error': str(e)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'gemini_evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"✅ Results saved to {results_path}")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = output_dir / 'gemini_evaluation_errors.csv'
        errors_df.to_csv(errors_path, index=False)
        logger.warning(f"⚠️ {len(errors)} errors saved to {errors_path}")

    return results


def compute_metrics(results_df: pd.DataFrame) -> Dict:
    """Compute evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    predictions = results_df['has_evidence_pred'].values
    gold = results_df['has_evidence_gold'].values

    # Try to compute AUROC with probabilities (confidence scores)
    try:
        auroc = roc_auc_score(gold, results_df['confidence'].values)
    except:
        auroc = None

    metrics = {
        'n_samples': len(results_df),
        'n_positive': int(gold.sum()),
        'n_negative': int((1-gold).sum()),
        'accuracy': float(accuracy_score(gold, predictions)),
        'precision': float(precision_score(gold, predictions, zero_division=0)),
        'recall': float(recall_score(gold, predictions, zero_division=0)),
        'f1': float(f1_score(gold, predictions, zero_division=0)),
        'auroc': float(auroc) if auroc is not None else None,
        'avg_confidence': float(results_df['confidence'].mean()),
        'avg_latency': float(results_df['latency'].mean()),
        'median_latency': float(results_df['latency'].median()),
        'total_latency': float(results_df['latency'].sum())
    }

    return metrics


def compare_with_phase1(gemini_results_df: pd.DataFrame, output_dir: Path) -> Dict:
    """Compare Gemini results with Phase 1 Qwen results."""
    # Load Phase 1 verifier results
    phase1_path = Path('outputs/llm_eval/phase1_qwen_fixed/llm_verifier_results.csv')

    if not phase1_path.exists():
        logger.warning(f"Phase 1 results not found at {phase1_path}")
        return {}

    phase1_df = pd.read_csv(phase1_path)

    # Phase 1 metrics
    phase1_predictions = phase1_df['llm_has_evidence'].astype(int).values
    phase1_gold = phase1_df['has_evidence_gold'].astype(int).values

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    phase1_metrics = {
        'n_samples': len(phase1_df),
        'accuracy': float(accuracy_score(phase1_gold, phase1_predictions)),
        'precision': float(precision_score(phase1_gold, phase1_predictions, zero_division=0)),
        'recall': float(recall_score(phase1_gold, phase1_predictions, zero_division=0)),
        'f1': float(f1_score(phase1_gold, phase1_predictions, zero_division=0)),
        'avg_confidence': float(phase1_df['llm_confidence'].mean()),
        'avg_self_consistency': float(phase1_df['self_consistency_score'].mean())
    }

    # Gemini metrics (computed separately)
    gemini_metrics = compute_metrics(gemini_results_df)

    comparison = {
        'phase1_qwen': phase1_metrics,
        'phase3_gemini': gemini_metrics,
        'delta': {
            'accuracy': gemini_metrics['accuracy'] - phase1_metrics['accuracy'],
            'precision': gemini_metrics['precision'] - phase1_metrics['precision'],
            'recall': gemini_metrics['recall'] - phase1_metrics['recall'],
            'f1': gemini_metrics['f1'] - phase1_metrics['f1']
        }
    }

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Execute Phase 3: Gemini API Evaluation')
    parser.add_argument('--mode', type=str, choices=['subsample', 'robustness', 'full'],
                        default='subsample', help='Evaluation mode')
    parser.add_argument('--n_queries', type=int, default=100,
                        help='Number of queries for subsample mode')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/llm_eval/phase3_gemini_validation',
                        help='Output directory')
    parser.add_argument('--gemini_model', type=str, default='gemini-2.0-flash-exp',
                        help='Gemini model name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Check API key
    if not os.getenv('GEMINI_API_KEY'):
        logger.error("❌ GEMINI_API_KEY environment variable not set")
        logger.error("Please run: export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    logger.info(f"Starting Phase 3 Gemini Evaluation at {timestamp}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.gemini_model}")

    # Load post texts
    logger.info("Loading post text mapping...")
    post_texts_path = output_dir / 'post_text_mapping.json'
    with open(post_texts_path) as f:
        post_texts = json.load(f)
    logger.info(f"Loaded {len(post_texts)} post texts")

    # Load stratified sample
    logger.info("Loading stratified sample...")
    sample_df = pd.read_csv(output_dir / 'stratified_sample_500.csv')
    logger.info(f"Loaded {len(sample_df)} queries")

    # Select queries based on mode
    if args.mode == 'subsample':
        queries_df = sample_df.sample(n=min(args.n_queries, len(sample_df)), random_state=args.seed)
        logger.info(f"Selected {len(queries_df)} queries for subsample evaluation")
    elif args.mode == 'robustness':
        queries_df = sample_df.sample(n=20, random_state=args.seed)
        logger.info(f"Selected {len(queries_df)} queries for robustness testing")
    else:  # full
        queries_df = sample_df
        logger.info(f"Using all {len(queries_df)} queries for full evaluation")

    # Initialize Gemini client
    logger.info(f"Initializing Gemini client ({args.gemini_model})...")
    gemini_client = GeminiClient(
        model_name=args.gemini_model,
        temperature=0.0
    )

    # Test connection
    logger.info("Testing Gemini API connection...")
    try:
        test_result = gemini_client.generate_json(
            prompt='Return a JSON object: {"status": "ok", "message": "Connection successful"}'
        )
        logger.info(f"✅ Connection successful: {test_result}")
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        sys.exit(1)

    # Run evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING {args.mode.upper()} EVALUATION")
    logger.info(f"{'='*60}\n")

    start_time = time.time()
    results = evaluate_with_gemini(gemini_client, queries_df, post_texts, output_dir)
    total_time = time.time() - start_time

    logger.info(f"\n✅ Evaluation complete in {total_time:.1f}s")
    logger.info(f"   Avg time per query: {total_time/len(results):.2f}s")

    # Compute metrics
    logger.info("\nComputing metrics...")
    results_df = pd.DataFrame(results)
    metrics = compute_metrics(results_df)

    logger.info("\n=== GEMINI PERFORMANCE ===")
    logger.info(f"Samples: {metrics['n_samples']}")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    logger.info(f"F1: {metrics['f1']:.3f}")
    if metrics['auroc']:
        logger.info(f"AUROC: {metrics['auroc']:.3f}")
    logger.info(f"Avg Confidence: {metrics['avg_confidence']:.3f}")
    logger.info(f"Avg Latency: {metrics['avg_latency']:.2f}s")

    # Compare with Phase 1
    logger.info("\nComparing with Phase 1 Qwen results...")
    comparison = compare_with_phase1(results_df, output_dir)

    if comparison:
        logger.info("\n=== QWEN VS GEMINI COMPARISON ===")
        logger.info(f"Qwen F1: {comparison['phase1_qwen']['f1']:.3f}")
        logger.info(f"Gemini F1: {comparison['phase3_gemini']['f1']:.3f}")
        logger.info(f"Delta F1: {comparison['delta']['f1']:+.3f}")
        logger.info(f"Delta Accuracy: {comparison['delta']['accuracy']:+.3f}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'mode': args.mode,
        'model': args.gemini_model,
        'n_queries': len(results),
        'total_time_seconds': total_time,
        'metrics': metrics,
        'comparison': comparison if comparison else None
    }

    summary_path = output_dir / f'phase3_{args.mode}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Summary saved to {summary_path}")
    logger.info(f"\nPhase 3 {args.mode} evaluation complete!")


if __name__ == '__main__':
    main()
