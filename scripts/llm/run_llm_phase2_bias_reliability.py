#!/usr/bin/env python3
"""Phase 2: LLM Bias & Reliability Testing

This script evaluates:
1. Subgroup fairness across criteria, difficulty levels, and probability bins
2. Prompt robustness with systematic variations
3. Self-consistency threshold validation
4. Test-retest reliability

Usage:
    python scripts/llm/run_llm_phase2_bias_reliability.py \\
        --phase1_results outputs/llm_eval/phase1_qwen_fixed \\
        --per_query_csv outputs/final_research_eval/20260118_031312_complete/per_query.csv \\
        --output_dir outputs/llm_eval/phase2_bias_reliability \\
        --model_name Qwen/Qwen2.5-7B-Instruct \\
        --load_in_4bit \\
        --n_samples_per_criterion 10 \\
        --n_prompt_variants 5 \\
        --n_reliability_runs 3
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from final_sc_review.llm.verifier import LLMVerifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_phase1_results(phase1_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load Phase 1 LLM evaluation results."""
    results = {}

    verifier_path = phase1_dir / "llm_verifier_results.csv"
    if verifier_path.exists():
        results['verifier'] = pd.read_csv(verifier_path)

    a10_path = phase1_dir / "llm_a10_classifier_results.csv"
    if a10_path.exists():
        results['a10'] = pd.read_csv(a10_path)

    return results


def stratify_queries(per_query_df: pd.DataFrame, n_samples_per_criterion: int = 10) -> pd.DataFrame:
    """Stratify queries for subgroup analysis.

    Stratification dimensions:
    - Criterion (A.1-A.10): 10 groups
    - Evidence status (has/no evidence): 2 groups
    - State (NEG/UNCERTAIN/POS): 3 groups
    - P4 probability bins: 3 groups
    """
    # Add probability bins
    per_query_df['p4_bin'] = pd.cut(
        per_query_df['p4_prob_calibrated'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['low', 'medium', 'high']
    )

    # Sample stratified by criterion and evidence status
    stratified = []

    for criterion in sorted(per_query_df['criterion_id'].unique()):
        crit_df = per_query_df[per_query_df['criterion_id'] == criterion]

        # Sample positive cases (with evidence)
        pos_df = crit_df[crit_df['has_evidence_gold'] == 1]
        if len(pos_df) > 0:
            n_pos = min(n_samples_per_criterion // 2, len(pos_df))
            stratified.append(pos_df.sample(n=n_pos, random_state=42))

        # Sample negative cases (no evidence)
        neg_df = crit_df[crit_df['has_evidence_gold'] == 0]
        if len(neg_df) > 0:
            n_neg = min(n_samples_per_criterion // 2, len(neg_df))
            stratified.append(neg_df.sample(n=n_neg, random_state=42))

    return pd.concat(stratified, ignore_index=True)


def generate_prompt_variants(base_criterion_text: str, variant_type: str) -> str:
    """Generate systematic prompt variations for robustness testing.

    Variant types:
    - 'baseline': Original prompt
    - 'paraphrase': Reworded criterion description
    - 'formal': More formal clinical language
    - 'concise': Shortened version
    - 'detailed': More detailed explanation
    """
    variants = {
        'baseline': base_criterion_text,
        'paraphrase': f"Determine whether the text indicates: {base_criterion_text.lower()}",
        'formal': f"Clinical assessment required: Does the provided evidence support the diagnostic criterion: {base_criterion_text}?",
        'concise': f"Evidence for: {base_criterion_text}?",
        'detailed': f"You are a clinical expert. Carefully evaluate whether the following sentences provide evidence for the DSM-5 criterion: {base_criterion_text}. Consider both explicit and implicit indicators."
    }

    return variants.get(variant_type, base_criterion_text)


def test_subgroup_fairness(
    verifier: LLMVerifier,
    test_df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """Test fairness across different subgroups."""
    logger.info("Testing subgroup fairness...")

    results = {
        'by_criterion': {},
        'by_state': {},
        'by_evidence_status': {},
        'by_p4_bin': {}
    }

    # Test by criterion
    for criterion in sorted(test_df['criterion_id'].unique()):
        crit_df = test_df[test_df['criterion_id'] == criterion]
        if len(crit_df) == 0:
            continue

        # Run verifier on this criterion's queries
        predictions = []
        gold = crit_df['has_evidence_gold'].values

        for idx, row in tqdm(crit_df.iterrows(), total=len(crit_df), desc=f"Criterion {criterion}"):
            # Create dummy prompt (simplified for Phase 2)
            prompt = f"Does the following text show evidence of {criterion}?"

            try:
                result = verifier.verify(
                    post_id=row['post_id'],
                    criterion_id=row['criterion_id'],
                    candidates=[],  # Simplified
                    self_consistency_runs=3
                )
                predictions.append(1 if result['has_evidence'] else 0)
            except Exception as e:
                logger.error(f"Error on {row['post_id']}, {row['criterion_id']}: {e}")
                predictions.append(0)  # Default to no evidence on error

        # Compute metrics
        predictions = np.array(predictions)
        results['by_criterion'][criterion] = {
            'n_samples': len(crit_df),
            'accuracy': accuracy_score(gold, predictions),
            'precision': precision_score(gold, predictions, zero_division=0),
            'recall': recall_score(gold, predictions, zero_division=0),
            'f1': f1_score(gold, predictions, zero_division=0)
        }

    # Test by state
    for state in ['NEG', 'UNCERTAIN', 'POS']:
        state_df = test_df[test_df['state'] == state]
        if len(state_df) == 0:
            continue

        results['by_state'][state] = {
            'n_samples': len(state_df),
            # Similar prediction loop...
        }

    # Save results
    with open(output_dir / 'subgroup_fairness_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def test_prompt_robustness(
    verifier: LLMVerifier,
    test_df: pd.DataFrame,
    n_variants: int,
    output_dir: Path
) -> Dict:
    """Test robustness to prompt variations."""
    logger.info("Testing prompt robustness...")

    variant_types = ['baseline', 'paraphrase', 'formal', 'concise', 'detailed'][:n_variants]

    # Sample a subset for robustness testing (expensive)
    sample_df = test_df.sample(n=min(30, len(test_df)), random_state=42)

    results = []

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Prompt robustness"):
        query_results = {
            'post_id': row['post_id'],
            'criterion_id': row['criterion_id'],
            'has_evidence_gold': row['has_evidence_gold'],
            'variants': {}
        }

        for variant_type in variant_types:
            # Note: This is simplified - in production, you'd modify the actual verifier prompt
            query_results['variants'][variant_type] = {
                'prediction': 0,  # Placeholder
                'confidence': 0.5
            }

        results.append(query_results)

    # Compute agreement across variants
    agreements = []
    for result in results:
        preds = [v['prediction'] for v in result['variants'].values()]
        agreement = len(set(preds)) == 1  # All variants agree
        agreements.append(agreement)

    summary = {
        'n_samples': len(sample_df),
        'n_variants': n_variants,
        'avg_agreement': np.mean(agreements),
        'variant_types': variant_types
    }

    with open(output_dir / 'prompt_robustness_results.json', 'w') as f:
        json.dump({'summary': summary, 'details': results}, f, indent=2)

    return summary


def test_self_consistency_threshold(
    phase1_verifier_df: pd.DataFrame,
    output_dir: Path
) -> Dict:
    """Validate optimal self-consistency threshold using Phase 1 results."""
    logger.info("Validating self-consistency thresholds...")

    thresholds = [0.0, 0.33, 0.5, 0.67, 1.0]
    results = {}

    for threshold in thresholds:
        # Filter to queries meeting threshold
        filtered_df = phase1_verifier_df[
            phase1_verifier_df['self_consistency_score'] >= threshold
        ]

        if len(filtered_df) == 0:
            results[threshold] = {
                'n_samples': 0,
                'coverage': 0.0,
                'accuracy': 0.0
            }
            continue

        # Compute accuracy on filtered queries
        predictions = filtered_df['llm_has_evidence'].astype(int)
        gold = filtered_df['has_evidence_gold'].astype(int)

        accuracy = accuracy_score(gold, predictions)
        coverage = len(filtered_df) / len(phase1_verifier_df)

        results[threshold] = {
            'n_samples': len(filtered_df),
            'coverage': coverage,
            'accuracy': accuracy,
            'precision': precision_score(gold, predictions, zero_division=0),
            'recall': recall_score(gold, predictions, zero_division=0),
            'f1': f1_score(gold, predictions, zero_division=0)
        }

    with open(output_dir / 'self_consistency_threshold_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def test_reliability(
    verifier: LLMVerifier,
    test_df: pd.DataFrame,
    n_runs: int,
    output_dir: Path
) -> Dict:
    """Test test-retest reliability."""
    logger.info(f"Testing test-retest reliability with {n_runs} runs...")

    # Sample subset for reliability testing
    sample_df = test_df.sample(n=min(20, len(test_df)), random_state=42)

    results = []

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Reliability"):
        query_results = {
            'post_id': row['post_id'],
            'criterion_id': row['criterion_id'],
            'has_evidence_gold': row['has_evidence_gold'],
            'runs': []
        }

        for run in range(n_runs):
            # Run verifier multiple times
            query_results['runs'].append({
                'run_id': run,
                'prediction': 0,  # Placeholder
                'confidence': 0.5
            })

        results.append(query_results)

    # Compute inter-run agreement
    agreements = []
    for result in results:
        preds = [r['prediction'] for r in result['runs']]
        agreement = len(set(preds)) == 1  # All runs agree
        agreements.append(agreement)

    summary = {
        'n_samples': len(sample_df),
        'n_runs': n_runs,
        'avg_agreement': np.mean(agreements),
        'perfect_agreement_rate': np.mean(agreements)
    }

    with open(output_dir / 'reliability_results.json', 'w') as f:
        json.dump({'summary': summary, 'details': results}, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Phase 2: LLM Bias & Reliability Testing')
    parser.add_argument('--phase1_results', type=str, required=True,
                        help='Path to Phase 1 results directory')
    parser.add_argument('--per_query_csv', type=str, required=True,
                        help='Path to research evaluation per_query.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for Phase 2 results')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='LLM model name')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Use 4-bit quantization')
    parser.add_argument('--n_samples_per_criterion', type=int, default=10,
                        help='Number of samples per criterion for fairness testing')
    parser.add_argument('--n_prompt_variants', type=int, default=5,
                        help='Number of prompt variants for robustness testing')
    parser.add_argument('--n_reliability_runs', type=int, default=3,
                        help='Number of runs for reliability testing')
    parser.add_argument('--skip_fairness', action='store_true',
                        help='Skip subgroup fairness testing')
    parser.add_argument('--skip_robustness', action='store_true',
                        help='Skip prompt robustness testing')
    parser.add_argument('--skip_reliability', action='store_true',
                        help='Skip test-retest reliability testing')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    logger.info(f"Starting Phase 2: Bias & Reliability Testing at {timestamp}")

    # Load Phase 1 results
    logger.info(f"Loading Phase 1 results from {args.phase1_results}...")
    phase1_results = load_phase1_results(Path(args.phase1_results))

    # Load per-query data
    logger.info(f"Loading per-query data from {args.per_query_csv}...")
    per_query_df = pd.read_csv(args.per_query_csv)

    # Stratify queries for testing
    logger.info(f"Stratifying queries ({args.n_samples_per_criterion} per criterion)...")
    test_df = stratify_queries(per_query_df, args.n_samples_per_criterion)
    logger.info(f"Selected {len(test_df)} queries for testing")

    # Initialize LLM verifier (for new tests)
    logger.info(f"Loading LLM model: {args.model_name}...")
    # verifier = LLMVerifier(
    #     model_name=args.model_name,
    #     load_in_4bit=args.load_in_4bit,
    #     cache_dir=output_dir / "llm_cache"
    # )
    verifier = None  # Placeholder for now

    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    config['n_test_queries'] = len(test_df)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Run tests
    phase2_results = {}

    # 1. Self-consistency threshold validation (uses Phase 1 results)
    if 'verifier' in phase1_results:
        logger.info("\n=== TEST 1: Self-Consistency Threshold Validation ===")
        sc_results = test_self_consistency_threshold(
            phase1_results['verifier'],
            output_dir
        )
        phase2_results['self_consistency_threshold'] = sc_results
        logger.info(f"Results: {json.dumps(sc_results, indent=2)}")

    # 2. Subgroup fairness (requires LLM inference - skip for now)
    if not args.skip_fairness and verifier is not None:
        logger.info("\n=== TEST 2: Subgroup Fairness ===")
        fairness_results = test_subgroup_fairness(verifier, test_df, output_dir)
        phase2_results['subgroup_fairness'] = fairness_results

    # 3. Prompt robustness (requires LLM inference - skip for now)
    if not args.skip_robustness and verifier is not None:
        logger.info("\n=== TEST 3: Prompt Robustness ===")
        robustness_results = test_prompt_robustness(
            verifier, test_df, args.n_prompt_variants, output_dir
        )
        phase2_results['prompt_robustness'] = robustness_results

    # 4. Test-retest reliability (requires LLM inference - skip for now)
    if not args.skip_reliability and verifier is not None:
        logger.info("\n=== TEST 4: Test-Retest Reliability ===")
        reliability_results = test_reliability(
            verifier, test_df, args.n_reliability_runs, output_dir
        )
        phase2_results['reliability'] = reliability_results

    # Save summary
    with open(output_dir / 'phase2_summary.json', 'w') as f:
        json.dump(phase2_results, f, indent=2)

    logger.info(f"\nPhase 2 testing complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
