#!/usr/bin/env python3
"""LLM Evidence Verifier for Borderline Cases.

Fast Track Integration Method 2:
- Targets borderline P4 probabilities (e.g., 0.4-0.6 range)
- Uses LLM to verify if retrieved evidence actually supports criterion
- Binary decision: SUPPORTS / DOES_NOT_SUPPORT
- Compares to P4 model predictions

Usage:
    # Local Qwen test
    python scripts/llm/llm_evidence_verifier.py --model qwen --subset 200

    # Gemini confirmation
    python scripts/llm/llm_evidence_verifier.py --model gemini --subset 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from final_sc_review.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result from LLM verification."""
    query_id: str
    criterion_id: str
    post_id: str

    # Retrieved evidence
    evidence_sentences: List[str]
    evidence_sent_uids: List[str]

    # P4 model prediction
    p4_prob: float
    p4_prediction: bool  # thresholded

    # LLM verification
    llm_decision: str  # "SUPPORTS" / "DOES_NOT_SUPPORT" / "UNCERTAIN"
    llm_explanation: str
    llm_confidence: float  # 0-1

    # Ground truth
    has_evidence_gold: bool
    gold_sent_uids: List[str]

    # Agreement
    p4_correct: bool
    llm_correct: bool
    p4_llm_agree: bool


class QwenVerifier:
    """Local Qwen2.5-7B-Instruct verifier."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        """Initialize Qwen model."""
        logger.info(f"Loading {model_name} on {device}...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device
            )
            self.device = device
            logger.info(f"Qwen model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")
            logger.warning("Falling back to mock LLM for testing")
            self.model = None
            self.tokenizer = None
            self.device = device

    def verify(
        self,
        query: str,
        criterion: str,
        evidence_sentences: List[str]
    ) -> Tuple[str, str, float]:
        """Verify if evidence supports criterion.

        Args:
            query: Post text (context)
            criterion: DSM-5 criterion text
            evidence_sentences: Retrieved evidence sentences

        Returns:
            (decision, explanation, confidence)
        """
        # Build prompt
        prompt = self._build_verification_prompt(query, criterion, evidence_sentences)

        # If no model, return mock decision
        if self.model is None:
            logger.warning("Using mock LLM verification")
            decision = np.random.choice(["SUPPORTS", "DOES_NOT_SUPPORT", "UNCERTAIN"])
            return decision, "Mock LLM", 0.5

        # Generate verification
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse response
            decision, explanation, confidence = self._parse_llm_response(response)
            return decision, explanation, confidence

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "UNCERTAIN", f"Error: {e}", 0.0

    def _build_verification_prompt(self, query: str, criterion: str, evidence: List[str]) -> str:
        """Build verification prompt."""

        # Format evidence
        evidence_list = "\n".join([
            f"  {i+1}. {sent}"
            for i, sent in enumerate(evidence)
        ])

        prompt = f"""You are a clinical evidence verification assistant. Your task is to determine if the provided evidence sentences from a Reddit post actually support a given DSM-5 depression criterion.

**Post Context:**
{query[:300]}... (truncated)

**Criterion:**
{criterion}

**Retrieved Evidence:**
{evidence_list}

**Instructions:**
1. Read the criterion carefully
2. Evaluate whether the evidence sentences contain information that supports the criterion
3. Make a binary decision: SUPPORTS or DOES_NOT_SUPPORT
4. Provide a brief explanation (1-2 sentences)
5. Rate your confidence (LOW/MEDIUM/HIGH)

**Output Format:**
DECISION: <SUPPORTS or DOES_NOT_SUPPORT>
EXPLANATION: <brief justification>
CONFIDENCE: <LOW or MEDIUM or HIGH>
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Tuple[str, str, float]:
        """Parse LLM response to extract decision, explanation, and confidence."""

        try:
            # Extract decision
            decision_line = [line for line in response.split('\n') if line.strip().startswith('DECISION:')]
            if not decision_line:
                raise ValueError("No DECISION: found in response")

            decision_str = decision_line[0].split('DECISION:')[1].strip().upper()

            if "SUPPORTS" in decision_str and "NOT" not in decision_str:
                decision = "SUPPORTS"
            elif "DOES_NOT_SUPPORT" in decision_str or "NOT" in decision_str:
                decision = "DOES_NOT_SUPPORT"
            else:
                decision = "UNCERTAIN"

            # Extract explanation
            explanation_line = [line for line in response.split('\n') if line.strip().startswith('EXPLANATION:')]
            explanation = explanation_line[0].split('EXPLANATION:')[1].strip() if explanation_line else "No explanation"

            # Extract confidence
            confidence_line = [line for line in response.split('\n') if line.strip().startswith('CONFIDENCE:')]
            confidence_str = confidence_line[0].split('CONFIDENCE:')[1].strip().upper() if confidence_line else "MEDIUM"

            # Map to numeric
            confidence_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
            confidence = confidence_map.get(confidence_str, 0.5)

            return decision, explanation, confidence

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return "UNCERTAIN", f"Parse error: {e}", 0.0


class GeminiVerifier:
    """Gemini 1.5 Flash verifier."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini API client."""
        logger.info("Initializing Gemini API client...")

        import os
        from google import genai

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-1.5-flash"
        logger.info(f"Gemini client initialized")

    def verify(
        self,
        query: str,
        criterion: str,
        evidence_sentences: List[str]
    ) -> Tuple[str, str, float]:
        """Verify if evidence supports criterion using Gemini."""

        prompt = self._build_verification_prompt(query, criterion, evidence_sentences)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 256,
                }
            )

            response_text = response.text
            decision, explanation, confidence = self._parse_llm_response(response_text)
            return decision, explanation, confidence

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return "UNCERTAIN", f"Gemini error: {e}", 0.0

    def _build_verification_prompt(self, query: str, criterion: str, evidence: List[str]) -> str:
        """Build verification prompt (same as Qwen)."""

        evidence_list = "\n".join([
            f"  {i+1}. {sent}"
            for i, sent in enumerate(evidence)
        ])

        prompt = f"""You are a clinical evidence verification assistant. Your task is to determine if the provided evidence sentences from a Reddit post actually support a given DSM-5 depression criterion.

**Post Context:**
{query[:300]}... (truncated)

**Criterion:**
{criterion}

**Retrieved Evidence:**
{evidence_list}

**Instructions:**
1. Read the criterion carefully
2. Evaluate whether the evidence sentences contain information that supports the criterion
3. Make a binary decision: SUPPORTS or DOES_NOT_SUPPORT
4. Provide a brief explanation (1-2 sentences)
5. Rate your confidence (LOW/MEDIUM/HIGH)

**Output Format:**
DECISION: <SUPPORTS or DOES_NOT_SUPPORT>
EXPLANATION: <brief justification>
CONFIDENCE: <LOW or MEDIUM or HIGH>
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Tuple[str, str, float]:
        """Parse LLM response (same logic as Qwen)."""

        try:
            decision_line = [line for line in response.split('\n') if line.strip().startswith('DECISION:')]
            if not decision_line:
                raise ValueError("No DECISION: found")

            decision_str = decision_line[0].split('DECISION:')[1].strip().upper()

            if "SUPPORTS" in decision_str and "NOT" not in decision_str:
                decision = "SUPPORTS"
            elif "DOES_NOT_SUPPORT" in decision_str or "NOT" in decision_str:
                decision = "DOES_NOT_SUPPORT"
            else:
                decision = "UNCERTAIN"

            explanation_line = [line for line in response.split('\n') if line.strip().startswith('EXPLANATION:')]
            explanation = explanation_line[0].split('EXPLANATION:')[1].strip() if explanation_line else "No explanation"

            confidence_line = [line for line in response.split('\n') if line.strip().startswith('CONFIDENCE:')]
            confidence_str = confidence_line[0].split('CONFIDENCE:')[1].strip().upper() if confidence_line else "MEDIUM"

            confidence_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
            confidence = confidence_map.get(confidence_str, 0.5)

            return decision, explanation, confidence

        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return "UNCERTAIN", f"Parse error: {e}", 0.0


def load_borderline_queries(eval_results_dir: Path, prob_min: float = 0.4, prob_max: float = 0.6, max_queries: int = 200) -> pd.DataFrame:
    """Load borderline queries from clinical evaluation.

    Args:
        eval_results_dir: Directory with clinical evaluation results
        prob_min: Minimum P4 probability
        prob_max: Maximum P4 probability
        max_queries: Maximum number of queries to sample

    Returns:
        DataFrame with borderline queries
    """
    logger.info(f"Loading borderline queries (P4 prob in [{prob_min}, {prob_max}])")

    fold_files = sorted(eval_results_dir.glob("fold_results/fold_*_predictions.csv"))

    dfs = []
    for fold_file in fold_files:
        df = pd.read_csv(fold_file)
        dfs.append(df)

    all_queries = pd.concat(dfs, ignore_index=True)

    # Filter to borderline probabilities
    borderline = all_queries[
        (all_queries['p4_prob_calibrated'] >= prob_min) &
        (all_queries['p4_prob_calibrated'] <= prob_max)
    ].copy()

    logger.info(f"Found {len(borderline)} borderline queries")

    if len(borderline) > max_queries:
        borderline = borderline.sample(n=max_queries, random_state=42)
        logger.info(f"Sampled {max_queries} queries")

    return borderline


def main():
    parser = argparse.ArgumentParser(description="LLM Evidence Verification")
    parser.add_argument(
        "--model",
        choices=["qwen", "gemini"],
        default="qwen",
        help="LLM model to use"
    )
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("outputs/clinical_high_recall/20260118_015913"),
        help="Clinical evaluation results directory"
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=200,
        help="Number of borderline queries to evaluate"
    )
    parser.add_argument(
        "--prob_min",
        type=float,
        default=0.4,
        help="Minimum P4 probability for borderline"
    )
    parser.add_argument(
        "--prob_max",
        type=float,
        default=0.6,
        help="Maximum P4 probability for borderline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for local models"
    )

    args = parser.parse_args()

    # Create output directory
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"outputs/llm_verification/{timestamp}_{args.model}")

    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

    # Load borderline queries
    borderline_df = load_borderline_queries(
        args.eval_dir,
        prob_min=args.prob_min,
        prob_max=args.prob_max,
        max_queries=args.subset
    )

    logger.info(f"Loaded {len(borderline_df)} borderline queries")
    logger.info(f"Positive rate: {borderline_df['has_evidence_gold'].mean():.2%}")
    logger.info(f"Mean P4 prob: {borderline_df['p4_prob_calibrated'].mean():.3f}")

    # Initialize verifier
    if args.model == "qwen":
        verifier = QwenVerifier(device=args.device)
    else:
        verifier = GeminiVerifier()

    # Save configuration
    config = {
        "model": args.model,
        "eval_dir": str(args.eval_dir),
        "n_queries": len(borderline_df),
        "subset_size": args.subset,
        "prob_range": [args.prob_min, args.prob_max],
        "output_dir": str(args.output),
        "status": "Configuration saved - full implementation pending"
    }

    config_file = args.output / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to: {config_file}")
    logger.info("NOTE: Full LLM verification implementation requires corpus loading")
    logger.info("      This is a framework/template for Fast Track assessment")

    return 0


if __name__ == "__main__":
    sys.exit(main())
