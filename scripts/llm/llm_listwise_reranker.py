#!/usr/bin/env python3
"""LLM Listwise Reranking for UNCERTAIN Cases.

Fast Track Integration Method 1:
- Targets queries in UNCERTAIN state (clinical gate output)
- Uses local Qwen2.5-7B-Instruct for listwise reranking
- Compares to baseline (P4 + calibration + gate, no LLM)
- Evaluates on small subset (~200-500 UNCERTAIN queries)

Usage:
    # Local Qwen test
    python scripts/llm/llm_listwise_reranker.py --model qwen --subset 200

    # Gemini confirmation (subset only)
    python scripts/llm/llm_listwise_reranker.py --model gemini --subset 200
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
class LLMRerankResult:
    """Result from LLM reranking."""
    query_id: str
    criterion_id: str
    post_id: str
    original_ranking: List[str]  # sent_uids
    llm_ranking: List[str]       # reranked sent_uids
    original_scores: List[float]
    llm_explanation: str

    # Ground truth
    has_evidence: bool
    gold_sent_uids: List[str]

    # Metrics
    original_recall_at_k: float
    llm_recall_at_k: float
    original_ndcg: float
    llm_ndcg: float
    k: int


class QwenReranker:
    """Local Qwen2.5-7B-Instruct reranker."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        """Initialize Qwen model.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
        """
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

    def rerank(
        self,
        query: str,
        criterion: str,
        candidates: List[str],
        top_k: int = 10
    ) -> Tuple[List[int], str]:
        """Rerank candidates using LLM.

        Args:
            query: Post text
            criterion: DSM-5 criterion text
            candidates: List of candidate sentence texts
            top_k: Number of results to return

        Returns:
            (reranked_indices, explanation)
        """
        # Build prompt
        prompt = self._build_rerank_prompt(query, criterion, candidates)

        # If no model loaded, return mock ranking
        if self.model is None:
            logger.warning("Using mock LLM ranking (random permutation)")
            indices = list(np.random.permutation(len(candidates))[:top_k])
            return indices, "Mock LLM (no model loaded)"

        # Generate ranking
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse response to extract ranking
            indices, explanation = self._parse_llm_response(response, len(candidates), top_k)
            return indices, explanation

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback to original order
            return list(range(min(top_k, len(candidates)))), f"Error: {e}"

    def _build_rerank_prompt(self, query: str, criterion: str, candidates: List[str]) -> str:
        """Build listwise reranking prompt."""

        # Format candidates
        cand_list = "\n".join([
            f"[{i+1}] {sent}"
            for i, sent in enumerate(candidates)
        ])

        prompt = f"""You are a clinical evidence extraction assistant. Your task is to rank evidence sentences from a Reddit post that best support a given DSM-5 depression criterion.

**Post:**
{query[:500]}... (truncated)

**Criterion:**
{criterion}

**Candidate Evidence Sentences:**
{cand_list}

**Instructions:**
1. Rank the sentences from most relevant to least relevant for supporting the criterion
2. Output ONLY the ranked sentence numbers, separated by commas (e.g., "3,1,5,2,4")
3. After the ranking, provide a brief explanation (1-2 sentences)

**Output Format:**
RANKING: <comma-separated numbers>
EXPLANATION: <brief justification>
"""
        return prompt

    def _parse_llm_response(self, response: str, n_candidates: int, top_k: int) -> Tuple[List[int], str]:
        """Parse LLM response to extract ranking and explanation."""

        try:
            # Extract ranking line
            ranking_line = [line for line in response.split('\n') if line.strip().startswith('RANKING:')]
            if not ranking_line:
                raise ValueError("No RANKING: found in response")

            ranking_str = ranking_line[0].split('RANKING:')[1].strip()

            # Parse numbers (1-indexed from LLM, convert to 0-indexed)
            indices = [int(x.strip()) - 1 for x in ranking_str.split(',')]

            # Validate indices
            indices = [i for i in indices if 0 <= i < n_candidates]

            # Truncate to top_k
            indices = indices[:top_k]

            # Extract explanation
            explanation_line = [line for line in response.split('\n') if line.strip().startswith('EXPLANATION:')]
            explanation = explanation_line[0].split('EXPLANATION:')[1].strip() if explanation_line else "No explanation provided"

            return indices, explanation

        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to original order
            return list(range(min(top_k, n_candidates))), f"Parse error: {e}"


class GeminiReranker:
    """Gemini 1.5 Flash reranker."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini API client.

        Args:
            api_key: Gemini API key (if None, reads from env GEMINI_API_KEY)
        """
        logger.info("Initializing Gemini API client...")

        import os
        from google import genai

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-1.5-flash"
        logger.info(f"Gemini client initialized with model: {self.model_name}")

    def rerank(
        self,
        query: str,
        criterion: str,
        candidates: List[str],
        top_k: int = 10
    ) -> Tuple[List[int], str]:
        """Rerank candidates using Gemini."""

        # Build prompt (same as Qwen)
        prompt = self._build_rerank_prompt(query, criterion, candidates)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 512,
                }
            )

            response_text = response.text

            # Parse response
            indices, explanation = self._parse_llm_response(response_text, len(candidates), top_k)
            return indices, explanation

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return list(range(min(top_k, len(candidates)))), f"Gemini error: {e}"

    def _build_rerank_prompt(self, query: str, criterion: str, candidates: List[str]) -> str:
        """Build listwise reranking prompt (same as Qwen)."""

        cand_list = "\n".join([
            f"[{i+1}] {sent}"
            for i, sent in enumerate(candidates)
        ])

        prompt = f"""You are a clinical evidence extraction assistant. Your task is to rank evidence sentences from a Reddit post that best support a given DSM-5 depression criterion.

**Post:**
{query[:500]}... (truncated)

**Criterion:**
{criterion}

**Candidate Evidence Sentences:**
{cand_list}

**Instructions:**
1. Rank the sentences from most relevant to least relevant for supporting the criterion
2. Output ONLY the ranked sentence numbers, separated by commas (e.g., "3,1,5,2,4")
3. After the ranking, provide a brief explanation (1-2 sentences)

**Output Format:**
RANKING: <comma-separated numbers>
EXPLANATION: <brief justification>
"""
        return prompt

    def _parse_llm_response(self, response: str, n_candidates: int, top_k: int) -> Tuple[List[int], str]:
        """Parse LLM response (same logic as Qwen)."""

        try:
            ranking_line = [line for line in response.split('\n') if line.strip().startswith('RANKING:')]
            if not ranking_line:
                raise ValueError("No RANKING: found in response")

            ranking_str = ranking_line[0].split('RANKING:')[1].strip()
            indices = [int(x.strip()) - 1 for x in ranking_str.split(',')]
            indices = [i for i in indices if 0 <= i < n_candidates]
            indices = indices[:top_k]

            explanation_line = [line for line in response.split('\n') if line.strip().startswith('EXPLANATION:')]
            explanation = explanation_line[0].split('EXPLANATION:')[1].strip() if explanation_line else "No explanation provided"

            return indices, explanation

        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
            return list(range(min(top_k, n_candidates))), f"Parse error: {e}"


def load_uncertain_queries(eval_results_dir: Path, max_queries: int = 200) -> pd.DataFrame:
    """Load UNCERTAIN queries from clinical evaluation.

    Args:
        eval_results_dir: Directory with clinical evaluation results
        max_queries: Maximum number of queries to sample

    Returns:
        DataFrame with UNCERTAIN queries
    """
    logger.info(f"Loading UNCERTAIN queries from {eval_results_dir}")

    # Load fold predictions
    fold_files = sorted(eval_results_dir.glob("fold_results/fold_*_predictions.csv"))

    dfs = []
    for fold_file in fold_files:
        df = pd.read_csv(fold_file)
        dfs.append(df)

    all_queries = pd.concat(dfs, ignore_index=True)

    # Filter to UNCERTAIN state
    uncertain = all_queries[all_queries['state'] == 'UNCERTAIN'].copy()

    logger.info(f"Found {len(uncertain)} UNCERTAIN queries total")

    # Sample subset if needed
    if len(uncertain) > max_queries:
        uncertain = uncertain.sample(n=max_queries, random_state=42)
        logger.info(f"Sampled {max_queries} queries for evaluation")

    return uncertain


def main():
    parser = argparse.ArgumentParser(description="LLM Listwise Reranking")
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
        help="Number of UNCERTAIN queries to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: outputs/llm_reranking/<timestamp>)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for local models (cuda/cpu)"
    )

    args = parser.parse_args()

    # Create output directory
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"outputs/llm_reranking/{timestamp}_{args.model}")

    args.output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output}")

    # Load UNCERTAIN queries
    uncertain_df = load_uncertain_queries(args.eval_dir, args.subset)

    logger.info(f"Loaded {len(uncertain_df)} UNCERTAIN queries")
    logger.info(f"Positive rate: {uncertain_df['has_evidence_gold'].mean():.2%}")

    # Initialize reranker
    if args.model == "qwen":
        reranker = QwenReranker(device=args.device)
    else:
        reranker = GeminiReranker()

    # For now, save configuration and exit
    # Full implementation requires loading post texts and evidence candidates

    config = {
        "model": args.model,
        "eval_dir": str(args.eval_dir),
        "n_queries": len(uncertain_df),
        "subset_size": args.subset,
        "output_dir": str(args.output),
        "status": "Configuration saved - full implementation pending"
    }

    config_file = args.output / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to: {config_file}")
    logger.info("NOTE: Full LLM reranking implementation requires corpus loading")
    logger.info("      This is a framework/template for Fast Track assessment")

    return 0


if __name__ == "__main__":
    sys.exit(main())
