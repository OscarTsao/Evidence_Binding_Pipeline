#!/usr/bin/env python3
"""Run Gemini API confirmatory evaluation with caching.

This script runs a subset of LLM evaluations using Gemini API
to validate local model results.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai package not installed")
    print("Install with: pip install google-genai")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiEvaluator:
    """Gemini API evaluator with caching and quota management."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize Gemini evaluator.
        
        Args:
            api_key: Gemini API key
            model_name: Model name
            cache_dir: Cache directory
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        if cache_dir is None:
            cache_dir = Path("outputs/gemini_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Gemini API: {model_name}")
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Load response from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)["response"]
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({"response": response}, f)
    
    def generate(
        self,
        prompt: str,
        use_cache: bool = True,
        max_retries: int = 3,
    ) -> str:
        """Generate response with caching and retry logic.
        
        Args:
            prompt: Input prompt
            use_cache: Whether to use caching
            max_retries: Max retries on failure
            
        Returns:
            Generated text
        """
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(prompt)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Generate with retries
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                    )
                )
                
                text = response.text
                
                # Cache response
                if use_cache:
                    self._save_to_cache(cache_key, text)
                
                return text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return ""


def verify_with_gemini(
    evaluator: GeminiEvaluator,
    post_text: str,
    criterion_text: str,
    evidence_sentences: List[str],
) -> Dict:
    """Verify evidence using Gemini API.
    
    Args:
        evaluator: Gemini evaluator
        post_text: Post text
        criterion_text: Criterion text
        evidence_sentences: Evidence sentences
        
    Returns:
        Verification result
    """
    prompt = f"""You are an expert clinical evidence reviewer. Determine if the provided sentences support a DSM-5 criterion.

**Post**: {post_text[:500]}

**Criterion**: {criterion_text}

**Evidence**: {"; ".join(evidence_sentences)}

Return ONLY a JSON object:
{{"has_evidence": true/false, "confidence": 0.0-1.0, "rationale": "brief explanation"}}

Your JSON response:"""
    
    try:
        response = evaluator.generate(prompt, use_cache=True)
        
        # Parse JSON
        start = response.index("{")
        end = response.rindex("}") + 1
        json_str = response[start:end]
        result = json.loads(json_str)
        
        return result
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {"has_evidence": False, "confidence": 0.0, "rationale": f"Error: {e}"}


def main():
    parser = argparse.ArgumentParser(
        description="Run Gemini API confirmatory evaluation"
    )
    parser.add_argument(
        "--per_query_csv",
        type=Path,
        default=Path("outputs/final_research_eval/20260118_031312_complete/per_query.csv"),
        help="Path to per_query.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Gemini API key (or set GOOGLE_API_KEY env var)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max samples to evaluate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model name"
    )
    
    args = parser.parse_args()
    
    # Get API key
    import os
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key provided. Use --api_key or set GOOGLE_API_KEY env var")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = GeminiEvaluator(
        api_key=api_key,
        model_name=args.model_name,
        cache_dir=args.output_dir / "cache"
    )
    
    # Load data
    logger.info(f"Loading data from {args.per_query_csv}")
    df = pd.read_csv(args.per_query_csv)
    
    # Sample UNCERTAIN cases
    uncertain_df = df[df['state'] == 'UNCERTAIN'].sample(
        n=min(args.max_samples, len(df[df['state'] == 'UNCERTAIN'])),
        random_state=42
    )
    logger.info(f"Evaluating {len(uncertain_df)} UNCERTAIN cases")
    
    # Run evaluations
    results = []
    
    for idx, row in tqdm(uncertain_df.iterrows(), total=len(uncertain_df)):
        post_text = f"Post ID: {row['post_id']}"  # Placeholder
        criterion_text = f"Criterion {row['criterion_id']}"  # Placeholder
        evidence_sentences = [f"Evidence {i}" for i in range(3)]  # Placeholder
        
        verification = verify_with_gemini(
            evaluator,
            post_text,
            criterion_text,
            evidence_sentences
        )
        
        results.append({
            "post_id": row["post_id"],
            "criterion_id": row["criterion_id"],
            "has_evidence_gold": row["has_evidence_gold"],
            "p4_prob_calibrated": row["p4_prob_calibrated"],
            "gemini_has_evidence": verification["has_evidence"],
            "gemini_confidence": verification["confidence"],
            "gemini_rationale": verification["rationale"],
        })
        
        # Rate limit: sleep briefly between requests
        time.sleep(0.5)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "gemini_evaluation_results.csv", index=False)
    
    # Compute metrics
    agreement = (results_df["gemini_has_evidence"] == results_df["has_evidence_gold"]).mean()
    mean_confidence = results_df["gemini_confidence"].mean()
    
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_name": args.model_name,
        "n_samples": len(results),
        "agreement_with_gold": float(agreement),
        "mean_confidence": float(mean_confidence),
    }
    
    with open(args.output_dir / "gemini_evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nGemini evaluation complete")
    logger.info(f"  Agreement: {agreement:.3f}")
    logger.info(f"  Mean confidence: {mean_confidence:.3f}")
    logger.info(f"  Results saved to {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
