"""LLM-based verifier for UNCERTAIN cases."""

import json
import logging
from typing import Dict, List

import numpy as np

from .base import LLMBase

logger = logging.getLogger(__name__)


class LLMVerifier(LLMBase):
    """LLM verifier for uncertain evidence cases with self-consistency."""
    
    VERIFY_PROMPT_TEMPLATE = """You are an expert clinical evidence reviewer. Your task is to determine if the provided sentences support a specific DSM-5 criterion.

**Post Context**: {post_text}

**DSM-5 Criterion**: {criterion_text}

**Evidence Sentences**:
{evidence_sentences}

**Task**: Determine if these sentences provide clear evidence for the criterion. Return ONLY a JSON object:
{{
  "has_evidence": true or false,
  "confidence": 0.0 to 1.0,
  "rationale": "brief explanation (1-2 sentences)"
}}

Guidelines:
- has_evidence=true: Sentences directly describe the criterion (e.g., depressed mood, loss of interest)
- has_evidence=false: Sentences are vague, tangential, or describe unrelated symptoms
- confidence: Your certainty in the judgment (1.0 = very certain, 0.5 = uncertain)

Your JSON response:"""
    
    def verify(
        self,
        post_text: str,
        criterion_text: str,
        evidence_sentences: List[str],
        self_consistency_runs: int = 3,
    ) -> Dict:
        """Verify if evidence supports criterion.
        
        Args:
            post_text: Full post text
            criterion_text: DSM-5 criterion description
            evidence_sentences: List of evidence sentences to verify
            self_consistency_runs: Number of runs for self-consistency check
            
        Returns:
            Dict with has_evidence, confidence, rationale, self_consistency_score
        """
        # Format evidence
        evidence_text = "\n".join([
            f"- {sent}" for sent in evidence_sentences
        ])
        
        # Create prompt
        prompt = self.VERIFY_PROMPT_TEMPLATE.format(
            post_text=post_text[:1000],
            criterion_text=criterion_text,
            evidence_sentences=evidence_text
        )
        
        # Generate multiple responses for self-consistency
        if self_consistency_runs > 1:
            responses = self.generate_multiple(
                prompt,
                n=self_consistency_runs,
                temperature=0.7
            )
        else:
            responses = [self.generate(prompt, use_cache=True, temperature=0.0)]
        
        # Parse responses
        parsed_responses = []
        for resp in responses:
            try:
                parsed = self._extract_json(resp)
                parsed_responses.append(parsed)
            except Exception as e:
                logger.warning(f"Failed to parse verification response: {e}")
        
        if not parsed_responses:
            # Fallback: uncertain
            return {
                "has_evidence": False,
                "confidence": 0.5,
                "rationale": "Failed to parse LLM response",
                "self_consistency_score": 0.0
            }
        
        # Compute self-consistency
        has_evidence_votes = [r.get("has_evidence", False) for r in parsed_responses]
        consistency_score = sum(has_evidence_votes) / len(has_evidence_votes)
        
        # Take majority vote
        majority_has_evidence = consistency_score >= 0.5
        
        # Average confidence
        confidences = [r.get("confidence", 0.5) for r in parsed_responses]
        avg_confidence = np.mean(confidences)
        
        # Use rationale from first response
        rationale = parsed_responses[0].get("rationale", "")
        
        return {
            "has_evidence": majority_has_evidence,
            "confidence": float(avg_confidence),
            "rationale": rationale,
            "self_consistency_score": float(consistency_score),
            "n_runs": len(parsed_responses),
        }
    
    # Note: _extract_json is now inherited from LLMBase which uses
    # the shared extract_json_from_response utility
