"""A.10-specific LLM classifier for suicidal ideation detection."""

import json
import logging
from typing import Dict, List

import numpy as np

from .base import LLMBase

logger = logging.getLogger(__name__)


class A10Classifier(LLMBase):
    """Specialized LLM classifier for A.10 (suicidal ideation)."""
    
    A10_PROMPT_TEMPLATE = """You are an expert clinical psychologist specializing in suicide risk assessment. Your task is to detect evidence of suicidal ideation in social media posts.

**Post**: {post_text}

**DSM-5 Criterion A.10**: Recurrent thoughts of death (not just fear of dying), recurrent suicidal ideation without a specific plan, or a suicide attempt or a specific plan for committing suicide.

**Task**: Determine if this post contains evidence of suicidal ideation. Return ONLY a JSON object:
{{
  "has_suicidal_ideation": true or false,
  "severity": "none" | "passive" | "active" | "plan",
  "confidence": 0.0 to 1.0,
  "evidence_sentences": ["quoted sentences that indicate suicidal ideation"],
  "rationale": "brief clinical explanation (1-2 sentences)"
}}

**Severity Guidelines**:
- none: No thoughts of death or suicide
- passive: Wishes to not exist, thoughts of death without active intent ("wish I wasn't here")
- active: Active thoughts of suicide without specific plan ("I want to kill myself")
- plan: Specific plan or prior attempt mentioned

**Important**: Look for euphemistic language ("end it all", "not wake up", "better off without me").

Your JSON response:"""
    
    def classify(
        self,
        post_text: str,
        self_consistency_runs: int = 3,
    ) -> Dict:
        """Classify A.10 suicidal ideation.
        
        Args:
            post_text: Full post text
            self_consistency_runs: Number of runs for self-consistency
            
        Returns:
            Dict with has_suicidal_ideation, severity, confidence, evidence_sentences, rationale
        """
        # Create prompt
        prompt = self.A10_PROMPT_TEMPLATE.format(post_text=post_text)
        
        # Generate multiple responses for high-stakes decision
        if self_consistency_runs > 1:
            responses = self.generate_multiple(
                prompt,
                n=self_consistency_runs,
                temperature=0.5  # Lower temp for clinical safety
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
                logger.warning(f"Failed to parse A.10 response: {e}")
        
        if not parsed_responses:
            # Fallback: assume positive for safety (conservative)
            return {
                "has_suicidal_ideation": True,
                "severity": "passive",
                "confidence": 0.3,
                "evidence_sentences": [],
                "rationale": "Failed to parse LLM response - flagged for manual review",
                "self_consistency_score": 0.0
            }
        
        # Compute self-consistency
        has_si_votes = [r.get("has_suicidal_ideation", False) for r in parsed_responses]
        consistency_score = sum(has_si_votes) / len(has_si_votes)
        
        # Conservative: flag if ANY run detected suicidal ideation
        majority_has_si = consistency_score > 0.0  # Even 1/3 triggers flag
        
        # Use most severe severity
        severities = [r.get("severity", "none") for r in parsed_responses]
        severity_order = ["none", "passive", "active", "plan"]
        max_severity = max(severities, key=lambda s: severity_order.index(s) if s in severity_order else 0)
        
        # Average confidence
        confidences = [r.get("confidence", 0.5) for r in parsed_responses]
        avg_confidence = np.mean(confidences)
        
        # Collect all evidence sentences
        all_evidence = []
        for r in parsed_responses:
            all_evidence.extend(r.get("evidence_sentences", []))
        unique_evidence = list(set(all_evidence))
        
        # Use rationale from first response
        rationale = parsed_responses[0].get("rationale", "")
        
        return {
            "has_suicidal_ideation": majority_has_si,
            "severity": max_severity,
            "confidence": float(avg_confidence),
            "evidence_sentences": unique_evidence,
            "rationale": rationale,
            "self_consistency_score": float(consistency_score),
            "n_runs": len(parsed_responses),
        }
    
    def _extract_json(self, response: str) -> Dict:
        """Extract JSON object from LLM response."""
        response = response.strip()
        
        # Remove markdown code blocks if present
        if "```json" in response:
            start = response.index("```json") + 7
            end = response.index("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.index("```") + 3
            end = response.index("```", start)
            response = response[start:end].strip()
        
        # Find first { and last }
        start = response.index("{")
        end = response.rindex("}") + 1
        json_str = response[start:end]
        
        return json.loads(json_str)
