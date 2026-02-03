"""A.9 Suicidal Ideation LLM classifier.

This classifier handles DSM-5 Criterion A.9 (Suicidal Ideation), NOT A.10.
A.10 is SPECIAL_CASE (expert discrimination cases per ReDSM5 taxonomy).
"""

import json
import logging
from typing import Dict, List

import numpy as np

from .base import LLMBase

logger = logging.getLogger(__name__)


class SuicidalIdeationClassifier(LLMBase):
    """Specialized LLM classifier for A.9 (Suicidal Ideation).

    DSM-5 Criterion A.9: Recurrent thoughts of death (not just fear of dying),
    recurrent suicidal ideation without a specific plan, or a suicide attempt
    or a specific plan for committing suicide.

    Note: This was previously misnamed A10Classifier. A.10 is SPECIAL_CASE
    (expert discrimination cases), not suicidal ideation.
    """

    PROMPT_TEMPLATE = """You are an expert clinical psychologist specializing in suicide risk assessment. Your task is to detect evidence of suicidal ideation in social media posts.

**Post**: {post_text}

**DSM-5 Criterion A.9 (Suicidal Ideation)**: Recurrent thoughts of death (not just fear of dying), recurrent suicidal ideation without a specific plan, or a suicide attempt or a specific plan for committing suicide.

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
        """Classify A.9 suicidal ideation.

        Args:
            post_text: Full post text
            self_consistency_runs: Number of runs for self-consistency

        Returns:
            Dict with has_suicidal_ideation, severity, confidence, evidence_sentences, rationale
        """
        # Create prompt
        prompt = self.PROMPT_TEMPLATE.format(post_text=post_text)

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
                logger.warning(f"Failed to parse A.9 classifier response: {e}")

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

    # Note: _extract_json is now inherited from LLMBase which uses
    # the shared extract_json_from_response utility


# Backward compatibility alias (deprecated)
A10Classifier = SuicidalIdeationClassifier
