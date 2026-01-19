"""LLM-based reranker with bias controls."""

import json
import logging
from typing import Dict, List, Tuple

import numpy as np

from .base import LLMBase

logger = logging.getLogger(__name__)


class LLMReranker(LLMBase):
    """LLM-based listwise reranker with position bias mitigation."""
    
    RERANK_PROMPT_TEMPLATE = """You are an expert evidence reviewer for mental health research. Your task is to rank candidate sentences by how well they support a specific DSM-5 criterion.

**Post**: {post_text}

**DSM-5 Criterion**: {criterion_text}

**Candidate Sentences** (in random order):
{candidates}

**Task**: Rank these sentences from most relevant to least relevant for supporting the criterion. Return ONLY a JSON object with this exact format:
{{
  "ranking": [sentence_ids in order from best to worst],
  "rationale": "brief explanation (1-2 sentences)"
}}

Example output:
{{"ranking": [3, 1, 5, 2, 4], "rationale": "Sentence 3 directly describes depressed mood, sentence 1 provides temporal context..."}}

Your JSON response:"""
    
    def rerank(
        self,
        post_text: str,
        criterion_text: str,
        candidates: List[Dict],
        top_k: int = 5,
        check_position_bias: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        """Rerank candidates using LLM.
        
        Args:
            post_text: Full post text
            criterion_text: DSM-5 criterion description
            candidates: List of candidate dicts with 'sent_uid', 'sentence', 'score'
            top_k: Number of top candidates to return
            check_position_bias: Whether to check for position bias
            
        Returns:
            Tuple of (reranked_candidates, metadata)
        """
        # Limit input to top 10 to avoid context length issues
        input_candidates = candidates[:10]
        
        # Format candidates
        cand_text = "\n".join([
            f"[{i}] {c['sentence']}"
            for i, c in enumerate(input_candidates)
        ])
        
        # Create prompt
        prompt = self.RERANK_PROMPT_TEMPLATE.format(
            post_text=post_text[:1000],  # Truncate very long posts
            criterion_text=criterion_text,
            candidates=cand_text
        )
        
        # Generate ranking
        response = self.generate(prompt, use_cache=True)
        
        # Parse JSON response
        try:
            ranking_data = self._extract_json(response)
            ranking = ranking_data["ranking"]
            rationale = ranking_data.get("rationale", "")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: keep original order
            ranking = list(range(len(input_candidates)))
            rationale = f"Parse error: {str(e)}"
        
        # Reorder candidates
        reranked = []
        for idx in ranking[:top_k]:
            if 0 <= idx < len(input_candidates):
                reranked.append(input_candidates[idx])
        
        # Fill remaining if needed
        for i, cand in enumerate(input_candidates):
            if len(reranked) >= top_k:
                break
            if i not in ranking[:top_k]:
                reranked.append(cand)
        
        metadata = {
            "rationale": rationale,
            "original_order": [c["sent_uid"] for c in input_candidates],
            "reranked_order": [c["sent_uid"] for c in reranked],
        }
        
        # Check position bias if requested
        if check_position_bias and len(input_candidates) >= 5:
            bias_score = self._check_position_bias(
                post_text, criterion_text, input_candidates
            )
            metadata["position_bias_score"] = bias_score
        
        return reranked[:top_k], metadata
    
    def _extract_json(self, response: str) -> Dict:
        """Extract JSON object from LLM response."""
        # Try to find JSON block
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
    
    def _check_position_bias(
        self,
        post_text: str,
        criterion_text: str,
        candidates: List[Dict],
    ) -> float:
        """Check for position bias by comparing forward and reverse orders.
        
        Returns:
            Disagreement rate (0 = perfect agreement, 1 = complete disagreement)
        """
        # Forward ranking
        forward_prompt = self.RERANK_PROMPT_TEMPLATE.format(
            post_text=post_text[:1000],
            criterion_text=criterion_text,
            candidates="\n".join([f"[{i}] {c['sentence']}" for i, c in enumerate(candidates[:5])])
        )
        
        # Reverse ranking (swap positions)
        reverse_cands = list(reversed(candidates[:5]))
        reverse_prompt = self.RERANK_PROMPT_TEMPLATE.format(
            post_text=post_text[:1000],
            criterion_text=criterion_text,
            candidates="\n".join([f"[{i}] {c['sentence']}" for i, c in enumerate(reverse_cands)])
        )
        
        try:
            forward_resp = self.generate(forward_prompt, use_cache=True)
            reverse_resp = self.generate(reverse_prompt, use_cache=True)
            
            forward_rank = self._extract_json(forward_resp)["ranking"]
            reverse_rank = self._extract_json(reverse_resp)["ranking"]
            
            # Reverse the reverse ranking to compare
            reverse_rank_aligned = [4 - r for r in reverse_rank]
            
            # Compute disagreement
            disagreements = sum([
                1 for i, j in zip(forward_rank, reverse_rank_aligned)
                if i != j
            ])
            disagreement_rate = disagreements / len(forward_rank)
            
            return disagreement_rate
            
        except Exception as e:
            logger.warning(f"Position bias check failed: {e}")
            return -1.0  # Indicator of failed check
