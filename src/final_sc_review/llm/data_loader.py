"""Data loader for LLM evaluation with actual post and criterion texts."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class LLMEvaluationDataLoader:
    """Load and prepare data for LLM evaluation."""
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        per_query_csv: Path = None,
    ):
        """Initialize data loader.
        
        Args:
            data_dir: Root data directory
            per_query_csv: Path to per_query.csv from main evaluation
        """
        self.data_dir = Path(data_dir)
        
        # Load criteria
        criteria_file = self.data_dir / "DSM5" / "MDD_Criteira.json"
        with open(criteria_file) as f:
            criteria_data = json.load(f)
        
        self.criteria_map = {
            c["id"]: c["text"]
            for c in criteria_data["criteria"]
        }
        
        logger.info(f"Loaded {len(self.criteria_map)} DSM-5 criteria")
        
        # Load sentence corpus (post texts)
        corpus_file = self.data_dir / "groundtruth" / "sentence_corpus.jsonl"
        self.sentences = {}  # sent_uid -> text
        self.post_sentences = {}  # post_id -> list of sentences
        
        with open(corpus_file) as f:
            for line in f:
                sent_data = json.loads(line)  # Fixed: json.loads() not json.load()
                sent_uid = sent_data["sent_uid"]
                post_id = sent_data["post_id"]
                text = sent_data["text"]
                
                self.sentences[sent_uid] = text
                
                if post_id not in self.post_sentences:
                    self.post_sentences[post_id] = []
                self.post_sentences[post_id].append(text)
        
        logger.info(f"Loaded {len(self.sentences)} sentences from {len(self.post_sentences)} posts")
        
        # Load per-query data if provided
        if per_query_csv:
            self.per_query_df = pd.read_csv(per_query_csv)
            logger.info(f"Loaded {len(self.per_query_df)} queries from {per_query_csv}")
        else:
            self.per_query_df = None
    
    def get_post_text(self, post_id: str) -> str:
        """Get full post text.
        
        Args:
            post_id: Post ID
            
        Returns:
            Full post text (all sentences concatenated)
        """
        sentences = self.post_sentences.get(post_id, [])
        return " ".join(sentences)
    
    def get_criterion_text(self, criterion_id: str) -> str:
        """Get criterion text.
        
        Args:
            criterion_id: Criterion ID (e.g., "A.1")
            
        Returns:
            Criterion description
        """
        return self.criteria_map.get(criterion_id, f"Criterion {criterion_id}")
    
    def get_query_data(self, post_id: str, criterion_id: str) -> Dict:
        """Get full query data.
        
        Args:
            post_id: Post ID
            criterion_id: Criterion ID
            
        Returns:
            Dict with post_text, criterion_text, post_id, criterion_id
        """
        return {
            "post_id": post_id,
            "criterion_id": criterion_id,
            "post_text": self.get_post_text(post_id),
            "criterion_text": self.get_criterion_text(criterion_id),
        }
    
    def get_stratified_sample(
        self,
        max_per_state: int = 100,
        states: List[str] = ['NEG', 'UNCERTAIN', 'POS'],
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Get stratified sample for evaluation.
        
        Args:
            max_per_state: Max samples per state
            states: States to sample from
            random_state: Random seed
            
        Returns:
            Stratified DataFrame
        """
        if self.per_query_df is None:
            raise ValueError("per_query_csv not provided during initialization")
        
        samples = []
        for state in states:
            state_df = self.per_query_df[self.per_query_df['state'] == state]
            n = min(max_per_state, len(state_df))
            if n > 0:
                state_sample = state_df.sample(n=n, random_state=random_state)
                samples.append(state_sample)
                logger.info(f"  {state}: {n} samples")
        
        result = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        logger.info(f"Total stratified samples: {len(result)}")
        
        return result
    
    def get_evidence_sentences(
        self,
        post_id: str,
        top_k: int = 5,
    ) -> List[str]:
        """Get top-K evidence sentences for a post.
        
        Args:
            post_id: Post ID
            top_k: Number of sentences to return
            
        Returns:
            List of sentence texts
        """
        sentences = self.post_sentences.get(post_id, [])
        return sentences[:top_k]
