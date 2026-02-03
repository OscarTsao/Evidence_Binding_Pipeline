"""Project-wide constants."""

CRITERION_TO_SYMPTOM = {
    "A.1": "DEPRESSED_MOOD",
    "A.2": "ANHEDONIA",
    "A.3": "APPETITE_CHANGE",
    "A.4": "SLEEP_ISSUES",
    "A.5": "PSYCHOMOTOR",
    "A.6": "FATIGUE",
    "A.7": "WORTHLESSNESS",
    "A.8": "COGNITIVE_ISSUES",
    "A.9": "SUICIDAL_THOUGHTS",
    "A.10": "SPECIAL_CASE",
}

# DSM-5 criteria (A.1-A.9) - used for training by default
# A.10 (SPECIAL_CASE) is excluded because:
# 1. It's not a standard DSM-5 criterion
# 2. It has low positive rate (5.8%) and poor performance (AUROC=0.665)
# 3. Ablation study showed removing A.10 improves nDCG@10 by +0.28%
DSM5_CRITERIA = ["A.1", "A.2", "A.3", "A.4", "A.5", "A.6", "A.7", "A.8", "A.9"]
EXCLUDED_CRITERIA = ["A.10"]
ALL_CRITERIA = DSM5_CRITERIA + EXCLUDED_CRITERIA

DEFAULT_SEED = 42

# Best HPO parameters for nv-embed-v2 + jina-reranker-v3
BEST_HPO_CONFIG = {
    "retriever_name": "nv-embed-v2",
    "reranker_name": "jina-reranker-v3",
    "top_k_retriever": 24,
    "top_k_final": 10,
    "fusion_method": "rrf",
    "rrf_k": 60,
    "reranker_max_length": 1024,
    "reranker_batch_size": 128,
    "reranker_use_listwise": True,
    "ndcg10": 0.8658,
}
