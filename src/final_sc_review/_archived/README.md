# Archived Research Modules

This directory contains archived research code from the evidence binding pipeline optimization process.
These modules were used during hyperparameter optimization and model selection but are no longer
needed for production inference.

## Why Archived (Not Deleted)

These files represent completed research work and may be useful for:
- Reproducing HPO experiments
- Understanding research decisions
- Reference implementations

## Contents

### baselines/
Baseline retrieval methods used for HPO comparisons. NV-Embed-v2 was selected as the best retriever.
- `base.py` - BM25, TF-IDF, random, E5, Contriever, BGE baselines

### reranker/
Research components for reranker training/optimization. Jina-Reranker-v3 was selected as best.
- `losses.py` - Custom loss functions (ListMLE, ListNet, etc.) for reranker training
- `maxout_trainer.py` - Training script for fine-tuning rerankers

### hpo/
Hyperparameter optimization objectives. Best parameters are now fixed in configs/default.yaml.
- `objective_training_v2.py` - Training objective for Optuna HPO
- `objective_inference.py` - Inference objective for Optuna HPO

### llm/
Experimental LLM components not used in production.
- `gemini_client.py` - Google Gemini API client (experimental)
- `base.py` - Base LLM abstractions

## Production Code

For production, use the modules in the parent directory:
- `retriever/` - NV-Embed-v2 retriever (best from 25 candidates)
- `reranker/` - Jina-Reranker-v3 (best from 15 candidates)
- `gnn/` - Graph neural network modules (P1-P4)
- `llm/` - Production LLM modules (LLMVerifier, SuicidalIdeationClassifier, HybridNEDetector)

## Date Archived

2026-01-30 - Comprehensive project optimization
