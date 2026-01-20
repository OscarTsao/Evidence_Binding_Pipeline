# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This is a sentence-criterion (S-C) evidence retrieval pipeline for mental health research. Given a post and a DSM-5 criterion, the system retrieves evidence sentences supporting the criterion.

### Best Model Configuration (HPO-Optimized)
- **Retriever:** NV-Embed-v2 (nvidia/NV-Embed-v2) - Best from 25 retrievers
- **Reranker:** Jina-Reranker-v3 (jinaai/jina-reranker-v3) - Best from 15 rerankers
- **Performance:** nDCG@10 = 0.8658 (from 324 model combinations tested)

## Conda Environment Requirements

**IMPORTANT:** This project uses two conda environments due to dependency conflicts:

| Environment | Purpose | Models |
|-------------|---------|--------|
| `nv-embed-v2` | Retriever encoding | NV-Embed-v2 (requires specific transformers version) |
| `llmhe` | Everything else | Jina-Reranker-v3, GNN models, LLM experiments |

### When to Use Each Environment

```bash
# nv-embed-v2: ONLY for encoding sentences with NV-Embed-v2 retriever
conda run -n nv-embed-v2 python scripts/encode_corpus.py

# llmhe: For all other operations
conda run -n llmhe python scripts/eval_zoo_pipeline.py
conda run -n llmhe python scripts/gnn/run_p3_integration.py
conda run -n llmhe python scripts/llm/run_llm_eval.py
```

### Why Two Environments?

- `nv-embed-v2`: Has older transformers that works with NV-Embed-v2 model's custom code
- `llmhe`: Has newer transformers (4.57+) required for Jina-Reranker-v3 (qwen3 architecture)

The NV-Embed-v2 embeddings are cached (`data/cache/retriever_zoo/nv-embed-v2/`), so most operations only need `llmhe`.

## Commands

### Setup
```bash
# Install in development mode (use llmhe environment)
conda activate llmhe
pip install -e .
```

### Build Data Artifacts
```bash
# Build groundtruth labels
python scripts/build_groundtruth.py --data_dir data --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus
python scripts/build_sentence_corpus.py --data_dir data --output data/groundtruth/sentence_corpus.jsonl
```

### Evaluation
```bash
# Evaluate with best HPO model combo
python scripts/eval_zoo_pipeline.py --config configs/default.yaml --split test

# Single query inference
python scripts/run_single_zoo.py --config configs/default.yaml --post_id <POST_ID> --criterion_id <CRITERION_ID>
```

### Testing
```bash
pytest -q                         # Run all tests
pytest tests/test_metrics.py -v   # Run specific test file
pytest -k "test_name"             # Run tests matching pattern
```

### Verification
```bash
# Verify paper bundle checksums
python scripts/verification/verify_checksums.py

# Audit splits for leakage
python scripts/audit_splits.py --data_dir data --seed 42 --k 5
```

## Architecture

### Zoo Pipeline (`src/final_sc_review/pipeline/zoo_pipeline.py`)
The recommended pipeline using retriever and reranker zoos:
1. **Stage 1 - Retrieval:** NV-Embed-v2 (top-24)
2. **Stage 2 - Reranking:** Jina-Reranker-v3 (top-10)

### Key Modules
- `src/final_sc_review/retriever/zoo.py` - Retriever zoo with 25+ models
- `src/final_sc_review/reranker/zoo.py` - Reranker zoo with 15+ models
- `src/final_sc_review/gnn/` - GNN modules (P1-P4)
- `src/final_sc_review/metrics/ranking.py` - Ranking metrics

### GNN Modules (`src/final_sc_review/gnn/`)

| Module | Status | Description | Key Metric |
|--------|--------|-------------|------------|
| P1 | Deprecated | NE Gate (no-evidence detection) | AUROC=0.577 |
| P2 | Production | Dynamic-K selection | Adaptive cutoff |
| P3 | Production | Graph Reranker | nDCG@10 +8.6% |
| P4 | Production | Criterion-Aware GNN | AUROC=0.8972 |

**P3 Graph Reranker** uses sentence graph structure to refine reranker scores:
- Checkpoints: `outputs/gnn_research/p3_retrained/20260120_190745/`
- Graph cache: `data/cache/gnn/rebuild_20260120/`
- Training: `scripts/gnn/train_p3_graph_reranker.py`
- Evaluation: `scripts/gnn/run_p3_integration.py`

### Deprecated Modules
- `src/final_sc_review/pipeline/three_stage.py` - Use `zoo_pipeline.py` instead
- `src/final_sc_review/hpo/objective_training.py` - Use `objective_training_v2.py` instead

## Key Invariants

1. **Post-ID Disjoint Splits:** No post appears in multiple splits
2. **Within-Post Retrieval:** Candidate pool is sentences from the same post
3. **Dual Protocol Metrics:**
   - `positives_only`: Ranking metrics (nDCG, Recall, MRR)
   - `all_queries`: Classification metrics (AUROC, AUPRC)
4. **No Gold Features:** Ground truth never used during inference

## Configuration

YAML configs control all pipeline parameters. Key settings in `configs/default.yaml`:

```yaml
models:
  retriever_name: nv-embed-v2
  reranker_name: jina-reranker-v3

retriever:
  top_k_retriever: 24
  top_k_final: 10
  fusion_method: rrf
  rrf_k: 60

split:
  seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

## Metrics Source of Truth

The single source of truth for all metrics is:
`results/paper_bundle/v2.0/metrics_master.json`

Primary metrics:
- AUROC: 0.8972 (all_queries protocol)
- Evidence Recall@K: 0.7043 (positives_only protocol)
- MRR: 0.3801 (positives_only protocol)
- nDCG@10: 0.8658 (positives_only protocol)

## Troubleshooting

### Common Issues

**"model type 'qwen3' not recognized"**
- Use `llmhe` environment (has transformers 4.57+)
- Don't use `nv-embed-v2` env for Jina-Reranker-v3

**"Unknown retriever: nvidia/NV-Embed-v2"**
- Use short name: `nv-embed-v2` (not full HuggingFace path)

**P3 checkpoint dimension mismatch**
- P3 must be trained with same embeddings used in graph cache
- Retrain P3 if changing retriever: `scripts/gnn/train_p3_graph_reranker.py`

**Tests fail on import**
- Ensure `from __future__` imports are first in file
- Run `pip install -e .` to install package

## Repository Structure

```
src/final_sc_review/
├── data/           # Data I/O and schemas
├── retriever/      # Retriever zoo (25+ models)
├── reranker/       # Reranker zoo (15+ models)
├── pipeline/       # ZooPipeline (recommended)
├── gnn/            # GNN models (P1-P4)
├── metrics/        # Ranking and classification metrics
├── postprocessing/ # Calibration and dynamic-K
├── hpo/            # Hyperparameter optimization
├── llm/            # LLM integration
├── clinical/       # Clinical gate logic
└── utils/          # Shared utilities

scripts/
├── gnn/            # GNN training and evaluation
├── verification/   # Metric verification and audits
├── llm/            # LLM experiments
└── ablation/       # Ablation studies
```
