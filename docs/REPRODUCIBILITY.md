# Reproducibility Guide

## Overview

This document provides complete instructions for reproducing all results in the paper.

**Paper Bundle Version:** v3.0
**Last Updated:** 2026-01-21

## Quick Verification (No Data Required)

```bash
# 1. Run all tests
pytest -q
# Expected: 232+ tests pass

# 2. Verify paper bundle integrity
python scripts/verification/verify_checksums.py
# Expected: All checksums match

# 3. Recompute metrics from predictions
python scripts/verification/recompute_metrics_from_csv.py
# Expected: Matches metrics_master.json
```

## Full Reproduction (Requires Data Access)

### Prerequisites

1. **Data Access**: See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)
2. **Hardware**: GPU with 24GB+ VRAM recommended
3. **Environments**: Set up both conda environments (see below)

### Environment Setup

```bash
# 1. Retriever environment (NV-Embed-v2)
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt

# 2. Main environment (reranking, GNN, evaluation)
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .
```

### Step 1: Build Data Artifacts

```bash
conda activate llmhe

# Build groundtruth labels
python scripts/build_groundtruth.py \
    --data_dir data \
    --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus
python scripts/build_sentence_corpus.py \
    --data_dir data \
    --output data/groundtruth/sentence_corpus.jsonl
```

### Step 2: Encode Corpus (Retriever Environment)

```bash
conda activate nv-embed-v2

# Pre-compute embeddings
python scripts/encode_corpus.py \
    --config configs/default.yaml \
    --retriever nv-embed-v2
```

### Step 3: Run Evaluation

```bash
conda activate llmhe

# Full pipeline evaluation
python scripts/eval_zoo_pipeline.py \
    --config configs/default.yaml \
    --split test \
    --output outputs/reproduction/eval_results.json
```

### Step 4: Verify Results

```bash
# Compare with expected metrics
python scripts/verification/metric_crosscheck.py \
    --computed outputs/reproduction/eval_results.json \
    --expected results/paper_bundle/v3.0/metrics_master.json
```

## Expected Results (5-Fold Cross-Validation)

| Model | nDCG@10 | MRR | Recall@10 |
|-------|---------|-----|-----------|
| Baseline (NV-Embed-v2 + Jina-v3) | 0.7428 ± 0.033 | 0.6862 ± 0.042 | 0.9485 ± 0.021 |
| + SAGE+Residual GNN | 0.8206 ± 0.030 | 0.7703 ± 0.035 | - |
| **GNN Improvement** | **+10.48%** | **+12.25%** | - |

Source: `outputs/comprehensive_ablation/`

### Quick Verification

```bash
# Run 5-fold evaluation from cached graphs
python scripts/experiments/evaluate_from_cache.py \
    --graph_dir data/cache/gnn/rebuild_20260120 \
    --output_dir outputs/unified_evaluation
```

Small variations may occur due to:
- Random seed differences in GNN training
- GPU architecture differences
- Library version differences

## GNN Training (Optional)

### Rebuild Graph Cache

```bash
conda activate llmhe

# Build graph cache (excludes A.10 by default)
python scripts/gnn/rebuild_graph_cache.py \
    --output_dir data/cache/gnn/rebuild_$(date +%Y%m%d)

# To include A.10 (not recommended):
python scripts/gnn/rebuild_graph_cache.py --include_a10
```

### Train P3 Graph Reranker

```bash
# Train P3 GNN (excludes A.10 by default)
python scripts/gnn/train_p3_graph_reranker.py \
    --graph_dir data/cache/gnn/rebuild_20260120 \
    --output_dir outputs/gnn_research/p3_retrained

# To include A.10 (not recommended):
python scripts/gnn/train_p3_graph_reranker.py --include_a10
```

### A.10 Exclusion

A.10 (SPECIAL_CASE) is excluded from GNN training by default because:
- It's not a standard DSM-5 criterion (expert discrimination cases)
- Low positive rate (5.8%) and poor AUROC (0.67)
- Removing it focuses training on the 9 standard DSM-5 criteria

**Current 5-fold CV results (A.10 excluded, SAGE+Residual architecture):**
- Baseline nDCG@10: 0.7428 ± 0.033
- SAGE+Residual nDCG@10: 0.8206 ± 0.030 (+10.48%)

See `outputs/comprehensive_ablation/` for full results.

## Ablation Studies

### A.10 Training Impact

```bash
python scripts/experiments/run_a10_ablation_experiment.py \
    --graph_dir data/cache/gnn/rebuild_20260120 \
    --output_dir outputs/a10_ablation_experiment
```

## Multi-Seed Robustness

```bash
python scripts/robustness/run_multi_seed_eval.py \
    --seeds 42,123,456,789,1337 \
    --output outputs/robustness/multi_seed.json
```

## Checksum Verification

All paper bundle files have SHA256 checksums:

```bash
cd results/paper_bundle/v3.0
sha256sum -c checksums.txt
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use smaller retriever (e.g., bge-large-en-v1.5)

### Metric Mismatch
- Verify data artifacts are correctly built
- Check environment versions match requirements

### Import Errors
- Ensure correct conda environment is active
- Reinstall: `pip install -e .`

## Source of Truth

The single source of truth for all metrics is:
```
results/paper_bundle/v3.0/metrics_master.json
```

All other metric reports should match this file.

## Available Baselines

The following baseline retrievers are available for comparison:

| Baseline | Type | Command |
|----------|------|---------|
| BM25 | Lexical | `get_baseline("bm25")` |
| TF-IDF | Lexical | `get_baseline("tfidf")` |
| E5-base | Dense | `get_baseline("e5-base")` |
| BGE | Dense | `get_baseline("bge")` |
| Contriever | Dense | `get_baseline("contriever")` |
| Cross-encoder | Reranker | `get_baseline("cross-encoder")` |
| Linear | TF-IDF features | `get_baseline("linear")` |
| LLM-embed | LLM embeddings | `get_baseline("llm-embed")` |

```bash
python scripts/baselines/run_baseline_comparison.py \
    --output outputs/baselines/ \
    --baselines bm25 tfidf e5-base bge contriever
```

## CITATION.cff

This repository includes a CITATION.cff file for proper academic citation.
To verify it's valid:

```bash
pytest tests/test_citation_cff.py -v
```
