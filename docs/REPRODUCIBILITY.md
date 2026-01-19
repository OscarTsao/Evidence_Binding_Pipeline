# Reproducibility Guide

## Overview

This document provides complete instructions for reproducing all results in the paper.

## Quick Verification (No Data Required)

```bash
# 1. Run all tests
pytest -q
# Expected: 197+ tests pass

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
    --expected results/paper_bundle/v2.0/metrics_master.json
```

## Expected Results

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| AUROC | 0.8972 | ±0.001 |
| Evidence Recall@K | 0.7043 | ±0.005 |
| MRR | 0.3801 | ±0.005 |
| nDCG@10 | 0.8658 | ±0.005 |

Small variations may occur due to:
- GPU architecture differences
- Library version differences
- Floating point precision

## Ablation Studies

### Retriever Comparison

```bash
python scripts/ablation/run_ablation_study.py \
    --study retriever_comparison \
    --output outputs/ablation/study_1.json
```

### Reranker Ablation

```bash
python scripts/ablation/run_ablation_study.py \
    --study reranker_ablation \
    --retriever nv-embed-v2 \
    --output outputs/ablation/study_2.json
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
cd results/paper_bundle/v2.0
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
results/paper_bundle/v2.0/metrics_master.json
```

All other metric reports should match this file.
