# Paper Bundle v2.0 - Manifest

## Overview

This bundle contains all artifacts required to verify the research results.

**Version:** v2.0
**Created:** 2026-01-20
**Status:** Production-ready

## Contents

### Core Files

| File | Description |
|------|-------------|
| `metrics_master.json` | Single source of truth for all metrics |
| `summary.json` | Bundle metadata |
| `MANIFEST.md` | This file |
| `checksums.txt` | SHA256 integrity verification |

### Tables

| File | Description |
|------|-------------|
| `tables/main_results.csv` | Primary metrics with CIs |
| `tables/per_criterion.csv` | Per-criterion AUROC breakdown |
| `tables/ablation.csv` | Top-10 model combinations |

### LLM Experiments

| File | Description |
|------|-------------|
| `llm_experiment_results.json` | Full LLM experiment results and analysis |

## Verification

```bash
# Verify all checksums
cd results/paper_bundle/v2.0
sha256sum -c checksums.txt

# Or use verification script
python scripts/verification/verify_checksums.py
```

## Metric Summary

| Metric | Value | Protocol |
|--------|-------|----------|
| AUROC | 0.8972 | all_queries |
| Evidence Recall@K | 0.7043 | positives_only |
| MRR | 0.3801 | positives_only |
| nDCG@10 | 0.8658 | positives_only |

## GNN Research Summary

| Model | Status | Key Metric | Improvement |
|-------|--------|------------|-------------|
| P4 Criterion-Aware GNN | Production | AUROC 0.8972 | Primary classifier |
| P3 Graph Reranker | Production | nDCG@10 +8.6% | Score refinement |
| P2 Dynamic-K | Production | - | Adaptive K selection |
| P1 NE Gate | Deprecated | AUROC 0.577 | Replaced by P4 |

### P3 Graph Reranker Details (Updated 2026-01-20)

P3 retrained with NV-Embed-v2 embeddings + Jina-Reranker-v3 scores:

| Metric | Baseline | With P3 | Improvement |
|--------|----------|---------|-------------|
| MRR | 0.6746 | 0.7485 | +10.9% |
| nDCG@5 | 0.6990 | 0.7716 | +10.4% |
| nDCG@10 | 0.7330 | 0.7959 | +8.6% |
| Recall@5 | 0.8439 | 0.8903 | +5.5% |
| Recall@10 | 0.9444 | 0.9619 | +1.9% |

**Checkpoints:** `outputs/gnn_research/p3_retrained/20260120_190745/`
**Graph cache:** `data/cache/gnn/rebuild_20260120/`

## LLM Experiment Summary (2026-01-20)

Model: Qwen/Qwen2.5-7B-Instruct (4-bit quantization)

| Component | Metric | Value | Notes |
|-----------|--------|-------|-------|
| LLM Verifier | AUROC | 0.8931 | Near-baseline, 87% accuracy |
| A.10 Classifier | AUROC | 0.5603 | Below baseline (0.66) |
| LLM Reranker | Position Bias | 0.507 | Needs mitigation |

## Provenance

- **Source data:** `outputs/final_research_eval/20260118_031312_complete/per_query.csv`
- **HPO results:** `outputs/hpo_inference_combos/full_results.csv`
- **GNN research:** `outputs/gnn_research/`
- **P3 checkpoints:** `outputs/gnn_research/p3_retrained/20260120_190745/`
- **P3 integration results:** `outputs/p3_integration/20260120_191158/`
- **Graph cache:** `data/cache/gnn/rebuild_20260120/`
- **Evaluation:** 5-fold cross-validation on 14,770 queries (1,379 positive)

## Regeneration

To regenerate this bundle from source:

```bash
# Recompute metrics
python scripts/verification/recompute_metrics_from_csv.py

# Regenerate checksums
cd results/paper_bundle/v2.0
sha256sum metrics_master.json summary.json MANIFEST.md tables/*.csv > checksums.txt
```
