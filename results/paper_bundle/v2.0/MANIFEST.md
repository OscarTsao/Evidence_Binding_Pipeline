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
| P3 Graph Reranker | Validated | MRR +15.4% | Available, not default |
| P2 Dynamic-K | Production | - | Adaptive K selection |
| P1 NE Gate | Deprecated | AUROC 0.577 | Replaced by P4 |

### P3 Graph Reranker Details

P3 uses graph structure to refine reranker scores, showing significant improvements:

| Metric | Original | Refined | Improvement |
|--------|----------|---------|-------------|
| MRR | 0.4159 | 0.5702 | +15.4% |
| nDCG@10 | 0.2996 | 0.3854 | +8.6% |
| Recall@10 | 0.6545 | 0.8072 | +15.3% |

**Note:** P3 is validated but requires graph cache reconstruction for integration.

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
- **P3 checkpoints:** `outputs/gnn_research/20260117_p3_final/20260117_030023/p3_graph_reranker/`
- **Evaluation:** 5-fold cross-validation on 14,770 queries

## Regeneration

To regenerate this bundle from source:

```bash
# Recompute metrics
python scripts/verification/recompute_metrics_from_csv.py

# Regenerate checksums
cd results/paper_bundle/v2.0
sha256sum metrics_master.json summary.json MANIFEST.md tables/*.csv > checksums.txt
```
