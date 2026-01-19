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

## Provenance

- **Source data:** `outputs/final_research_eval/20260118_031312_complete/per_query.csv`
- **HPO results:** `outputs/hpo_inference_combos/full_results.csv`
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
