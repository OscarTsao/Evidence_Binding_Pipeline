# Paper Bundle v3.0 Manifest

Generated: 2026-01-20T21:58:46.613173

## Contents

| File | Description |
|------|-------------|
| metrics_master.json | Single source of truth for all metrics |
| summary.json | Bundle metadata and high-level summary |
| tables/main_results.csv | Primary metrics with 95% CIs |
| tables/per_criterion.csv | Per-criterion performance breakdown |
| tables/baselines.csv | Baseline comparison (if available) |
| tables/robustness.csv | Robustness analysis (if available) |
| checksums.txt | SHA256 integrity verification |
| MANIFEST.md | This file |

## Regeneration

To regenerate this bundle:

```bash
python scripts/reporting/build_paper_bundle.py \
    --version v3.0 \
    --source_run outputs/final_research_eval/20260118_031312_complete \
    --output results/paper_bundle/v3.0
```

## Verification

To verify bundle integrity:

```bash
python scripts/verification/verify_checksums.py \
    --bundle results/paper_bundle/v3.0
```

To cross-check metrics:

```bash
python scripts/verification/metric_crosscheck.py \
    --bundle results/paper_bundle/v3.0
```

## Source Data

| Source | Path |
|--------|------|
| Per-query results | outputs/final_research_eval/20260118_031312_complete/per_query.csv |
| Criteria registry | configs/criteria_registry.yaml |
| Metric contract | docs/METRIC_CONTRACT.md |

## Protocols

- **positives_only**: Ranking metrics (nDCG, Recall, MRR) computed on queries with evidence
- **all_queries**: Classification metrics (AUROC, AUPRC) computed on all queries

## Notes

- Evaluation: 5-fold cross-validation, positives_only protocol
- Criteria: A.1-A.9 (A.10 excluded)

## 5-Fold Cross-Validation Results

| Model | nDCG@10 | MRR | Recall@10 |
|-------|---------|-----|-----------|
| Baseline (NV-Embed-v2 + Jina-v3) | 0.7428 ± 0.033 | 0.6862 ± 0.042 | 0.9485 ± 0.021 |
| + P3 GNN (SAGE+Residual) | 0.8206 ± 0.030 | 0.7703 ± 0.035 | - |
| **Improvement** | **+10.48%** | **+12.25%** | - |

Source: `outputs/comprehensive_ablation/`

## Criteria

| Criterion | Description | Type | In Training |
|-----------|-------------|------|-------------|
| A.1-A.9 | DSM-5 MDD Criteria | Standard | Yes |
| A.10 | SPECIAL_CASE (expert discrimination) | Non-DSM-5 | **No (excluded)** |

### A.10 Exclusion

A.10 is excluded from training because:
- Not a standard DSM-5 criterion
- Low positive rate (5.8%) and poor AUROC (0.67)
- Focuses training on the 9 standard criteria

## Pipeline Configuration

| Parameter | Value |
|-----------|-------|
| Retriever | NV-Embed-v2 |
| Reranker | Jina-Reranker-v3 |
| top_k_retriever | 24 |
| top_k_rerank | 20 |
| top_k_final | 10 |
| fusion_method | rrf |
| rrf_k | 60 |
