# Metric Contract

## Overview

This document defines the evaluation protocols and metric definitions used in this research.

## Dual Protocol System

### Protocol 1: `positives_only`

**Used for:** Ranking metrics
**Scope:** Only queries where `has_evidence=1`
**Rationale:** Ranking quality only meaningful when there is something to rank

**Metrics:**
- nDCG@K
- Evidence Recall@K
- MRR (Mean Reciprocal Rank)
- MAP@K (Mean Average Precision)

### Protocol 2: `all_queries`

**Used for:** Classification metrics
**Scope:** All queries (both `has_evidence=0` and `has_evidence=1`)
**Rationale:** Binary classification of evidence presence/absence

**Metrics:**
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- Sensitivity/Specificity at thresholds

## Metric Definitions

### nDCG@K (Normalized Discounted Cumulative Gain)

```
DCG@K = sum_{i=1}^{K} rel_i / log2(i+1)
nDCG@K = DCG@K / IDCG@K
```

Where:
- `rel_i` = 1 if position i contains gold evidence, 0 otherwise
- `IDCG@K` = ideal DCG with all positives ranked first

**Interpretation:** Quality of ranking (1.0 = perfect)

### Evidence Recall@K

```
Recall@K = |retrieved_gold ∩ top_K| / |retrieved_gold|
```

**Interpretation:** Fraction of gold evidence in top-K

### MRR (Mean Reciprocal Rank)

```
MRR = mean(1 / rank_of_first_relevant)
```

**Interpretation:** Average position of first relevant result

### AUROC (Area Under ROC Curve)

```
AUROC = P(score(positive) > score(negative))
```

**Interpretation:** Probability of ranking positive higher than negative

### AUPRC (Area Under Precision-Recall Curve)

```
AUPRC = integral(Precision(Recall))
```

**Interpretation:** Trade-off between precision and recall

## Evaluation Splits

| Split | Purpose | Posts | Queries |
|-------|---------|-------|---------|
| TRAIN | Model training | 80% | ~11,800 |
| VAL (DEV) | HPO, threshold tuning | 10% | ~1,470 |
| TEST | Final evaluation | 10% | ~1,500 |

**Critical invariant:** Post-ID disjoint (no post appears in multiple splits)

## Criteria

| Criterion | Description | Type | Training |
|-----------|-------------|------|----------|
| A.1-A.9 | DSM-5 MDD Criteria | Standard | Included |
| A.10 | SPECIAL_CASE (expert discrimination) | Non-DSM-5 | **Excluded by default** |

### A.10 Exclusion

A.10 is excluded from GNN training by default because:
- Not a standard DSM-5 criterion
- Low positive rate (5.8%) and poor AUROC (0.665)
- Ablation study showed +0.28% nDCG@10 improvement on A.1-A.9 when excluded

**Total queries:** 14,770 (all criteria) or 13,293 (DSM-5 only)

## Statistical Confidence

### Bootstrap Confidence Intervals

For key metrics, 95% CIs computed via bootstrap (n=10,000):

```python
from scipy.stats import bootstrap
result = bootstrap((y_true, y_score), metric_func, n_resamples=10000)
ci = result.confidence_interval
```

### Reported Results (5-Fold Cross-Validation)

| Model | nDCG@10 | MRR | Recall@10 |
|-------|---------|-----|-----------|
| Baseline (Jina-v3) | 0.7330 ± 0.031 | 0.6746 ± 0.037 | 0.9444 ± 0.022 |
| + SAGE+Residual GNN | 0.8206 ± 0.030 | 0.7703 ± 0.035 | 0.9606 ± 0.019 |
| **Improvement** | +10.48% | +12.02% | +1.71% |

Source: `outputs/comprehensive_ablation/`

Standard deviations are across 5 folds (post-ID disjoint).

### Extended Metrics (K = 1, 3, 5, 10, 20)

| Metric | Baseline | GNN (HPO) | Improvement |
|--------|----------|-----------|-------------|
| nDCG@1 | 0.5540 ± 0.063 | 0.6497 ± 0.052 | +17.27% |
| nDCG@3 | 0.6660 ± 0.042 | 0.7532 ± 0.039 | +13.09% |
| nDCG@5 | 0.7086 ± 0.037 | 0.7832 ± 0.038 | +10.53% |
| nDCG@10 | 0.7330 ± 0.031 | 0.8206 ± 0.030 | +10.48% |
| nDCG@20 | 0.7566 ± 0.032 | 0.8223 ± 0.026 | +8.68% |
| Precision@1 | 0.5540 ± 0.063 | 0.6605 ± 0.047 | +19.22% |
| Precision@3 | 0.2709 ± 0.008 | 0.3018 ± 0.008 | +11.41% |
| Precision@5 | 0.1859 ± 0.003 | 0.2002 ± 0.005 | +7.69% |
| Precision@10 | 0.1048 ± 0.002 | 0.1079 ± 0.001 | +2.96% |
| Hit@1 | 0.5540 ± 0.063 | 0.6605 ± 0.047 | +19.22% |
| Hit@3 | 0.7691 ± 0.030 | 0.8483 ± 0.027 | +10.30% |
| Hit@5 | 0.8631 ± 0.020 | 0.9187 ± 0.031 | +6.44% |
| Hit@10 | 0.9534 ± 0.017 | 0.9762 ± 0.012 | +2.39% |
| MAP@1 | 0.5540 ± 0.063 | 0.6605 ± 0.047 | +19.22% |
| MAP@3 | 0.6322 ± 0.047 | 0.7308 ± 0.037 | +15.60% |
| MAP@5 | 0.6574 ± 0.044 | 0.7508 ± 0.038 | +14.21% |
| MAP@10 | 0.6732 ± 0.042 | 0.7612 ± 0.035 | +13.07% |

## Implementation

### Ranking Metrics
```python
# src/final_sc_review/metrics/ranking.py
def compute_ndcg_at_k(y_true, y_score, k=10):
    ...
```

### Classification Metrics
```python
from sklearn.metrics import roc_auc_score, average_precision_score
auroc = roc_auc_score(y_true, y_score)
auprc = average_precision_score(y_true, y_score)
```

## Validation

Tests ensure metric implementations are correct:
```bash
pytest tests/metrics/test_ranking_metrics.py -v
```

## Source of Truth

All reported metrics come from:
```
results/paper_bundle/v3.0/metrics_master.json
```

Any discrepancy should be reported as a bug.
