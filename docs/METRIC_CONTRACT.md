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

## Statistical Confidence

### Bootstrap Confidence Intervals

For key metrics, 95% CIs computed via bootstrap (n=10,000):

```python
from scipy.stats import bootstrap
result = bootstrap((y_true, y_score), metric_func, n_resamples=10000)
ci = result.confidence_interval
```

### Reported CIs

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUROC | 0.8972 | [0.8941, 0.9003] |
| Evidence Recall@K | 0.7043 | [0.6921, 0.7165] |

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
results/paper_bundle/v2.0/metrics_master.json
```

Any discrepancy should be reported as a bug.
