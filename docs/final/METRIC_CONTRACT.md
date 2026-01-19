# Metric Contract

This document defines the exact mathematical specification for all metrics used in the paper.
**All implementations MUST match these definitions.**

---

## 1. Ranking Metrics (Retrieval Evaluation)

### 1.1 Notation

| Symbol | Definition |
|--------|------------|
| Q | Set of queries (post_id, criterion_id pairs) |
| G(q) | Set of gold (relevant) sentence UIDs for query q |
| R(q) | Ranked list of sentence UIDs for query q |
| R(q)_k | Top-k items from R(q) |
| K_eff | min(K, |R(q)|) - effective K for short posts |

### 1.2 Recall@K

**Definition:**
```
Recall@K(q) = |R(q)_K ∩ G(q)| / |G(q)|       if |G(q)| > 0
           = 0                                if |G(q)| = 0
```

**Implementation:** `src/final_sc_review/metrics/ranking.py:recall_at_k()`

**Range:** [0, 1]

**Edge cases:**
- Empty gold set returns 0.0
- K_eff applied when K > |R(q)|

### 1.3 MRR@K (Mean Reciprocal Rank)

**Definition:**
```
MRR@K(q) = 1/r   where r = rank of first gold item in R(q)_K
         = 0     if no gold item in top-K
```

**Implementation:** `src/final_sc_review/metrics/ranking.py:mrr_at_k()`

**Range:** [0, 1]

**Notes:**
- Rank is 1-indexed (first position has rank 1)
- Only considers first relevant hit

### 1.4 MAP@K (Mean Average Precision)

**Definition:**
```
AP@K(q) = (1/min(|G(q)|, K)) * Σ_{i=1}^{K} [rel(i) * P(i)]

where:
- rel(i) = 1 if R(q)_i ∈ G(q), else 0
- P(i) = precision at position i = (# relevant in top-i) / i
```

**Implementation:** `src/final_sc_review/metrics/ranking.py:map_at_k()`

**Range:** [0, 1]

### 1.5 nDCG@K (Normalized Discounted Cumulative Gain)

**Definition:**
```
DCG@K(q) = Σ_{i=1}^{K} rel(i) / log₂(i+1)

IDCG@K(q) = Σ_{i=1}^{min(|G(q)|,K)} 1 / log₂(i+1)

nDCG@K(q) = DCG@K / IDCG@K    if IDCG@K > 0
          = 0                  if IDCG@K = 0
```

**Implementation:** `src/final_sc_review/metrics/ranking.py:ndcg_at_k()`

**Range:** [0, 1]

**Notes:**
- Binary relevance (rel ∈ {0, 1})
- Log base 2

---

## 2. Aggregate Metrics

### 2.1 Mean Across Queries

All ranking metrics are macro-averaged:
```
Metric = (1/|Q_eff|) * Σ_{q ∈ Q_eff} Metric(q)
```

Where Q_eff depends on `skip_no_positives`:
- `skip_no_positives=True`: Q_eff = {q : |G(q)| > 0}
- `skip_no_positives=False`: Q_eff = Q (all queries)

### 2.2 Dual Evaluation Protocol

**Rationale:** Papers must report both modes to avoid metric inflation.

| Mode | Population | Use Case |
|------|------------|----------|
| `positives_only` | Queries with gold positives | Measures ranking quality |
| `all_queries` | All queries | Measures system hit rate |

**Implementation:** `src/final_sc_review/metrics/retrieval_eval.py:dual_evaluate()`

---

## 3. K Policy

### 3.1 Paper-Standard K Values

| Category | K Values | Rationale |
|----------|----------|-----------|
| Primary | [3, 5, 10] | Mandatory deployment metrics |
| Optional | [1] | Very short outputs |
| Ceiling | ALL | Sanity check only |

### 3.2 K_eff (Fair Evaluation)

**Problem:** Posts vary in length. K=10 is meaningless for 5-sentence posts.

**Solution:**
```
K_eff = min(K, n_candidates)
```

**Implementation:** `src/final_sc_review/metrics/k_policy.py:compute_k_eff()`

---

## 4. Classification Metrics (GNN Models)

### 4.1 AUROC (Area Under ROC Curve)

**Definition:** Area under the curve plotting TPR vs FPR at all thresholds.

**Population:** All (query, candidate) pairs in evaluation set.

**Implementation:** `sklearn.metrics.roc_auc_score()`

**Range:** [0, 1] (random = 0.5)

### 4.2 AUPRC (Area Under Precision-Recall Curve)

**Definition:** Area under the curve plotting Precision vs Recall at all thresholds.

**Implementation:** `sklearn.metrics.average_precision_score()`

**Range:** [0, 1] (baseline = class prior)

### 4.3 TPR at FPR

**Definition:**
```
TPR@FPR=α = max TPR such that FPR ≤ α
```

**Reported levels:** FPR = 3%, 5%, 10%

**Implementation:** `src/final_sc_review/gnn/evaluation/metrics.py:NEGateMetrics.compute()`

---

## 5. Edge Case Handling

| Condition | Behavior |
|-----------|----------|
| Empty gold set | Recall=0, MRR=0, nDCG=0, MAP=0 |
| Empty ranked list | All metrics = 0 |
| K > |ranked| | Use K_eff = |ranked| |
| Single class (binary) | AUROC=0.5, AUPRC=class_prior |
| Zero denominator | Return 0.0 |

---

## 6. Verification Requirements

### 6.1 Independent Cross-Check

Run `scripts/verification/metric_crosscheck.py` to verify:
- All metrics computed by two independent implementations
- Maximum allowed deviation: <1% relative difference
- Must pass before publication

### 6.2 Range Assertions

All metrics must satisfy:
- 0.0 ≤ recall@K ≤ 1.0
- 0.0 ≤ mrr@K ≤ 1.0
- 0.0 ≤ map@K ≤ 1.0
- 0.0 ≤ ndcg@K ≤ 1.0
- 0.0 ≤ auroc ≤ 1.0
- 0.0 ≤ auprc ≤ 1.0

**Test:** `tests/test_publication_gate.py::TestMetricRangeAssertions`

---

## 7. Implementation Locations

| Metric | File | Function |
|--------|------|----------|
| recall@K | `metrics/ranking.py` | `recall_at_k()` |
| mrr@K | `metrics/ranking.py` | `mrr_at_k()` |
| map@K | `metrics/ranking.py` | `map_at_k()` |
| ndcg@K | `metrics/ranking.py` | `ndcg_at_k()` |
| K_eff | `metrics/k_policy.py` | `compute_k_eff()` |
| dual_evaluate | `metrics/retrieval_eval.py` | `dual_evaluate()` |
| paper_evaluate | `metrics/retrieval_eval.py` | `paper_evaluate()` |
| AUROC/AUPRC | `gnn/evaluation/metrics.py` | `NEGateMetrics.compute()` |
| TPR@FPR | `gnn/evaluation/metrics.py` | `NEGateMetrics.compute()` |

---

**Generated:** 2026-01-19
**Version:** 1.0
