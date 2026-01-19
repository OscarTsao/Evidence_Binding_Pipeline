# METRICS CONTRACT
## Evidence Retrieval Pipeline - Gold Standard Definitions

**Version:** 1.0.0
**Date:** 2026-01-18
**Purpose:** Single source of truth for ALL metrics used in evaluation

---

## 1. DATASET SPECIFICATIONS

### 1.1 Query Definition
- **Query:** A (post_id, criterion_id) pair
- **Total Queries:** 14,770 = 1,477 posts × 10 criteria (A.1 through A.10)
- **Positive Queries:** 1,379 (9.34%) - queries where has_evidence_gold = 1
- **Negative Queries:** 13,391 (90.66%) - queries where has_evidence_gold = 0

### 1.2 Split Definition
- **Split Method:** Post-ID disjoint (NO post appears in multiple splits)
- **Folds:** 5-fold cross-validation
- **Fold Sizes:**
  - Folds 0-3: 295 posts each (2,950 queries)
  - Fold 4: 297 posts (2,970 queries)
- **Verified:** ✅ ZERO overlap between folds (checked per_query.csv)

### 1.3 Within-Post Retrieval
- **Candidate Pool:** All sentences from the SAME post (within-post only)
- **Retrieval Constraint:** Cannot retrieve from other posts
- **Average Candidates per Query:** ~20 sentences

---

## 2. CLASSIFICATION METRICS (NE Gate - No-Evidence Detection)

**Scope:** Query-level binary classification
**Task:** Predict has_evidence ∈ {0, 1} for each (post, criterion) pair
**Total Samples:** 14,770 queries
**Model Output:** Calibrated probability p ∈ [0, 1]

### 2.1 ROC Metrics

#### AUROC (Area Under ROC Curve)
- **Definition:** Area under True Positive Rate vs False Positive Rate curve
- **Range:** [0, 1], random = 0.5, perfect = 1.0
- **Computation:**
  ```python
  from sklearn.metrics import roc_auc_score
  auroc = roc_auc_score(y_true, y_prob)
  ```
- **Subset:** ALL 14,770 queries
- **Interpretation:** Probability that a random positive query has higher score than random negative query
- **Target:** ≥ 0.85 (excellent discrimination)

#### TPR @ FPR Targets
- **Definition:** True Positive Rate achieved at specific False Positive Rate thresholds
- **FPR Targets:** {1%, 3%, 5%, 10%}
- **Computation:**
  1. Sort queries by descending probability
  2. For each FPR target, find threshold τ where FPR ≤ target
  3. Report TPR at that threshold
- **Use Case:** Clinical operating point selection
- **Example:** TPR@1%FPR = 0.326 means "at 1% false alarm rate, we catch 32.6% of evidence cases"

### 2.2 Precision-Recall Metrics

#### AUPRC (Area Under Precision-Recall Curve)
- **Definition:** Area under Precision vs Recall curve
- **Range:** [0, 1], random baseline = positive rate (0.0934), perfect = 1.0
- **Computation:**
  ```python
  from sklearn.metrics import average_precision_score
  auprc = average_precision_score(y_true, y_prob)
  ```
- **Subset:** ALL 14,770 queries
- **Interpretation:** Average precision across all recall levels
- **Target:** ≥ 0.55 (above random baseline)

### 2.3 Confusion Matrix Metrics

**Operating Point:** Default threshold τ = 0.5 (or optimized threshold)

| Metric | Formula | Definition | Range |
|--------|---------|------------|-------|
| **TP** | - | True Positives: correctly predicted has_evidence=1 | [0, n_pos] |
| **TN** | - | True Negatives: correctly predicted has_evidence=0 | [0, n_neg] |
| **FP** | - | False Positives: predicted 1, actual 0 | [0, n_neg] |
| **FN** | - | False Negatives: predicted 0, actual 1 | [0, n_pos] |
| **Sensitivity (TPR, Recall)** | TP / (TP + FN) | Fraction of evidence cases detected | [0, 1] |
| **Specificity (TNR)** | TN / (TN + FP) | Fraction of no-evidence cases correctly rejected | [0, 1] |
| **FPR** | FP / (TN + FP) | False alarm rate | [0, 1] |
| **Precision (PPV)** | TP / (TP + FP) | Fraction of positive predictions that are correct | [0, 1] |
| **NPV** | TN / (TN + FN) | Negative predictive value | [0, 1] |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall | [0, 1] |
| **MCC** | (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] | Matthews Correlation Coefficient | [-1, 1] |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Average of TPR and TNR | [0, 1] |

### 2.4 Calibration Metrics

**Purpose:** Assess whether predicted probabilities match true frequencies

#### ECE (Expected Calibration Error)
- **Definition:** Average absolute difference between predicted probability and true frequency across bins
- **Range:** [0, 1], perfect calibration = 0
- **Computation:**
  1. Bin predictions into M bins (default M=10)
  2. For bin m: compute |predicted_prob_m - true_freq_m|
  3. ECE = Σ (n_m / N) × |predicted_prob_m - true_freq_m|
- **Target:** < 0.05 (well-calibrated)

#### Brier Score
- **Definition:** Mean squared error between predicted probabilities and true labels
- **Range:** [0, 1], perfect = 0
- **Computation:**
  ```python
  from sklearn.metrics import brier_score_loss
  brier = brier_score_loss(y_true, y_prob)
  ```
- **Interpretation:** Lower is better
- **Target:** < 0.1

---

## 3. RANKING METRICS (Evidence Retrieval)

**Scope:** Query-level ranking of evidence sentences
**Subset:** Queries with has_evidence=1 ONLY (1,379 queries)
**Task:** Rank candidate sentences, measure how well gold evidence sentences are ranked

### 3.1 Recall @ K

#### Definition
- **Recall@K:** Fraction of gold evidence sentences that appear in the top-K ranked candidates
- **Range:** [0, 1], higher is better
- **Formula:** Recall@K = |GoldInTopK| / |GoldTotal|
  - GoldInTopK = {gold sentences in top-K}
  - GoldTotal = {all gold sentences for this query}

#### K Values
Report for K ∈ {1, 3, 5, 10, 20}

#### Computation (per query)
```python
def recall_at_k(gold_sent_uids: Set[str], ranked_sent_uids: List[str], k: int) -> float:
    """Compute recall@K for a single query."""
    if not gold_sent_uids:
        return 0.0  # No gold to retrieve
    top_k = set(ranked_sent_uids[:k])
    retrieved_gold = gold_sent_uids & top_k
    return len(retrieved_gold) / len(gold_sent_uids)
```

#### Aggregation
- **Mean Recall@K:** Average across all has_evidence=1 queries
- **Median Recall@K:** Median across all has_evidence=1 queries
- **Report both:** Mean ± Std, Median, P25, P75

### 3.2 Precision @ K

#### Definition
- **Precision@K:** Fraction of top-K ranked sentences that are gold evidence
- **Range:** [0, 1], higher is better
- **Formula:** Precision@K = |GoldInTopK| / K

#### Computation (per query)
```python
def precision_at_k(gold_sent_uids: Set[str], ranked_sent_uids: List[str], k: int) -> float:
    """Compute precision@K for a single query."""
    top_k = set(ranked_sent_uids[:k])
    retrieved_gold = gold_sent_uids & top_k
    return len(retrieved_gold) / k
```

### 3.3 Hit Rate @ K

#### Definition
- **HitRate@K:** Fraction of queries where at least one gold sentence appears in top-K
- **Range:** [0, 1], higher is better
- **Formula:** HitRate@K = (# queries with ≥1 gold in top-K) / (total queries)

### 3.4 MRR (Mean Reciprocal Rank)

#### Definition
- **MRR:** Mean of 1/rank_first_gold across all queries
- **Range:** [0, 1], higher is better
- **rank_first_gold:** Rank (1-indexed) of the first gold sentence

#### Computation
```python
def mrr_single_query(gold_sent_uids: Set[str], ranked_sent_uids: List[str]) -> float:
    """Compute MRR for a single query."""
    for rank, sent_uid in enumerate(ranked_sent_uids, start=1):
        if sent_uid in gold_sent_uids:
            return 1.0 / rank
    return 0.0  # No gold found in ranking
```

#### Aggregation
- **Mean MRR:** Average across all has_evidence=1 queries
- **Interpretation:** 1/MRR ≈ expected rank of first gold sentence
- **Example:** MRR=0.38 → first gold appears around rank 2.6 on average

### 3.5 MAP @ K (Mean Average Precision)

#### Definition
- **MAP@K:** Mean of Average Precision@K across all queries
- **Average Precision@K:** Average of Precision@i for i where gold sentence is found, up to K

#### Computation
```python
def average_precision_at_k(gold_sent_uids: Set[str], ranked_sent_uids: List[str], k: int) -> float:
    """Compute AP@K for a single query."""
    if not gold_sent_uids:
        return 0.0

    precisions = []
    n_gold_found = 0

    for i, sent_uid in enumerate(ranked_sent_uids[:k], start=1):
        if sent_uid in gold_sent_uids:
            n_gold_found += 1
            precision_at_i = n_gold_found / i
            precisions.append(precision_at_i)

    if not precisions:
        return 0.0
    return sum(precisions) / len(gold_sent_uids)
```

### 3.6 nDCG @ K (Normalized Discounted Cumulative Gain)

#### Definition
- **nDCG@K:** Normalized discounted cumulative gain at rank K
- **Range:** [0, 1], perfect = 1.0
- **Relevance:** Binary (1 for gold, 0 for non-gold)

#### Computation
```python
def ndcg_at_k(gold_sent_uids: Set[str], ranked_sent_uids: List[str], k: int) -> float:
    """Compute nDCG@K for a single query."""
    # Relevance labels: 1 for gold, 0 for non-gold
    relevance = [1 if sent in gold_sent_uids else 0 for sent in ranked_sent_uids[:k]]

    # DCG = Σ (2^rel_i - 1) / log2(i + 1)
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance))

    # IDCG = DCG for perfect ranking (all golds first)
    n_gold = len(gold_sent_uids)
    ideal_relevance = [1] * min(n_gold, k) + [0] * max(0, k - n_gold)
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    if idcg == 0:
        return 0.0
    return dcg / idcg
```

---

## 4. DYNAMIC-K EXTRACTION METRICS

**Scope:** Queries with has_evidence=1 ONLY (1,379 queries)
**Task:** Evaluate quality of adaptive K selection and extracted evidence set

### 4.1 Selected K Distribution

**Metrics:**
- **Mean Selected K:** Average number of sentences extracted per query
- **Median Selected K:** Median K
- **P90 Selected K:** 90th percentile K (workload planning)
- **K Range:** [K_min, K_max] observed

**Breakdown:**
- Overall (all queries)
- By state (NEG/UNCERTAIN/POS)
- By has_evidence (0/1)

### 4.2 Evidence Recall @ Selected K

#### Definition
- **Evidence Recall:** Fraction of gold evidence sentences included in the extracted K sentences
- **Range:** [0, 1], higher is better
- **Subset:** has_evidence=1 queries ONLY

#### Computation
```python
def evidence_recall_at_selected_k(gold_sent_uids: Set[str], extracted_sent_uids: Set[str]) -> float:
    """Compute recall for extracted set (dynamic K)."""
    if not gold_sent_uids:
        return 0.0
    retrieved_gold = gold_sent_uids & extracted_sent_uids
    return len(retrieved_gold) / len(gold_sent_uids)
```

#### Aggregation
- **Mean Evidence Recall:** Average across has_evidence=1 queries
- **Target:** ≥ 0.70 (retrieve 70% of evidence on average)

### 4.3 Evidence Precision @ Selected K

#### Definition
- **Evidence Precision:** Fraction of extracted sentences that are gold evidence
- **Range:** [0, 1], higher is better
- **Subset:** has_evidence=1 queries ONLY

#### Computation
```python
def evidence_precision_at_selected_k(gold_sent_uids: Set[str], extracted_sent_uids: Set[str]) -> float:
    """Compute precision for extracted set (dynamic K)."""
    if not extracted_sent_uids:
        return 0.0
    retrieved_gold = gold_sent_uids & extracted_sent_uids
    return len(retrieved_gold) / len(extracted_sent_uids)
```

#### Aggregation
- **Mean Evidence Precision:** Average across has_evidence=1 queries
- **Interpretation:** Clinical burden (how many non-evidence sentences to review)

---

## 5. THREE-STATE GATE METRICS (Clinical Deployment)

**Scope:** All 14,770 queries
**States:** NEG (skip), UNCERTAIN (conservative), POS (standard)
**Thresholds:** τ_neg (skip threshold), τ_pos (alert threshold)

### 5.1 State Definitions

```python
def assign_state(prob: float, tau_neg: float, tau_pos: float) -> str:
    """Assign 3-state gate decision."""
    if prob < tau_neg:
        return "NEG"      # Skip - likely no evidence
    elif prob < tau_pos:
        return "UNCERTAIN"  # Conservative - low confidence
    else:
        return "POS"      # Standard - high confidence alert
```

### 5.2 Workload Metrics

**Purpose:** Understand clinical review workload distribution

| Metric | Definition | Computation |
|--------|------------|-------------|
| **NEG Rate** | Fraction of queries skipped | n_NEG / 14,770 |
| **UNCERTAIN Rate** | Fraction needing conservative review | n_UNCERTAIN / 14,770 |
| **POS Rate** | Fraction flagged as high-confidence | n_POS / 14,770 |
| **Alert Rate per 1000** | High-confidence alerts per 1000 queries | (n_POS / 14,770) × 1000 |

### 5.3 Screening Stage Metrics

**Screening Decision:** "Screen in" = NOT NEG (i.e., UNCERTAIN or POS)

#### Screening Sensitivity
- **Definition:** Fraction of has_evidence=1 queries that are screened in (not skipped)
- **Formula:** TP_screen / (TP_screen + FN_screen)
  - TP_screen = has_evidence=1 AND state ≠ NEG
  - FN_screen = has_evidence=1 AND state = NEG
- **Target:** ≥ 99.5% (very few evidence cases missed at screening)

#### Screening FN per 1000
- **Definition:** Number of false negatives (missed evidence) per 1000 queries
- **Formula:** (FN_screen / 14,770) × 1000
- **Target:** ≤ 5 per 1000 (0.5% miss rate)

### 5.4 Alert Stage Metrics

**Alert Decision:** High-confidence alert = POS state

#### Alert Precision
- **Definition:** Fraction of POS alerts that truly have evidence
- **Formula:** TP_alert / (TP_alert + FP_alert)
  - TP_alert = has_evidence=1 AND state = POS
  - FP_alert = has_evidence=0 AND state = POS
- **Target:** ≥ 90% (most high-confidence alerts are correct)

#### Alert Rate per 1000
- **Definition:** Number of high-confidence alerts per 1000 queries
- **Formula:** (n_POS / 14,770) × 1000
- **Clinical Utility:** Lower is better (reduces review burden)

---

## 6. PER-POST MULTI-LABEL METRICS

**Scope:** Post-level aggregation of 10 criteria predictions
**Total Posts:** 1,477
**Vector Length:** 10 (one binary label per criterion A.1-A.10)

### 6.1 Exact Match Rate

#### Definition
- **Exact Match:** Fraction of posts where ALL 10 criteria predictions match ground truth
- **Range:** [0, 1], higher is better
- **Formula:** ExactMatch = (# posts with all 10 correct) / 1,477

### 6.2 Hamming Score (Subset Accuracy)

#### Definition
- **Hamming Score:** Average per-criterion accuracy across posts
- **Range:** [0, 1], higher is better
- **Formula:** HammingScore = (Σ across posts: # correct criteria / 10) / 1,477

### 6.3 Hamming Loss

#### Definition
- **Hamming Loss:** Average per-criterion error rate across posts
- **Range:** [0, 1], lower is better
- **Formula:** HammingLoss = 1 - HammingScore

### 6.4 Per-Post F1 Metrics

**Computation:** Treat each post as a multi-label classification instance

| Metric | Definition | Aggregation |
|--------|------------|-------------|
| **Micro F1** | F1 computed globally (pool all post×criterion pairs) | Single F1 score |
| **Macro F1** | Average of per-post F1 scores | Mean across 1,477 posts |
| **Weighted F1** | Weighted average by number of positive labels per post | sklearn default |

---

## 7. PER-CRITERION BREAKDOWN METRICS

**Scope:** Criterion-level analysis (A.1 through A.10)
**Purpose:** Identify which criteria are easier/harder to detect

### 7.1 Per-Criterion NE Detection

**For each criterion c ∈ {A.1, A.2, ..., A.10}:**

| Metric | Subset | Interpretation |
|--------|--------|----------------|
| **AUROC_c** | All 1,477 queries for criterion c | Discrimination quality |
| **AUPRC_c** | All 1,477 queries for criterion c | Precision-recall tradeoff |
| **Sensitivity_c** | Queries for criterion c | Fraction of evidence detected |
| **Precision_c** | Queries for criterion c | Positive predictive value |
| **n_queries_c** | - | Total queries (should be 1,477) |
| **n_positive_c** | - | Queries with evidence for criterion c |
| **positive_rate_c** | - | n_positive_c / 1,477 |

### 7.2 Per-Criterion Ranking

**For each criterion c (only queries with has_evidence=1):**

| Metric | Subset | Interpretation |
|--------|--------|----------------|
| **Evidence Recall@10_c** | has_evidence=1 for criterion c | How well evidence is ranked |
| **nDCG@10_c** | has_evidence=1 for criterion c | Ranking quality |
| **MRR_c** | has_evidence=1 for criterion c | First evidence rank |

---

## 8. ABLATION STUDY METRICS

**Purpose:** Quantify contribution of each pipeline component

### 8.1 Module Ablations

**Configurations:**
1. Retriever only (NV-Embed-v2, top-24)
2. + Jina Reranker (top-10)
3. + P3 Graph Reranker
4. + P2 Dynamic-K Selection
5. + P4 NE Gate (GNN)
6. + 3-State Clinical Gate (full pipeline)

**Metrics to Report (per configuration):**
- nDCG@10 (mean ± std across folds)
- AUROC (NE detection)
- Evidence Recall@10
- Alert Precision

### 8.2 Policy Ablations

**Dynamic-K γ Sweep:**
- Test γ ∈ {0.7, 0.8, 0.9, 0.95}
- Report: Mean K, Evidence Recall, Workload (# sentences extracted)

**Threshold Grid (τ_neg, τ_pos):**
- Tune only on TUNE split (nested CV)
- Report: Screening Sensitivity, Alert Precision, Workload

---

## 9. CONSISTENCY CHECKS (MUST PASS)

### 9.1 Range Checks

**All probabilities and rates MUST be in [0, 1]:**
```python
assert 0 <= auroc <= 1
assert 0 <= auprc <= 1
assert 0 <= precision <= 1
assert 0 <= recall <= 1
assert 0 <= fpr <= 1
```

### 9.2 Confusion Matrix Invariants

```python
assert tp + tn + fp + fn == 14770
assert tp + fn == 1379  # has_evidence=1 count
assert tn + fp == 13391  # has_evidence=0 count
assert sensitivity == tp / (tp + fn)
assert specificity == tn / (tn + fp)
```

### 9.3 Metric Consistency

**If same metric appears in multiple places, values MUST match OR be clearly labeled as different subsets:**
- AUPRC (all queries) vs AUPRC (per-criterion)
- Recall@10 (ranking) vs Evidence Recall (extraction)

---

## 10. REPORTING REQUIREMENTS

### 10.1 Mean ± Std Reporting

**For cross-validation metrics:**
- Report: Mean ± Std across folds
- Include: 95% CI if available (bootstrap recommended)
- Example: "AUROC = 0.8972 ± 0.0015 (95% CI: [0.8941, 0.9003])"

### 10.2 Tables

**All metric tables MUST include:**
- Metric name
- Value (mean ± std)
- Subset (which queries)
- Interpretation (what it means clinically)

### 10.3 Plots

**Required visualizations:**
- ROC curve (with 95% CI band)
- PR curve (with random baseline)
- Calibration plot (reliability diagram)
- Per-criterion bar charts
- Dynamic-K histograms

---

## 11. REFERENCE IMPLEMENTATIONS

### 11.1 Primary Implementation
- **Location:** `scripts/gnn/run_e2e_eval_and_report.py`
- **Outputs:** `outputs/final_research_eval/.../summary.json`

### 11.2 Independent Verification
- **Location:** `scripts/verification/recompute_metrics_from_csv.py`
- **Input:** `per_query.csv`
- **Purpose:** Cross-check all metrics from raw predictions

### 11.3 Leakage Tests
- **Location:** `tests/test_gnn_no_leakage.py`, `tests/clinical/test_no_leakage.py`
- **Purpose:** Ensure no gold-derived features used

---

## APPENDIX: Common Pitfalls

### A.1 Denominator Errors
- ❌ Recall@K computed on ALL queries (should be has_evidence=1 only)
- ❌ AUPRC baseline compared to 0.5 (should compare to positive rate)
- ❌ MRR averaged over all queries (should exclude has_evidence=0)

### A.2 Threshold Leakage
- ❌ Tuning τ_neg/τ_pos on TEST split (should use TUNE only)
- ❌ Reporting metrics at multiple thresholds without specifying which is deployment threshold

### A.3 K Confusion
- ❌ Mixing fixed K (retriever K=24) with dynamic K (extracted K=variable)
- ❌ Reporting Recall@10 when using dynamic K (should report Evidence Recall @ Selected K)

### A.4 Split Leakage
- ❌ Posts appearing in multiple folds
- ❌ Using gold labels from TEST in feature engineering

---

**END OF METRICS CONTRACT**

This document is the authoritative reference for all metrics.
Any discrepancy between code and this contract should be resolved by updating the code.
