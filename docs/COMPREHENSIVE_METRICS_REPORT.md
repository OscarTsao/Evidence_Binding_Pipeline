# Comprehensive Pipeline Evaluation Report
Generated: 2026-02-03

## Executive Summary

This report consolidates all available metrics for the Evidence Binding Pipeline.

---

## 1. P3 Graph Reranker (SAGE+Residual+GELU) - COMPLETE

**Source:** `outputs/comprehensive_eval_20260203/p3_eval/summary.json`
**Evaluation:** 5-Fold CV, 1,260 queries with evidence

### Performance Comparison

| Metric | Baseline | With P3 GNN | Improvement |
|--------|----------|-------------|-------------|
| **nDCG@10** | 0.7330 ± 0.309 | **0.8008 ± 0.286** | **+9.24%** |
| **nDCG@5** | 0.6990 ± 0.359 | 0.7759 ± 0.332 | +11.00% |
| **MRR** | 0.6746 ± 0.367 | 0.7557 ± 0.343 | +12.02% |
| **Recall@5** | 0.8439 ± 0.352 | 0.8886 ± 0.306 | +5.30% |
| **Recall@10** | 0.9444 ± 0.223 | 0.9606 ± 0.188 | +1.71% |

---

## 2. P4 No-Evidence Detection (HeteroGNN) - FROM PREVIOUS EVAL

**Source:** `outputs/final_research_eval/20260118_031312_complete/`
**Evaluation:** All 14,770 queries

### Classification Metrics

| Metric | Value |
|--------|-------|
| **AUROC** | 0.8972 |
| **AUPRC** | 0.5709 |
| **Sensitivity** | 0.4329 |
| **Specificity** | 0.9808 |
| **Precision (PPV)** | 0.6991 |
| **F1 Score** | 0.5347 |
| **MCC** | 0.5157 |
| **Balanced Accuracy** | 0.7069 |

### TPR at FPR Thresholds

| FPR Budget | TPR | Threshold |
|------------|-----|-----------|
| 1% | 0.3263 | 0.6364 |
| 3% | 0.5199 | 0.3333 |
| 5% | 0.6026 | 0.2222 |
| 10% | 0.7397 | 0.1592 |

### Calibration
- ECE (Expected Calibration Error): 0.0084
- ECE Improvement: 7.85%

---

## 3. P2 Dynamic-K Selection - FROM PREVIOUS EVAL

**Source:** `outputs/final_research_eval/20260118_031312_complete/verification/stage_summary.json`

| Metric | Value |
|--------|-------|
| Mean Selected K | 6.82 |
| Evidence Recall (UNCERTAIN) | 0.7056 |
| Evidence Recall (POS) | 0.7132 |

---

## 4. Per-Criterion Performance

**Source:** `results/paper_bundle/v3.0/tables/per_criterion.csv`

| Criterion | n_queries | Positive Rate | AUROC | AUPRC | Recall@K | MRR |
|-----------|-----------|---------------|-------|-------|----------|-----|
| A.1 (Depressed Mood) | 1477 | 22.2% | 0.8274 | 0.5761 | 0.7096 | 0.3940 |
| A.2 (Anhedonia) | 1477 | 8.4% | 0.8761 | 0.4642 | 0.8380 | 0.4265 |
| A.3 (Weight Change) | 1477 | 3.0% | 0.9079 | 0.2504 | 0.6591 | 0.4012 |
| A.4 (Sleep) | 1477 | 6.9% | 0.8859 | 0.3784 | 0.6928 | 0.4332 |
| A.5 (Psychomotor) | 1477 | 2.4% | 0.8006 | 0.1099 | 0.6286 | 0.3079 |
| A.6 (Fatigue) | 1477 | 8.4% | 0.9264 | 0.6033 | 0.7144 | 0.2966 |
| A.7 (Worthlessness) | 1477 | 21.1% | 0.9183 | 0.7735 | 0.7169 | 0.4061 |
| A.8 (Concentration) | 1477 | 4.0% | 0.7985 | 0.1151 | 0.6610 | 0.3402 |
| A.9 (Suicidal Ideation) | 1477 | 11.2% | 0.9466 | 0.7444 | 0.6767 | 0.3736 |
| A.10 (SPECIAL_CASE) | 1477 | 5.8% | 0.6648 | 0.0856 | 0.5814 | 0.2821 |

**Note:** A.10 excluded from GNN training due to poor performance.

---

## 5. Extended Ranking Metrics (All K values)

**Source:** `docs/METRIC_CONTRACT.md`

| Metric | Baseline | With GNN | Δ |
|--------|----------|----------|---|
| nDCG@1 | 0.5540 | 0.6497 | +17.27% |
| nDCG@3 | 0.6660 | 0.7532 | +13.09% |
| nDCG@5 | 0.7086 | 0.7832 | +10.53% |
| nDCG@10 | 0.7330 | 0.8206 | +10.48% |
| nDCG@20 | 0.7566 | 0.8223 | +8.68% |
| Hit@1 | 0.5540 | 0.6605 | +19.22% |
| Hit@3 | 0.7691 | 0.8483 | +10.30% |
| Hit@5 | 0.8631 | 0.9187 | +6.44% |
| Hit@10 | 0.9534 | 0.9762 | +2.39% |
| MAP@10 | 0.6732 | 0.7612 | +13.07% |

---

## 6. GNN Architecture Summary

| Module | Architecture | Status | Key Metric |
|--------|--------------|--------|------------|
| P1 | Simple GCN | ❌ Deprecated | AUROC 0.577 |
| P2 | GCN + Regressor | ✅ Production | Mean K = 6.82 |
| **P3** | **SAGE+Residual+GELU** | ✅ Production | **nDCG@10 +9.24%** |
| P4 | HeteroGNN | ✅ Production | AUROC 0.8972 |

---

## 7. Metrics Not Yet Computed

| Metric | Status | Notes |
|--------|--------|-------|
| Statistical significance (p-values) | ⚠️ Pending | Requires permutation tests |
| Multi-seed robustness | ⚠️ Pending | Only seed=42 tested |
| Inference latency | ⚠️ Pending | No timing benchmarks |
| End-to-end with all GNNs | ⚠️ Partial | P2/P4 eval scripts need fixing |

---

## Source of Truth

- P3 Comprehensive Ablation: `outputs/comprehensive_ablation/`
- P4 Stage Summary: `outputs/final_research_eval/20260118_031312_complete/`
- Per-Criterion: `results/paper_bundle/v3.0/tables/per_criterion.csv`
- Extended Metrics: `docs/METRIC_CONTRACT.md`
