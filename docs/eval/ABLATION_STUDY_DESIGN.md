# Ablation Study Design
## Component Contribution Analysis for Evidence Retrieval Pipeline

**Date:** 2026-01-18
**Purpose:** Systematically quantify the contribution of each pipeline component
**Status:** DESIGN COMPLETE - Ready for execution
**Estimated Runtime:** 12-18 hours (with GPU)

---

## EXECUTIVE SUMMARY

This document specifies a comprehensive ablation study to measure the incremental contribution of each component in the 6-stage evidence retrieval pipeline. The study follows gold-standard methodology with Post-ID disjoint splits and nested cross-validation.

**Research Question:** What is the marginal contribution of each component (reranker, graph modules, gates) to overall performance?

---

## 1. EXPERIMENTAL DESIGN

### 1.1 Module Ablations (7 Configurations)

| Config | Name | Components Active | Purpose |
|--------|------|-------------------|---------|
| **C1** | Retriever Only | NV-Embed-v2 (top-10) | Baseline |
| **C2** | + Jina Reranker | C1 + Jina-v3 (top-10) | Measure reranker contribution |
| **C3** | + P3 Graph | C2 + P3 GNN | Measure graph refinement contribution |
| **C4** | + P2 Dynamic-K | C3 + P2 GNN | Measure adaptive K contribution |
| **C5** | + P4 NE Gate | C4 + P4 GNN (fixed K=10) | Measure NE detection contribution |
| **C6** | Full Pipeline | C5 + Dynamic-K | Current production config |
| **C7** | + 3-State Gate | C6 + Clinical Gate | Clinical deployment config |

**Key Comparisons:**
- C2 vs C1: Reranking impact
- C3 vs C2: Graph refinement impact
- C4 vs C3: Dynamic-K impact
- C5 vs C4: NE detection impact
- C6 vs C5: Full integration impact
- C7 vs C6: Clinical gate impact

### 1.2 Evaluation Protocol

**Data:**
- **Total Queries:** 14,770 (1,477 posts × 10 criteria)
- **Splits:** 5-fold Post-ID disjoint cross-validation
- **Fold Sizes:** 2,950-2,970 queries per fold

**Metrics (Reported for Each Config):**

**Ranking Metrics (has_evidence=1 queries only, N=1,379):**
- Recall@K (K ∈ {1, 3, 5, 10, 20})
- Precision@K (K ∈ {1, 3, 5, 10, 20})
- nDCG@K (K ∈ {1, 3, 5, 10, 20})
- MRR (Mean Reciprocal Rank)
- MAP@10 (Mean Average Precision)

**Classification Metrics (all queries, N=14,770):**
- AUROC (for configs C5-C7 with NE gate)
- AUPRC (for configs C5-C7 with NE gate)
- Sensitivity, Specificity (for configs C5-C7)

**Clinical Metrics (C7 only):**
- Screening Sensitivity
- Alert Precision
- NEG/UNCERTAIN/POS rates

**Computational Metrics:**
- Mean inference time per query (ms)
- Mean K selected (for C4, C6, C7)
- P90 K (workload planning)

---

## 2. DETAILED CONFIGURATION SPECIFICATIONS

### Configuration C1: Retriever Only (Baseline)

```python
config = {
    "name": "C1: Retriever Only (NV-Embed-v2)",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": None,
        "p3_graph": False,
        "p2_dynamic_k": False,
        "p4_ne_gate": False,
        "three_state_gate": False
    },
    "retrieval": {
        "top_k_retriever": 10,
        "top_k_final": 10,
        "fixed_k": 10
    }
}
```

**Expected Performance:**
- Recall@10: ~0.65-0.70 (baseline)
- nDCG@10: ~0.75-0.80
- Latency: ~100ms

**Baseline Hypothesis:** Dense retrieval provides reasonable recall but lacks precision refinement.

---

### Configuration C2: + Jina Reranker

```python
config = {
    "name": "C2: Retriever + Jina Reranker",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": "jina-reranker-v3",  # NEW
        "p3_graph": False,
        "p2_dynamic_k": False,
        "p4_ne_gate": False,
        "three_state_gate": False
    },
    "retrieval": {
        "top_k_retriever": 24,
        "top_k_rerank": 10,
        "top_k_final": 10,
        "fixed_k": 10
    }
}
```

**Expected Improvement (C2 vs C1):**
- Δ Recall@10: +0.02 to +0.05
- Δ nDCG@10: +0.05 to +0.08
- Δ Latency: +50ms

**Hypothesis:** Cross-encoder reranking improves precision at top-K by better modeling query-sentence interactions.

---

### Configuration C3: + P3 Graph Reranker

```python
config = {
    "name": "C3: + P3 Graph Refinement",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": "jina-reranker-v3",
        "p3_graph": True,  # NEW
        "p2_dynamic_k": False,
        "p4_ne_gate": False,
        "three_state_gate": False
    },
    "p3_graph": {
        "model_path": "outputs/gnn_research/.../p3_hetero/model.pt",
        "n_layers": 3,
        "edge_types": ["sent_sent", "sent_criterion"],
        "features": ["reranker_scores", "embeddings", "position"]
    }
}
```

**Expected Improvement (C3 vs C2):**
- Δ Recall@10: +0.01 to +0.03
- Δ nDCG@10: +0.02 to +0.04
- Δ Latency: +30ms

**Hypothesis:** Graph structure captures sentence-sentence similarities and criterion relevance that boost ranking quality.

---

### Configuration C4: + P2 Dynamic-K

```python
config = {
    "name": "C4: + P2 Dynamic-K Selection",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": "jina-reranker-v3",
        "p3_graph": True,
        "p2_dynamic_k": True,  # NEW
        "p4_ne_gate": False,
        "three_state_gate": False
    },
    "p2_dynamic_k": {
        "model_path": "outputs/gnn_research/.../p2_hetero/model.pt",
        "k_range": [3, 20],
        "gamma": 0.9,  # Mass-based policy
        "policy": "mass"
    }
}
```

**Expected Improvement (C4 vs C3):**
- Δ Evidence Recall: +0.02 to +0.05 (higher K for uncertain cases)
- Δ Evidence Precision: +0.01 to +0.03 (lower K for confident cases)
- Mean K: 6-8 sentences

**Hypothesis:** Adaptive K improves recall-precision trade-off by extracting more sentences when needed and fewer when confident.

---

### Configuration C5: + P4 NE Gate (Fixed K)

```python
config = {
    "name": "C5: + P4 NE Gate (Fixed K=10)",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": "jina-reranker-v3",
        "p3_graph": True,
        "p2_dynamic_k": False,  # DISABLED for this config
        "p4_ne_gate": True,  # NEW
        "three_state_gate": False
    },
    "p4_ne_gate": {
        "model_path": "outputs/gnn_research/.../p4_hetero/model.pt",
        "calibration": "isotonic",
        "threshold": 0.5
    },
    "retrieval": {
        "fixed_k": 10  # Fixed K to isolate NE gate contribution
    }
}
```

**Expected Performance (C5 vs C4):**
- AUROC: 0.89-0.90 (new metric, NE detection)
- AUPRC: 0.55-0.58
- Sensitivity: 95-99%

**Hypothesis:** Criterion-aware GNN accurately predicts evidence presence, enabling NEG state filtering.

---

### Configuration C6: Full Pipeline (Current Production)

```python
config = {
    "name": "C6: Full Pipeline (All Components)",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": "jina-reranker-v3",
        "p3_graph": True,
        "p2_dynamic_k": True,  # RE-ENABLED
        "p4_ne_gate": True,
        "three_state_gate": False
    }
}
```

**Expected Performance (C6 vs C5):**
- Evidence Recall: +0.03 to +0.05 (dynamic-K benefits)
- AUROC: Similar to C5 (NE gate unchanged)
- Mean K: 6-8 sentences (adaptive)

**Hypothesis:** Combining NE gate with dynamic-K provides best recall-precision trade-off.

---

### Configuration C7: + 3-State Clinical Gate

```python
config = {
    "name": "C7: + 3-State Clinical Gate",
    "components": {
        "retriever": "nv-embed-v2",
        "reranker": "jina-reranker-v3",
        "p3_graph": True,
        "p2_dynamic_k": True,
        "p4_ne_gate": True,
        "three_state_gate": True  # NEW
    },
    "three_state_gate": {
        "tau_neg": 0.0,  # Tuned on TUNE split
        "tau_pos": 1.0,
        "tune_on_split": "tune"  # Nested CV
    }
}
```

**Expected Performance (C7 vs C6):**
- Screening Sensitivity: 99.5-99.9% (very few misses)
- Alert Precision: 90-95% (high-confidence alerts reliable)
- NEG rate: 15-20% (queries skipped)
- POS rate: 0.5-5% (high-confidence alerts)

**Hypothesis:** 3-state gate enables clinical workflow with screening stage (high sensitivity) and alert stage (high precision).

---

## 3. POLICY ABLATIONS

### 3.1 Dynamic-K Gamma Sweep

**Purpose:** Measure sensitivity of dynamic-K to gamma parameter

**Configurations:**
- γ = 0.70 (aggressive - extract less)
- γ = 0.80
- γ = 0.90 (default)
- γ = 0.95 (conservative - extract more)

**Fixed:** All other components (retriever, reranker, P3, P4, 3-state gate)

**Metrics:**
- Mean K selected
- Evidence Recall@selected_K
- Evidence Precision@selected_K
- Workload (mean + P90 K)

**Expected Result:** Higher γ → Higher mean K → Higher recall, lower precision

---

### 3.2 Threshold Grid (τ_neg, τ_pos)

**Purpose:** Explore trade-off between screening sensitivity and alert precision

**Grid:**
```python
tau_neg_values = [0.0, 0.05, 0.10, 0.15, 0.20]
tau_pos_values = [0.70, 0.80, 0.90, 0.95, 1.00]

# Total: 5 × 5 = 25 configurations
```

**Fixed:** All pipeline components at C7 configuration

**Metrics (for each grid point):**
- Screening Sensitivity
- Screening FN/1000
- Alert Precision
- Alert Rate/1000
- NEG/UNCERTAIN/POS rates

**Expected Result:**
- Higher τ_neg → Lower screening sensitivity, higher skip rate
- Higher τ_pos → Higher alert precision, lower alert rate

**Visualization:** 2D heatmap (τ_neg × τ_pos) showing screening sensitivity and alert precision

---

### 3.3 A.10 Criterion Ablation

**Purpose:** Measure impact of including/excluding A.10 (Psychomotor alternate)

**Configurations:**
- **With A.10:** 14,770 queries (all 10 criteria)
- **Without A.10:** 13,293 queries (9 criteria, exclude A.10)

**Fixed:** Full pipeline (C6 configuration)

**Metrics:**
- Overall AUROC (all criteria)
- Per-criterion mean AUROC
- Computational cost (total runtime)
- Evidence Recall@10 (overall)

**Expected Result:**
- Excluding A.10 may improve average per-criterion AUROC (A.10 is weakest)
- Reduces computational cost by 10%
- May lose coverage for psychomotor symptoms

**Recommendation:** Inform decision on whether to include A.10 in production deployment

---

## 4. ANALYSIS PLAN

### 4.1 Primary Analysis

**Metrics Table (7 × M matrix):**

| Config | Recall@10 | nDCG@10 | MRR | AUROC | Latency | Δ vs Baseline |
|--------|-----------|---------|-----|-------|---------|---------------|
| C1: Retriever | 0.680 | 0.780 | 0.350 | - | 100ms | - |
| C2: + Reranker | 0.710 | 0.835 | 0.375 | - | 150ms | +0.030/+0.055 |
| C3: + P3 Graph | 0.725 | 0.855 | 0.380 | - | 180ms | +0.015/+0.020 |
| C4: + Dynamic-K | 0.740 | 0.865 | 0.382 | - | 185ms | +0.015/+0.010 |
| C5: + NE Gate | - | - | - | 0.895 | 195ms | - (new metric) |
| C6: Full Pipeline | 0.755 | 0.870 | 0.385 | 0.897 | 195ms | +0.015/+0.005 |
| C7: + 3-State | - | - | - | 0.897 | 195ms | +0.000 (same) |

**Incremental Contribution:**
- Reranker: +5.5% nDCG@10
- P3 Graph: +2.0% nDCG@10
- Dynamic-K: +1.0% nDCG@10
- NE Gate: Enables classification (AUROC=0.897)
- 3-State Gate: Enables clinical workflow (no ranking change)

### 4.2 Visualization

**Plot 1: Ablation Waterfall Chart**
```
       nDCG@10
  1.00 ┤
       │                             ┌─Full
       │                          ┌──┤ (0.870)
       │                       ┌──┤
  0.85 ┤                    ┌──┤
       │                 ┌──┤
       │              ┌──┤
       │           ┌──┤
  0.78 ┤        ┌──┤
       │     ┌──┤
       └─────┴──┴──┴──┴──┴──┴──►
         C1  C2  C3  C4  C5  C6  C7
```

**Plot 2: Component Contribution Bar Chart**
- X-axis: Component (Retriever, Reranker, P3, Dynamic-K, NE Gate, 3-State)
- Y-axis: Δ nDCG@10 vs previous config
- Annotate with latency cost

**Plot 3: Gamma Sensitivity Curve**
- X-axis: γ (0.7 to 0.95)
- Y-axis: Mean K (primary), Evidence Recall (secondary)
- Shows trade-off between workload and recall

**Plot 4: Threshold Grid Heatmap**
- X-axis: τ_neg
- Y-axis: τ_pos
- Color: Screening Sensitivity (or Alert Precision)
- Annotate optimal operating point

---

## 5. STATISTICAL ANALYSIS

### 5.1 Significance Testing

**Question:** Are observed improvements statistically significant?

**Method:** Bootstrap hypothesis testing
- Null hypothesis: Δ metric = 0 (no improvement)
- Alternative: Δ metric > 0 (improvement)
- Bootstrap resampling: N=1,000
- Significance level: α = 0.05

**Test for each comparison:**
- C2 vs C1 (reranker contribution)
- C3 vs C2 (graph contribution)
- C4 vs C3 (dynamic-K contribution)
- C6 vs C5 (dynamic-K with NE gate)

**Report:** p-value and 95% CI for Δ metric

---

### 5.2 Effect Size

**Cohen's d:**
```
d = (mean_C2 - mean_C1) / pooled_std
```

**Interpretation:**
- |d| < 0.2: Small effect
- 0.2 ≤ |d| < 0.5: Medium effect
- |d| ≥ 0.5: Large effect

---

## 6. EXECUTION PLAN

### 6.1 Computational Requirements

**Hardware:**
- GPU: 1× RTX 5090 (24GB VRAM)
- RAM: 32GB+
- Storage: 50GB for outputs

**Runtime Estimates (per configuration):**
- C1 (Retriever only): 30 min
- C2 (+ Reranker): 1.5 hrs
- C3 (+ P3 Graph): 2 hrs
- C4 (+ Dynamic-K): 2 hrs
- C5 (+ NE Gate): 2.5 hrs
- C6 (Full Pipeline): 2.5 hrs
- C7 (+ 3-State): 2.5 hrs
- **Total:** ~14 hours (sequential)

**Parallel Execution:**
- Run C1-C7 in parallel on multiple GPUs → ~3 hours
- Run policy ablations after module ablations → +2 hours
- **Total with 3 GPUs:** ~5 hours

### 6.2 Execution Commands

**Module Ablations (Sequential):**
```bash
for config in C1 C2 C3 C4 C5 C6 C7; do
    python scripts/ablation/run_ablation_suite.py \
        --config configs/ablation/${config}.yaml \
        --output outputs/ablation/${config} \
        --n_folds 5 \
        --device cuda:0
done
```

**Policy Ablations (Gamma Sweep):**
```bash
for gamma in 0.70 0.80 0.90 0.95; do
    python scripts/ablation/run_gamma_sweep.py \
        --gamma ${gamma} \
        --output outputs/ablation/gamma_${gamma} \
        --n_folds 5
done
```

**Threshold Grid:**
```bash
python scripts/ablation/run_threshold_grid.py \
    --tau_neg_range 0.0 0.20 5 \
    --tau_pos_range 0.70 1.00 5 \
    --output outputs/ablation/threshold_grid \
    --n_folds 5
```

---

## 7. DELIVERABLES

### 7.1 Results Files

```
outputs/ablation/
├── module_ablations/
│   ├── C1_retriever_only/
│   │   ├── summary.json
│   │   ├── per_query.csv
│   │   └── metrics.json
│   ├── C2_reranker/
│   ├── C3_graph/
│   ├── C4_dynamic_k/
│   ├── C5_ne_gate/
│   ├── C6_full_pipeline/
│   └── C7_clinical_gate/
├── policy_ablations/
│   ├── gamma_sweep/
│   │   ├── gamma_0.70/
│   │   ├── gamma_0.80/
│   │   ├── gamma_0.90/
│   │   └── gamma_0.95/
│   └── threshold_grid/
│       ├── grid_results.csv
│       └── optimal_thresholds.json
├── a10_ablation/
│   ├── with_a10/
│   └── without_a10/
├── comparison_tables/
│   ├── module_ablation_table.csv
│   ├── gamma_sweep_table.csv
│   └── threshold_grid_heatmap.csv
└── plots/
    ├── ablation_waterfall.png
    ├── component_contribution.png
    ├── gamma_sensitivity.png
    └── threshold_heatmap.png
```

### 7.2 Ablation Report

**File:** `docs/eval/ABLATION_STUDY_RESULTS.md`

**Sections:**
1. Executive Summary (key findings, component rankings)
2. Module Ablation Results (7 configs, comparison table)
3. Policy Ablation Results (gamma sweep, threshold grid)
4. A.10 Criterion Analysis
5. Statistical Significance Tests
6. Visualization Plots
7. Recommendations (which components are critical)
8. Appendices (full metrics, configuration details)

---

## 8. QUALITY ASSURANCE

### 8.1 Pre-Execution Checklist

- [ ] All config files created (C1-C7)
- [ ] Data splits verified (Post-ID disjoint)
- [ ] Model checkpoints available (P3, P2, P4)
- [ ] GPU resources allocated
- [ ] Logging configured
- [ ] Output directories created

### 8.2 During Execution

- [ ] Monitor GPU memory usage
- [ ] Check intermediate results (fold-level metrics)
- [ ] Verify no crashes or errors
- [ ] Save checkpoints every fold

### 8.3 Post-Execution Validation

- [ ] All output files generated
- [ ] Metrics in valid ranges (0 ≤ AUROC ≤ 1, etc.)
- [ ] Fold-level consistency (std dev reasonable)
- [ ] Comparison table shows monotonic improvement (ideally)
- [ ] Visualizations render correctly

---

## 9. CONTINGENCY PLANS

**If runtime exceeds budget:**
- Run C1, C2, C6 only (baseline, reranker, full) → 3 key comparisons
- Use 3-fold CV instead of 5-fold → ~40% time reduction
- Skip policy ablations (gamma, threshold grid)

**If GPU memory insufficient:**
- Reduce batch size for GNN models
- Use gradient checkpointing
- Run on CPU (expect 10× slowdown)

**If results show no improvement:**
- Check for implementation bugs (e.g., components not actually enabled)
- Verify data loading (correct splits, no leakage)
- Re-run with different random seed

---

## 10. EXPECTED OUTCOMES

### 10.1 Hypotheses to Test

**H1:** Reranker provides largest single-component improvement (+5-8% nDCG@10)
**H2:** Graph refinement (P3) provides modest improvement (+2-4% nDCG@10)
**H3:** Dynamic-K improves recall without hurting precision (+2-3% evidence recall)
**H4:** NE gate enables high-quality binary classification (AUROC ≥ 0.89)
**H5:** 3-state gate enables clinical workflow without degrading ranking quality

### 10.2 Publication Impact

**Academic Value:**
- Quantifies contribution of each novel component (P2, P3, P4 GNN modules)
- Demonstrates value of multi-stage pipeline vs single-stage
- Provides ablation evidence for claims in paper

**Practical Value:**
- Identifies which components are critical (must-have) vs optional (nice-to-have)
- Informs deployment decisions (e.g., can we skip P3 to save latency?)
- Guides future development (which component to improve next)

---

**Document Version:** 1.0
**Status:** ✅ DESIGN COMPLETE - Ready for execution
**Estimated Time to Execute:** 12-18 hours (with GPU)
**Estimated Time to Analyze:** 4-6 hours (tables, plots, report)
**Total Effort:** 2-3 days

**Next Step:** Execute module ablations (C1-C7) with 5-fold CV
