# Phase 1: Ablation Study

**Date:** 2026-01-18
**Status:** IN PROGRESS
**Branch:** gnn_e2e_gold_standard_report

---

## Executive Summary

Phase 1 systematically evaluates each pipeline component to quantify its contribution to overall performance. We run 7 configurations with identical evaluation protocol:

- **Identical splits:** Same 5-fold Post-ID disjoint splits across all configs
- **Identical candidate pools:** Within-post retrieval only
- **Identical metrics:** Recall@K, nDCG@K, MRR@K (K ∈ {1,3,5,10,20})
- **Fair comparison:** Each config evaluated on same 14,770 queries

---

## Ablation Configurations

### 1. Retriever Only (NV-Embed-v2)

**Components:** Dense retrieval baseline only
**Purpose:** Establish retrieval ceiling without reranking
**Parameters:**
- Retriever: NV-Embed-v2
- Top-K: 10
- No reranking, no GNN

**Expected Performance:**
- Recall@10: ~0.75-0.80 (based on HPO results)
- nDCG@10: ~0.75-0.80
- Establishes upper bound on retrieval quality

### 2. Retriever + Jina Reranker

**Components:** Dense retrieval + cross-encoder reranking
**Purpose:** Quantify reranking improvement over retrieval-only
**Parameters:**
- Retriever: NV-Embed-v2 (top-24)
- Reranker: Jina-Reranker-v3 (select top-10)

**Expected Improvement:**
- +5-10% nDCG@10 vs retriever-only
- Better ranking quality for top-K

### 3. + P3 Graph Reranker

**Components:** Add GNN-based graph reranking
**Purpose:** Quantify benefit of graph-based candidate refinement
**Parameters:**
- All from config 2
- + P3 Graph Reranker (sentence-sentence edges)

**GNN Architecture:**
- Node features: Sentence embeddings + scores
- Edge features: Semantic similarity + position
- Output: Refined ranking scores

**Expected Improvement:**
- +2-5% nDCG@10 vs config 2
- Better handling of context dependencies

### 4. + P2 Dynamic-K Selection

**Components:** Add GNN-based adaptive K selection
**Purpose:** Quantify benefit of query-adaptive evidence count
**Parameters:**
- All from config 3
- + P2 Dynamic-K (K ∈ [3,20])

**Dynamic-K Policy:**
- GNN predicts optimal K per query
- Adapts to evidence density
- Improves precision without hurting recall

**Expected Improvement:**
- +1-3% evidence precision
- Reduced false positives (fewer irrelevant sentences)

### 5. + P4 NE Gate (Fixed K)

**Components:** Add GNN-based no-evidence detection
**Purpose:** Quantify benefit of filtering no-evidence queries
**Parameters:**
- All from config 3 (graph reranker)
- + P4 NE Gate (binary classifier)
- Fixed K=10 (no dynamic-K)

**NE Gate:**
- Binary classifier: has_evidence vs no_evidence
- AUROC: ~0.90
- Filters out queries unlikely to have evidence

**Expected Improvement:**
- Improved system efficiency (skip retrieval for NE queries)
- Better precision on has-evidence queries

### 6. Full Pipeline (All Components)

**Components:** Complete system with all modules
**Purpose:** Establish state-of-the-art performance
**Parameters:**
- Retriever: NV-Embed-v2
- Reranker: Jina-Reranker-v3
- P3 Graph Reranker ✓
- P2 Dynamic-K ✓
- P4 NE Gate ✓
- 3-state gate (NEG/UNCERTAIN/POS) ✓

**Performance (from Phase 0):**
- AUROC (NE detection): 0.8972
- Recall (evidence retrieval): 0.7043
- Precision: 0.2263
- MRR: 0.3801
- Screening sensitivity: 99.78%
- Alert precision: 93.5%

**Status:** ✅ Results available from Phase 0 (can be reused)

### 7. Full Pipeline (Exclude A.10)

**Components:** Same as config 6, exclude suicidal ideation criterion
**Purpose:** Sensitivity analysis - is performance driven by one criterion?
**Parameters:**
- All from config 6
- Exclude A.10 (Suicidal Ideation) from evaluation

**Rationale:**
- A.10 may be easier/harder to detect than other criteria
- Check if overall performance is robust to criterion choice
- Assess generalization across criteria

---

## Evaluation Protocol

### Splits
- **5-fold cross-validation**
- **Post-ID disjoint** (no post in multiple splits)
- **Stratified by has_evidence** (balanced positive rate)
- **Fixed seed (42)** for reproducibility

### Metrics

**Ranking Metrics (queries with evidence):**
- Recall@K: Fraction of gold sentences retrieved in top-K
- nDCG@K: Normalized discounted cumulative gain
- MRR@K: Mean reciprocal rank
- K ∈ {1, 3, 5, 10, 20}

**Classification Metrics (NE gate, configs 5-7 only):**
- AUROC: Area under ROC curve
- AUPRC: Area under PR curve
- Precision, Recall, F1 at optimal threshold

**Clinical Metrics (config 6-7 only):**
- Screening sensitivity (minimize false negatives)
- Alert precision (minimize false positive alerts)
- Workload distribution (NEG/UNCERTAIN/POS rates)

### Output Structure

```
outputs/final_eval/phase1_ablations/
├── 1_retriever_only/
│   ├── fold_0_per_query.csv
│   ├── fold_1_per_query.csv
│   ├── fold_2_per_query.csv
│   ├── fold_3_per_query.csv
│   ├── fold_4_per_query.csv
│   └── summary.json
├── 2_retriever_jina/
│   ├── ...
│   └── summary.json
├── ...
├── ablation_suite_results.json          # Combined results
├── ablation_comparison.csv              # Comparison table
└── PHASE1_ABLATION_STUDY.md            # This file
```

---

## Expected Timeline

| Configuration | Runtime (5-fold CV) | Bottleneck |
|--------------|---------------------|------------|
| 1. Retriever only | 30-60 min | Embedding computation |
| 2. Retriever + Jina | 1-2 hours | Reranker inference |
| 3. + P3 Graph | 2-3 hours | GNN inference |
| 4. + P2 Dynamic-K | 2-3 hours | GNN inference |
| 5. + P4 NE Gate | 2-3 hours | GNN inference |
| 6. Full pipeline | 0 min (reuse) | N/A |
| 7. Exclude A.10 | 2-3 hours | GNN inference |
| **Total** | **10-17 hours** | GPU memory |

**Parallelization:** Configs 1-2 can run in parallel (no GNN), then configs 3-5,7 sequentially.

---

## Key Questions

1. **What is the retrieval ceiling?** (Config 1)
   - How good can dense retrieval alone get?
   - Is there room for reranking improvement?

2. **What is the reranking benefit?** (Config 2 vs 1)
   - How much does Jina-v3 improve over NV-Embed-v2?
   - Is reranking worth the compute cost?

3. **What is the GNN contribution?** (Config 3 vs 2)
   - Does graph structure help?
   - Is the added complexity justified?

4. **What is the Dynamic-K benefit?** (Config 4 vs 3)
   - Does adaptive K improve precision?
   - How much variance is there in optimal K?

5. **What is the NE gate value?** (Config 5 vs 3)
   - Can we skip retrieval for no-evidence queries?
   - What's the efficiency gain?

6. **What is the full system performance?** (Config 6)
   - Do all components work together?
   - Any negative interactions?

7. **Is performance criterion-dependent?** (Config 7 vs 6)
   - Is A.10 easier/harder than other criteria?
   - Does performance generalize?

---

## Success Criteria

1. **Monotonic improvement:** Each added component should improve performance (or at worst be neutral)
2. **Statistical significance:** Improvements should be significant across folds
3. **Component independence:** No negative interactions between components
4. **Reproducibility:** Results match within ±1% when re-run with same seed

---

## Deliverables

1. ✅ Ablation configurations defined (this document)
2. ⏳ Ablation runner script (`scripts/ablation/run_ablation_suite.py`)
3. ⏳ Per-configuration results (`ablation_suite_results.json`)
4. ⏳ Comparison table (`ablation_comparison.csv`)
5. ⏳ Visualization plots:
   - Component contribution bar chart
   - nDCG@K across configs
   - Recall@K across configs
   - Dynamic-K distribution analysis
6. ⏳ Analysis report (Section in `ACADEMIC_EVAL_REPORT.md`)

---

## Next Steps

1. Run configs 1-2 (no GNN required)
2. Run configs 3-5, 7 (requires GNN models)
3. Generate comparison plots
4. Analyze component contributions
5. Update academic report with findings

---

**Status:** Ablation runner implemented, ready for execution
**Last Updated:** 2026-01-18
