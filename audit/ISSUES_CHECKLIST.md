# Publication Readiness Audit - Issues Checklist

**Generated:** 2026-01-20
**Updated:** 2026-01-20
**Commit:** d64ffbe

---

## PHASE 0: Audit Results

### A.10 Definition Issues (CRITICAL - FIXED)

**Correct Definition (per ReDSM5 & constants.py):**
- A.10 = SPECIAL_CASE (expert discrimination cases; non-DSM-5 clinical/positive discriminations)

**Files FIXED:**

| File | Status |
|------|--------|
| `configs/criteria_registry.yaml` | FIXED - now SPECIAL_CASE |
| `data/DSM5/MDD_Criteira.json` | FIXED - now SPECIAL_CASE |
| `tests/test_criteria_registry.py` | FIXED - tests for SPECIAL_CASE |
| `scripts/analysis/error_analysis.py` | FIXED - now SPECIAL_CASE |
| `scripts/gnn/rebuild_graph_cache.py` | FIXED - now SPECIAL_CASE |
| `scripts/gnn/rebuild_graph_cache_phase2.py` | FIXED - now SPECIAL_CASE |
| `results/paper_bundle/v2.0/metrics_master.json` | FIXED - now SPECIAL_CASE |
| `results/paper_bundle/v3.0/metrics_master.json` | FIXED - now SPECIAL_CASE |
| `results/paper_bundle/v2.0/tables/per_criterion.csv` | FIXED - now SPECIAL_CASE |
| `results/paper_bundle/v3.0/tables/per_criterion.csv` | FIXED - now SPECIAL_CASE |
| `results/paper_bundle/v2.0/reproduction/REPRODUCTION_REPORT.md` | FIXED - now SPECIAL_CASE |
| `docs/final/PUBLICATION_READINESS_REPORT.md` | FIXED - now SPECIAL_CASE |
| `docs/submission/REVIEWER_FAQ.md` | FIXED - now SPECIAL_CASE |

**Lint Test Added:**
- `tests/test_a10_consistency.py` - Prevents A.10 definition drift

**Canonical Source (CORRECT - keep as-is):**
- `src/final_sc_review/constants.py:13` - `"A.10": "SPECIAL_CASE"` ✓

---

### Simulation/Placeholder Code Issues

| File | Line | Issue | Action |
|------|------|-------|--------|
| `scripts/analysis/efficiency_metrics.py` | 132-137 | `mock_load()` with simulated timing | Remove or replace with real measurement |
| `scripts/analysis/efficiency_metrics.py` | 183 | "Simulated throughput calculation" | Replace with real measurement |
| `scripts/clinical/generate_plots.py` | 46,61,72 | Placeholder ROC/PR curves | Generate from real data |
| `scripts/verification/compute_ablation_ci.py` | 108 | "placeholder for CI" | Implement real CI computation |

---

### Missing Baselines

**Required (must implement):**
- [x] BM25 - exists in `src/final_sc_review/baselines/base.py`
- [x] TF-IDF - exists in `src/final_sc_review/baselines/base.py`
- [x] E5-base - exists in `src/final_sc_review/baselines/base.py`
- [x] Contriever - exists in `src/final_sc_review/baselines/base.py`
- [ ] BGE - NOT IMPLEMENTED
- [ ] Cross-encoder reranker baseline - NOT IMPLEMENTED
- [ ] No-retrieval LLM baseline - NOT IMPLEMENTED
- [ ] Linear model baseline (LogReg/SVM on TF-IDF) - NOT IMPLEMENTED

---

### Metric Issues

**Required metrics not found:**
- [ ] Precision@k verification
- [ ] MAP@k implementation verification
- [ ] Evidence coverage metric
- [ ] Micro/Macro F1 for multi-label
- [ ] Reliability diagram generation

---

## Action Items

### Phase 1: Fix A.10 Definition (BLOCKER) - COMPLETE
- [x] Update `configs/criteria_registry.yaml`
- [x] Update `data/DSM5/MDD_Criteira.json`
- [x] Update `tests/test_criteria_registry.py`
- [x] Update `scripts/analysis/error_analysis.py`
- [x] Update `scripts/gnn/rebuild_graph_cache.py`
- [x] Update `scripts/gnn/rebuild_graph_cache_phase2.py`
- [x] Update paper bundles with correct A.10
- [x] Add lint test for A.10 consistency (`tests/test_a10_consistency.py`)

### Phase 2: Remove Simulations
- [ ] Fix `scripts/analysis/efficiency_metrics.py`
- [ ] Fix `scripts/clinical/generate_plots.py`
- [ ] Fix `scripts/verification/compute_ablation_ci.py`

### Phase 3: Add Missing Baselines
- [ ] Add BGE baseline
- [ ] Add cross-encoder reranker baseline
- [ ] Add no-retrieval LLM baseline
- [ ] Add linear model baseline

### Phase 4: Complete Metrics
- [ ] Verify all metrics implemented
- [ ] Add missing metrics
- [ ] Create metrics registry

### Phase 5: Ablations
- [ ] Verify ablation suite complete
- [ ] Add missing ablations

### Phase 6: Paper Bundle
- [x] Regenerate with correct A.10
- [ ] Add CITATION.cff check
- [ ] Verify reproducibility guide

### Phase 7: Final Validation
- [ ] All tests pass
- [x] A.10 = SPECIAL_CASE everywhere
- [ ] No simulated metrics
- [ ] Baselines complete
