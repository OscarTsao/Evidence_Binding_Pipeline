# Cleanup Plan

**Date:** 2026-01-20
**Purpose:** Deep cleanup for 10/10 production readiness

---

## Files to PRESERVE (Essential)

### Source Code (src/)
- [x] src/final_sc_review/data/ (data loading, splits, I/O)
- [x] src/final_sc_review/retriever/ (NV-Embed-v2, BGE-M3, zoo)
- [x] src/final_sc_review/reranker/ (Jina-v3, zoo)
- [x] src/final_sc_review/gnn/ (P1-P4 models)
- [x] src/final_sc_review/metrics/ (ranking metrics)
- [x] src/final_sc_review/pipeline/ (ZooPipeline, ThreeStage)
- [x] src/final_sc_review/llm/ (LLM integration)
- [x] src/final_sc_review/postprocessing/ (calibration, dynamic-k)
- [x] src/final_sc_review/utils/ (logging, hashing, seed)
- [x] src/final_sc_review/clinical/ (three-state gate)

### Tests (tests/)
- [x] tests/metrics/test_ranking_metrics.py
- [x] tests/leakage/test_general_leakage.py
- [x] tests/clinical/test_no_leakage.py
- [x] tests/test_*.py (21 test files)

### Critical Scripts (referenced in docs/tests)
- [x] scripts/build_groundtruth.py
- [x] scripts/build_sentence_corpus.py
- [x] scripts/eval_zoo_pipeline.py
- [x] scripts/audit_splits.py
- [x] scripts/encode_corpus.py
- [x] scripts/verification/verify_checksums.py
- [x] scripts/verification/metric_crosscheck.py
- [x] scripts/gnn/train_eval_hetero_graph.py
- [x] scripts/ablation/run_ablation_study.py

### Validated Results (SOURCE OF TRUTH)
- [x] outputs/final_research_eval/20260118_031312_complete/per_query.csv
  - 14,770 query predictions with all metrics
  - This file generates ALL reported metrics

### Configuration
- [x] envs/requirements-retriever.txt
- [x] envs/requirements-main.txt
- [x] pyproject.toml
- [x] configs/default.yaml

### Git/CI
- [x] .git/
- [x] .gitignore
- [x] .github/workflows/ci.yml

---

## Files to DELETE

### Documentation (will regenerate)
- [x] README.md
- [x] CONTRIBUTING.md
- [x] CLAUDE.md
- [x] CITATION.cff
- [x] LICENSE
- [x] docs/* (entire directory)
- [x] paper/* (entire directory)

### Results/Artifacts (will regenerate)
- [x] results/paper_bundle/* (v1.0 - will create v2.0)
- [x] results/figures/
- [x] results/tables/
- [x] results/audit/

### Outputs (keep only per_query.csv)
- [x] outputs/hpo_cache/ (2.4GB - cache only)
- [x] outputs/dual_track/ (347MB)
- [x] outputs/gnn_research_nvembed/ (150MB)
- [x] outputs/gnn_research/ (100MB - keep cv_results.json)
- [x] outputs/e2e_full_eval/
- [x] outputs/clinical_high_recall/
- [x] outputs/reproduction/
- [x] outputs/version_a_full_audit/
- [x] outputs/llm_dev_eval/
- [x] outputs/final_eval/
- [x] outputs/llm_eval/
- [x] outputs/audit/
- [x] outputs/repro_baseline/
- [x] outputs/analysis/
- [x] outputs/audit_full_eval/
- [x] outputs/gnn_e2e_report/
- [x] outputs/llm_pilot/
- [x] outputs/ablation/ (empty results)
- [x] outputs/hpo_inference_combos/ (keep full_results.csv)

### Cache/Temporary
- [x] data/cache/*
- [x] .pytest_cache/
- [x] **/__pycache__/
- [x] artifacts/ (if exists)

### Unused Scripts (not referenced)
- [x] scripts/train_reranker_hybrid.py
- [x] scripts/launch_hpo_multi_gpu.py
- [x] scripts/run_full_academic_evaluation.py
- [x] scripts/final_test_evaluation.py
- [x] scripts/final_eval.py
- [x] scripts/export_best_config.py
- [x] scripts/hpo_inference.py
- [x] scripts/run_single.py (legacy)
- [x] scripts/eval_sc_pipeline.py (legacy)

---

## Files to CREATE (Fresh)

### Core Documentation
- [ ] README.md (complete project overview)
- [ ] CONTRIBUTING.md (developer guide)
- [ ] LICENSE (MIT)
- [ ] CITATION.cff (proper metadata)
- [ ] CLAUDE.md (AI assistant context)

### Research Documentation (docs/)
- [ ] docs/ARCHITECTURE.md (system design)
- [ ] docs/DATA_AVAILABILITY.md (access procedures)
- [ ] docs/ETHICS.md (privacy + IRB)
- [ ] docs/ENVIRONMENT_SETUP.md (dual conda)
- [ ] docs/REPRODUCIBILITY.md (full guide)
- [ ] docs/METRIC_CONTRACT.md (definitions)

### Paper Bundle v2.0
- [ ] results/paper_bundle/v2.0/report.md
- [ ] results/paper_bundle/v2.0/metrics_master.json
- [ ] results/paper_bundle/v2.0/summary.json
- [ ] results/paper_bundle/v2.0/tables/main_results.csv
- [ ] results/paper_bundle/v2.0/tables/per_criterion.csv
- [ ] results/paper_bundle/v2.0/tables/ablation.csv
- [ ] results/paper_bundle/v2.0/figures/*.png
- [ ] results/paper_bundle/v2.0/checksums.txt
- [ ] results/paper_bundle/v2.0/MANIFEST.md

---

## Execution Order

1. Backup critical files (per_query.csv, full_results.csv, cv_results.json)
2. Delete all marked files
3. Regenerate documentation
4. Regenerate paper bundle
5. Run tests
6. Verify checksums
7. Commit and push
