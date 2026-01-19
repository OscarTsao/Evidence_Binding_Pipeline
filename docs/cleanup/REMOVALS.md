# Paper Release Cleanup - Removal Summary

**Date:** 2026-01-19
**Branch:** paper_release_cleanup
**Backup:** `research_full_backup` branch, `pre_paper_cleanup_20260119` tag

## Summary

- **Files before:** 345
- **Files after:** 161
- **Reduction:** 184 files removed (53%)

---

## Removed Directories

### 1. `notebook/` (5 files)
**Reason:** Exploratory Jupyter notebooks not required for paper reproduction.
- `sc_pipeline_no_postprocessing.ipynb`
- `sc_post_processing.ipynb`
- `sc_post_processing_executed.ipynb`
- `sc_reranker_pipeline_no_postProcessing.ipynb`
- `sc_retrieval_pipeline.ipynb`

### 2. `scripts/supersearch/` (5 files)
**Reason:** Experimental feature not used in final paper results.
- `build_feature_store.py`
- `build_oof_cache.py`
- `run_stage1_sweep.py`
- `run_supersearch.py`
- `train_noevidence_reranker_fixed.py`

### 3. `scripts/llm/` (10 files)
**Reason:** LLM experiment scripts; core LLM code retained in `src/final_sc_review/llm/`.
- `README.md`
- `execute_phase3_evaluation.py`
- `generate_phase_reports.py`
- `llm_evidence_verifier.py`
- `llm_listwise_reranker.py`
- `run_gemini_evaluation.py`
- `run_llm_evaluation.py`
- `run_llm_evaluation_v2.py`
- `run_llm_phase2_bias_reliability.py`
- `run_phase3_gemini_validation.py`

### 4. `scripts/llm_integration/` (3 files)
**Reason:** Development scripts for LLM integration.
- `analyze_dev_reranker.py`
- `analyze_dev_reranker_v2.py`
- `run_llm_pilot.py`

### 5. `scripts/reranker/` (7 files)
**Reason:** Development/training scripts superseded by zoo implementation.
- `benchmark_optimized.py`
- `build_candidates.py`
- `eval_rerankers.py`
- `hpo_inference.py`
- `hpo_inference_fast.py`
- `train_all_rerankers.py`
- `train_maxout.py`

### 6. `scripts/retriever/` (10 files)
**Reason:** Development scripts superseded by zoo implementation.
- `build_cache.py`
- `build_candidates.py`
- `check_gold_alignment.py`
- `check_retriever_coverage.py`
- `compare_all_retrievers.py`
- `dev_select_eval.py`
- `eval_nv_embed_v2.py`
- `hpo_frozen.py`
- `model_zoo_smoke_test.py`
- `retriever_driver.py`

### 7. `scripts/visualization/` (2 files)
**Reason:** Superseded by `scripts/verification/generate_publication_plots.py`.
- `plot_baseline_metrics.py`
- `plot_per_criterion.py`

### 8. `docs/plans/` (3 files)
**Reason:** Research planning artifacts, not needed for reproduction.
- `MAXOUT_PLAN_0109.md`
- `reranker_research_plan.md`
- `retriever_research_plan.md`

### 9. `docs/llm_integration/` (3 files)
**Reason:** Development planning docs, superseded by final reports.
- `IMPLEMENTATION_PLAN.md`
- `MULTI_KEY_SETUP.md`
- `README.md`

### 10. `src/final_sc_review/supersearch/` (2 files)
**Reason:** Experimental feature code, not used in final pipeline.
- `__init__.py`
- `registry.py`

---

## Removed Root Files (4 files)

| File | Reason |
|------|--------|
| `AUDIT_COMPLETE_SUMMARY.md` | Superseded by `docs/final/` reports |
| `FINAL_AUDIT_SUMMARY.md` | Superseded by `docs/final/` reports |
| `PROJECT_EXECUTION_PLAN.md` | Research planning artifact |
| `run_registry.csv` | Development tracking file |

---

## Removed Individual Scripts (~50 files)

### Analysis/Debug Scripts
- `scripts/analyze_diagnostics.py`
- `scripts/assess_no_evidence_model.py`
- `scripts/compute_ceiling_from_queries.py`
- `scripts/gold_count_stats.py`
- `scripts/gpu_accumulate.py`
- `scripts/gpu_time_tracker.py`
- `scripts/post_length_stats.py`
- `scripts/analysis/per_criterion_analysis.py`
- `scripts/clinical/debug_alert_precision.py`

### Superseded Audit Scripts
- `scripts/audit_label_leakage_dynamic.py` → use `verification/audit_splits_and_leakage.py`
- `scripts/audit_label_leakage_static.py` → use `verification/audit_splits_and_leakage.py`
- `scripts/audit_pushed_results.py`

### Deprecated Evaluation Scripts
- `scripts/eval_decision_gates.py`
- `scripts/eval_ensemble_stacking.py`
- `scripts/eval_finetuned_retriever.py`
- `scripts/eval_inference_baselines.py`
- `scripts/eval_multiquery.py`
- `scripts/eval_no_evidence_hybrid.py`
- `scripts/eval_no_evidence_reranker.py`
- `scripts/eval_nv_embed_wrapper.sh`
- `scripts/eval_postprocessing.py`
- `scripts/eval_reranker_comparison.py`
- `scripts/eval_retriever_reranker_combinations.py`
- `scripts/eval_retriever_zoo.py`
- `scripts/eval_with_cached_embeddings.py`
- `scripts/eval_with_precomputed_embeddings.py`

### HPO Variant Scripts (superseded by `hpo_inference.py`)
- `scripts/hpo_dynamic_k.py`
- `scripts/hpo_ensemble_classifier.py`
- `scripts/hpo_finetuning_combo.py`
- `scripts/hpo_ne_gate.py`
- `scripts/hpo_nv_embed_retriever.py`
- `scripts/hpo_per_criterion_thresholds.py`
- `scripts/hpo_postprocessing.py`
- `scripts/hpo_postprocessing_on_cached.py`
- `scripts/hpo_postprocessing_on_reranker.py`
- `scripts/hpo_retriever_reranker_combinations.py`
- `scripts/hpo_training.py`
- `scripts/hpo_training_v2.py`
- `scripts/precompute_hpo_cache.py`
- `scripts/reranker_hpo.py`
- `scripts/retriever_hpo.py`

### Development/Research Scripts
- `scripts/build_multiquery_templates.py`
- `scripts/create_splits.py`
- `scripts/encode_corpus_nv_embed.py`
- `scripts/encode_queries_nv_embed.py`
- `scripts/generate_training_data.py`
- `scripts/research_driver.py`
- `scripts/retrieval_ceiling.py`
- `scripts/validate_runs.py`
- `scripts/verify_invariants.py`
- `scripts/verify_phase.py`

### Training Scripts (model training, not inference)
- `scripts/train_reranker_multi_seed.py`
- `scripts/train_reranker_with_no_evidence.py`
- `scripts/train_retriever.py`

### Shell Scripts (orchestration/cluster)
- `scripts/run_all_finetuning_hpo.py`
- `scripts/run_all_incomplete_hpo.sh`
- `scripts/run_deployment_assessment.py`
- `scripts/run_full_assessment.py`
- `scripts/run_full_optimization_pipeline.py`
- `scripts/run_full_reranker_research.sh`
- `scripts/run_incomplete_finetuning_hpo.sh`
- `scripts/run_incomplete_hpo.sh`
- `scripts/run_inference_hpo_all_combos.py`
- `scripts/run_postproc_optimization.sh`
- `scripts/run_research_pipeline.py`
- `scripts/slurm_hpo_worker.sbatch`

### GNN Development Scripts (keep core eval scripts)
- `scripts/gnn/ablation_study.py`
- `scripts/gnn/build_graph_dataset.py`
- `scripts/gnn/build_oof_cache_from_embeddings.py`
- `scripts/gnn/debug_dynamic_k_sanity.py`
- `scripts/gnn/make_gnn_e2e_plots.py`
- `scripts/gnn/run_all_e2e_report.sh`
- `scripts/gnn/run_graph_reranker.py`
- `scripts/gnn/run_graph_stats_baselines.py`

### Verification Scripts (superseded)
- `scripts/verification/a10_investigation.py`
- `scripts/verification/ablation_analysis.py`
- `scripts/verification/build_deployable_features.py`
- `scripts/verification/run_clean_supersearch_eval.py`
- `scripts/verification/stage_summary.py`
- `scripts/verification/test_ranking_crosscheck.py`

---

## Removed Configs (26 files)

### HPO Configs
- `configs/hpo_inference.yaml`
- `configs/hpo_inference_exhaustive.yaml`
- `configs/hpo_inference_v2.yaml`
- `configs/hpo_retriever_only.yaml`
- `configs/hpo_training.yaml`
- `configs/hpo_training_maxout.yaml`
- `configs/hpo_training_v2.yaml`

### Development Configs
- `configs/ablation_large_pool.yaml`
- `configs/best_v2.yaml`
- `configs/bge_m3_config.yaml`
- `configs/budgets_maxout.yaml`
- `configs/default_v2.yaml`
- `configs/default_val.yaml`
- `configs/default_val_optimized.yaml`
- `configs/deployment_low_fpr.yaml`
- `configs/deployment_targets.yaml`
- `configs/locked_best_stageC_reranker.yaml`
- `configs/locked_best_stageE_deploy.yaml`
- `configs/model_lists.yaml`
- `configs/reranker_extended.yaml`
- `configs/reranker_hybrid.yaml`
- `configs/reranker_with_no_evidence.yaml`
- `configs/retriever_finetune.yaml`
- `configs/retriever_zoo.yaml`
- `configs/sota_baselines.yaml`
- `configs/stageB_best.yaml`
- `configs/training_data.yaml`

---

## Removed Documentation (25 files)

### Superseded Status/Summary Docs
- `docs/CURRENT_PIPELINE.md`
- `docs/FINAL_COMPLETE_SUMMARY.md`
- `docs/LLM_INTEGRATION_COMPLETE.md`
- `docs/LLM_PHASES_1_4_COMPLETE.md`
- `docs/MODEL_INVENTORY.md`
- `docs/PLAN.md`
- `docs/PROJECT_STATUS.md`
- `docs/metric_policy.md`
- `docs/optimization.md`
- `docs/optimization_plan_baseline.md`
- `docs/retriever_comparison_results.md`
- `docs/retriever_lists.md`

### Clinical Implementation Docs (keep final report)
- `docs/clinical/COMPLETION_REPORT.md`
- `docs/clinical/FINAL_INTEGRATION_SUMMARY.md`
- `docs/clinical/IMPLEMENTATION_STATUS.md`
- `docs/clinical/IMPLEMENTATION_SUMMARY.md`

### Eval Planning Docs
- `docs/eval/ABLATION_STUDY_DESIGN.md`
- `docs/eval/PRODUCTION_READINESS_CHECKLIST.md`

### GNN Planning Docs
- `docs/gnn/SOTA_RESEARCH_AND_PLAN.md`
- `docs/gnn/graph_data_spec.md`

### Verification Notes
- `docs/verification/reproduce_supersearch.md`
- `docs/verification/research_notes_ne_and_dynk.md`

---

## Kept Files Summary

### Core Source Code (src/)
- All core pipeline modules retained
- All GNN modules retained
- All clinical modules retained
- All metrics/evaluation modules retained
- Removed: `supersearch/` (experimental)

### Scripts
- Data preparation: `build_groundtruth.py`, `build_sentence_corpus.py`
- Main evaluation: `eval_zoo_pipeline.py`, `eval_sc_pipeline.py`
- Single inference: `run_single.py`, `run_single_zoo.py`
- Final evaluation: `final_eval.py`, `final_test_evaluation.py`
- Clinical: `clinical/run_clinical_high_recall_eval.py`, `clinical/generate_plots.py`
- Verification: 8 key verification scripts
- Ablation: `ablation/run_ablation_study.py`, `ablation/run_ablation_suite.py`
- GNN: 6 key GNN evaluation scripts
- HPO: `hpo_inference.py`, `export_best_config.py`, `launch_hpo_multi_gpu.py`
- Academic: `run_full_academic_evaluation.py`
- Training: `train_reranker_hybrid.py`
- NEW: `reporting/package_paper_bundle.py`, `run_paper_reproduce.sh`

### Configs (4 files)
- `default.yaml` - Main configuration
- `locked_best_config.yaml` - Best HPO configuration
- `deployment_high_recall.yaml` - Clinical deployment
- `final_test_eval.yaml` - Test evaluation

### Tests (All retained)
- All 22 test files kept for research integrity verification

### Documentation
- `docs/final/` - All final reports
- `docs/verification/` - Verification reports and figures
- `docs/clinical/` - Final clinical report and quick start
- `docs/gnn/` - GNN final reports
- `docs/eval/` - Metrics contract and academic report
- `docs/cleanup/` - Cleanup documentation (new)
