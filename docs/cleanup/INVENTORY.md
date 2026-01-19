# Repository Inventory for Paper Release Cleanup

**Date:** 2026-01-19
**Branch:** paper_release_cleanup
**Total tracked files:** 345

## Summary

This inventory classifies all tracked files for paper release cleanup.

### Classification Legend
- **KEEP** - Paper-critical, required for reproduction
- **KEEP_OPTIONAL** - Nice to have but not strictly required
- **REMOVE** - Unused, exploratory, intermediate, or superseded

---

## 1. Root Files

| File | Size | Classification | Reason |
|------|------|----------------|--------|
| `.gitignore` | 0.3K | KEEP | Required for repo |
| `pyproject.toml` | 0.8K | KEEP | Package definition |
| `README.md` | 2.2K | KEEP | Entry point documentation |
| `CLAUDE.md` | 8.0K | KEEP | Development guidance |
| `AUDIT_COMPLETE_SUMMARY.md` | 4K | REMOVE | Superseded by final reports |
| `FINAL_AUDIT_SUMMARY.md` | 6K | REMOVE | Superseded by docs/final/ |
| `PROJECT_EXECUTION_PLAN.md` | 8K | REMOVE | Research planning artifact |
| `run_registry.csv` | 2K | REMOVE | Development tracking |

---

## 2. Source Code (src/)

### 2.1 Core Pipeline (KEEP ALL)
```
src/final_sc_review/
├── __init__.py
├── constants.py
├── data/                    # KEEP - Data loading and schemas
│   ├── __init__.py
│   ├── io.py
│   ├── schemas.py
│   └── splits.py
├── metrics/                 # KEEP - Evaluation metrics
│   ├── __init__.py
│   ├── k_policy.py
│   ├── ranking.py
│   └── retrieval_eval.py
├── pipeline/                # KEEP - Core pipelines
│   ├── __init__.py
│   ├── run.py
│   ├── three_stage.py
│   └── zoo_pipeline.py
├── retriever/               # KEEP - Retriever implementations
│   ├── __init__.py
│   ├── bge_m3.py
│   ├── hybrid.py
│   └── zoo.py
├── reranker/                # KEEP - Reranker implementations
│   ├── __init__.py
│   ├── dataset.py
│   ├── jina_v3.py
│   ├── losses.py
│   ├── maxout_trainer.py
│   ├── optimized_inference.py
│   ├── trainer.py
│   └── zoo.py
├── postprocessing/          # KEEP - Post-processing modules
│   ├── __init__.py
│   ├── calibration.py
│   ├── dynamic_k.py
│   └── no_evidence.py
├── clinical/                # KEEP - Clinical deployment
│   ├── __init__.py
│   ├── config.py
│   ├── dynamic_k.py
│   ├── metrics_reference.py
│   ├── model_inference.py
│   └── three_state_gate.py
├── gnn/                     # KEEP - GNN models (paper contribution)
│   ├── __init__.py
│   ├── config.py
│   ├── evaluation/
│   ├── graphs/
│   ├── models/
│   └── training/
├── llm/                     # KEEP_OPTIONAL - LLM integration
│   ├── __init__.py
│   ├── a10_classifier.py
│   ├── base.py
│   ├── data_loader.py
│   ├── gemini_client.py
│   ├── hybrid_pipeline.py
│   ├── reranker.py
│   └── verifier.py
├── hpo/                     # KEEP_OPTIONAL - HPO utilities
│   ├── __init__.py
│   ├── cache_builder.py
│   ├── constraints.py
│   ├── multi_gpu.py
│   ├── objective_inference.py
│   ├── objective_training.py
│   ├── objective_training_v2.py
│   ├── reporting.py
│   ├── search_space.py
│   └── storage.py
├── training/                # KEEP_OPTIONAL - Training utilities
│   ├── __init__.py
│   └── hard_negative_miner.py
├── supersearch/             # REMOVE - Experimental feature
│   ├── __init__.py
│   └── registry.py
└── utils/                   # KEEP - Utility functions
    ├── __init__.py
    ├── gpu_optimize.py
    ├── hashing.py
    ├── logging.py
    ├── seed.py
    └── text.py
```

---

## 3. Scripts (scripts/)

### 3.1 KEEP - Paper-Critical Scripts
```
scripts/
├── build_groundtruth.py           # Data preparation
├── build_sentence_corpus.py       # Data preparation
├── eval_zoo_pipeline.py           # Main evaluation
├── eval_sc_pipeline.py            # Legacy evaluation
├── run_single.py                  # Single query inference
├── run_single_zoo.py              # Single query (zoo)
├── final_eval.py                  # Final evaluation
├── final_test_evaluation.py       # Test set evaluation
├── clinical/
│   ├── run_clinical_high_recall_eval.py  # Clinical evaluation
│   └── generate_plots.py                 # Clinical plots
├── verification/
│   ├── metric_crosscheck.py              # Independent metric verification
│   ├── audit_splits_and_leakage.py       # Split/leakage audit
│   ├── generate_publication_plots.py     # Paper figures
│   ├── full_pipeline_eval.py             # Full pipeline eval
│   └── recompute_metrics_from_csv.py     # Metric recomputation
├── ablation/
│   ├── run_ablation_study.py             # Ablation study
│   └── run_ablation_suite.py             # Ablation suite
├── gnn/
│   ├── run_e2e_eval_and_report.py        # GNN E2E evaluation
│   ├── train_eval_hetero_graph.py        # P4 hetero GNN
│   ├── train_eval_ne_gnn.py              # P1 NE gate
│   ├── eval_dynamic_k_gnn.py             # P2 dynamic-K
│   ├── generate_verification_plots.py   # GNN plots
│   └── recompute_metrics_independent.py  # Independent metrics
└── audit_splits.py                       # Split audit
```

### 3.2 KEEP_OPTIONAL - HPO/Training Scripts
```
scripts/
├── hpo_inference.py
├── export_best_config.py
├── precompute_hpo_cache.py
├── launch_hpo_multi_gpu.py
├── train_reranker_hybrid.py
├── train_retriever.py
└── run_full_academic_evaluation.py
```

### 3.3 REMOVE - Experimental/Unused Scripts
```
scripts/
├── analyze_diagnostics.py                # Debug utility
├── assess_no_evidence_model.py           # Exploratory
├── audit_label_leakage_dynamic.py        # Superseded by verification/
├── audit_label_leakage_static.py         # Superseded by verification/
├── audit_pushed_results.py               # Dev utility
├── build_multiquery_templates.py         # Unused experiment
├── compute_ceiling_from_queries.py       # Analysis utility
├── create_splits.py                      # One-time utility
├── encode_corpus_nv_embed.py             # Superseded by zoo
├── encode_queries_nv_embed.py            # Superseded by zoo
├── eval_decision_gates.py                # Exploratory
├── eval_ensemble_stacking.py             # Unused experiment
├── eval_finetuned_retriever.py           # Superseded
├── eval_inference_baselines.py           # Dev evaluation
├── eval_multiquery.py                    # Unused experiment
├── eval_no_evidence_hybrid.py            # Exploratory
├── eval_no_evidence_reranker.py          # Exploratory
├── eval_nv_embed_wrapper.sh              # Dev utility
├── eval_postprocessing.py                # Superseded
├── eval_reranker_comparison.py           # Dev evaluation
├── eval_retriever_reranker_combinations.py  # Dev evaluation
├── eval_retriever_zoo.py                 # Dev evaluation
├── eval_with_cached_embeddings.py        # Dev utility
├── eval_with_precomputed_embeddings.py   # Dev utility
├── generate_training_data.py             # Training utility
├── gold_count_stats.py                   # Analysis
├── gpu_accumulate.py                     # Dev utility
├── gpu_time_tracker.py                   # Dev utility
├── hpo_*.py (multiple)                   # Superseded HPO variants
├── post_length_stats.py                  # Analysis
├── research_driver.py                    # Dev orchestration
├── retrieval_ceiling.py                  # Analysis
├── run_ablations.py                      # Superseded
├── run_all_*.py                          # Dev orchestration
├── run_deployment_assessment.py          # Dev evaluation
├── run_full_assessment.py                # Dev evaluation
├── run_full_optimization_pipeline.py     # Dev pipeline
├── run_inference_hpo_all_combos.py       # Dev HPO
├── run_research_pipeline.py              # Dev pipeline
├── slurm_hpo_worker.sbatch               # Cluster script
├── train_reranker_multi_seed.py          # Dev training
├── train_reranker_with_no_evidence.py    # Exploratory
├── validate_runs.py                      # Dev validation
├── verify_invariants.py                  # Dev utility
├── verify_phase.py                       # Dev utility
├── llm/                                  # LLM experiments
├── llm_integration/                      # LLM experiments
├── reranker/                             # Dev reranker scripts
├── retriever/                            # Dev retriever scripts
├── supersearch/                          # Experimental
└── visualization/                        # Superseded by verification/
```

---

## 4. Configs (configs/)

### 4.1 KEEP - Paper-Critical Configs
```
configs/
├── default.yaml                   # Main config
├── locked_best_config.yaml        # Best HPO config
├── deployment_high_recall.yaml    # Clinical config
└── final_test_eval.yaml           # Test evaluation config
```

### 4.2 REMOVE - Development/Experimental Configs
```
configs/
├── ablation_large_pool.yaml
├── best_v2.yaml
├── bge_m3_config.yaml
├── budgets_maxout.yaml
├── default_v2.yaml
├── default_val.yaml
├── default_val_optimized.yaml
├── deployment_low_fpr.yaml
├── deployment_targets.yaml
├── hpo_inference.yaml
├── hpo_inference_exhaustive.yaml
├── hpo_inference_v2.yaml
├── hpo_retriever_only.yaml
├── hpo_training.yaml
├── hpo_training_maxout.yaml
├── hpo_training_v2.yaml
├── locked_best_stageC_reranker.yaml
├── locked_best_stageE_deploy.yaml
├── model_lists.yaml
├── reranker_extended.yaml
├── reranker_hybrid.yaml
├── reranker_with_no_evidence.yaml
├── retriever_finetune.yaml
├── retriever_zoo.yaml
├── sota_baselines.yaml
├── stageB_best.yaml
└── training_data.yaml
```

---

## 5. Tests (tests/)

### 5.1 KEEP - Critical Tests
```
tests/
├── test_no_leakage_splits.py          # Split integrity
├── test_candidate_pool_within_post_only.py  # Retrieval constraint
├── test_hpo_never_uses_test_split.py  # HPO integrity
├── test_gnn_no_leakage.py             # GNN leakage
├── test_no_leaky_features.py          # Feature leakage
├── test_metrics.py                    # Metric correctness
├── test_dynamic_k_caps.py             # Dynamic-K constraints
├── test_dynamic_k_gamma_effect.py     # Dynamic-K behavior
├── test_postprocessing.py             # Post-processing
├── clinical/test_no_leakage.py        # Clinical leakage
├── leakage/test_no_leakage.py         # Leakage tests
├── metrics/test_ranking_metrics.py    # Metric tests
└── verification/test_split_postid_disjoint.py  # Split verification
```

### 5.2 KEEP_OPTIONAL - Utility Tests
```
tests/
├── test_cache_fingerprint.py
├── test_cache_key_stability.py
├── test_decoupled_pool_sizes.py
├── test_fixed_k_behavior.py
├── test_hpo_cache_determinism.py
├── test_hpo_objective_requires_gold.py
├── test_id_mapping_no_collision.py
├── test_no_evidence_candidate.py
└── test_sentence_splitting.py
```

---

## 6. Documentation (docs/)

### 6.1 KEEP - Final Reports
```
docs/
├── final/
│   ├── ACADEMIC_EVAL_REPORT.md
│   ├── PHASE1_ABLATION_STUDY.md
│   ├── PHASE2_LLM_INTEGRATION.md
│   └── PRODUCTION_READINESS.md
├── verification/
│   ├── COMPREHENSIVE_VERIFICATION_REPORT.md
│   ├── FINAL_REPORT_leakage_verification.md
│   ├── VERIFICATION_SUMMARY.md
│   └── figures/ (all .png files)
├── clinical/
│   ├── FINAL_CLINICAL_HIGH_RECALL_REPORT.md
│   ├── QUICK_START.md
│   └── README.md
├── gnn/
│   ├── GNN_E2E_FINAL_REPORT.md
│   ├── GNN_FINAL_REPORT.md
│   └── METRICS_SPEC_E2E.md
└── eval/
    ├── FINAL_ACADEMIC_REPORT.md
    └── METRICS_CONTRACT.md
```

### 6.2 REMOVE - Superseded/Planning Docs
```
docs/
├── CURRENT_PIPELINE.md              # Superseded
├── FINAL_COMPLETE_SUMMARY.md        # Redundant
├── LLM_INTEGRATION_COMPLETE.md      # Superseded
├── LLM_PHASES_1_4_COMPLETE.md       # Superseded
├── MODEL_INVENTORY.md               # Dev reference
├── PLAN.md                          # Planning artifact
├── PROJECT_STATUS.md                # Status tracking
├── metric_policy.md                 # Internal policy
├── optimization.md                  # Dev notes
├── optimization_plan_baseline.md    # Planning
├── retriever_comparison_results.md  # Dev analysis
├── retriever_lists.md               # Dev reference
├── clinical/
│   ├── COMPLETION_REPORT.md
│   ├── FINAL_INTEGRATION_SUMMARY.md
│   ├── IMPLEMENTATION_STATUS.md
│   └── IMPLEMENTATION_SUMMARY.md
├── eval/
│   ├── ABLATION_STUDY_DESIGN.md
│   └── PRODUCTION_READINESS_CHECKLIST.md
├── gnn/
│   ├── SOTA_RESEARCH_AND_PLAN.md
│   └── graph_data_spec.md
├── llm_integration/                 # All files - dev planning
├── plans/                           # All files - planning
└── verification/
    ├── reproduce_supersearch.md
    └── research_notes_ne_and_dynk.md
```

---

## 7. Notebooks (notebook/)

| File | Classification | Reason |
|------|----------------|--------|
| `sc_pipeline_no_postprocessing.ipynb` | REMOVE | Exploratory |
| `sc_post_processing.ipynb` | REMOVE | Exploratory |
| `sc_post_processing_executed.ipynb` | REMOVE | Exploratory |
| `sc_reranker_pipeline_no_postProcessing.ipynb` | REMOVE | Exploratory |
| `sc_retrieval_pipeline.ipynb` | REMOVE | Exploratory |

**Decision:** Remove entire notebook/ directory - not required for paper reproduction.

---

## 8. File Size Summary

Large directories (on disk, not in git):
- `data/cache/` = 8.5G (embedding caches - in .gitignore)
- `outputs/hpo_cache/` = 2.4G (HPO caches - in .gitignore)
- `outputs/gnn_research_nvembed/` = 150M (in .gitignore)
- `outputs/gnn_research/` = 100M (in .gitignore)
- `outputs/dual_track/` = 347M (in .gitignore)

Git repo size: ~4MB (tracked files only)

---

## 9. Removal Summary

### Directories to Remove
1. `notebook/` - All notebooks (exploratory)
2. `scripts/supersearch/` - Experimental feature
3. `scripts/llm/` - LLM experiments (keep src/llm/ only)
4. `scripts/llm_integration/` - LLM dev scripts
5. `scripts/reranker/` - Dev reranker scripts
6. `scripts/retriever/` - Dev retriever scripts
7. `scripts/visualization/` - Superseded by verification/
8. `docs/plans/` - Planning artifacts
9. `docs/llm_integration/` - Dev planning
10. `src/final_sc_review/supersearch/` - Experimental

### Files to Remove (Root)
- `AUDIT_COMPLETE_SUMMARY.md`
- `FINAL_AUDIT_SUMMARY.md`
- `PROJECT_EXECUTION_PLAN.md`
- `run_registry.csv`

### Configs to Remove
- 26 of 30 configs (keep 4 paper-critical)

### Scripts to Remove
- ~60 scripts (keep ~25 paper-critical)

### Docs to Remove
- ~25 doc files (keep ~15 final reports)

---

## 10. Final Keep List

### Essential for Paper Reproduction
1. **src/** - All core modules (minus supersearch/)
2. **scripts/build_*.py** - Data preparation
3. **scripts/eval_zoo_pipeline.py** - Main evaluation
4. **scripts/final_*.py** - Final evaluation
5. **scripts/clinical/** - Clinical evaluation
6. **scripts/verification/** - Verification scripts
7. **scripts/ablation/** - Ablation study
8. **scripts/gnn/** - GNN evaluation (subset)
9. **tests/** - All tests (critical for integrity)
10. **configs/default.yaml, locked_best_config.yaml** - Main configs
11. **docs/final/, docs/verification/, docs/clinical/, docs/gnn/** - Final reports
