# Paper Bundle Manifest

**Version:** v1.0
**Created:** 2026-01-19
**Last Updated:** 2026-01-20
**Git Commit:** 9403a7f

---

## Provenance

This bundle was generated using the Evidence Binding Pipeline evaluation scripts.

### How to Regenerate

```bash
# 1. Ensure environment is set up (see docs/ENVIRONMENT_SETUP.md)
conda activate nv-embed-v2  # For NV-Embed-v2 evaluation

# 2. Run evaluation pipeline
python scripts/eval_zoo_pipeline.py \
    --config configs/default.yaml \
    --split test \
    --output outputs/eval/

# 3. Generate publication plots
conda activate llmhe
python scripts/verification/generate_publication_plots.py \
    --per_query_csv outputs/eval/per_query.csv \
    --output_dir paper/figures/

# 4. Package bundle
python scripts/reporting/package_paper_bundle.py \
    --results_dir outputs/eval/ \
    --output results/paper_bundle/v1.0 \
    --version v1.0
```

---

## Contents

### Core Documents

| File | Description |
|------|-------------|
| `report.md` | Comprehensive academic evaluation report |
| `summary.json` | Machine-readable results summary |
| `metrics_master.json` | Single source of truth for all metrics |
| `checksums.txt` | SHA256 checksums for verification |
| `MANIFEST.md` | This file |

### Figures (10 files)

| File | Description | Manuscript Reference |
|------|-------------|---------------------|
| `1_roc_curve_with_ci.png` | ROC curve with confidence intervals | Figure 1 |
| `2_pr_curve_with_baseline.png` | Precision-Recall curve | Figure 2 |
| `3_calibration_diagram.png` | Calibration diagram | Figure 3 |
| `4_confusion_matrix.png` | Confusion matrix | Figure 4 |
| `5_per_criterion_auroc.png` | Per-criterion AUROC | Figure 5 |
| `6_dynamic_k_analysis.png` | Dynamic-K analysis | Figure 6 |
| `7_threshold_sensitivity.png` | Threshold sensitivity | Figure 7 |
| `calibration_plots.png` | Additional calibration | Supplementary |
| `component_comparison.png` | Component comparison | Supplementary |
| `roc_pr_curves.png` | Combined ROC/PR | Supplementary |

### Tables

Tables are embedded in `report.md`. Key tables:
- Table 1: Main results (nDCG@10, Recall@10, AUROC)
- Table 2: Per-criterion performance
- Table 3: Ablation study results
- Table 4: Clinical deployment metrics

---

## Key Metrics (from metrics_master.json)

| Metric | Value | Protocol | Split |
|--------|-------|----------|-------|
| nDCG@10 | 0.8658 | positives_only | TEST |
| Evidence Recall@10 | 0.7043 | positives_only | TEST |
| MRR | 0.3801 | positives_only | TEST |
| AUROC | 0.8972 [0.8941, 0.9003] | all_queries | TEST |
| AUPRC | 0.7043 [0.6921, 0.7165] | all_queries | TEST |

See `metrics_master.json` for complete metric definitions and provenance.

---

## Verification

### Quick Check

```bash
# Verify checksums
python scripts/verification/verify_checksums.py

# Or using sha256sum directly
cd results/paper_bundle/v1.0
sha256sum -c checksums.txt
```

### Full Verification

```bash
# 1. Run all tests
pytest -q

# 2. Verify split integrity (no data leakage)
python scripts/audit_splits.py --data_dir data --seed 42 --k 5

# 3. Cross-check metrics independently
python scripts/verification/metric_crosscheck.py \
    --fold_results_dir outputs/eval/ \
    --pipeline_summary outputs/eval/summary.json
```

---

## Dependencies

- Python 3.10+
- Two conda environments required (see docs/ENVIRONMENT_SETUP.md):
  - `nv-embed-v2`: For NV-Embed-v2 (transformers<=4.44)
  - `llmhe`: For reranking, GNN, evaluation (transformers>=4.45)

---

## Contact

For questions about this bundle:
- See README.md for project overview
- See docs/DATA_AVAILABILITY.md for data access
- Open an issue on GitHub for technical questions
