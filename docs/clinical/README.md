# Clinical High-Recall Deployment Mode

This module implements a clinically-validated, high-recall evidence retrieval system with rigorous verification to prevent data leakage and ensure reproducibility.

## Overview

### Clinical Intent

The clinical deployment mode addresses the real-world requirement for:

1. **High Sensitivity**: Minimize false negatives (missed evidence/symptoms)
2. **Workload Management**: Prevent alert fatigue via tiered decision system
3. **Transparent Decisions**: Provide evidence basis for clinical review

### Three-State Decision System

The system classifies each query-criterion pair into one of three states:

- **NEG** (No Evidence): High confidence that no evidence exists → Skip extraction (K=0)
- **UNCERTAIN**: Low confidence → Conservative extraction (higher K, higher gamma)
- **POS** (Positive): High confidence evidence exists → Standard extraction

### Key Features

✅ **No Data Leakage**: Post-ID disjoint 5-fold CV with nested threshold selection
✅ **Calibrated Probabilities**: Isotonic or Platt calibration on TUNE split
✅ **Dynamic-K Selection**: Per-state policies adapting to candidate count
✅ **Comprehensive Metrics**: NE gate, evidence extraction, deployment metrics
✅ **Reproducible Artifacts**: Full configuration and results saved

## Architecture

### Pipeline Flow

```
Query + Criterion
       ↓
   Retriever (nv-embed-v2)
       ↓
   Reranker (jina-reranker-v3)
       ↓
   P3 Graph Reranker (score refinement)
       ↓
   P4 NE Gate (calibrated probability p)
       ↓
Three-State Classification
  ↙       ↓       ↘
NEG   UNCERTAIN   POS
K=0   K=5-12     K=3-12
  ↘       ↓       ↙
   Dynamic-K Selection
       ↓
  Evidence Output
```

### Nested Cross-Validation

**Critical**: Threshold selection must use TUNE split only (no test leakage).

```
For each fold:
  TRAIN split → Train P4 model
  TUNE split  → Fit calibration, select tau_neg and tau_pos
  TEST split  → Apply thresholds, evaluate (held-out)
```

## Installation

### Requirements

```bash
pip install -e .  # Install main package
pip install pytest scikit-learn matplotlib seaborn  # Additional dependencies
```

### Verify Installation

```bash
python -c "from final_sc_review.clinical import ClinicalConfig; print('✓ Clinical module installed')"
```

## Usage

### Quick Start

```bash
# Run 5-fold evaluation with default configuration
python scripts/clinical/run_clinical_high_recall_eval.py \
    --graph_dir data/cache/gnn/20260117_003135 \
    --output_dir outputs/clinical_high_recall

# Results saved to: outputs/clinical_high_recall/<timestamp>/
```

### Custom Configuration

1. Create custom config:

```yaml
# my_clinical_config.yaml
threshold_config:
  sensitivity_target: 0.99
  max_fpr_pos: 0.03
  calibration_method: "isotonic"

uncertain_config:
  k_min: 5
  k_max1: 12
  k_max_ratio: 0.70
  gamma: 0.95

pos_config:
  k_min: 3
  k_max1: 12
  k_max_ratio: 0.60
  gamma: 0.90
```

2. Run with custom config:

```bash
python scripts/clinical/run_clinical_high_recall_eval.py \
    --graph_dir data/cache/gnn/20260117_003135 \
    --config my_clinical_config.yaml
```

## Verification

### Run Leakage Tests

```bash
cd tests/clinical
python test_no_leakage.py

# Or use pytest
pytest test_no_leakage.py -v -s
```

### Expected Test Output

```
✓ Verified 5 folds are post-ID disjoint
✓ No forbidden gold-derived features found
✓ Threshold selection uses TUNE split only
✓ Calibration fitted on TUNE, applied to TEST
✓ Recall@K computed correctly
✓ MRR computed correctly
```

## Output Structure

```
outputs/clinical_high_recall/<timestamp>/
├── config.yaml                          # Configuration used
├── summary.json                         # Machine-readable results
├── CLINICAL_DEPLOYMENT_REPORT.md        # Human-readable report
├── fold_results/
│   ├── fold_0_predictions.csv           # Per-query predictions
│   ├── fold_1_predictions.csv
│   └── ...
└── curves/                              # Visualizations (future)
    ├── roc_pr_curves.png
    ├── calibration_plot.png
    └── tradeoff_curves.png
```

## Key Metrics

### NE Gate Metrics (P4)

- **AUROC**: Area under ROC curve
- **AUPRC**: Area under PR curve
- **TPR@{1,3,5,10}%FPR**: True positive rate at fixed false positive rates
- **ECE**: Expected calibration error

### Deployment Metrics

**Screening Tier** (NOT NEG = flagged):
- **Sensitivity**: Fraction of evidence queries correctly flagged
- **NPV**: Negative predictive value for NEG decisions
- **FN/1000**: False negatives per 1000 queries

**Alert Tier** (POS = high confidence):
- **Precision**: Fraction of POS decisions that are correct
- **Recall**: Fraction of evidence queries classified as POS
- **FPR**: False positive rate for POS decisions

### Evidence Extraction Metrics

- **Recall@K**: Fraction of gold evidence retrieved in top-K
- **Precision@K**: Fraction of top-K that are gold evidence
- **MRR**: Mean reciprocal rank of first gold evidence
- **nDCG@K**: Normalized discounted cumulative gain

## Configuration Parameters

### Threshold Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensitivity_target` | 0.99 | Target sensitivity for screening tier |
| `max_fpr_pos` | 0.05 | Maximum FPR for alert tier |
| `calibration_method` | "isotonic" | Calibration method ("isotonic" or "platt") |

### Dynamic-K Parameters

**NEG State:**
- `k_min = 0`, `k_max1 = 0`, `gamma = None`
- **Result**: K = 0 (no extraction)

**UNCERTAIN State:**
- `k_min = 5`, `k_max1 = 12`, `k_max_ratio = 0.70`, `gamma = 0.95`
- **Result**: Conservative extraction (higher K)

**POS State:**
- `k_min = 3`, `k_max1 = 12`, `k_max_ratio = 0.60`, `gamma = 0.90`
- **Result**: Standard extraction

## Clinical Validation

### Required Steps Before Deployment

1. **Clinical Expert Review**:
   - Review false negative cases
   - Validate state assignments on sample queries
   - Confirm clinical appropriateness of thresholds

2. **External Validation**:
   - Test on independent dataset
   - Verify generalization across different populations

3. **Prospective Clinical Trial**:
   - Deploy in clinical setting with monitoring
   - Measure real-world sensitivity and workload
   - Collect clinician feedback

### Limitations

⚠️ **NOT FOR AUTONOMOUS DIAGNOSIS**: This is decision support only
⚠️ **CLINICIAN REVIEW REQUIRED**: All outputs must be reviewed
⚠️ **FALSE NEGATIVES POSSIBLE**: System has {FN_rate} FN/1000 queries
⚠️ **DATASET-SPECIFIC**: Trained on specific mental health data

## Research Gold Standards

### Leakage Prevention

✅ Post-ID disjoint 5-fold CV
✅ No gold-derived features (mrr, recall_at_*, gold_rank)
✅ Nested threshold selection (TUNE split only)
✅ Calibration fitted on TUNE, applied to TEST

### Metric Validation

✅ Independent reference implementation
✅ Cross-checked against existing metrics
✅ Unit tests for all metric functions

### Reproducibility

✅ Configuration saved with results
✅ Random seeds fixed
✅ Fold-level details preserved
✅ Single command to reproduce full evaluation

## Commands Reference

### Reproduce Full Evaluation

```bash
# Single command to reproduce everything
python scripts/clinical/run_clinical_high_recall_eval.py \
    --graph_dir data/cache/gnn/20260117_003135 \
    --output_dir outputs/clinical_high_recall \
    --n_folds 5
```

### Run Verification Tests

```bash
# All tests
pytest tests/clinical/ -v -s

# Specific test
pytest tests/clinical/test_no_leakage.py::TestSplitDisjointness -v -s
```

### Generate Plots (Future)

```bash
# Generate visualizations from summary.json
python scripts/clinical/generate_plots.py \
    --summary outputs/clinical_high_recall/<timestamp>/summary.json
```

## Troubleshooting

### Common Issues

**Issue**: "Graph cache not found"
**Solution**: Run GNN evaluation first to generate graph cache:
```bash
python scripts/gnn/run_e2e_eval_and_report.py --graph_dir data/cache/gnn/20260117_003135
```

**Issue**: "Sensitivity target cannot be met"
**Solution**: Lower `sensitivity_target` in configuration (e.g., 0.95 instead of 0.99)

**Issue**: "All queries classified as UNCERTAIN"
**Solution**: Check tau_neg and tau_pos thresholds. Increase `max_fpr_pos` to allow more POS decisions.

## Contributing

When modifying the clinical module:

1. ✅ **Add tests** for any new functionality
2. ✅ **Verify no leakage** by running `test_no_leakage.py`
3. ✅ **Document clinical intent** in docstrings
4. ✅ **Update this README** with new parameters/features

## References

### Clinical Decision Support

- [FDA Guidance on Clinical Decision Support](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
- [TRIPOD Statement for Prediction Models](https://www.tripod-statement.org/)

### Calibration Methods

- Platt Scaling: [Platt, 1999](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- Isotonic Regression: [Zadrozny & Elkan, 2002](https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf)

### Evaluation Standards

- [PROBAST: Risk of Bias Tool](https://www.probast.org/)
- [STARD: Diagnostic Accuracy](https://www.equator-network.org/reporting-guidelines/stard/)
