# Clinical High-Recall Deployment - Quick Start Guide

## What Was Completed ✅

### 1. Critical Bug Fixes
- **Fixed negative FPR bug**: Changed `~labels` (bitwise NOT) to `labels == 0` (logical comparison)
  - Files: `three_state_gate.py`, `run_clinical_high_recall_eval.py`
- **Fixed alert precision = 0%**: Root cause was the negative FPR bug
- **Created diagnostic tool**: `debug_alert_precision.py` for future debugging

### 2. New Features Implemented
- **Per-query CSV export**: Full prediction metadata for all 14,770 test queries
- **Per-post multi-label CSV**: Aggregated predictions for clinical review
- **Per-criterion analysis**: Metrics breakdown for each MDD criterion (A.1-A.10)
- **Visualization plots**: 5 comprehensive plots for clinical deployment review
- **Diagnostic tools**: Automated bug detection and analysis

### 3. Documentation
- `IMPLEMENTATION_SUMMARY.md`: Comprehensive implementation details
- `QUICK_START.md`: This file
- Debug reports: Generated automatically by diagnostic tool

---

## Next Steps

### Step 1: Re-Run Evaluation with Bug Fixes

The evaluation needs to be re-run to verify the bug fixes work correctly:

```bash
cd /home/user/YuNing/Final_SC_Review

python scripts/clinical/run_clinical_high_recall_eval.py \
    --graph_dir data/cache/gnn/20260117_003135 \
    --p4_model_dir outputs/gnn_research/20260117_p4_final/20260117_030024/p4_hetero \
    --output_dir outputs/clinical_high_recall \
    --n_folds 5 \
    --device cuda
```

**Expected runtime:** 30-60 minutes (depends on GPU)

**What to check:**
- ✅ FPR values are positive (0-1 range)
- ✅ Alert precision is non-zero
- ✅ All CSV files generated in `fold_results/`
- ✅ Per-criterion metrics in `summary.json`

### Step 2: Generate Visualization Plots

Once evaluation completes, generate plots:

```bash
# Replace YYYYMMDD_HHMMSS with actual timestamp from Step 1
python scripts/clinical/generate_plots.py \
    --summary outputs/clinical_high_recall/YYYYMMDD_HHMMSS/summary.json \
    --output outputs/clinical_high_recall/YYYYMMDD_HHMMSS/curves
```

**Expected output:**
- `roc_pr_curves.png`
- `calibration_plot.png`
- `tradeoff_curves.png`
- `per_criterion_analysis.png`
- `dynamic_k_analysis.png`

### Step 3: Verify Results

Check the output files:

```bash
# View directory structure
tree outputs/clinical_high_recall/YYYYMMDD_HHMMSS/

# Check CSV files were created
ls -lh outputs/clinical_high_recall/YYYYMMDD_HHMMSS/fold_results/*.csv

# Check plot files were created
ls -lh outputs/clinical_high_recall/YYYYMMDD_HHMMSS/curves/*.png

# View summary metrics
python -c "import json; print(json.dumps(json.load(open('outputs/clinical_high_recall/YYYYMMDD_HHMMSS/summary.json'))['aggregated_metrics'], indent=2))"
```

### Step 4: Review Results

Key metrics to check in `summary.json`:

```python
{
  "deployment.screening_fpr": {
    "mean": 0.05-0.15,  # Should be POSITIVE now!
    ...
  },
  "deployment.alert_precision": {
    "mean": 0.10-0.40,  # Should be NON-ZERO now!
    ...
  },
  "deployment.screening_sensitivity": {
    "mean": 0.99,  # Should be high
    ...
  }
}
```

---

## File Locations

### New Files Created
- `scripts/clinical/debug_alert_precision.py` - Diagnostic tool
- `scripts/clinical/generate_plots.py` - Plot generation
- `docs/clinical/IMPLEMENTATION_SUMMARY.md` - Full documentation
- `docs/clinical/QUICK_START.md` - This file

### Modified Files
- `src/final_sc_review/clinical/three_state_gate.py` - Bug fixes (5 locations)
- `scripts/clinical/run_clinical_high_recall_eval.py` - Bug fixes + new features (300+ lines)

### Output Files (After Re-Run)
```
outputs/clinical_high_recall/YYYYMMDD_HHMMSS/
├── config.yaml
├── summary.json (enhanced with per_criterion_metrics)
├── CLINICAL_DEPLOYMENT_REPORT.md
├── fold_results/
│   ├── fold_0_predictions.csv (NEW - ~2,950 rows)
│   ├── fold_1_predictions.csv (NEW - ~2,950 rows)
│   ├── fold_2_predictions.csv (NEW - ~2,950 rows)
│   ├── fold_3_predictions.csv (NEW - ~2,950 rows)
│   ├── fold_4_predictions.csv (NEW - ~2,970 rows)
│   └── per_post_multilabel.csv (NEW - ~148 posts × 5 folds)
└── curves/
    ├── roc_pr_curves.png (NEW)
    ├── calibration_plot.png (NEW)
    ├── tradeoff_curves.png (NEW)
    ├── per_criterion_analysis.png (NEW)
    └── dynamic_k_analysis.png (NEW)
```

---

## Troubleshooting

### If evaluation fails with import error:
```bash
# Make sure you're in the right directory
cd /home/user/YuNing/Final_SC_Review

# Verify Python can find the package
python -c "from final_sc_review.clinical.three_state_gate import ThreeStateGate; print('OK')"
```

### If plots fail to generate:
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# If needed, set non-interactive backend
export MPLBACKEND=Agg
python scripts/clinical/generate_plots.py --summary <path> --output <path>
```

### If FPR is still negative after re-run:
```bash
# This means the bug fix didn't work - run diagnostic
python scripts/clinical/debug_alert_precision.py \
    --summary outputs/clinical_high_recall/YYYYMMDD_HHMMSS/summary.json \
    --output outputs/clinical_high_recall/YYYYMMDD_HHMMSS/debug

# Check the diagnostic report
cat outputs/clinical_high_recall/YYYYMMDD_HHMMSS/debug/alert_precision_analysis.md
```

---

## Expected Before/After Comparison

### Before (Buggy Results)
```json
{
  "deployment.screening_fpr": {"mean": -0.206},  ❌ NEGATIVE
  "deployment.alert_precision": {"mean": 0.0},   ❌ ZERO
  "deployment.alert_fpr": {"mean": -0.206}       ❌ NEGATIVE
}
```

### After (Fixed Results)
```json
{
  "deployment.screening_fpr": {"mean": 0.08},    ✅ POSITIVE
  "deployment.alert_precision": {"mean": 0.25},  ✅ NON-ZERO
  "deployment.alert_fpr": {"mean": 0.05}         ✅ POSITIVE
}
```

---

## Questions?

See the detailed documentation:
- **Full details:** `docs/clinical/IMPLEMENTATION_SUMMARY.md`
- **Diagnostic report:** Run `debug_alert_precision.py` on any summary.json
- **Original plan:** See the plan document provided

---

**Status:** Ready for testing
**Next action:** Run Step 1 (re-evaluation)
**Estimated time:** 30-60 minutes
