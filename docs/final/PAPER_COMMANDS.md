# Paper-Critical Commands

This document lists the exact commands required to reproduce all paper results.

## Prerequisites

```bash
# Activate environment
source .venv/bin/activate  # or conda activate <env>

# Verify installation
pip install -e .
pytest -q  # All tests must pass
```

## One-Command Full Reproduction

```bash
bash scripts/run_paper_reproduce.sh
```

This script executes all steps below in sequence.

---

## Individual Commands

### 1. Data Preparation (if starting fresh)

```bash
# Build groundtruth from annotations
python scripts/build_groundtruth.py \
    --data_dir data \
    --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus
python scripts/build_sentence_corpus.py \
    --data_dir data \
    --output data/groundtruth/sentence_corpus.jsonl
```

### 2. Split Audit (Verify No Leakage)

```bash
python scripts/audit_splits.py --config configs/default.yaml
python scripts/verification/audit_splits_and_leakage.py --config configs/default.yaml
```

Expected output: "PASS: All splits are post-ID disjoint"

### 3. Main Evaluation

```bash
# Zoo pipeline evaluation (recommended)
python scripts/eval_zoo_pipeline.py \
    --config configs/default.yaml \
    --split test \
    --output outputs/eval/

# Full academic evaluation
python scripts/run_full_academic_evaluation.py \
    --config configs/default.yaml \
    --output outputs/academic_eval/
```

### 4. Clinical High-Recall Evaluation

```bash
python scripts/clinical/run_clinical_high_recall_eval.py \
    --config configs/deployment_high_recall.yaml \
    --output outputs/clinical/
```

### 5. GNN Evaluation (if applicable)

```bash
python scripts/gnn/run_e2e_eval_and_report.py \
    --config configs/default.yaml \
    --output outputs/gnn_eval/
```

### 6. Metric Cross-Check (Independent Verification)

```bash
python scripts/verification/metric_crosscheck.py \
    --predictions outputs/eval/predictions.csv \
    --groundtruth data/groundtruth/evidence_sentence_groundtruth.csv \
    --summary outputs/eval/summary.json \
    --output outputs/crosscheck/
```

Expected output: "PASS: All metrics match within tolerance"

### 7. Generate Publication Plots

```bash
python scripts/verification/generate_publication_plots.py \
    --results_dir outputs/eval/ \
    --output paper/figures/
```

Outputs:
- `roc_pr_curves.pdf`
- `calibration_plot.pdf`
- `per_criterion_performance.pdf`
- `dynamic_k_analysis.pdf`

### 8. Generate Tables

```bash
python scripts/clinical/generate_plots.py \
    --results_dir outputs/clinical/ \
    --output paper/tables/
```

### 9. Ablation Study (Optional)

```bash
python scripts/ablation/run_ablation_suite.py \
    --config configs/default.yaml \
    --output outputs/ablation/
```

### 10. Package Paper Bundle

```bash
python scripts/reporting/package_paper_bundle.py \
    --results_dir outputs/eval/ \
    --output results/paper_bundle/v1.0 \
    --version v1.0
```

---

## Output Locations

| Artifact | Location |
|----------|----------|
| Main evaluation | `outputs/eval/` |
| Clinical evaluation | `outputs/clinical/` |
| GNN evaluation | `outputs/gnn_eval/` |
| Cross-check report | `outputs/crosscheck/` |
| Publication figures | `paper/figures/` |
| Publication tables | `paper/tables/` |
| Paper bundle | `results/paper_bundle/v1.0/` |

---

## Verification Checklist

After running all commands, verify:

1. [ ] All tests pass: `pytest -q`
2. [ ] Split audit passes: No post-ID overlap
3. [ ] Metric cross-check passes: <1% deviation
4. [ ] Paper bundle exists: `results/paper_bundle/v1.0/MANIFEST.md`
5. [ ] All figures generated: `paper/figures/*.pdf`
