# Paper-Critical Commands

This document lists the exact commands required to reproduce all paper results.

## Prerequisites

**IMPORTANT:** This project requires two conda environments due to NV-Embed-v2's dependency constraints.

```bash
# 1. Create and setup main environment
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .

# 2. Create and setup retriever environment
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt
pip install -e .

# 3. Verify installation (in main env)
conda activate llmhe
pytest -q  # All tests must pass
```

See `docs/ENVIRONMENT_SETUP.md` for detailed setup instructions.

## One-Command Full Reproduction

```bash
bash scripts/run_paper_reproduce.sh
```

This script automatically switches between environments as needed.

---

## Individual Commands

### 1. Data Preparation (if starting fresh)

```bash
conda activate llmhe

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
conda activate llmhe

python scripts/audit_splits.py \
    --data_dir data \
    --seed 42 \
    --k 5 \
    --output outputs/audit/split_audit_report.md
```

Expected output: "PASS: All splits are post-ID disjoint"

### 3. Main Evaluation

**Note:** NV-Embed-v2 evaluation requires the `nv-embed-v2` environment.

```bash
# Switch to retriever environment for NV-Embed-v2
conda activate nv-embed-v2

# Zoo pipeline evaluation (recommended)
python scripts/eval_zoo_pipeline.py \
    --config configs/default.yaml \
    --split test \
    --output outputs/eval/
```

### 4. Clinical High-Recall Evaluation

```bash
conda activate llmhe

python scripts/clinical/run_clinical_high_recall_eval.py \
    --config configs/deployment_high_recall.yaml \
    --output outputs/clinical/
```

### 5. GNN Evaluation (if applicable)

```bash
conda activate llmhe

python scripts/gnn/run_e2e_eval_and_report.py \
    --config configs/default.yaml \
    --output outputs/gnn_eval/
```

### 6. Metric Cross-Check (Independent Verification)

```bash
conda activate llmhe

python scripts/verification/metric_crosscheck.py \
    --fold_results_dir outputs/eval/ \
    --pipeline_summary outputs/eval/summary.json \
    --output outputs/crosscheck/crosscheck_report.json
```

Expected output: "PASS: All metrics match within tolerance"

### 7. Generate Publication Plots

```bash
conda activate llmhe

python scripts/verification/generate_publication_plots.py \
    --per_query_csv outputs/eval/per_query.csv \
    --output_dir paper/figures/
```

Outputs:
- `1_roc_curve_with_ci.png`
- `2_pr_curve_with_baseline.png`
- `3_calibration_diagram.png`
- `4_confusion_matrix.png`
- `5_per_criterion_auroc.png`
- `6_dynamic_k_analysis.png`
- `7_threshold_sensitivity.png`

### 8. Ablation Study (Optional)

```bash
conda activate llmhe

python scripts/ablation/run_ablation_suite.py \
    --config configs/default.yaml \
    --output outputs/ablation/
```

### 9. Package Paper Bundle

```bash
conda activate llmhe

python scripts/reporting/package_paper_bundle.py \
    --results_dir outputs/eval/ \
    --output results/paper_bundle/v1.0 \
    --version v1.0
```

---

## Environment Summary

| Task | Environment | Reason |
|------|-------------|--------|
| Tests, audits | `llmhe` | Main environment |
| NV-Embed-v2 evaluation | `nv-embed-v2` | Requires transformers<=4.44 |
| GNN, clinical eval | `llmhe` | Main environment |
| Plotting, packaging | `llmhe` | Main environment |

---

## Output Locations

| Artifact | Location |
|----------|----------|
| Main evaluation | `outputs/eval/` |
| Clinical evaluation | `outputs/clinical/` |
| GNN evaluation | `outputs/gnn_eval/` |
| Cross-check report | `outputs/crosscheck/` |
| Publication figures | `paper/figures/` |
| Paper bundle | `results/paper_bundle/v1.0/` |

---

## Verification Checklist

After running all commands, verify:

1. [ ] All tests pass: `pytest -q` (all tests should pass)
2. [ ] Split audit passes: No post-ID overlap
3. [ ] Metric cross-check passes: <1% deviation
4. [ ] Paper bundle exists: `results/paper_bundle/v1.0/MANIFEST.md`
5. [ ] All figures generated: `paper/figures/*.png`
