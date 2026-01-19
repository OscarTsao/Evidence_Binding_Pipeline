# Paper Reproducibility Guide

This document provides exact instructions to reproduce all results reported in the paper.

## Quick Start (One Command)

```bash
# Run complete paper reproduction pipeline
bash scripts/run_paper_reproduce.sh
```

This script will:
1. Run all tests (pytest)
2. Verify split integrity and no data leakage
3. Run final evaluation on test set
4. Generate publication figures and tables
5. Package results into `results/paper_bundle/`

---

## Environment Setup

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU inference)
- 32GB+ GPU VRAM (for NV-Embed-v2)
- Conda (for environment management)

### Installation (Dual Environment Required)

**IMPORTANT:** This project requires two conda environments due to NV-Embed-v2's dependency on older transformers.

```bash
# 1. Create main environment (reranking, GNN, tests)
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .

# Optional: Install PyTorch Geometric for GNN modules
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 2. Create retriever environment (NV-Embed-v2)
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt
pip install -e .
```

**Why two environments?** NV-Embed-v2 requires `transformers<=4.44`, while Jina-Reranker-v3 and sentence-transformers 5.x need `transformers>=4.45`.

See `docs/ENVIRONMENT_SETUP.md` for detailed setup instructions.

### Hardware Notes
- Primary evaluation uses NVIDIA RTX 5090 (32GB)
- CPU-only evaluation possible but significantly slower
- Minimum 16GB RAM recommended

---

## Data Setup

### Required Data Files
Data is NOT included in this repository due to privacy considerations.

```
data/
  redsm5/
    redsm5_posts.csv       # Reddit posts (columns: post_id, text)
    redsm5_annotations.csv # Evidence annotations
  DSM5/
    MDD_Criteira.json      # DSM-5 MDD criteria definitions
```

### Data Availability
- Contact authors for data access
- IRB approval required for clinical data use
- See paper Section X for data description

### Build Groundtruth (Required First)

```bash
# Build groundtruth labels from annotations
python scripts/build_groundtruth.py \
    --data_dir data \
    --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus with canonical splitting
python scripts/build_sentence_corpus.py \
    --data_dir data \
    --output data/groundtruth/sentence_corpus.jsonl
```

---

## Reproduce Paper Results

### Main Evaluation (Table 1)

```bash
# Run evaluation with best model configuration
python scripts/eval_zoo_pipeline.py \
    --config configs/default.yaml \
    --split test
```

Expected output:
- nDCG@10: 0.8658 ± 0.02
- Recall@10: 0.892 ± 0.03
- MRR@10: 0.812 ± 0.02

### Ablation Study (Table 2)

```bash
# Run ablation suite
python scripts/ablation/run_ablation_suite.py \
    --config configs/default.yaml \
    --output results/ablation/
```

### Clinical Evaluation (Table 3)

```bash
# Run clinical high-recall evaluation
python scripts/clinical/run_clinical_high_recall_eval.py \
    --config configs/deployment_high_recall.yaml \
    --output results/clinical/
```

### GNN Evaluation (Table 4)

```bash
# Run GNN E2E evaluation
python scripts/gnn/run_e2e_eval_and_report.py \
    --config configs/default.yaml \
    --output results/gnn/
```

---

## Verification Scripts

### Split Integrity

```bash
# Verify no data leakage between splits
python scripts/verification/audit_splits_and_leakage.py \
    --config configs/default.yaml
```

### Metric Cross-Check

```bash
# Independent metric recomputation
python scripts/verification/metric_crosscheck.py \
    --predictions results/predictions.csv \
    --groundtruth data/groundtruth/evidence_sentence_groundtruth.csv
```

---

## Generate Publication Figures

```bash
# Generate all paper figures
python scripts/verification/generate_publication_plots.py \
    --results_dir results/ \
    --output paper/figures/
```

Generates:
- `roc_pr_curves.pdf` - ROC and PR curves (Figure 1)
- `calibration_plot.pdf` - Calibration curve (Figure 2)
- `per_criterion_performance.pdf` - Per-criterion breakdown (Figure 3)
- `dynamic_k_analysis.pdf` - Dynamic-K visualization (Figure 4)

---

## Configuration

### Best Model Configuration (configs/default.yaml)

```yaml
models:
  retriever_name: nv-embed-v2
  reranker_name: jina-reranker-v3

retriever:
  top_k_retriever: 24
  top_k_final: 10
  fusion_method: rrf
  rrf_k: 60

split:
  seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `top_k_retriever` | 24 | Candidates from retriever |
| `top_k_final` | 10 | Final output size |
| `rrf_k` | 60 | RRF fusion parameter |

---

## Expected Outputs

### Results Directory Structure

```
results/
├── paper_bundle/
│   ├── MANIFEST.md
│   ├── summary.json
│   ├── figures/
│   │   ├── roc_pr_curves.pdf
│   │   ├── calibration_plot.pdf
│   │   └── ...
│   └── tables/
│       ├── main_results.csv
│       ├── ablation_results.csv
│       └── ...
├── predictions.csv
└── evaluation_summary.json
```

### Checksums (for verification)

```
main_results.csv:    sha256:TBD
ablation_results.csv: sha256:TBD
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use smaller retriever model (e.g., bge-m3)

2. **Missing Data Files**
   - Ensure all data files are in correct paths
   - Run build_groundtruth.py first

3. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -e .` to install package

### Support
- Open issue on GitHub repository
- Contact: [authors]

---

## Citation

If you use this code, please cite:

```bibtex
@article{author2026evidence,
  title={Evidence Sentence Retrieval for Mental Health Assessment},
  author={Author et al.},
  journal={TBD},
  year={2026}
}
```
