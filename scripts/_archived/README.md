# Archived Scripts

These scripts have been archived after completing the optimization research phase.

## experiments/

Contains 43 optimization experiment scripts that tested various GNN configurations:

- Activation functions (GELU, ReLU, SiLU, etc.)
- Aggregation methods (mean, max, sum)
- Optimizers and schedulers
- Regularization techniques
- Graph augmentation
- Alternative architectures (GIN)

**Key findings documented in:** `outputs/experiments/EXPERIMENT_SUMMARY.md`

**Final best configuration:**
- SAGE + Residual + GELU
- nDCG@10 = 0.8237 ± 0.030 (+10.89% improvement)

## hpo/

Contains hyperparameter optimization scripts used during the research phase:

- `run_full_pipeline_hpo.py` - Full pipeline HPO
- `run_gnn_hpo_optimized.py` - GNN-specific HPO
- `run_gnn_hpo_single_split.py` - Single-split quick HPO
- `run_jina_hpo_full.py` - Jina reranker HPO

**Best HPO parameters are now frozen in:** `configs/default.yaml`

---

These scripts are preserved for reproducibility but are no longer needed for production use.
