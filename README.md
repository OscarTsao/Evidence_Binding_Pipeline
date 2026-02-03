# Evidence Binding Pipeline

**Publication-Ready Research Repository**

Sentence-criterion evidence retrieval for mental health assessment using DSM-5 Major Depressive Disorder criteria.

## Performance Summary (5-Fold Cross-Validation)

| Model | nDCG@10 | MRR | Recall@10 |
|-------|---------|-----|-----------|
| **Baseline (NV-Embed-v2 + Jina-v3)** | 0.7330 ± 0.031 | 0.6746 ± 0.037 | 0.9444 ± 0.022 |
| **+ P3 GNN (SAGE+Residual)** | 0.8206 ± 0.030 | 0.7703 ± 0.035 | 0.9606 ± 0.019 |
| **Improvement** | **+10.48%** | **+12.02%** | **+1.71%** |

**Dataset:** 13,293 queries across 1,477 posts (DSM-5 criteria A.1-A.9, A.10 excluded)
**Evaluation:** 5-fold cross-validation, positives_only protocol
**Reproducibility:** Complete artifact bundle with SHA256 checksums

## Quick Start

### 1. Environment Setup

Two conda environments required (NV-Embed-v2 requires older transformers):

```bash
# Environment 1: NV-Embed-v2 retriever (transformers==4.44.2)
conda env create -f environment_nv_embed.yaml
conda activate nv-embed-v2

# Environment 2: Main pipeline (reranking, GNN, LLM, evaluation)
conda env create -f environment_llmhe.yaml
conda activate llmhe
```

### 2. Encode Corpus (one-time setup)

```bash
# In nv-embed-v2 environment - encodes corpus embeddings
conda activate nv-embed-v2
python scripts/encode_nv_embed.py --config configs/default.yaml
```

This caches NV-Embed-v2 embeddings to disk. Run once per corpus change.

### 3. Verify Installation

```bash
# In llmhe environment
conda activate llmhe

# Run tests
pytest -q

# Verify paper bundle integrity
python scripts/verification/verify_checksums.py
```

### 4. Reproduce Results

```bash
# Full reproduction guide
cat docs/REPRODUCIBILITY.md

# Quick evaluation (requires data access)
python scripts/eval_zoo_pipeline.py --config configs/default.yaml --split test
```

## Pipeline Architecture

```
Query (DSM-5 Criterion) + Post
    |
[Stage 1] NV-Embed-v2 Retriever (top-24 candidates)
    |
[Stage 2] Jina-Reranker-v3 Cross-Encoder (top-10)
    |
[Stage 3] P3 GNN Graph Reranker (optional refinement)
    |
[Stage 4] P2 GNN Dynamic-K Selection (K in [3,20])
    |
[Stage 5] P4 GNN NE Gate (No-Evidence Detection)
    |
Evidence Sentences (ranked by relevance)
```

**Key Features:**
- Post-ID disjoint splits (zero data leakage)
- Within-post retrieval only (clinical constraint)
- Dual-protocol metrics (positives_only + all_queries)
- HPO-optimized over 324 model combinations
- A.10 excluded from training (improves A.1-A.9 performance)

**Best HPO Parameters (Jina-Reranker-v3):**
| Parameter | Value |
|-----------|-------|
| top_k_retriever | 24 |
| top_k_final | 10 |
| fusion_method | rrf |
| rrf_k | 60 |
| reranker_max_length | 1024 |

**GNN Enhancements (P3 Graph Reranker SAGE+Residual, 5-Fold CV):**
- nDCG@10: +10.48% (0.7330 → 0.8206)
- MRR: +12.02% (0.6746 → 0.7703)

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and components |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Full reproduction protocol |
| [DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) | Data access procedures |
| [ETHICS.md](docs/ETHICS.md) | Privacy and IRB compliance |
| [METRIC_CONTRACT.md](docs/METRIC_CONTRACT.md) | Metric definitions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Developer guide |

## Research Artifacts

All results verified and checksummed in `results/paper_bundle/v3.0/`:

| File | Purpose |
|------|---------|
| `metrics_master.json` | Single source of truth for all metrics |
| `summary.json` | Bundle metadata |
| `tables/*.csv` | Machine-readable results |
| `checksums.txt` | SHA256 integrity verification |

## Per-Criterion Performance

| Criterion | Description | AUROC | Notes |
|-----------|-------------|-------|-------|
| A.1 | Depressed Mood | 0.83 | DSM-5 |
| A.2 | Anhedonia | 0.88 | DSM-5 |
| A.3 | Weight/Appetite Change | 0.91 | DSM-5 |
| A.4 | Sleep Disturbance | 0.89 | DSM-5 |
| A.5 | Psychomotor Changes | 0.80 | DSM-5 |
| A.6 | Fatigue/Loss of Energy | 0.93 | DSM-5 |
| A.7 | Worthlessness/Guilt | 0.92 | DSM-5 |
| A.8 | Concentration Difficulty | 0.80 | DSM-5 |
| A.9 | Suicidal Ideation | 0.95 | DSM-5 |
| A.10 | SPECIAL_CASE | 0.67 | Excluded from training* |

*A.10 (expert discrimination cases) is excluded from GNN training by default because it's not a standard DSM-5 criterion and ablation study showed removing it improves nDCG@10 by +0.28%.

## Citation

```bibtex
@software{evidence_binding_2026,
  title = {Evidence Binding Pipeline for Mental Health Assessment},
  author = {Tsao, Oscar},
  year = {2026},
  url = {https://github.com/OscarTsao/Evidence_Binding_Pipeline}
}
```

See [CITATION.cff](CITATION.cff) for structured metadata.

## License

MIT License - see [LICENSE](LICENSE)

## Data Access

Research data is not publicly available due to privacy/IRB requirements.
See [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) for access procedures.

## Production Readiness

This repository meets gold-standard criteria for reproducible research:
- CI/CD with automated tests + checksum verification
- Zero data leakage (verified via 12+ independent tests)
- Complete metrics provenance (git commit + seeds + configs)
- Machine-readable result tables
- Publication compliance (LICENSE + CITATION + ethics)
