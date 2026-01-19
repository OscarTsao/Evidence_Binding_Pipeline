# Evidence Binding Pipeline

**Publication-Ready Research Repository**

Sentence-criterion evidence retrieval for mental health assessment using DSM-5 Major Depressive Disorder criteria.

## Performance Summary

| Metric | Value | Protocol | Split |
|--------|-------|----------|-------|
| **AUROC** | 0.8972 | all_queries | TEST |
| **Evidence Recall@K** | 0.7043 | positives_only | TEST |
| **MRR** | 0.3801 | positives_only | TEST |
| **nDCG@10** | 0.8658 | positives_only | TEST |

**Dataset:** 14,770 queries across 1,641 posts
**Validation:** Gold-standard evaluation with post-ID disjoint splits
**Reproducibility:** Complete artifact bundle with SHA256 checksums

## Quick Start

### 1. Environment Setup

Two conda environments required (dependency conflicts):

```bash
# Retriever environment (NV-Embed-v2 requires transformers<=4.44)
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt

# Main environment (reranking, GNN, evaluation)
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .
```

### 2. Verify Installation

```bash
# Run tests
pytest -q

# Verify paper bundle integrity
python scripts/verification/verify_checksums.py
```

### 3. Reproduce Results

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

All results verified and checksummed in `results/paper_bundle/v2.0/`:

| File | Purpose |
|------|---------|
| `metrics_master.json` | Single source of truth for all metrics |
| `summary.json` | Bundle metadata |
| `tables/*.csv` | Machine-readable results |
| `checksums.txt` | SHA256 integrity verification |

## Per-Criterion Performance

| Criterion | Description | AUROC |
|-----------|-------------|-------|
| A.1 | Depressed Mood | 0.83 |
| A.2 | Anhedonia | 0.88 |
| A.3 | Weight/Appetite Change | 0.91 |
| A.4 | Sleep Disturbance | 0.89 |
| A.5 | Psychomotor Changes | 0.80 |
| A.6 | Fatigue/Loss of Energy | 0.93 |
| A.7 | Worthlessness/Guilt | 0.92 |
| A.8 | Concentration Difficulty | 0.80 |
| A.9 | Suicidal Ideation | 0.95 |

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
