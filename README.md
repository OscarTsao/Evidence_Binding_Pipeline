# Evidence Binding Pipeline

Sentence-Criterion (S-C) evidence retrieval for mental health research. Given a Reddit post and a DSM-5 criterion, retrieve sentences that serve as evidence.

## Best Model Configuration

| Component | Model | Performance |
|-----------|-------|-------------|
| **Retriever** | NV-Embed-v2 (`nvidia/NV-Embed-v2`) | 4096d embeddings |
| **Reranker** | Jina-Reranker-v3 (`jinaai/jina-reranker-v3`) | Cross-encoder |
| **nDCG@10** | **0.8658** | From 324 model combinations |
| **AUROC (NE Gate)** | **0.8972** | P4 Criterion-Aware GNN |

## Quick Start

### 1. Environment Setup

This project requires **two conda environments** due to dependency conflicts:

```bash
# Retriever environment (NV-Embed-v2, transformers<=4.44)
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt

# Main environment (reranking, GNN, evaluation)
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .
```

See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for detailed setup instructions.

### 2. Data Setup

Place data locally (not tracked in git):

```
data/
├── redsm5/
│   ├── redsm5_posts.csv
│   └── redsm5_annotations.csv
├── DSM5/
│   └── MDD_Criteira.json
└── groundtruth/           # Generated
    ├── evidence_sentence_groundtruth.csv
    └── sentence_corpus.jsonl
```

### 3. Build Groundtruth

```bash
conda activate llmhe

# Build sentence-level labels
python scripts/build_groundtruth.py \
    --data_dir data \
    --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus
python scripts/build_sentence_corpus.py \
    --data_dir data \
    --output data/groundtruth/sentence_corpus.jsonl
```

### 4. Reproduce Paper Results

```bash
# Full reproduction (handles environment switching)
bash scripts/run_paper_reproduce.sh
```

Or run individual steps:

```bash
# 1. Run tests (main env)
conda activate llmhe
pytest -q

# 2. Audit splits (main env)
python scripts/audit_splits.py --data_dir data --seed 42 --k 5

# 3. Evaluate with NV-Embed-v2 (requires nv-embed-v2 env)
conda activate nv-embed-v2
pip install -e .  # Install package in this env too
python scripts/eval_zoo_pipeline.py \
    --config configs/default.yaml \
    --split test

# Alternative: Encode corpus only, then use cached embeddings
python scripts/encode_corpus.py \
    --retriever nv-embed-v2 \
    --corpus data/groundtruth/sentence_corpus.jsonl \
    --output data/cache/nv-embed-v2
```

## Project Structure

```
├── configs/               # YAML configuration files
├── data/                  # Data files (not tracked)
├── docs/                  # Documentation
│   ├── ENVIRONMENT_SETUP.md
│   ├── final/            # Paper reproduction docs
│   └── verification/     # Audit reports
├── envs/                  # Environment requirements
│   ├── requirements-retriever.txt
│   └── requirements-main.txt
├── outputs/               # Generated outputs (not tracked)
├── paper/                 # Paper figures and tables
├── results/               # Paper bundle
├── scripts/               # Executable scripts
├── src/                   # Source code
│   └── final_sc_review/
│       ├── data/         # Data loading
│       ├── gnn/          # GNN modules (P1-P4)
│       ├── metrics/      # Evaluation metrics
│       ├── pipeline/     # Pipeline implementations
│       ├── reranker/     # Reranker models
│       └── retriever/    # Retriever models
└── tests/                 # Test suite
```

## Key Commands

| Command | Description |
|---------|-------------|
| `bash scripts/run_paper_reproduce.sh` | Full paper reproduction |
| `python scripts/eval_zoo_pipeline.py` | Evaluate pipeline |
| `python scripts/audit_splits.py` | Verify no data leakage |
| `python scripts/encode_corpus.py` | Pre-compute embeddings |
| `pytest` | Run all tests |

## Documentation

- [ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) - Conda environment setup
- [PAPER_COMMANDS.md](docs/final/PAPER_COMMANDS.md) - Paper reproduction commands
- [PAPER_REPRODUCIBILITY.md](docs/final/PAPER_REPRODUCIBILITY.md) - Full reproduction guide
- [METRIC_CONTRACT.md](docs/final/METRIC_CONTRACT.md) - Metric definitions

## Key Design Decisions

1. **Post-ID Disjoint Splits**: All train/val/test splits ensure no post appears in multiple splits (prevents data leakage)

2. **Within-Post Retrieval**: Candidate pool is always sentences from the same post (clinically meaningful)

3. **Dual Environment**: NV-Embed-v2 requires `transformers<=4.44`, other components need `transformers>=4.45`

4. **Caching**: Embeddings are cached with corpus fingerprints for reproducibility

## Results Summary

| Metric | Value | Protocol | Split |
|--------|-------|----------|-------|
| nDCG@10 | 0.8658 | positives_only | TEST |
| Evidence Recall@10 | 0.7043 | positives_only | TEST |
| AUROC (NE Gate) | 0.8972 [0.8941, 0.9003] | all_queries | TEST |
| Screening Sensitivity | 99.78% | clinical | TEST |
| Alert Precision | 93.5% | clinical | TEST |

### Results Provenance

- **Protocol**: Ranking metrics (nDCG, Recall, MRR) computed on queries with evidence only (`has_evidence=1`). Classification metrics (AUROC, AUPRC) computed on all queries. See [METRIC_CONTRACT.md](docs/final/METRIC_CONTRACT.md) for definitions.
- **Split**: TEST split (10% of posts, seed=42, post-ID disjoint)
- **Single source of truth**: [metrics_master.json](results/paper_bundle/v1.0/metrics_master.json)

## Citation

```bibtex
@article{evidence_binding_2026,
  title={Evidence Binding Pipeline for Mental Health Assessment},
  author={...},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Data Availability

The data used in this research is not publicly available due to privacy considerations and IRB requirements. Researchers interested in accessing the data should:

1. Contact the authors with a description of the intended use
2. Provide evidence of IRB approval (or equivalent ethics approval)
3. Sign a data use agreement

The data includes:
- Reddit posts from mental health communities (anonymized)
- DSM-5 MDD criteria definitions
- Human annotations mapping sentences to criteria

See [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) for detailed data access procedures.

## Ethics

This research involves analysis of social media posts from mental health communities. We follow strict privacy and ethical guidelines. See [docs/ETHICS.md](docs/ETHICS.md) for details on:
- Data anonymization procedures
- Privacy protections
- Intended use and misuse prevention
- Compliance with platform terms of service
