# Repository Overview

This document provides a comprehensive map of the Evidence Binding Pipeline repository structure, key invariants, and reproducibility artifacts.

---

## What This System Does

The Evidence Binding Pipeline is a **sentence-criterion (S-C) evidence retrieval system** for mental health research. Given:
- A Reddit post from mental health communities
- A DSM-5 Major Depressive Disorder (MDD) criterion (e.g., "A.1 Depressed Mood")

The system retrieves sentences from the post that serve as **evidence** supporting or refuting the criterion.

### Pipeline Stages (HPO-Optimized)

```
Query (DSM-5 Criterion) + Post
    ↓
[Stage 1] NV-Embed-v2 Retriever (top-24 candidates)
    ↓
[Stage 2] Jina-Reranker-v3 Cross-Encoder (top-10)
    ↓
[Stage 3] P3 GNN Graph Reranker (optional refinement)
    ↓
[Stage 4] P2 GNN Dynamic-K Selection (K ∈ [3,20])
    ↓
[Stage 5] P4 GNN NE Gate (No-Evidence Detection)
    ↓
[Stage 6] Three-State Clinical Gate (NEG/UNCERTAIN/POS)
    ↓
Evidence Sentences (ranked by relevance)
```

### Model Selection

From 324 retriever×reranker combinations tested:
- **Retriever:** NV-Embed-v2 (nvidia/NV-Embed-v2) - Best of 25 retrievers
- **Reranker:** Jina-Reranker-v3 (jinaai/jina-reranker-v3) - Best of 15 rerankers
- **HPO Result:** nDCG@10 = 0.8658 on DEV split

---

## Architecture Map

### Source Code (`src/final_sc_review/`)

| Package | Purpose |
|---------|---------|
| `data/` | Data loading, splits, I/O utilities |
| `retriever/` | Retriever implementations (NV-Embed-v2, BGE-M3, zoo) |
| `reranker/` | Reranker implementations (Jina-v3, zoo) |
| `pipeline/` | Pipeline orchestration (ZooPipeline, ThreeStage) |
| `gnn/` | GNN modules (P1-P4: NE Gate, Dynamic-K, Graph Reranker) |
| `metrics/` | Ranking and classification metrics |
| `clinical/` | Clinical deployment utilities (three-state gate) |
| `hpo/` | Hyperparameter optimization |
| `llm/` | LLM integration (reranker, verifier) |
| `postprocessing/` | Post-processing utilities |
| `utils/` | Logging, config, helpers |

### Scripts (`scripts/`)

| Directory/Script | Purpose |
|------------------|---------|
| `ablation/` | Ablation study framework |
| `analysis/` | Error analysis, efficiency metrics |
| `baselines/` | Baseline comparison (BM25, TF-IDF) |
| `clinical/` | Clinical high-recall evaluation |
| `eval/` | Evaluation utilities |
| `gnn/` | GNN-specific evaluation |
| `reporting/` | Paper bundle packaging |
| `robustness/` | Multi-seed robustness evaluation |
| `verification/` | Checksums, metric cross-check, plots |
| `build_groundtruth.py` | Build sentence-level labels |
| `build_sentence_corpus.py` | Build canonical sentence corpus |
| `eval_zoo_pipeline.py` | Main evaluation script (recommended) |
| `audit_splits.py` | Verify post-ID disjoint splits |
| `encode_corpus.py` | Pre-compute embeddings |
| `run_paper_reproduce.sh` | Full paper reproduction pipeline |

### Tests (`tests/`)

| Directory/File | Coverage |
|----------------|----------|
| `metrics/` | Ranking metrics (Recall@K, nDCG@K, MRR, MAP) |
| `leakage/` | Data leakage prevention (12+ tests) |
| `clinical/` | Clinical no-leakage tests |
| `llm/` | LLM integration tests |
| `test_publication_gate.py` | Publication readiness gate (required files, forbidden paths) |
| `test_gnn_no_leakage.py` | GNN-specific leakage tests |
| `test_dynamic_k_*.py` | Dynamic-K behavior tests |
| `test_no_evidence_*.py` | No-evidence handling tests |

---

## Key Invariants

### 1. Post-ID Disjoint Splits

**Critical invariant:** No post appears in multiple splits (TRAIN/VAL/TEST).

- **Why:** Sentences from the same post share context; mixing would cause data leakage
- **Implementation:** `src/final_sc_review/data/splits.py`
- **Verification:** `scripts/audit_splits.py`, `tests/test_no_leakage_splits.py`

### 2. Within-Post Retrieval

**Invariant:** Candidate pool for retrieval is always sentences from the same post.

- **Why:** Clinically meaningful - we retrieve evidence from the post being analyzed
- **Implementation:** `ZooPipeline.retrieve()` filters by `post_id`

### 3. Dual Protocol Metrics

**Invariant:** Different metric protocols for different tasks:

| Protocol | Applies To | Description |
|----------|------------|-------------|
| `positives_only` | Ranking metrics (nDCG, Recall, MRR) | Computed only on queries where `has_evidence=1` |
| `all_queries` | Classification metrics (AUROC, AUPRC) | Computed on all queries (binary classification) |

- **Documentation:** `docs/final/METRIC_CONTRACT.md`
- **Implementation:** `src/final_sc_review/metrics/ranking.py`

### 4. Nested Threshold Selection

**Invariant:** Thresholds and hyperparameters selected on TUNE/DEV split only.

- **Why:** Prevents information leakage from TEST split
- **Verification:** `tests/test_hpo_never_uses_test_split.py`

### 5. No Gold Features in Inference

**Invariant:** Ground truth labels never used as features during inference.

- **Verification:** `tests/test_gnn_no_leakage.py`, feature audit in `report.md`

---

## Reproducibility Artifacts

### Paper Bundle (`results/paper_bundle/v1.0/`)

| File | Purpose |
|------|---------|
| `report.md` | Comprehensive academic evaluation report |
| `summary.json` | Machine-readable results summary |
| `metrics_master.json` | **Single source of truth** for all metrics |
| `MANIFEST.md` | Bundle contents and regeneration instructions |
| `checksums.txt` | SHA256 checksums for verification |
| `figures/` | Publication figures (10 PNG files) |
| `tables/` | Machine-readable tables (to be populated) |

### Verification Tools

| Tool | Command | Purpose |
|------|---------|---------|
| Checksum verification | `python scripts/verification/verify_checksums.py` | Verify bundle integrity |
| Metric cross-check | `python scripts/verification/metric_crosscheck.py` | Independent metric recomputation |
| Split audit | `python scripts/audit_splits.py` | Verify no data leakage |
| Publication gate | `pytest tests/test_publication_gate.py` | Required files, forbidden paths |

### Environment Requirements

**Dual conda environments required:**

| Environment | Purpose | Key Constraint |
|-------------|---------|----------------|
| `nv-embed-v2` | NV-Embed-v2 retriever | `transformers<=4.44` |
| `llmhe` | Reranking, GNN, evaluation | `transformers>=4.45` |

See `docs/ENVIRONMENT_SETUP.md` for setup instructions.

---

## Data Flow

```
data/
├── redsm5/
│   ├── redsm5_posts.csv        # Reddit posts (post_id, text)
│   └── redsm5_annotations.csv  # Evidence annotations
├── DSM5/
│   └── MDD_Criteira.json       # DSM-5 criterion definitions
└── groundtruth/                # Generated artifacts
    ├── evidence_sentence_groundtruth.csv  # Sentence-level labels
    └── sentence_corpus.jsonl              # Canonical sentences
```

**Groundtruth schema:**
- `post_id`: Unique post identifier
- `criterion`: DSM-5 criterion (A.1-A.9)
- `sid`: Sentence index within post
- `sent_uid`: Unique sentence ID (`{post_id}_{sid}`)
- `sentence`: Sentence text
- `groundtruth`: Binary label (0/1)

---

## Configuration

### Primary Config (`configs/default.yaml`)

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

---

## Quick Reference Commands

```bash
# Run all tests
pytest -q

# Run publication gate
pytest tests/test_publication_gate.py -v

# Verify checksums
python scripts/verification/verify_checksums.py

# Audit splits for leakage
python scripts/audit_splits.py --data_dir data --seed 42 --k 5

# Full paper reproduction
bash scripts/run_paper_reproduce.sh
```

---

## Related Documentation

- [README.md](../README.md) - Project overview and quick start
- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Dual environment setup
- [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) - Data access procedures
- [ETHICS.md](ETHICS.md) - Privacy and ethical considerations
- [docs/final/PAPER_REPRODUCIBILITY.md](final/PAPER_REPRODUCIBILITY.md) - Paper reproduction guide
- [docs/final/METRIC_CONTRACT.md](final/METRIC_CONTRACT.md) - Metric definitions
