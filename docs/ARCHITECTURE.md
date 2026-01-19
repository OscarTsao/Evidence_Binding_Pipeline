# System Architecture

## Overview

The Evidence Binding Pipeline is a multi-stage retrieval system for identifying evidence sentences supporting DSM-5 MDD criteria in social media posts.

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing                          │
│  Post Text + DSM-5 Criterion Query                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: Dense Retrieval                        │
│  NV-Embed-v2 Encoder → Top-24 candidates                    │
│  (Within-post constraint: candidates from same post only)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 2: Cross-Encoder Reranking               │
│  Jina-Reranker-v3 → Top-10 candidates                       │
│  (Listwise scoring with query-document attention)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 3: Graph Reranking (Optional)            │
│  P3 GNN: Models sentence relationships                      │
│  (+19.0% Recall@10 improvement)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 4: Dynamic-K Selection                    │
│  P2 GNN: Adaptive cutoff K ∈ [3, 20]                        │
│  (+2.7% hit rate vs fixed-K)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 5: No-Evidence Detection                  │
│  P4 GNN (NE Gate): Binary classification                    │
│  AUROC = 0.8972                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output                                    │
│  Ranked evidence sentences + confidence scores              │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Retriever Zoo (`src/final_sc_review/retriever/zoo.py`)

Supports 25+ retrievers including:
- **NV-Embed-v2** (default): Best performance in HPO
- BGE-M3: Hybrid dense/sparse/ColBERT
- E5-Mistral-7B: Large language model embeddings
- Qwen3-Embed: Multilingual support

### Reranker Zoo (`src/final_sc_review/reranker/zoo.py`)

Supports 15+ rerankers including:
- **Jina-Reranker-v3** (default): Best performance in HPO
- BGE-Reranker-v2-M3: Multilingual
- MxBAI-Rerank: Fast inference

### GNN Modules (`src/final_sc_review/gnn/`)

| Module | Purpose | Performance |
|--------|---------|-------------|
| P1 NE Gate | No-evidence detection (deprecated) | - |
| P2 Dynamic-K | Adaptive cutoff selection | +2.7% hit rate |
| P3 Graph Reranker | Sentence relationship modeling | +19.0% Recall@10 |
| P4 Criterion-Aware | Heterogeneous graph classification | AUROC 0.8972 |

### Metrics (`src/final_sc_review/metrics/`)

Two evaluation protocols:
1. **positives_only**: Ranking metrics computed only on queries with evidence
2. **all_queries**: Classification metrics computed on all queries

## Data Flow

```
data/
├── redsm5/
│   ├── redsm5_posts.csv        # Reddit posts
│   └── redsm5_annotations.csv  # Evidence annotations
├── DSM5/
│   └── MDD_Criteira.json       # Criterion definitions
└── groundtruth/
    ├── evidence_sentence_groundtruth.csv
    └── sentence_corpus.jsonl
```

## Key Invariants

1. **Post-ID Disjoint Splits**: Train/Val/Test have no overlapping posts
2. **Within-Post Retrieval**: Candidates come from the queried post only
3. **No Feature Leakage**: Gold labels never used as input features
4. **Deterministic Evaluation**: Fixed seeds for reproducibility (seed=42)

## Configuration

Primary configuration in `configs/default.yaml`:

```yaml
models:
  retriever_name: nv-embed-v2
  reranker_name: jina-reranker-v3

retriever:
  top_k_retriever: 24    # First-stage pool size
  top_k_final: 10        # Final output size

split:
  seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```
