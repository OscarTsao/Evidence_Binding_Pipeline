# System Architecture

## Overview

The Evidence Binding Pipeline is a multi-stage retrieval system for identifying evidence sentences supporting DSM-5 MDD criteria in social media posts. The pipeline combines dense retrieval, cross-encoder reranking, and graph neural networks for state-of-the-art evidence retrieval.

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
│              Stage 3: Graph Reranking (P3 GNN)              │
│  Models sentence relationships via graph convolution         │
│  +10.48% nDCG@10 improvement (0.7330 → 0.8206)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 4: No-Evidence Detection (P4 GNN)        │
│  Criterion-Aware Heterogeneous Graph Classification         │
│  AUROC = 0.8972                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output                                    │
│  Ranked evidence sentences + confidence scores              │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Dense Retrieval (NV-Embed-v2)

### Model Architecture
- **Model**: `nvidia/NV-Embed-v2` (4096-dim embeddings)
- **Architecture**: Decoder-only LLM with bi-directional attention for embedding
- **Instruction-tuned**: Uses task-specific prompts for retrieval

### Method
1. Encode query (criterion description) with instruction prefix
2. Encode all sentences in the post as document embeddings
3. Compute cosine similarity between query and document embeddings
4. Return top-K candidates based on similarity scores

### Configuration
```yaml
# configs/default.yaml
models:
  retriever_name: nv-embed-v2
retriever:
  top_k_retriever: 24         # First-stage pool size
  embedding_dim: 4096         # NV-Embed-v2 output dimension
  batch_size: 32              # Encoding batch size
  instruction_prefix: "Instruct: Given a mental health criterion, retrieve evidence sentences that support it.\nQuery: "
```

### Environment Requirement
NV-Embed-v2 requires older transformers version. Use dedicated environment:
```bash
conda run -n nv-embed-v2 python scripts/encode_corpus.py
```

### Performance
- Encoding speed: ~500 sentences/second on RTX 5090
- Memory: ~16GB GPU RAM for model loading
- Recall@24: 0.97+ (measured on test set)

---

## Stage 2: Cross-Encoder Reranking (Jina-Reranker-v3)

### Model Architecture
- **Model**: `jinaai/jina-reranker-v3` (Qwen3-based)
- **Architecture**: Cross-encoder with bidirectional attention over query-document pairs
- **Scoring**: Listwise scoring with softmax over candidate set

### Method
1. Construct query-document pairs for each candidate
2. Feed pairs through cross-encoder
3. Apply listwise scoring (normalized across candidates)
4. Return top-K candidates with calibrated scores

### Configuration
```yaml
# configs/default.yaml
models:
  reranker_name: jina-reranker-v3
reranker:
  top_k_final: 10             # Final output size
  max_length: 1024            # Max sequence length
  batch_size: 128             # Reranking batch size
  use_listwise: true          # Enable listwise scoring
```

### Fusion Method (when using multiple retrievers)
```yaml
retriever:
  fusion_method: rrf          # Reciprocal Rank Fusion
  rrf_k: 60                   # RRF smoothing parameter
```

### Performance
- Reranking speed: ~200 queries/second
- nDCG@10 (baseline): 0.7330 ± 0.031 (5-fold CV)
- MRR: 0.6746 ± 0.037

---

## Stage 3: P3 Graph Reranker

### Model Architecture
The P3 Graph Reranker models sentence-to-sentence relationships within a post to refine reranker scores.

```
Input Graph Structure:
┌───────────────────────────────────────────────────────────┐
│  Nodes: Candidate sentences (with features)               │
│  Edges: Sentence similarity edges (cosine > threshold)    │
│  Node Features: [NV-Embed embedding, Jina score, position]│
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│  Input Projection: Linear(input_dim, hidden_dim)          │
│  → ReLU → Dropout                                          │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│  GNN Layers (configurable: gcn, sage, gat, gatv2)         │
│  h^(l+1) = ReLU(GNN(h^(l), A)) + Residual                │
│  → Dropout between layers                                  │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│  Output Projection: Linear(hidden_dim, 1) → GNN Score     │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│  Score Fusion:                                             │
│  final_score = α × reranker_score + (1-α) × gnn_score    │
│  α = sigmoid(learnable_alpha) ∈ (0, 1)                    │
└───────────────────────────────────────────────────────────┘
```

### Graph Construction
```python
# Edge construction (see scripts/gnn/rebuild_graph_cache.py)
for i, j in candidate_pairs:
    sim = cosine_similarity(embed[i], embed[j])
    if sim > edge_threshold:  # default: 0.5
        add_edge(i, j)
        add_edge(j, i)  # undirected
```

### Configuration
```yaml
# Best parameters (verified via comprehensive ablation)
gnn:
  gnn_type: sage              # Options: gcn, sage, gat, gatv2
  hidden_dim: 128             # GNN hidden dimension
  num_layers: 1               # Number of GNN layers
  num_heads: 2                # Attention heads (for gat/gatv2)
  dropout: 0.05               # Dropout rate
  alpha_init: 0.65            # Initial α for score fusion
  learn_alpha: true           # Make α learnable
  use_residual: true          # Residual connections (+0.78%)
  use_layer_norm: false       # LayerNorm hurts (-2.38%)

training:
  lr: 3.69e-05                # Learning rate
  weight_decay: 9.06e-06      # L2 regularization
  batch_size: 32              # Graph batch size
  max_epochs: 25              # Maximum training epochs
  patience: 10                # Early stopping patience

loss:
  type: margin_ranking        # Pairwise margin ranking loss
  margin: 0.1                 # Margin value
  alpha_rank: 1.0             # Ranking loss weight
  alpha_align: 0.5            # Alignment loss weight
  alpha_reg: 0.1              # Regularization weight
```

### Training Details
- **Loss Function**: Pairwise margin ranking loss
  ```
  L = max(0, margin - (s_pos - s_neg))
  ```
- **Sampling**: Up to 10 positive and 10 negative pairs per graph
- **Early Stopping**: Based on validation nDCG@10
- **Final Alpha**: Typically converges to ~0.70 (70% reranker, 30% GNN)

### Performance (5-Fold CV, SAGE+Residual)
| Metric | Baseline (Jina-v3) | With P3 GNN | Improvement |
|--------|-------------------|-------------|-------------|
| nDCG@10 | 0.7330 ± 0.031 | 0.8206 ± 0.030 | +10.48% |
| MRR | 0.6746 ± 0.037 | 0.7703 ± 0.035 | +12.02% |

Source: `outputs/comprehensive_ablation/`

---

## Stage 4: P4 Criterion-Aware GNN (No-Evidence Detection)

### Model Architecture
Heterogeneous graph neural network for binary classification (evidence exists vs. no evidence).

```
Heterogeneous Graph Structure:
┌───────────────────────────────────────────────────────────┐
│  Node Types:                                               │
│  - sentence: Candidate sentence embeddings                 │
│  - criterion: DSM-5 criterion embedding                    │
│                                                            │
│  Edge Types:                                               │
│  - (sentence, similar_to, sentence): Similarity edges      │
│  - (sentence, supports, criterion): Candidate-query edges  │
│  - (criterion, has_candidate, sentence): Query-candidate   │
└───────────────────────────────────────────────────────────┘
```

### Method
1. Build heterogeneous graph with sentence and criterion nodes
2. Apply heterogeneous graph convolution (HeteroConv)
3. Pool sentence representations
4. Binary classification: P(evidence exists | graph)

### Configuration
```yaml
p4_gnn:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.3
  pooling: mean              # Graph pooling method

classification:
  threshold: 0.5             # Classification threshold
  positive_weight: 2.0       # Weight for positive class (imbalanced)
```

### Performance
- AUROC: 0.8972
- AUPRC: 0.5709 (note: imbalanced classes)

---

## GNN Module Summary

| Module | Status | Purpose | Architecture | Key Metric |
|--------|--------|---------|--------------|------------|
| P1 | Deprecated | No-evidence gate | Simple GCN | AUROC 0.577 |
| P2 | Production | Dynamic-K selection | GCN + regressor | +2.7% hit rate |
| P3 | Production | Graph reranker | SAGE+Residual | nDCG@10 +10.48% |
| P4 | Production | Criterion-aware classification | HeteroGNN | AUROC 0.8972 |

**Available GNN architectures for P3:** GCN, SAGE (default), GAT, GATv2

---

## Data Pipeline

### Input Data
```
data/
├── redsm5/
│   ├── redsm5_posts.csv        # Reddit posts (post_id, text, ...)
│   └── redsm5_annotations.csv  # Evidence annotations (post_id, criterion_id, sent_idx, label)
├── DSM5/
│   └── MDD_Criteira.json       # Criterion definitions
└── groundtruth/
    ├── evidence_sentence_groundtruth.csv  # Processed ground truth
    └── sentence_corpus.jsonl              # Sentence-level corpus
```

### Cache Structure
```
data/cache/
├── retriever_zoo/
│   └── nv-embed-v2/           # Pre-computed embeddings
│       ├── corpus_embeddings.npy
│       └── query_embeddings.npy
└── gnn/
    └── rebuild_20260120/       # Graph cache for GNN
        ├── metadata.json
        ├── fold_0.pt           # Graphs for fold 0
        ├── fold_1.pt
        └── ...
```

### Graph Cache Format
Each `fold_X.pt` contains:
```python
{
    "graphs": [
        Data(
            x=tensor,              # Node features [N, D]
            edge_index=tensor,     # Edge connectivity [2, E]
            node_labels=tensor,    # Binary labels [N]
            reranker_scores=tensor,# Jina-v3 scores [N]
            post_id=str,
            criterion_id=str,
            sentence_indices=list,
        ),
        ...
    ],
    "metadata": {...}
}
```

---

## Evaluation Protocols

### 1. Positives-Only Protocol (Ranking)
- Computes ranking metrics only on queries with at least one positive
- Used for: nDCG@K, Recall@K, Precision@K, MRR, MAP@K, Hit@K

### 2. All-Queries Protocol (Classification)
- Computes classification metrics on all queries
- Used for: AUROC, AUPRC

### Metrics Computed
```python
# Ranking metrics at K = [1, 3, 5, 10, 20]
- nDCG@K      # Normalized Discounted Cumulative Gain
- Recall@K    # Proportion of positives in top-K
- Precision@K # Proportion of top-K that are positive
- Hit@K       # 1 if any positive in top-K, else 0
- MAP@K       # Mean Average Precision at K
- MRR         # Mean Reciprocal Rank

# Classification metrics
- AUROC       # Area Under ROC Curve
- AUPRC       # Area Under Precision-Recall Curve
```

---

## Configuration Files

### Main Configuration (`configs/default.yaml`)
```yaml
# DEFAULT CONFIGURATION - BEST MODEL COMBO
# Updated: 2026-01-24 based on 5-fold CV validated results

models:
  retriever_name: nv-embed-v2
  reranker_name: jina-reranker-v3

retriever:
  top_k_retriever: 24
  top_k_final: 10
  fusion_method: rrf
  rrf_k: 60

reranker:
  max_length: 1024
  batch_size: 128
  use_listwise: true

gnn:
  enabled: true
  checkpoint_dir: outputs/gnn_research/p3_retrained/20260120_190745/
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
  alpha_init: 0.7
  learn_alpha: true

split:
  seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  n_folds: 5

evaluation:
  protocols: [positives_only, all_queries]
  k_values: [1, 3, 5, 10, 20]
```

### Criteria Registry (`configs/criteria_registry.yaml`)
```yaml
# DSM-5 MDD Criteria
criteria:
  A.1: "Depressed mood most of the day, nearly every day"
  A.2: "Markedly diminished interest or pleasure"
  A.3: "Significant weight loss or gain"
  A.4: "Insomnia or hypersomnia"
  A.5: "Psychomotor agitation or retardation"
  A.6: "Fatigue or loss of energy"
  A.7: "Feelings of worthlessness or guilt"
  A.8: "Diminished ability to think or concentrate"
  A.9: "Recurrent thoughts of death or suicidal ideation"

excluded_from_gnn:
  - A.10  # SPECIAL_CASE (expert discrimination cases)
```

---

## Key Invariants

1. **Post-ID Disjoint Splits**: Train/Val/Test have no overlapping posts
2. **Within-Post Retrieval**: Candidates come from the queried post only
3. **No Feature Leakage**: Gold labels never used as input features
4. **Deterministic Evaluation**: Fixed seeds for reproducibility (seed=42)
5. **A.10 Exclusion**: A.10 (SPECIAL_CASE) is excluded from GNN training by default

### A.10 Exclusion Rationale

A.10 (SPECIAL_CASE - expert discrimination cases) is excluded from GNN training because:
- Not a standard DSM-5 criterion
- Low positive rate (5.8%) and poor performance (AUROC=0.665)
- Ablation study showed removing A.10 **improves nDCG@10 by +0.28%** on A.1-A.9

DSM-5 criteria (A.1-A.9) are used for training by default. See `src/final_sc_review/constants.py`.

---

## Training Scripts

### GNN Training
```bash
# Rebuild graph cache (after changing retriever/reranker)
python scripts/gnn/rebuild_graph_cache.py

# Train P3 Graph Reranker
python scripts/gnn/train_p3_graph_reranker.py

# Run HPO for GNN
python scripts/hpo/run_gnn_hpo_single_split.py --n_trials 100
```

### Evaluation
```bash
# Full 5-fold CV evaluation
python scripts/experiments/evaluate_from_cache.py

# Single split evaluation
python scripts/gnn/run_p3_integration.py --split test
```

---

## LLM Integration (Optional)

### LLM Verifier
- **Model**: Qwen2.5-7B-Instruct
- **Purpose**: Verify evidence-criterion matches
- **Performance**: 87% accuracy, AUROC 0.8855

### Suicidal Ideation Classifier (A.9)
- **Model**: Qwen2.5-7B-Instruct
- **Purpose**: Detect A.9 (suicidal ideation) mentions
- **Performance**: 75% accuracy, 91% recall, AUROC 0.8902

### LLM Reranker (Deprecated)
- **Status**: Experimental only
- **Issue**: High position bias (only 9.8% samples achieved low bias)
- **Recommendation**: Use Jina-Reranker-v3 instead
