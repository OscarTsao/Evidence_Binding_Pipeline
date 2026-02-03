# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

This is a sentence-criterion (S-C) evidence retrieval pipeline for mental health research. Given a post and a DSM-5 criterion, the system retrieves evidence sentences supporting the criterion.

### Best Model Configuration (5-Fold CV Validated)
- **Retriever:** NV-Embed-v2 (nvidia/NV-Embed-v2) - Best from 25 retrievers
- **Reranker:** Jina-Reranker-v3 (jinaai/jina-reranker-v3) - Best from 15 rerankers
- **GNN Reranker:** SAGE + Residual (2-layer) + GELU activation
- **Performance (5-Fold CV):**
  - Baseline (NV-Embed + Jina): nDCG@10 = 0.7428 ± 0.033
  - With SAGE + Residual + GELU: nDCG@10 = **0.8237 ± 0.030** (+10.89%)
  - MRR: 0.7823 ± 0.032

> **Source of Truth:** `outputs/experiments/activation_validation/` (GELU is new best)

### Graph Construction
```yaml
# Graph construction parameters (original configuration is optimal)
knn_k: 5
knn_threshold: 0.5  # Original threshold is best
edge_types: [semantic_knn, adjacency]
node_features:
  - embedding (4096 dim)
  - reranker_score (normalized)
  - rank_percentile
```
Graph cache: `data/cache/gnn/rebuild_20260120/`

### Best HPO Parameters (Jina-Reranker-v3)
```yaml
top_k_retriever: 24
top_k_final: 10
fusion_method: rrf
rrf_k: 60
reranker_max_length: 1024
reranker_batch_size: 128
reranker_use_listwise: true
```

### GNN Architecture (SAGE + Residual - Verified Best)
Comprehensive ablation tested 17+ configurations including architectures, layer counts, hidden dimensions, dropout, and normalization:
```yaml
# SAGE + Residual + GELU parameters (verified best)
architecture: sage
hidden_dim: 128
num_layers: 2  # 2 layers with residual connections
dropout: 0.05
activation: gelu  # GELU > ReLU: +0.52% nDCG@10
alpha_init: 0.65
learn_alpha: true
use_residual: true
use_layer_norm: false  # LayerNorm hurts performance (-2.38%)
lr: 3.69e-05
n_epochs: 25
weight_decay: 9.06e-06
batch_size: 32
# IMPORTANT: Simple margin loss only! Complex loss hurts performance.
loss: PairwiseMarginLoss (margin=0.1)
alpha_align: 0.0  # Do NOT use alpha_align/alpha_reg
alpha_reg: 0.0    # Complex loss reduces nDCG by ~0.7%
```

**Verified Experiment Results (5-Fold CV):**
| Rank | Configuration | nDCG@10 | MRR | vs SAGE 1-layer |
|------|---------------|---------|-----|-----------------|
| 1 | **SAGE + Residual** | **0.8206 ± 0.030** | 0.7703 | **+0.78%** |
| 2 | SAGE 2-layer | 0.8170 ± 0.030 | - | +0.33% |
| 3 | SAGE hidden=256 | 0.8166 ± 0.030 | - | +0.29% |
| 4 | SAGE hidden=512 | 0.8160 ± 0.031 | - | +0.21% |
| 5 | SAGE aggr=max | 0.8149 ± 0.030 | - | +0.08% |
| 6 | SAGE 1-layer | 0.8143 ± 0.030 | - | 0.00% |
| 7 | **GAT v2 (heads=2)** | 0.8047 ± 0.029 | 0.7588 | -1.18% |
| - | SAGE + LayerNorm | 0.7948 ± 0.025 | - | -2.38% |

**Key Findings:**
- ✅ SAGE + Residual + GELU is the best: nDCG@10 = **0.8237** (NEW!)
- ✅ GELU activation > ReLU: +0.52% improvement
- ✅ Simple margin loss is optimal; complex loss hurts performance
- ✅ Residual connections help significantly
- ❌ GAT v2 underperforms SAGE: -1.97%
- ❌ LayerNorm hurts significantly: -2.38%
- ❌ Complex loss (alpha_align, alpha_reg > 0) hurts: -0.85%

**Available GNN Types:** gcn, sage, gat, gatv2 (use `--gnn_type` flag)

Run training:
```bash
conda activate llmhe
# Best configuration (SAGE + Residual with simple loss)
python scripts/gnn/train_p3_graph_reranker.py \
  --gnn_type sage --num_layers 2 --dropout 0.05 \
  --lr 3.69e-5 --alpha_align 0.0 --alpha_reg 0.0 --margin 0.1
```

### A.10 Exclusion (Default)
A.10 (SPECIAL_CASE) is **excluded from GNN training by default** because:
- It's not a standard DSM-5 criterion (it's "expert discrimination cases")
- Low positive rate (5.8%) and poor performance (AUROC=0.665)
- Ablation study showed removing A.10 **improves nDCG@10 by +0.28%**

To include A.10 (not recommended):
```bash
python scripts/gnn/rebuild_graph_cache.py --include_a10
python scripts/gnn/train_p3_graph_reranker.py --include_a10
```

## Conda Environment Requirements

**IMPORTANT:** This project uses two conda environments due to dependency conflicts:

| Environment | Purpose | Models |
|-------------|---------|--------|
| `nv-embed-v2` | Retriever encoding | NV-Embed-v2 (requires specific transformers version) |
| `llmhe` | Everything else | Jina-Reranker-v3, GNN models, LLM experiments |

### When to Use Each Environment

```bash
# nv-embed-v2: ONLY for encoding sentences with NV-Embed-v2 retriever
conda run -n nv-embed-v2 python scripts/encode_corpus.py

# llmhe: For all other operations
conda run -n llmhe python scripts/eval_zoo_pipeline.py
conda run -n llmhe python scripts/gnn/run_p3_integration.py
conda run -n llmhe python scripts/llm/run_llm_eval.py
```

### Why Two Environments?

- `nv-embed-v2`: Has older transformers that works with NV-Embed-v2 model's custom code
- `llmhe`: Has newer transformers (4.57+) required for Jina-Reranker-v3 (qwen3 architecture)

The NV-Embed-v2 embeddings are cached (`data/cache/retriever_zoo/nv-embed-v2/`), so most operations only need `llmhe`.

## Commands

### Setup
```bash
# Install in development mode (use llmhe environment)
conda activate llmhe
pip install -e .
```

### Build Data Artifacts
```bash
# Build groundtruth labels
python scripts/build_groundtruth.py --data_dir data --output data/groundtruth/evidence_sentence_groundtruth.csv

# Build sentence corpus
python scripts/build_sentence_corpus.py --data_dir data --output data/groundtruth/sentence_corpus.jsonl
```

### Evaluation
```bash
# Evaluate with best HPO model combo
python scripts/eval_zoo_pipeline.py --config configs/default.yaml --split test

# Single query inference
python scripts/run_single_zoo.py --config configs/default.yaml --post_id <POST_ID> --criterion_id <CRITERION_ID>
```

### Testing
```bash
pytest -q                         # Run all tests
pytest tests/test_metrics.py -v   # Run specific test file
pytest -k "test_name"             # Run tests matching pattern
```

### Verification
```bash
# Verify paper bundle checksums
python scripts/verification/verify_checksums.py

# Audit splits for leakage
python scripts/audit_splits.py --data_dir data --seed 42 --k 5
```

## Architecture

### Zoo Pipeline (`src/final_sc_review/pipeline/zoo_pipeline.py`)
The recommended pipeline using retriever and reranker zoos:
1. **Stage 1 - Retrieval:** NV-Embed-v2 (top-24)
2. **Stage 2 - Reranking:** Jina-Reranker-v3 (top-10)

### Key Modules
- `src/final_sc_review/retriever/zoo.py` - NV-Embed-v2 retriever (simplified)
- `src/final_sc_review/reranker/zoo.py` - Jina-Reranker-v3 (simplified)
- `src/final_sc_review/gnn/` - GNN modules (P1-P4)
- `src/final_sc_review/llm/` - LLM integration (verifier, A.9 suicidal ideation classifier)
- `src/final_sc_review/metrics/ranking.py` - Ranking metrics

### GNN Modules (`src/final_sc_review/gnn/`)

| Module | Status | Description | Key Metric |
|--------|--------|-------------|------------|
| P1 | Deprecated | NE Gate (no-evidence detection) | AUROC=0.577 |
| P2 | Production | Dynamic-K selection | Adaptive cutoff |
| P3 | Production | Graph Reranker (SAGE+Residual) | nDCG@10 +10.48% (5-fold CV) |
| P4 | Production | Criterion-Aware GNN | AUROC=0.8972 |

**P3 Graph Reranker** uses sentence graph structure to refine reranker scores:
- Checkpoints: `outputs/gnn_research/p3_retrained/20260120_190745/`
- Graph cache: `data/cache/gnn/rebuild_20260120/`
- Training: `scripts/gnn/train_p3_graph_reranker.py`
- Evaluation: `scripts/gnn/run_p3_integration.py`

### Removed/Deprecated Code
- Legacy three-stage pipeline (BGE-M3 retriever) - removed
- HPO/Ablation scripts - removed (best config is now fixed)
- 24 other retrievers - removed (only NV-Embed-v2 kept)
- 14 other rerankers - removed (only Jina-Reranker-v3 kept)

## Key Invariants

1. **Post-ID Disjoint Splits:** No post appears in multiple splits
2. **Within-Post Retrieval:** Candidate pool is sentences from the same post
3. **Dual Protocol Metrics:**
   - `positives_only`: Ranking metrics (nDCG, Recall, MRR)
   - `all_queries`: Classification metrics (AUROC, AUPRC)
4. **No Gold Features:** Ground truth never used during inference

## Configuration

YAML configs control all pipeline parameters. Key settings in `configs/default.yaml`:

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

## Metrics Source of Truth

The single source of truth for all metrics is:
- `outputs/comprehensive_ablation/` (original ablation)
- `outputs/experiments/combined_optimization/sage_residual_simple_loss/` (validated 2026-01-30)

Primary metrics (5-fold CV, positives_only protocol):
| Metric | Baseline | GNN (SAGE+Residual) | Improvement |
|--------|----------|---------------------|-------------|
| nDCG@10 | 0.7428 ± 0.033 | 0.8207 ± 0.027 | +10.49% |
| MRR | 0.6862 ± 0.042 | 0.7795 ± 0.030 | +13.60% |
| Recall@10 | 0.9485 ± 0.021 | 0.9659 ± 0.015 | +1.83% |

Classification (all_queries protocol):
- AUROC: 0.8972
- AUPRC: 0.5709

### LLM Integration (Evaluated 2026-01-24)
| Module | Status | Performance |
|--------|--------|-------------|
| LLMVerifier | Production-ready | 87% acc, 0.8855 AUROC |
| A.9 Classifier | Production-ready | 75% acc, 0.8902 AUROC, 91% recall |
| LLMReranker | Experimental | High position bias (not recommended) |

## Troubleshooting

### Common Issues

**"model type 'qwen3' not recognized"**
- Use `llmhe` environment (has transformers 4.57+)
- Don't use `nv-embed-v2` env for Jina-Reranker-v3

**"Unknown retriever: nvidia/NV-Embed-v2"**
- Use short name: `nv-embed-v2` (not full HuggingFace path)

**P3 checkpoint dimension mismatch**
- P3 must be trained with same embeddings used in graph cache
- Retrain P3 if changing retriever: `scripts/gnn/train_p3_graph_reranker.py`

**Tests fail on import**
- Ensure `from __future__` imports are first in file
- Run `pip install -e .` to install package

## Performance Profile (2026-01-30)

**Pipeline Latency Breakdown (per query):**
| Component | Latency | Notes |
|-----------|---------|-------|
| Reranker (Jina-v3) | ~14ms | **Main bottleneck** |
| GNN (SAGE) | ~0.5ms | Very fast |
| Retrieval | ~1ms | Uses cached embeddings |

**Reranker Details:**
- Short text (24 candidates): 14.0ms
- Long text (24 candidates): 27.5ms (2x slower)
- Batch processing: 1.94x faster than sequential
- GPU memory: ~1.2GB

**Scaling:**
- 10 candidates: 10ms
- 24 candidates: 14ms
- 50 candidates: 26ms
- 100 candidates: 48ms

**Optimization Recommendations:**
1. Use batch mode where possible (1.94x speedup)
2. Truncate long sentences (reduces latency 2x)
3. Keep top_k_retriever at 24 (good latency/quality tradeoff)
4. Enable `use_torch_compile=True` for ~1.4x speedup (requires warmup)

**Experimental Results (2026-01-30):**
- torch.compile: 17.7ms → 12.6ms (1.40x speedup) ✅ Recommended
- batch_size=32+ optimal (16 has high variance)
- max_length has minimal impact (256-1024 similar for typical text)
- INT8 quantization: 36% memory savings but 5x slower ❌ Not recommended
- FP16 + torch.compile is optimal for RTX 5090

## Code Optimization Status (2026-01-30)

The codebase has been thoroughly optimized through 9 improvement cycles:

**Performance Optimizations:**
- Query embedding cache in retriever (avoids re-encoding repeated queries)
- FAISS-based kNN for graph building (O(n log n) vs O(n²))
- Vectorized metric computation (replaced pandas iterrows)
- Generator expressions for memory efficiency

**Code Quality:**
- Centralized JSON extraction in `llm/base.py`
- Centralized device/dtype handling in `utils/gpu_optimize.py`
- Profiling utilities in `utils/profiling.py`
- Fail-fast config validation in `pipeline/zoo_pipeline.py`
- All file operations use explicit UTF-8 encoding
- 100% type coverage in core modules
- 258 tests passing (33.5% test-to-code ratio)

**Consolidated Functions:**
- `extract_json_from_response()` - shared LLM response parsing
- `focal_loss()` - single implementation in `gnn/training/losses.py`
- `get_device()`, `get_optimal_dtype()` - centralized GPU utilities

## Repository Structure

```
src/final_sc_review/
├── data/           # Data I/O and schemas
├── retriever/      # NV-Embed-v2 retriever (best from 25 candidates)
├── reranker/       # Jina-Reranker-v3 (best from 15 candidates)
├── pipeline/       # ZooPipeline (recommended)
├── gnn/            # GNN models (P1-P4)
├── metrics/        # Ranking and classification metrics
├── postprocessing/ # Calibration and dynamic-K
├── hpo/            # Hyperparameter optimization utilities
├── llm/            # LLM integration (verifier, A.9 classifier, NE detector)
├── clinical/       # Clinical gate logic
├── utils/          # Shared utilities
└── _archived/      # Archived research code (baselines, research losses, HPO objectives)

scripts/
├── encode_nv_embed.py  # Encode corpus with NV-Embed-v2 (run in nv-embed-v2 env)
├── eval_zoo_pipeline.py # Evaluate pipeline
├── run_single_zoo.py    # Single query inference
├── gnn/                 # GNN training and evaluation
├── verification/        # Metric verification and audits
├── llm/                 # LLM experiments
└── _archived/           # Archived HPO and experiment scripts
```
