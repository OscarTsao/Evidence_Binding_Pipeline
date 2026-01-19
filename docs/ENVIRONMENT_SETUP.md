# Environment Setup Guide

This project requires **two separate conda environments** due to dependency conflicts between NV-Embed-v2 and other components.

## Why Two Environments?

The NV-Embed-v2 retriever model requires `transformers<=4.44.x`, but newer components (Jina-Reranker-v3, sentence-transformers 5.x) require `transformers>=4.45.0`. Running NV-Embed-v2 with newer transformers causes:

```
TypeError: cannot unpack non-iterable NoneType object
```

## Environment Overview

| Environment | Purpose | Key Constraint |
|-------------|---------|----------------|
| `nv-embed-v2` | NV-Embed-v2 retriever encoding | `transformers>=4.42,<4.45` |
| `llmhe` | Reranking, GNN, evaluation, tests | `transformers>=4.45` |

## Quick Setup

```bash
# 1. Create retriever environment (for NV-Embed-v2)
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt

# 2. Create main environment (for everything else)
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .  # Install this package

# 3. Install PyTorch Geometric (main env only)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Detailed Setup

### Step 1: Retriever Environment (nv-embed-v2)

```bash
# Create environment
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r envs/requirements-retriever.txt

# Verify NV-Embed-v2 works
python -c "from transformers import AutoModel; print('OK')"
```

### Step 2: Main Environment (llmhe)

```bash
# Create environment
conda create -n llmhe python=3.10 -y
conda activate llmhe

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r envs/requirements-main.txt

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install this package in editable mode
pip install -e .

# Verify installation
pytest -q  # All tests should pass
```

## Usage in Scripts

The paper reproduction script automatically switches environments:

```bash
# Full reproduction (handles env switching internally)
bash scripts/run_paper_reproduce.sh
```

### Manual Environment Switching

```bash
# For retrieval/encoding tasks
conda activate nv-embed-v2
python scripts/encode_corpus.py --retriever nv-embed-v2

# For everything else (reranking, GNN, evaluation)
conda activate llmhe
python scripts/eval_zoo_pipeline.py --config configs/default.yaml
```

## Embedding Cache

To avoid repeated encoding, embeddings are cached:

```
data/cache/nv-embed-v2/    # NV-Embed-v2 embeddings (3.4GB)
data/cache/bge_m3/         # BGE-M3 embeddings (2.4GB)
```

The cache is automatically used if the corpus hash matches. Pre-computed embeddings can be shared between environments.

## Troubleshooting

### "TypeError: cannot unpack non-iterable NoneType object"

**Cause:** Running NV-Embed-v2 with `transformers>=4.45`

**Solution:** Use the `nv-embed-v2` environment:
```bash
conda activate nv-embed-v2
```

### "ModuleNotFoundError: No module named 'torch_geometric'"

**Cause:** PyTorch Geometric not installed in current environment

**Solution:** Install in the main environment:
```bash
conda activate llmhe
pip install torch-geometric
```

### GPU Memory Issues

NV-Embed-v2 requires ~8GB VRAM. If OOM:
```bash
# Reduce batch size
export CUDA_VISIBLE_DEVICES=0
python scripts/encode_corpus.py --batch_size 4
```

## Version Matrix

| Package | nv-embed-v2 env | llmhe env |
|---------|-----------------|-----------|
| Python | 3.10 | 3.10 |
| torch | 2.0+ | 2.0+ |
| transformers | 4.44.2 | 4.57.3 |
| sentence-transformers | 2.7.0 | 5.2.0 |
| torch-geometric | - | 2.7.0 |

## Frozen Requirements

Full frozen requirements for exact reproducibility:
- `envs/requirements-nv-embed-v2.txt` - Complete nv-embed-v2 env
- `envs/requirements-llmhe.txt` - Complete llmhe env

---

**Last Updated:** 2026-01-19
