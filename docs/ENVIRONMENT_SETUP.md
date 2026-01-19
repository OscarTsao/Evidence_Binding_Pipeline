# Environment Setup Guide

## Why Two Environments?

This project requires two separate conda environments due to dependency conflicts:

| Environment | Purpose | Key Constraint |
|-------------|---------|----------------|
| `nv-embed-v2` | NV-Embed-v2 retriever | `transformers<=4.44` |
| `llmhe` | Reranking, GNN, evaluation | `transformers>=4.45` |

The NV-Embed-v2 model requires an older version of transformers that conflicts with other dependencies.

## Setup Instructions

### 1. Create Retriever Environment

```bash
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt
```

**Key packages:**
- `transformers==4.44.0`
- `sentence-transformers`
- `torch>=2.0`

### 2. Create Main Environment

```bash
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .
```

**Key packages:**
- `transformers>=4.45`
- `torch>=2.0`
- `torch-geometric`
- `scikit-learn`
- `pandas`

## Usage Workflow

### Encoding (Retriever Environment)

```bash
conda activate nv-embed-v2
python scripts/encode_corpus.py --config configs/default.yaml
```

### Evaluation (Main Environment)

```bash
conda activate llmhe
python scripts/eval_zoo_pipeline.py --config configs/default.yaml --split test
```

### Testing (Main Environment)

```bash
conda activate llmhe
pytest -q
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16GB | 24GB+ |
| RAM | 16GB | 32GB+ |
| Storage | 50GB | 100GB+ |

## Troubleshooting

### CUDA Version Mismatch

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Import Errors

```bash
# Verify correct environment
conda activate llmhe
python -c "import final_sc_review; print('OK')"
```

### Memory Issues

Reduce batch sizes in `configs/default.yaml`:
```yaml
retriever:
  batch_size: 16  # Reduce from default
```

## Environment Verification

```bash
# Verify retriever env
conda activate nv-embed-v2
python -c "from transformers import AutoModel; print('Retriever env OK')"

# Verify main env
conda activate llmhe
python -c "import torch_geometric; print('Main env OK')"
pytest tests/test_metrics.py -v
```

## Docker Alternative (Optional)

For containerized deployment:

```dockerfile
# Dockerfile.retriever
FROM nvidia/cuda:12.4-runtime-ubuntu22.04
RUN pip install -r requirements-retriever.txt

# Dockerfile.main
FROM nvidia/cuda:12.4-runtime-ubuntu22.04
RUN pip install -r requirements-main.txt
```
