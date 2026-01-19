# Contributing Guide

## Development Setup

### Prerequisites
- Python 3.10+
- CUDA 12.4+ (for GPU inference)
- 32GB+ RAM recommended

### Environment Setup

Two conda environments required due to dependency conflicts:

```bash
# 1. Retriever environment (NV-Embed-v2 requires transformers<=4.44)
conda create -n nv-embed-v2 python=3.10 -y
conda activate nv-embed-v2
pip install -r envs/requirements-retriever.txt

# 2. Main environment (reranking, GNN, evaluation)
conda create -n llmhe python=3.10 -y
conda activate llmhe
pip install -r envs/requirements-main.txt
pip install -e .
```

### Running Tests

```bash
# All tests
pytest -q

# Specific test suite
pytest tests/test_metrics.py -v

# Leakage tests only
pytest tests/leakage/ -v
```

## Development Workflow

### 1. Making Changes

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ...

# Run tests
pytest -q

# Verify checksums (if modifying bundle)
python scripts/verification/verify_checksums.py
```

### 2. Updating Metrics

If you change any metrics:

1. Update `results/paper_bundle/v2.0/metrics_master.json`
2. Update `results/paper_bundle/v2.0/summary.json`
3. Regenerate checksums:
   ```bash
   cd results/paper_bundle/v2.0/
   sha256sum *.json *.md tables/*.csv > checksums.txt
   ```
4. Verify: `python scripts/verification/verify_checksums.py`

### 3. Adding Tests

All new features must have tests:

```python
# tests/test_your_feature.py
def test_your_feature():
    # Arrange
    # Act
    # Assert
    pass
```

### 4. Submitting PR

```bash
# Push branch
git push origin feature/your-feature

# Open PR with:
# - Clear description
# - Verification commands run
# - Updated docs if needed
```

## Code Standards

### Style
- Follow PEP 8
- Use type hints where feasible
- Maximum line length: 100 characters

### Testing
- All tests must be deterministic (use fixed seeds)
- No GPU required for unit tests
- No private data in tests (use fixtures)

### Documentation
- Update README if adding features
- Update metrics_master.json if metrics change
- Regenerate checksums if bundle changes

## Key Invariants

When contributing, maintain these invariants:

1. **Post-ID Disjoint Splits:** No post in multiple splits
2. **Within-Post Retrieval:** Candidates from same post only
3. **Dual Protocol:** positives_only vs all_queries metrics
4. **No Gold Features:** Ground truth never used in inference

## Repository Structure

```
src/final_sc_review/     # Core source code
tests/                   # Test suite
scripts/                 # Executable scripts
results/paper_bundle/    # Immutable release bundle
docs/                    # Documentation
configs/                 # Configuration files
envs/                    # Environment specifications
```

## Questions?

- Open an issue for bugs
- Start a discussion for features
- Email maintainers for data access
