# Contributing to Evidence Binding Pipeline

Thank you for your interest in contributing to this research project. This document provides guidelines for contributing code, documentation, and reporting issues.

---

## Code of Conduct

- Be respectful and constructive in all interactions
- Prioritize scientific rigor and reproducibility
- Protect participant privacy - never commit raw data or identifiable information

---

## Development Setup

### Prerequisites

1. **Dual conda environments** are required due to dependency conflicts:
   - `nv-embed-v2`: For NV-Embed-v2 retriever (`transformers<=4.44`)
   - `llmhe`: For reranking, GNN, and evaluation (`transformers>=4.45`)

2. See [docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md) for detailed setup instructions.

### Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd Evidence_Binding_Pipeline

# Install in development mode
pip install -e .

# Run tests to verify setup
pytest -q
```

---

## Code Style

### Python Style Guide

- Follow PEP 8 conventions
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

### Docstrings

Use Google-style docstrings:

```python
def compute_ndcg(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at K.

    Args:
        y_true: Binary relevance labels (0 or 1).
        y_score: Predicted scores (higher = more relevant).
        k: Cutoff position for evaluation.

    Returns:
        nDCG@K score in range [0, 1].

    Raises:
        ValueError: If k < 1 or arrays have mismatched lengths.
    """
```

### Imports

Organize imports in this order:
1. Standard library
2. Third-party packages
3. Local modules

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_metrics.py

# Run tests matching pattern
pytest -k "test_ndcg"

# Run with verbose output
pytest -v
```

### Test Requirements

- All new code must include unit tests
- Maintain >90% test coverage for core modules
- Tests must pass before merging

### Key Test Categories

| Category | Purpose |
|----------|---------|
| `tests/metrics/` | Ranking and classification metrics |
| `tests/leakage/` | Data leakage prevention (12+ tests) |
| `tests/clinical/` | Clinical deployment safety |

---

## Pull Request Process

### Before Submitting

1. **Run all tests**: `pytest`
2. **Verify checksums**: `python scripts/verification/verify_checksums.py`
3. **Audit splits**: `python scripts/audit_splits.py --data_dir data --seed 42 --k 5`
4. **Check publication gate**: `pytest tests/test_publication_gate.py -v`

### PR Requirements

- Clear description of changes
- Reference any related issues
- Include test coverage for new code
- Update documentation if needed

### Commit Messages

Use clear, descriptive commit messages:

```
Add per-criterion AUROC breakdown to evaluation output

- Add criterion-level AUROC computation
- Update summary.json schema to include per-criterion metrics
- Add unit tests for new functionality
```

---

## Reporting Issues

### Bug Reports

Include:
- Python version and environment
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)

---

## Key Invariants

When contributing, ensure these invariants are maintained:

1. **Post-ID Disjoint Splits**: No post appears in multiple splits (TRAIN/VAL/TEST)
2. **Within-Post Retrieval**: Candidate pool is always sentences from the same post
3. **Dual Protocol Metrics**: `positives_only` for ranking, `all_queries` for classification
4. **No Gold Features in Inference**: Ground truth labels never used as features

See [docs/REPO_OVERVIEW.md](docs/REPO_OVERVIEW.md) for detailed explanations.

---

## Documentation

### Updating Documentation

- Update relevant .md files when changing functionality
- Keep `metrics_master.json` as single source of truth for metrics
- Update checksums when modifying paper bundle files

### Paper Bundle Changes

If modifying `results/paper_bundle/v1.0/`:
1. Update the relevant files
2. Regenerate checksums: `cd results/paper_bundle/v1.0 && sha256sum <files> > checksums.txt`
3. Verify: `sha256sum -c checksums.txt`

---

## Questions?

For questions about contributing, please open an issue or contact the maintainers.
