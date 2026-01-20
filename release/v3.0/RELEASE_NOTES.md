# Release v3.0

**Date:** 2026-01-20
**Commit:** 3f3d3e7e
**Branch:** master

---

## Overview

This release contains the complete Evidence Binding Pipeline for psychiatric symptom detection,
ready for academic publication.

## Key Metrics

| Metric | Value |
|--------|-------|
| AUROC | 0.8972 |
| AUPRC | 0.5709 |
| Evidence Recall@K | 0.7043 |
| MRR | 0.3801 |

## What's Included

### Paper Bundle v3.0
- `metrics_master.json` - Single source of truth for all metrics
- `tables/` - Publication-ready tables
- `checksums.txt` - SHA256 verification

### Documentation
- Complete reproducibility instructions
- Data statement with ethics/IRB documentation
- Error analysis report

### Code
- Full pipeline implementation
- Baseline implementations (BM25, TF-IDF, E5, Contriever)
- Robustness and significance testing scripts
- 227+ automated tests

## Verification

```bash
# Verify checksums
python scripts/verification/verify_checksums.py --bundle results/paper_bundle/v3.0

# Verify metrics
python scripts/verification/metric_crosscheck.py --bundle results/paper_bundle/v3.0

# Run tests
pytest -q
```

## Citation

See `CITATION.cff` for citation information.

## License

MIT License - see `LICENSE` file.
