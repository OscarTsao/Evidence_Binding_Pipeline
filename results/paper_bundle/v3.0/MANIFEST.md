# Paper Bundle v3.0 Manifest

Generated: 2026-01-20T21:58:46.613173

## Contents

| File | Description |
|------|-------------|
| metrics_master.json | Single source of truth for all metrics |
| summary.json | Bundle metadata and high-level summary |
| tables/main_results.csv | Primary metrics with 95% CIs |
| tables/per_criterion.csv | Per-criterion performance breakdown |
| tables/baselines.csv | Baseline comparison (if available) |
| tables/robustness.csv | Robustness analysis (if available) |
| checksums.txt | SHA256 integrity verification |
| MANIFEST.md | This file |

## Regeneration

To regenerate this bundle:

```bash
python scripts/reporting/build_paper_bundle.py \
    --version v3.0 \
    --source_run outputs/final_research_eval/20260118_031312_complete \
    --output results/paper_bundle/v3.0
```

## Verification

To verify bundle integrity:

```bash
python scripts/verification/verify_checksums.py \
    --bundle results/paper_bundle/v3.0
```

To cross-check metrics:

```bash
python scripts/verification/metric_crosscheck.py \
    --bundle results/paper_bundle/v3.0
```

## Source Data

| Source | Path |
|--------|------|
| Per-query results | outputs/final_research_eval/20260118_031312_complete/per_query.csv |
| Criteria registry | configs/criteria_registry.yaml |
| Metric contract | docs/METRIC_CONTRACT.md |

## Protocols

- **positives_only**: Ranking metrics (nDCG, Recall, MRR) computed on queries with evidence
- **all_queries**: Classification metrics (AUROC, AUPRC) computed on all queries

## Notes

- AUROC/AUPRC: 95% bootstrap CIs (n=2000, seed=42)
- Ranking metrics: positives_only protocol
- Split: TEST (10% posts, post-ID disjoint)
