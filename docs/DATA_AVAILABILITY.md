# Data Availability Statement

## Overview

The research data used in this study (RedSM5 dataset) is **not publicly available** due to privacy considerations and IRB requirements for mental health research involving social media data.

## Dataset Description

| Attribute | Value |
|-----------|-------|
| Name | RedSM5 |
| Source | Reddit mental health communities |
| Posts | 1,641 unique posts |
| Queries | 14,770 (post × criterion pairs) |
| Criteria | DSM-5 MDD A.1-A.9 + duration |
| Annotations | Sentence-level evidence labels |

## Access Procedures

### For Researchers

To request data access:

1. **Institutional Affiliation**: Must be affiliated with an accredited research institution
2. **IRB Approval**: Provide documentation of IRB/ethics board approval for mental health research
3. **Data Use Agreement**: Sign a DUA covering:
   - No re-identification attempts
   - No redistribution
   - Secure storage requirements
   - Publication restrictions on raw data

### Contact

Contact the dataset maintainers with:
- Research proposal (1-2 pages)
- IRB approval documentation
- Institutional affiliation verification

## Reproducibility Without Raw Data

The repository supports reproducibility verification without raw data:

### 1. Verified Outputs
Pre-computed predictions are provided:
```
outputs/final_research_eval/20260118_031312_complete/per_query.csv
```

### 2. Metric Recomputation
Recompute metrics from predictions:
```bash
python scripts/verification/recompute_metrics_from_csv.py \
    --input outputs/final_research_eval/*/per_query.csv
```

### 3. Test Suite
All tests use synthetic fixtures:
```bash
pytest -q  # No real data required
```

### 4. Architecture Validation
Pipeline components can be tested with dummy data:
```python
from final_sc_review.pipeline.zoo_pipeline import ZooPipeline
# Create pipeline with test config
# Validate component behavior
```

## Synthetic Data for Development

For development and testing, use the provided fixtures:
```
tests/fixtures/
├── sample_posts.csv       # Anonymized example posts
├── sample_annotations.csv # Example annotations
└── sample_corpus.jsonl    # Example sentence corpus
```

## Citation Requirements

If you use the dataset, cite:
1. This repository (see CITATION.cff)
2. Original Reddit data collection methodology
3. DSM-5 criterion definitions (APA)

## Ethical Considerations

See [ETHICS.md](ETHICS.md) for:
- Privacy protection measures
- IRB compliance details
- Data handling procedures
- Participant consent (public data)
