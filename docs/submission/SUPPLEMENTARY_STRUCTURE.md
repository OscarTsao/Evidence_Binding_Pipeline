# Supplementary Materials Structure

**Version:** 1.0  
**Date:** 2026-01-20

This document describes the structure and contents of supplementary materials for submission.

---

## Recommended Supplementary Organization

```
supplementary/
├── A_detailed_results/
│   ├── main_results_with_cis.csv        # Full results table with 95% CIs
│   ├── per_criterion_breakdown.csv      # Performance by DSM-5 criterion
│   ├── confusion_matrices.json          # Overall and per-criterion
│   └── robustness_analysis.csv          # Bootstrap stability results
│
├── B_error_analysis/
│   ├── failure_modes.csv                # Categorized failure modes
│   ├── errors_by_criterion.csv          # Error rates stratified by criterion
│   ├── errors_by_density.csv            # Error rates by evidence density
│   └── clinical_implications.md         # Clinical interpretation
│
├── C_baselines/
│   ├── baseline_comparison.csv          # All baseline results
│   ├── significance_tests.csv           # Statistical significance
│   └── baseline_descriptions.md         # Baseline method descriptions
│
├── D_ablations/
│   ├── ablation_results.csv             # Component ablation study
│   └── ablation_descriptions.md         # What each ablation tests
│
├── E_reproducibility/
│   ├── REPRODUCIBILITY.md               # Full reproduction instructions
│   ├── checksums.txt                    # SHA256 for all artifacts
│   └── environment.yml                  # Conda environment specification
│
└── F_data_documentation/
    ├── DATA_STATEMENT.md                # Full data statement
    ├── ETHICS.md                        # Ethics documentation
    └── annotation_guidelines.md         # Annotation protocol
```

---

## Contents Mapping

### A. Detailed Results

| File | Source | Description |
|------|--------|-------------|
| main_results_with_cis.csv | `results/paper_bundle/v3.0/tables/main_results.csv` | Primary metrics with CIs |
| per_criterion_breakdown.csv | `results/paper_bundle/v3.0/tables/per_criterion.csv` | Per-criterion AUROC/AUPRC |
| confusion_matrices.json | `outputs/error_analysis/confusion_matrix_overall.json` | Classification matrices |
| robustness_analysis.csv | `outputs/robustness/*/robustness_summary.csv` | Bootstrap stability |

### B. Error Analysis

| File | Source | Description |
|------|--------|-------------|
| failure_modes.csv | `outputs/error_analysis/failure_modes.csv` | Failure mode counts |
| errors_by_criterion.csv | `outputs/error_analysis/errors_by_criterion.csv` | Stratified errors |
| errors_by_density.csv | `outputs/error_analysis/errors_by_density.csv` | By evidence density |

### C. Baselines

| File | Source | Description |
|------|--------|-------------|
| baseline_comparison.csv | `outputs/baselines/comparison_summary.csv` | All baseline results |
| significance_tests.csv | `outputs/significance/*/significance_*.csv` | p-values |

### D. Ablations

| File | Source | Description |
|------|--------|-------------|
| ablation_results.csv | `outputs/ablation/ablation_summary.csv` | Component removal effects |

### E. Reproducibility

| File | Source | Description |
|------|--------|-------------|
| checksums.txt | `results/paper_bundle/v3.0/checksums.txt` | Artifact verification |

---

## Build Script

To generate supplementary materials package:

```bash
python scripts/reporting/build_supplementary.py \
    --output supplementary/ \
    --format zip
```

---

## Venue-Specific Requirements

### ACL/EMNLP/NAACL

- Appendix length: No strict limit
- Ethics statement: Required
- Reproducibility checklist: Required
- Software/data availability: Required

### Clinical Venues (JAMIA, npj Digital Medicine)

- IRB documentation: Required
- CONSORT/TRIPOD checklist: If applicable
- Clinical validation: Recommended
- Data access statement: Required

---

## Checklist Before Submission

- [ ] All tables match main paper values
- [ ] CIs are 95% bootstrap (2000 iterations)
- [ ] Error analysis is complete
- [ ] Baseline comparisons include all methods
- [ ] Checksums verify correctly
- [ ] Ethics statement is complete
- [ ] Data statement is complete
- [ ] Code repository is clean (no secrets)
