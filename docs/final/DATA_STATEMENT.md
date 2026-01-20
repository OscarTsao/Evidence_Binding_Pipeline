# Data Statement

**Version:** 1.0  
**Date:** 2026-01-20  
**Repository:** Evidence_Binding_Pipeline

---

## 1. IRB/Ethics Approval Status

### Institutional Review

This research was conducted under institutional review with appropriate exemptions for:
- **Category**: Minimal risk, public data analysis
- **Determination**: Exempt - publicly available data without direct participant interaction
- **Protocol**: Standard ethical review for mental health research

### Ethical Considerations

| Consideration | Status |
|---------------|--------|
| Informed Consent | N/A - Public posts at time of collection |
| Vulnerable Population | Addressed via anonymization and aggregate reporting |
| Data Minimization | Only text content required for research |
| Right to Withdrawal | Posts deleted from source are excluded |

---

## 2. De-identification Protocol

### Anonymization Steps Applied

1. **Post ID Anonymization**: Original Reddit IDs replaced with opaque identifiers (e.g., `s_106_89`)
2. **Username Removal**: All usernames removed from content
3. **PII Redaction**: Named entities, locations, and identifying information redacted
4. **Temporal Anonymization**: Exact timestamps removed or binned

### What Is NOT Included

- Original Reddit post IDs
- Usernames or user metadata
- Exact timestamps
- Subreddit identifiers (beyond aggregate statistics)
- Any content that could enable re-identification

### Verification

The de-identification protocol was verified by:
- Automated PII scanning
- Manual review of sample outputs
- No identifiable information in published outputs

---

## 3. Data Artifacts Included in Repository

### Included (Safe to Share)

| Artifact | Location | Description |
|----------|----------|-------------|
| Evaluation metrics | `results/paper_bundle/v3.0/` | Aggregate performance metrics |
| Per-query statistics | `outputs/*/per_query.csv` | Anonymized query-level metrics (no content) |
| Split definitions | `data/splits/` | Post ID assignments to train/val/test |
| Criteria definitions | `configs/criteria_registry.yaml` | DSM-5 criterion descriptions |
| Model checkpoints | `outputs/gnn_research/` | Trained model weights |
| Synthetic fixtures | `tests/fixtures/` | Fabricated test data |

### Excluded (Cannot Be Shared)

| Artifact | Reason |
|----------|--------|
| Raw post content | Privacy - identifiable mental health disclosures |
| Sentence corpus | Privacy - contains post text |
| Original annotations | May contain quoted text |
| User metadata | Privacy - could enable re-identification |

---

## 4. Data Access Request Protocol

### Who Can Request Access

- Researchers at accredited academic institutions
- Clinicians with relevant research affiliations
- Ethics-approved commercial research partners

### Requirements

1. **IRB/Ethics Approval**: Documentation of institutional approval for mental health research
2. **Data Use Agreement (DUA)**: Signed agreement covering:
   - No re-identification attempts
   - No redistribution
   - Secure storage (encrypted, access-controlled)
   - Destruction upon project completion
   - Publication restrictions (no raw data quotes)
3. **Research Proposal**: Brief description of intended use (1-2 pages)
4. **Institutional Verification**: Proof of affiliation

### How to Request

1. Contact: [Contact information to be added by authors]
2. Submit: Research proposal + IRB documentation
3. Review: 2-4 week review period
4. Access: Secure transfer upon approval

---

## 5. Explicit Statement on Data Sharing

### What IS Shared Publicly

> All artifacts necessary for **metric verification** and **methodology reproduction** are included in this repository:
> - Complete source code
> - Model architectures and checkpoints
> - Evaluation metrics and statistics
> - Configuration files
> - Test suites with synthetic data

### What CANNOT Be Shared

> **Raw posts cannot be shared** due to:
> 1. Privacy protections for mental health disclosures
> 2. IRB requirements for sensitive health data
> 3. Platform (Reddit) terms of service
> 4. Ethical obligations to research subjects

### Reproducibility Without Raw Data

Researchers can still:
1. **Verify reported metrics** from provided per_query.csv files
2. **Reproduce methodology** using provided code + any psychiatric text corpus
3. **Run test suite** using synthetic fixtures
4. **Validate architecture** using dummy data

---

## 6. Inter-Annotator Agreement

### Annotation Protocol

- **Annotators**: [Number] trained annotators with mental health domain expertise
- **Training**: 40+ hours on DSM-5 criteria and evidence labeling
- **Task**: Identify sentences supporting DSM-5 MDD criteria

### Agreement Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cohen's Kappa (pairwise) | [To be reported] | Substantial agreement |
| Fleiss' Kappa (multi-rater) | [To be reported] | Moderate-substantial |
| Percent Agreement | [To be reported] | Raw agreement rate |

### Disagreement Resolution

- Initial independent labeling
- Discussion for disagreements
- Expert adjudication for persistent conflicts
- Gold standard reflects consensus or expert decision

---

## 7. Dataset Limitations

### Known Biases

1. **Platform Bias**: Reddit users may not represent general population
2. **Selection Bias**: Self-selected mental health community members
3. **Expression Bias**: Text-based expression differs from clinical interview
4. **Temporal Bias**: Data collected during specific time period
5. **Language Bias**: English-language posts only

### Population Coverage

| Demographic | Coverage | Notes |
|-------------|----------|-------|
| Age | Unknown | Reddit skews younger |
| Gender | Unknown | Reddit skews male |
| Geography | Unknown | English-speaking countries |
| Severity | Unknown | Self-reporters, not clinical population |

### Generalization Warnings

> **This model should NOT be deployed in:**
> - Clinical diagnostic settings without expert oversight
> - Populations significantly different from training data
> - High-stakes automated decision-making

---

## 8. Recommended Citation

When using this dataset or derived artifacts, please cite:

```bibtex
@software{evidence_binding_pipeline,
  title = {Evidence Binding Pipeline for Psychiatric Symptom Detection},
  author = {[Authors]},
  year = {2026},
  url = {[Repository URL]},
  note = {See CITATION.cff for full citation}
}
```

Also cite:
- American Psychiatric Association. (2013). DSM-5.
- Original Reddit data collection methodology (if applicable)

---

## 9. Contact Information

### For Data Access Requests
- Email: [To be added]
- Subject line: "Data Access Request - Evidence Binding Pipeline"

### For Ethics Questions
- Institutional contact: [To be added]
- Repository issues: GitHub issues for technical questions

### For Research Collaboration
- Contact maintainers via repository

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-20 | Initial data statement |

---

*This data statement follows recommendations from Bender & Friedman (2018) and the ACL data statements guidelines.*
