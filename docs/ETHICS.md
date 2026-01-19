# Ethics Statement

## Overview

This research involves analysis of social media content related to mental health. We take privacy, safety, and ethical considerations seriously.

## Data Privacy

### Source Data
- **Origin**: Public Reddit posts from mental health communities
- **Collection**: Posts were publicly available at time of collection
- **No Private Data**: No private messages, deleted posts, or non-public content

### Privacy Protections

1. **No Raw Data in Repository**: Original post content is never committed
2. **Anonymization**: All post IDs are anonymized (e.g., `s_106_89`)
3. **No Re-identification**: Analyses designed to prevent user identification
4. **Aggregate Reporting**: Results reported at aggregate level only

### Data Handling

- Raw data stored on secured, access-controlled servers
- No cloud storage of identifiable content
- Data access requires signed DUA
- Retention limited to research duration + archival period

## IRB Compliance

### Status
Research conducted under institutional review with appropriate exemptions for:
- Public data analysis
- No direct participant interaction
- Minimal risk category

### Considerations Addressed
- Public nature of source data
- Sensitive mental health content
- Potential for identification
- Benefit to mental health research

## Responsible AI

### Clinical Use Disclaimers

**This system is NOT intended for:**
- Clinical diagnosis
- Treatment recommendations
- Unsupervised mental health assessment
- Direct patient-facing applications

**Appropriate uses:**
- Research tool with clinical oversight
- Screening assistance (not replacement)
- Evidence aggregation for expert review

### Limitations Acknowledged

1. **Not a Diagnostic Tool**: Model outputs are evidence indicators, not diagnoses
2. **Population Bias**: Training data from Reddit may not generalize
3. **False Negatives**: Missing evidence does not mean absence of symptoms
4. **Context Dependency**: Sentence-level analysis may miss context

### Safety Measures

- High-sensitivity threshold for clinical deployment (99.78% screening sensitivity)
- Three-state output (NEG/UNCERTAIN/POS) to flag uncertain cases
- Mandatory human review for all flagged content
- No automated actions without expert verification

## Content Sensitivity

### Mental Health Content
This research involves content discussing:
- Depression symptoms
- Suicidal ideation
- Emotional distress
- Personal struggles

### Researcher Wellbeing
- Exposure to sensitive content managed through:
  - Batch processing (not real-time reading)
  - Aggregate analysis focus
  - Support resources available

## Dual Use Considerations

### Beneficial Uses
- Improving mental health screening efficiency
- Supporting clinical research
- Reducing assessment burden

### Potential Misuses (Mitigated)
- **Surveillance**: Data access requires ethical approval
- **Stigmatization**: No individual-level outputs published
- **Commercial exploitation**: MIT license allows review of use

## Compliance

### Regulations Considered
- HIPAA (for any clinical deployment)
- GDPR (for any EU data)
- Platform ToS (Reddit API)

### Best Practices Followed
- ACM Code of Ethics
- NeurIPS ethics guidelines
- Mental health research standards

## Contact

For ethics-related questions:
- Raise issues in the repository
- Contact institutional ethics board
- Email maintainers directly
