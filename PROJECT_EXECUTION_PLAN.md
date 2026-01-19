# PROJECT EXECUTION PLAN
## Dual-Track: Academic Publication + Clinical Deployment

**Launch Date:** 2026-01-18
**Project Code:** EVID-RETRIEVAL-DUAL-TRACK
**Timeline:** 4-6 months to completion
**Status:** 🚀 LAUNCHED

---

## EXECUTIVE OVERVIEW

This execution plan coordinates two parallel tracks:

**Track 1: Academic Publication**
- Goal: Submit to top-tier venue (ACL, EMNLP, or JAMIA)
- Timeline: 4-6 months to submission
- Deliverable: Publication-ready manuscript with all experiments

**Track 2: Clinical Deployment**
- Goal: Pilot deployment with N=5-10 clinicians
- Timeline: 4-8 weeks to pilot launch
- Deliverable: Production system with monitoring

**Shared Components:**
- External validation (benefits both tracks)
- Clinical expert review (benefits both tracks)
- Ablation studies (primarily publication, but informs deployment)

---

## TIMELINE OVERVIEW

```
Month 1 (Weeks 1-4): Foundation & Validation
├─ Week 1: Setup, External Validation Prep
├─ Week 2: External Validation Execution
├─ Week 3: Ablation Studies Execution
└─ Week 4: Clinical Expert Review

Month 2 (Weeks 5-8): Pilot Launch & Analysis
├─ Week 5: Pilot Study Preparation
├─ Week 6: Pilot Launch (N=5-10 clinicians)
├─ Week 7: Monitoring & Data Collection
└─ Week 8: Pilot Analysis & Manuscript Drafting

Month 3 (Weeks 9-12): Manuscript & Expanded Pilot
├─ Week 9: Manuscript First Draft
├─ Week 10: Expanded Pilot (N=20-50 clinicians)
├─ Week 11: Results Integration
└─ Week 12: Internal Review

Month 4 (Weeks 13-16): Refinement
├─ Week 13: Manuscript Revision
├─ Week 14: Additional Experiments (if needed)
├─ Week 15: Final Polishing
└─ Week 16: Pre-submission Review

Month 5-6 (Optional): Submission & Production Prep
├─ Month 5: Submission + Monitoring Refinement
└─ Month 6: Production Deployment Planning
```

---

## TRACK 1: ACADEMIC PUBLICATION

### Phase 1.1: Ablation Studies (Week 3)

**Objective:** Quantify component contributions

**Tasks:**
- [ ] Execute 7 module ablations (C1-C7) with 5-fold CV
- [ ] Execute gamma sweep (4 configurations)
- [ ] Execute threshold grid (25 configurations)
- [ ] Execute A.10 ablation (include vs exclude)

**Resources:**
- GPU: 1× RTX 5090
- Runtime: 12-18 hours (can run overnight)
- Script: Ready (`docs/eval/ABLATION_STUDY_DESIGN.md`)

**Deliverables:**
- `outputs/ablation/module_ablations/` (7 configs)
- `outputs/ablation/policy_ablations/` (29 configs)
- Ablation comparison table
- Ablation waterfall chart

**Owner:** Research Engineer
**Estimated Effort:** 1 week (2 days execution + 3 days analysis)

---

### Phase 1.2: External Validation (Week 2)

**Objective:** Validate on independent dataset

**Options:**
1. **Different Reddit Community** (Recommended)
   - Sample N=1,000 posts from /r/depression (not in training)
   - Annotation: Use existing annotations if available, or sample 100 for spot-check
   - Metrics: AUROC, AUPRC, Screening Sensitivity
   - Timeline: 1-2 weeks

2. **Different Time Period**
   - Sample posts from 2024 (if training used pre-2024)
   - Check for drift over time
   - Timeline: 1 week

3. **Different Platform** (Ambitious)
   - Twitter mental health dataset
   - Requires adaptation layer
   - Timeline: 3-4 weeks

**Recommended:** Option 1 (different Reddit community)

**Tasks:**
- [ ] Identify external dataset (Reddit /r/depression, N=1,000 posts)
- [ ] Prepare data in same format as training
- [ ] Run full pipeline evaluation
- [ ] Compare metrics to internal validation
- [ ] Document any performance degradation

**Deliverables:**
- `outputs/external_validation/summary.json`
- `outputs/external_validation/comparison_report.md`
- Performance comparison table (internal vs external)

**Owner:** Research Engineer
**Estimated Effort:** 1-2 weeks

---

### Phase 1.3: Manuscript Preparation (Weeks 8-16)

**Objective:** Submission-ready manuscript

**Structure (Following ACL/EMNLP format):**

1. **Abstract** (250 words)
2. **Introduction** (1.5 pages)
   - Problem statement
   - Contributions (3-4 bullet points)
   - Paper organization
3. **Related Work** (1.5 pages)
   - Evidence retrieval
   - Mental health NLP
   - GNN applications
4. **Dataset & Task** (1 page)
   - RedSM5 dataset
   - Post-ID disjoint splits
   - Task definition
5. **Method** (3 pages)
   - 6-stage pipeline architecture
   - GNN modules (P2, P3, P4)
   - 3-state clinical gate
6. **Experiments** (2 pages)
   - Evaluation protocol
   - Ablation studies
   - External validation
7. **Results** (2 pages)
   - Main results table
   - Ablation analysis
   - Per-criterion breakdown
8. **Analysis** (1.5 pages)
   - Clinical utility
   - Error analysis
   - Limitations
9. **Conclusion** (0.5 page)

**Figures (8 total, 4 main + 4 supplementary):**
- Figure 1: Pipeline architecture diagram
- Figure 2: ROC curve with 95% CI
- Figure 3: Ablation waterfall chart
- Figure 4: Per-criterion AUROC bars
- Supp. A: PR curve with baseline
- Supp. B: Calibration diagram
- Supp. C: Dynamic-K analysis
- Supp. D: Threshold sensitivity

**Timeline:**
- Week 8: First draft (8 pages, rough)
- Week 9: Second draft (refine, add results)
- Week 10-11: Incorporate pilot data (prospective validation)
- Week 12: Internal review
- Week 13-15: Revisions
- Week 16: Final polishing

**Owner:** Lead Researcher + Domain Expert (collaboration)
**Estimated Effort:** 8 weeks

---

### Phase 1.4: Submission (Week 17+)

**Target Venues:**

**Tier 1 (AI/NLP):**
- ACL (Association for Computational Linguistics) - Deadline: Feb 2026
- EMNLP (Empirical Methods in NLP) - Deadline: Jun 2026
- NAACL (North American ACL) - Deadline: Oct 2026

**Tier 1 (Medical Informatics):**
- JAMIA (Journal of American Medical Informatics Association)
- JMIR (Journal of Medical Internet Research)
- NPJ Digital Medicine

**Tier 2 (Specialized):**
- ACL-BioNLP Workshop
- LOUHI (Health Document Processing)
- CLPsych (Computational Linguistics and Clinical Psychology)

**Recommendation:**
- Primary: ACL 2026 (Feb deadline) or EMNLP 2026 (Jun deadline)
- Backup: JAMIA (rolling submission)

---

## TRACK 2: CLINICAL DEPLOYMENT

### Phase 2.1: Pilot Study Preparation (Weeks 4-5)

**Objective:** Prepare for pilot deployment with N=5-10 clinicians

**Tasks:**

**Technical Setup:**
- [ ] Deploy pipeline to dedicated server (GPU-enabled)
- [ ] Create web interface for clinicians (simple UI)
- [ ] Implement logging (all predictions, user interactions)
- [ ] Setup monitoring dashboard (real-time metrics)
- [ ] Create user authentication (secure access)

**Clinical Setup:**
- [ ] Recruit N=5-10 clinicians (Reddit-familiar psychiatrists/psychologists)
- [ ] IRB approval (if required by institution)
- [ ] Create user guide (2-page quick reference)
- [ ] Design feedback form (usability, accuracy, time savings)
- [ ] Schedule training session (1 hour onboarding)

**Deliverables:**
- Deployed system (URL: https://evidence-retrieval.example.com)
- User guide (PDF, 2 pages)
- Monitoring dashboard (Grafana/Kibana)
- IRB approval documentation (if needed)

**Owner:** Software Engineer + Clinical Coordinator
**Estimated Effort:** 2 weeks

---

### Phase 2.2: Pilot Study Execution (Weeks 6-8)

**Objective:** Collect real-world usage data and feedback

**Study Design:**

**Participants:** N=5-10 clinicians
- Psychiatrists or psychologists
- Familiar with Reddit/social media mental health content
- Willing to provide feedback

**Task:**
- Review N=50 Reddit posts each (total 250-500 posts)
- For each post, assess evidence for 10 MDD criteria
- Compare system suggestions with their own judgment
- Provide feedback on each query (helpful/not helpful, time saved)

**Data Collection:**
- System predictions (p(has_evidence), ranked evidence sentences, state)
- Clinician judgments (agree/disagree, time saved, comments)
- Usage patterns (time per query, queries completed, drop-off rate)
- Feedback surveys (weekly + final)

**Metrics:**
- **Agreement Rate:** % queries where clinician agrees with system
- **Time Savings:** Mean time reduction vs manual review
- **Satisfaction:** Clinician satisfaction score (1-5 Likert)
- **False Negative Rate:** Clinician-identified misses
- **False Positive Rate:** Clinician-identified false alarms

**Timeline:**
- Week 6: Pilot launch, onboarding, first queries
- Week 7: Data collection, weekly check-in
- Week 8: Data collection, feedback survey

**Deliverables:**
- `outputs/pilot_study/clinician_judgments.csv` (250-500 rows)
- `outputs/pilot_study/agreement_analysis.md`
- `outputs/pilot_study/feedback_summary.md`
- Prospective validation metrics (AUROC on clinician labels)

**Owner:** Clinical Coordinator + Data Analyst
**Estimated Effort:** 3 weeks

---

### Phase 2.3: Monitoring Dashboard (Week 5)

**Objective:** Real-time system monitoring

**Dashboard Components:**

**Panel 1: Request Metrics**
- Queries per hour (time series)
- Queries per clinician (bar chart)
- Query completion rate

**Panel 2: Performance Metrics**
- AUROC (daily rolling average, N=100 queries)
- Positive rate (% has_evidence predictions)
- Alert rate (% POS state)
- Screening FN rate (clinician-reported misses)

**Panel 3: System Health**
- Latency (P50, P95, P99)
- Error rate (% failed queries)
- GPU utilization
- API uptime

**Panel 4: Clinician Feedback**
- Agreement rate (% agree with system)
- Time savings (mean minutes saved per query)
- Satisfaction scores (daily average)

**Alerting Rules:**
- 🚨 AUROC < 0.80 → Alert research team
- 🚨 Error rate > 5% → Alert engineering team
- 🚨 P95 latency > 1s → Scale resources
- 🚨 Satisfaction < 3.0 → Review UX issues

**Technology Stack:**
- Grafana (dashboards)
- Prometheus (metrics collection)
- PostgreSQL (data storage)
- Python FastAPI (backend)

**Deliverables:**
- Live dashboard (URL: https://monitor.evidence-retrieval.example.com)
- Alert configuration
- Metrics documentation

**Owner:** DevOps Engineer
**Estimated Effort:** 1 week

---

### Phase 2.4: Expanded Pilot (Weeks 10-12)

**Objective:** Scale to N=20-50 clinicians

**Tasks:**
- [ ] Recruit additional clinicians (N=15-40)
- [ ] Onboard new users
- [ ] Collect larger-scale usage data
- [ ] Monitor for drift or degradation
- [ ] Iterate on UI based on Phase 2.2 feedback

**Success Criteria:**
- Maintain AUROC ≥ 0.85
- Maintain agreement rate ≥ 70%
- Maintain satisfaction ≥ 4.0/5.0
- Zero critical incidents (patient safety)

**Deliverables:**
- Larger validation dataset (N=1,000+ queries)
- Updated agreement analysis
- Production readiness assessment

**Owner:** Clinical Coordinator + Product Manager
**Estimated Effort:** 3 weeks

---

## SHARED COMPONENTS

### External Validation (Week 2)

**Owner:** Research Engineer
**Benefits:**
- **Publication:** Required for Methods section
- **Deployment:** Demonstrates generalizability

**Tasks:** See Track 1, Phase 1.2

---

### Clinical Expert Review (Week 4)

**Objective:** Review edge cases and false negatives

**Tasks:**
- [ ] Identify 3 false negative cases (from verification)
- [ ] Identify 20 random high-confidence alerts (POS state)
- [ ] Identify 10 edge cases (very long posts, rare criteria)
- [ ] Recruit N=2-3 clinical experts
- [ ] Conduct structured review session
- [ ] Document findings

**Review Questions:**
1. Are the false negatives truly errors or annotation issues?
2. Are high-confidence alerts clinically meaningful?
3. How would you improve the system?
4. What failure modes concern you most?

**Deliverables:**
- `outputs/clinical_review/expert_judgments.csv`
- `outputs/clinical_review/recommendations.md`
- Updated error analysis for manuscript

**Owner:** Clinical Lead
**Estimated Effort:** 1 week

---

## RESOURCE ALLOCATION

### Team Roles

| Role | Responsibility | Time Commitment |
|------|----------------|-----------------|
| **Lead Researcher** | Overall coordination, manuscript writing | 50% (20 hrs/week) |
| **Research Engineer** | Ablation studies, external validation | 100% (40 hrs/week, Weeks 1-4) |
| **Software Engineer** | Pilot deployment, monitoring | 100% (40 hrs/week, Weeks 4-8) |
| **Clinical Coordinator** | Recruit clinicians, manage pilot | 50% (20 hrs/week, Weeks 4-12) |
| **Domain Expert** | Manuscript collaboration, clinical review | 25% (10 hrs/week, Weeks 8-16) |
| **DevOps Engineer** | Infrastructure, monitoring dashboard | 50% (20 hrs/week, Week 5) |
| **Data Analyst** | Pilot data analysis, metrics | 50% (20 hrs/week, Weeks 8-10) |

### Hardware Resources

- **GPU Server:** 1× RTX 5090 (24GB VRAM) - for ablations and inference
- **Web Server:** 4 vCPU, 16GB RAM - for pilot deployment
- **Database:** PostgreSQL, 100GB storage - for logging
- **Monitoring:** Grafana/Prometheus instance

### Budget Estimate (Rough)

| Item | Cost | Notes |
|------|------|-------|
| **GPU Server** | $50/day × 30 days | $1,500 (cloud or on-prem) |
| **Web Server** | $100/month × 3 months | $300 |
| **Clinician Compensation** | $100/hr × 10 clinicians × 5 hrs | $5,000 (pilot study) |
| **IRB Fees** | $500 | If required |
| **Miscellaneous** | $500 | Data annotation, tools |
| **Total** | ~$7,800 | |

---

## RISK MANAGEMENT

### Critical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **External validation shows degradation** | Medium | High | Have backup dataset ready, analyze domain shift |
| **Clinicians unavailable for pilot** | Medium | High | Recruit 15-20 (target N=10, expect 50% drop) |
| **Ablation studies show no improvement** | Low | Medium | Document honestly, frame as negative result |
| **Pilot reveals safety issue** | Low | Critical | Immediate rollback, incident review |

### Contingency Plans

**If External Validation AUROC < 0.80:**
- Analyze domain shift
- Retrain with domain adaptation
- Report as limitation in paper
- Adjust deployment scope

**If Pilot Feedback is Negative (Satisfaction < 3.0):**
- Conduct user interviews
- Iterate on UI/UX
- Consider system as research tool only (not clinical deployment)

**If Timeline Slips:**
- Priority 1: External validation (required for both tracks)
- Priority 2: Ablation studies (required for publication)
- Priority 3: Pilot study (can be reduced to N=5 minimum)

---

## SUCCESS METRICS

### Track 1 (Publication)

**Minimum Success:**
- [ ] External validation AUROC ≥ 0.80
- [ ] Ablation studies complete (7 configs)
- [ ] Manuscript submitted to top-tier venue

**Target Success:**
- [ ] External validation AUROC ≥ 0.85
- [ ] All ablations + policy studies complete
- [ ] Pilot data included as prospective validation
- [ ] Accepted to ACL/EMNLP/JAMIA

### Track 2 (Deployment)

**Minimum Success:**
- [ ] Pilot deployed with N=5 clinicians
- [ ] Agreement rate ≥ 60%
- [ ] Time savings ≥ 5%

**Target Success:**
- [ ] Pilot deployed with N=10 clinicians
- [ ] Agreement rate ≥ 70%
- [ ] Time savings ≥ 10%
- [ ] Satisfaction ≥ 4.0/5.0
- [ ] Zero critical incidents

---

## MILESTONES & GATES

### Month 1 Gate: Foundation Complete

**Criteria:**
- ✅ External validation complete (AUROC ≥ 0.80)
- ✅ Ablation studies complete (7 configs)
- ✅ Clinical expert review complete

**Decision:** GO/NO-GO for pilot study
- **GO:** Proceed with pilot deployment (Week 5)
- **NO-GO:** Focus on publication only, defer deployment

---

### Month 2 Gate: Pilot Launch

**Criteria:**
- ✅ Monitoring dashboard live
- ✅ N=5-10 clinicians recruited
- ✅ IRB approval obtained (if needed)
- ✅ User guide created

**Decision:** LAUNCH pilot study (Week 6)

---

### Month 3 Gate: Manuscript Draft

**Criteria:**
- ✅ First manuscript draft complete
- ✅ Pilot data collected (N=250-500 queries)
- ✅ Ablation analysis written up

**Decision:** Proceed to internal review

---

### Month 4 Gate: Pre-Submission

**Criteria:**
- ✅ Manuscript revised (internal review complete)
- ✅ All figures finalized
- ✅ Expanded pilot launched (if proceeding to production)

**Decision:** SUBMIT to venue

---

## COMMUNICATION PLAN

### Weekly Standup (Every Monday, 30 min)

**Attendees:** Full team
**Agenda:**
1. Progress updates (5 min per track)
2. Blockers and issues
3. Next week priorities

### Bi-Weekly Review (Every Other Friday, 1 hr)

**Attendees:** Lead Researcher + Stakeholders
**Agenda:**
1. Metrics review (publication + deployment)
2. Risk assessment
3. Timeline adjustments

### Monthly Executive Briefing (1 hr)

**Attendees:** Executive stakeholders
**Format:** Slide deck (10 slides)
**Content:**
1. Progress summary
2. Key metrics
3. Risks and mitigations
4. Next month priorities

---

## DELIVERABLES TRACKER

### Track 1: Academic Publication

- [ ] External validation complete (`outputs/external_validation/`)
- [ ] Ablation studies complete (`outputs/ablation/`)
- [ ] Manuscript first draft (`manuscript/draft_v1.pdf`)
- [ ] Manuscript submitted (`manuscript/final.pdf`)

### Track 2: Clinical Deployment

- [ ] Pilot system deployed (URL live)
- [ ] Monitoring dashboard live (URL)
- [ ] User guide created (`docs/clinical/USER_GUIDE.pdf`)
- [ ] Pilot study complete (`outputs/pilot_study/`)
- [ ] Expanded pilot launched

### Shared

- [ ] Clinical expert review (`outputs/clinical_review/`)
- [ ] Project plan finalized (this document)

---

## NEXT IMMEDIATE ACTIONS (Week 1)

**Day 1-2:**
- [ ] Finalize team assignments
- [ ] Kickoff meeting (all stakeholders)
- [ ] Setup project tracking (GitHub Projects or Jira)
- [ ] Identify external validation dataset

**Day 3-5:**
- [ ] Start ablation studies (launch overnight runs)
- [ ] Prepare external validation dataset
- [ ] Begin clinician recruitment for pilot

**Day 6-7:**
- [ ] Review ablation study results (partial)
- [ ] Run external validation
- [ ] Schedule clinical expert review session

---

**Project Plan Version:** 1.0
**Last Updated:** 2026-01-18
**Next Review:** 2026-01-25 (Week 1 standup)
**Status:** 🚀 LAUNCHED - Dual-track execution in progress

**For Questions:** Contact Lead Researcher

**END OF PROJECT EXECUTION PLAN**
