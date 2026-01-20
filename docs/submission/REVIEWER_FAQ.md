# Anticipated Reviewer Questions

**Version:** 1.0  
**Date:** 2026-01-20

This document anticipates common reviewer questions and provides pre-drafted responses.

---

## Q1: Why is the AUPRC relatively low compared to AUROC?

**Response:**

The AUPRC (0.57) is lower than AUROC (0.90) due to the class imbalance in our dataset:
- Positive rate: 9.3% of queries have evidence
- AUPRC is sensitive to class imbalance while AUROC is not

This is expected behavior. For highly imbalanced problems, AUPRC is a more informative metric than AUROC [Saito & Rehmsmeier, 2015]. Our AUPRC of 0.57 represents a ~6x improvement over the random baseline (0.093).

**Evidence:**
- `results/paper_bundle/v3.0/metrics_master.json` - Positive rate statistics
- Random baseline AUPRC = positive_rate = 0.093

---

## Q2: How do you ensure no data leakage between train/val/test?

**Response:**

We enforce **post-ID-disjoint splits**:
1. Each unique post appears in exactly one split
2. Splits are created at the post level, not query level
3. Automated test (`test_splits.py`) verifies disjointness on every run

**Verification command:**
```bash
python scripts/audit_splits.py --data_dir data --seed 42 --k 5
```

**Output shows:**
- Train/Val/Test: Disjoint = YES
- 5-Fold CV: Disjoint = YES
- Test overlaps: 0

---

## Q3: Why is A.10 (Duration) performance lower than other criteria?

**Response:**

A.10 (Duration: 2+ weeks) has the lowest AUROC (0.66) because:

1. **Implicit Information**: Duration is rarely stated explicitly ("I've felt this way for months")
2. **Temporal Reasoning**: Requires inference across sentences/posts
3. **Low Base Rate**: Only 5.8% positive rate (fewest examples)
4. **Text Limitation**: Single-post analysis misses longitudinal patterns

This is a known limitation documented in Section 5.2 of the paper. Future work could incorporate temporal modeling across multiple posts.

---

## Q4: How do you handle the privacy of mental health content?

**Response:**

Multiple privacy protections are in place:

1. **No raw data in repository**: Post content never committed
2. **Anonymized IDs**: Original Reddit IDs replaced with opaque identifiers
3. **Aggregate reporting**: Only statistics reported, no individual examples
4. **Data access requires DUA**: Signed agreement for any data access

See `docs/final/DATA_STATEMENT.md` for complete protocol.

---

## Q5: Can your method be used for clinical diagnosis?

**Response:**

**No. This is explicitly NOT intended for diagnosis.**

Appropriate uses:
- Research tool with clinical oversight
- Screening assistance (not replacement)
- Evidence aggregation for expert review

We implement a three-state output (NEG/UNCERTAIN/POS) to flag uncertain cases for mandatory human review. See `docs/ETHICS.md` for full clinical use disclaimers.

---

## Q6: How does your method compare to existing psychiatric NLP methods?

**Response:**

Our method differs from prior work in several ways:

| Aspect | Prior Work | Our Method |
|--------|-----------|------------|
| Task | Post-level classification | Sentence-level evidence retrieval |
| Output | Binary (has symptom) | Ranked evidence sentences |
| Criteria | Often single symptom | All 10 DSM-5 MDD criteria |
| Explainability | Implicit | Explicit evidence binding |

We provide IR baselines (BM25, TF-IDF, E5, Contriever) for comparison.

---

## Q7: What is the computational cost of the pipeline?

**Response:**

| Component | Latency (per query) | Hardware |
|-----------|---------------------|----------|
| Retriever (NV-Embed-v2) | ~50ms | A100 GPU |
| Reranker (Jina-v3) | ~30ms | A100 GPU |
| GNN (P4) | ~5ms | A100 GPU |
| **Total (end-to-end)** | **~85ms** | A100 GPU |

See `scripts/analysis/measure_latency.py` for real measurement methodology.

---

## Q8: How reproducible are your results?

**Response:**

Full reproducibility is ensured through:

1. **Fixed seeds**: All randomness controlled (seed=42)
2. **Deterministic operations**: CuDNN deterministic mode enabled
3. **Version pinning**: All dependencies pinned in `requirements.txt`
4. **Checksums**: All artifacts have SHA256 verification
5. **Automated tests**: 227 tests verify correctness

**Verification:**
```bash
python scripts/verification/verify_checksums.py --bundle results/paper_bundle/v3.0
# Output: VERIFICATION PASSED
```

---

## Q9: Why use bootstrap CIs instead of cross-validation?

**Response:**

We use bootstrap CIs for several reasons:

1. **Computational efficiency**: Avoids re-training 5x
2. **Post-level correlation**: Bootstrap samples posts, preserving within-post structure
3. **Standard practice**: Recommended for IR evaluation [Sakai, 2018]
4. **Interpretability**: Direct CI on metric of interest

We also provide 5-fold CV results in supplementary materials for completeness.

---

## Q10: Can you share the trained models?

**Response:**

Yes, model checkpoints are included:

```
outputs/gnn_research/p3_retrained/  # GNN checkpoints
```

The pipeline can be run with provided checkpoints:
```bash
python scripts/eval_zoo_pipeline.py --config configs/default.yaml --split test
```

Note: This requires access to the sentence corpus (see data access request process).
