# Phase 2: LLM Integration - Research Matrix

**Date:** 2026-01-18
**Status:** IN PROGRESS
**Branch:** gnn_e2e_gold_standard_report
**Hardware:** RTX 5090 (32GB VRAM)

---

## Executive Summary

Phase 2 explores LLM-enhanced evidence retrieval through 5 experimental modules:

- **M1:** LLM Listwise Reranker (rerank top-N candidates)
- **M2:** LLM Evidence Verifier (binary evidence classification)
- **M3:** LLM Query Refinement (generate criterion variations)
- **M4:** LLM Self-Reflection (reconsider UNCERTAIN predictions)
- **M5:** LLM-as-Judge (evaluate evidence quality with controls)

**Backend:** Qwen2.5-7B-Instruct (local, 4-bit quantization, temperature=0.0)
**Validation:** Gemini 1.5 Flash (optional API confirmation, N=3 self-consistency)

All modules designed with:
- ✅ Deterministic inference (temperature=0.0)
- ✅ Strict prompt templates (no free-form generation)
- ✅ Response caching (reproducibility)
- ✅ Fail-safe fallbacks (if LLM unavailable)
- ✅ Gold-standard evaluation (same metrics as Phase 0/1)

---

## Module Descriptions

### M1: LLM Listwise Reranker

**Purpose:** Use LLM to rerank top-K candidates from base retriever

**Input:**
- Query (criterion text)
- Candidate sentences (top-24 from retriever)

**Output:**
- Reranked candidate list (top-10)
- Relevance scores per candidate

**Prompt Template:**
```
You are an expert mental health researcher. Given a DSM-5 criterion and a list of sentences, rank the sentences by how well they provide evidence for the criterion.

Criterion: {criterion_text}

Sentences:
[1] {sentence_1}
[2] {sentence_2}
...
[24] {sentence_24}

Instructions:
- Rank sentences from most relevant (1) to least relevant (24)
- Output ONLY the ranked indices, one per line
- No explanations, no additional text

Output:
```

**Evaluation:**
- Compare vs Jina-Reranker-v3 baseline
- Metrics: nDCG@10, Recall@10, MRR@10
- Latency: Inference time per query

**Expected Performance:**
- Competitive with Jina-v3 (nDCG@10 ≈ 0.85-0.87)
- Slower inference (~2-5s vs 50ms for Jina-v3)

### M2: LLM Evidence Verifier

**Purpose:** Binary classification - is this sentence evidence for the criterion?

**Input:**
- Query (criterion text)
- Single candidate sentence

**Output:**
- Binary decision: EVIDENCE / NOT_EVIDENCE
- Confidence score (0-1)

**Prompt Template:**
```
You are an expert mental health researcher. Determine if the given sentence provides direct evidence for the DSM-5 criterion.

Criterion: {criterion_text}

Sentence: {candidate_sentence}

Instructions:
- Answer "YES" if the sentence directly supports the criterion
- Answer "NO" if the sentence does not provide evidence
- Output ONLY "YES" or "NO", nothing else

Answer:
```

**Evaluation:**
- Use as post-filter on top-K from reranker
- Metrics: Precision (after filtering), Recall (coverage)
- Compare vs P4 NE Gate (AUROC=0.8972)

**Expected Performance:**
- High precision (reduce false positives)
- May reduce recall slightly (conservative filtering)

### M3: LLM Query Refinement

**Purpose:** Generate criterion paraphrases for multi-query retrieval

**Input:**
- Original criterion text

**Output:**
- N paraphrased versions (N=3-5)
- Semantic variations of the criterion

**Prompt Template:**
```
You are an expert mental health researcher. Generate {N} paraphrased versions of the following DSM-5 criterion. Each paraphrase should:
- Preserve the clinical meaning
- Use different wording
- Maintain specificity

Original Criterion: {criterion_text}

Instructions:
- Generate exactly {N} paraphrases
- Each paraphrase on a new line, numbered
- No explanations, no additional text

Paraphrases:
```

**Evaluation:**
- Retrieve with each paraphrase, fuse results (RRF)
- Compare vs single-query baseline
- Metrics: Recall@10 (coverage improvement)

**Expected Performance:**
- +2-5% recall improvement (query diversity)
- Increased inference cost (N queries vs 1)

### M4: LLM Self-Reflection (UNCERTAIN Only)

**Purpose:** Reconsider predictions in UNCERTAIN zone (NEG < prob < POS)

**Input:**
- Criterion text
- Top-K retrieved candidates
- P4 probability (from NE gate)
- State: UNCERTAIN

**Output:**
- Revised decision: PROMOTE_TO_POS / DEMOTE_TO_NEG / KEEP_UNCERTAIN
- Reasoning (for clinical review)

**Prompt Template:**
```
You are an expert mental health researcher. A screening system flagged this case as UNCERTAIN. Review the evidence and decide if there is enough support for the criterion.

Criterion: {criterion_text}

Retrieved Evidence:
[1] {sentence_1} (score: {score_1})
[2] {sentence_2} (score: {score_2})
...
[K] {sentence_K} (score: {score_K})

Model Confidence: {p4_probability:.2f} (UNCERTAIN zone)

Instructions:
- If evidence clearly supports the criterion: Answer "PROMOTE"
- If evidence is clearly insufficient: Answer "DEMOTE"
- If genuinely uncertain: Answer "KEEP"
- Output ONLY one of these three words

Decision:
```

**Evaluation:**
- Apply only to UNCERTAIN queries (14.9% of dataset)
- Metrics: Alert precision change, screening sensitivity change
- Clinical utility: Reduce uncertain queue size

**Expected Performance:**
- Reduce UNCERTAIN rate by 20-40%
- Maintain screening sensitivity ≥99.5%
- Improve alert precision (better confidence)

### M5: LLM-as-Judge (Optional, Strict Controls)

**Purpose:** Evaluate evidence quality for clinical review

**Input:**
- Criterion text
- Retrieved evidence sentences
- Gold labels (for evaluation only)

**Output:**
- Quality score (1-5)
- Brief justification

**Prompt Template:**
```
You are an expert mental health researcher. Rate the quality of evidence retrieved for this criterion on a scale of 1-5.

Criterion: {criterion_text}

Retrieved Evidence:
{top_k_sentences}

Scale:
1 - No relevant evidence
2 - Weak or tangential evidence
3 - Moderate evidence, some support
4 - Strong evidence, clear support
5 - Excellent evidence, comprehensive support

Instructions:
- Output ONLY a number (1-5)
- No explanations

Rating:
```

**Strict Controls:**
- ⚠️ Use ONLY for evaluation, never for training
- ⚠️ Compare LLM ratings vs gold labels (correlation)
- ⚠️ Report agreement statistics (Pearson r, Spearman ρ)
- ⚠️ NOT used for system decisions (audit only)

**Evaluation:**
- Correlation with gold labels
- Inter-rater reliability (LLM vs human)
- Identify systematic biases

---

## Implementation Plan

### Phase 2A: Local LLM Setup (1-2 hours)

**Model:** Qwen2.5-7B-Instruct
- HuggingFace: `Qwen/Qwen2.5-7B-Instruct`
- Quantization: 4-bit (VRAM ≈8GB, fits on RTX 5090 with room for retriever)
- Temperature: 0.0 (deterministic)
- Max tokens: 512 (sufficient for ranking/classification)

**Infrastructure:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
)
```

**Caching:**
- Response cache: `{prompt_hash: response}` (disk-based)
- Ensures reproducibility and reduces inference cost

### Phase 2B: Implement Modules (6-8 hours)

**Files to Create:**
```
src/final_sc_review/llm/
├── __init__.py
├── base.py                    # Base LLM wrapper
├── qwen_local.py             # Qwen2.5 local inference
├── gemini_api.py             # Gemini API (optional validation)
├── m1_reranker.py            # Listwise reranker
├── m2_verifier.py            # Evidence verifier
├── m3_query_expansion.py     # Query refinement
├── m4_reflection.py          # Self-reflection
└── m5_judge.py               # LLM-as-judge (optional)

scripts/llm/
├── test_llm_modules.py       # Unit tests
├── run_m1_reranker_eval.py   # M1 evaluation
├── run_m2_verifier_eval.py   # M2 evaluation
├── run_m3_expansion_eval.py  # M3 evaluation
├── run_m4_reflection_eval.py # M4 evaluation
└── run_llm_full_suite.py     # All modules
```

### Phase 2C: Evaluation (4-6 hours)

**Evaluation Protocol:**
1. Use DEV split for initial experiments (fast iteration)
2. Run 5-fold CV for final results (same folds as Phase 0)
3. Report same metrics as baseline for comparison

**Metrics Per Module:**
- M1: nDCG@10, Recall@10, MRR@10, latency
- M2: Precision, Recall, F1 (after filtering), AUROC
- M3: Recall@10 improvement, latency overhead
- M4: UNCERTAIN rate change, alert precision, screening sensitivity
- M5: Correlation with gold (Pearson r), agreement statistics

### Phase 2D: Analysis (2-3 hours)

**Questions to Answer:**
1. **Performance:** Do LLMs improve over neural baselines?
2. **Efficiency:** Is latency acceptable for deployment?
3. **Reliability:** Are LLM outputs consistent (self-consistency N=3)?
4. **Failure modes:** Where do LLMs make mistakes?
5. **Clinical utility:** Which modules have practical value?

**Deliverables:**
- Performance comparison table (LLM vs baselines)
- Latency analysis (inference time distribution)
- Failure mode analysis (error categories)
- Clinical utility assessment (which modules to deploy)

---

## Expected Timeline

| Task | Duration | Deliverable |
|------|----------|-------------|
| Phase 2A: LLM setup | 1-2 hours | Qwen2.5 loaded, response cache working |
| Phase 2B: M1-M2 implementation | 3-4 hours | Reranker + verifier modules |
| Phase 2B: M3-M4 implementation | 3-4 hours | Query expansion + reflection |
| Phase 2B: M5 implementation (optional) | 1-2 hours | LLM-as-judge |
| Phase 2C: DEV split experiments | 2-3 hours | Initial results for all modules |
| Phase 2C: 5-fold CV evaluation | 2-4 hours | Final results (if promising on DEV) |
| Phase 2D: Analysis + report | 2-3 hours | Comparison tables, findings |
| **Total** | **14-22 hours** | Complete LLM integration study |

---

## Success Criteria

1. **M1 (Reranker):** Competitive with Jina-v3 (nDCG@10 ≥ 0.83)
2. **M2 (Verifier):** Precision improvement ≥ +5% at same recall
3. **M3 (Query Expansion):** Recall improvement ≥ +2%
4. **M4 (Reflection):** UNCERTAIN rate reduced by ≥20%, sensitivity ≥99.5%
5. **M5 (Judge):** Correlation with gold ≥ 0.70 (Pearson r)

**Minimum Viable Output:** At least 1 module shows clear improvement over baseline

---

## Risk Mitigation

**Risk 1: LLM inference too slow**
- **Mitigation:** Batch inference, response caching
- **Fallback:** Use only for high-value queries (UNCERTAIN zone)

**Risk 2: LLM outputs inconsistent**
- **Mitigation:** Temperature=0.0, self-consistency N=3
- **Validation:** Gemini API confirmation

**Risk 3: No performance gain over baselines**
- **Mitigation:** Focus on clinical utility (explainability, confidence)
- **Value:** Document when LLMs don't help (negative result publishable)

**Risk 4: Prompt engineering required**
- **Mitigation:** Start with simple templates, iterate
- **Fallback:** Use chain-of-thought if needed

---

## Next Steps

1. ✅ Phase 2 design complete (this document)
2. ⏳ Implement base LLM wrapper (Qwen2.5 + caching)
3. ⏳ Implement M1-M2 (core modules)
4. ⏳ Run DEV split experiments
5. ⏳ Decide: Full 5-fold CV or move to Phase 3

---

**Status:** Design Complete - Ready for Implementation
**Last Updated:** 2026-01-18
**Next Action:** Implement base LLM infrastructure
