# EntityResolve-X: Final Engineering & Evaluation Report
**Person 4 Independent ML Pipeline**  
**Amazon ML Unstop Business Entity Resolution Challenge**  
**Date:** September 2026 | **Author:** Person 4 (Senior ML Engineer)

---

## 1. Executive Summary

This report documents the design, empirical validation, error analysis, and test performance of **EntityResolve-X**, the independent machine learning pipeline developed by Person 4 for the Amazon ML Unstop Business Entity Resolution Challenge.

Unlike single-pass or black-box classifiers, EntityResolve-X implements a **two-stage retrieval and reranking architecture**:
1. **Multi-Pass Blocking (Blocks A–F):** Achieved **100.00% Candidate Recall** on the held-out Source 1 validation split with an average candidate pool of **28.6** records per entity and a **94.69% Reduction Ratio**.
2. **Stage 1 (Broad Filtering):** Uses lightweight set-theoretic token and n-gram overlap, successfully pruning **89.4%** of non-matching candidate noise in **0.01 seconds** while retaining **100.00% Candidate Recall**.
3. **Stage 2 (Detailed Reranking):** Evaluates multi-scale string similarities (token sort, partial ratio), street number agreement, postal code verification, and cross-field consistency.
4. **Architectural Staging Validation:** Compared to a single-stage baseline (scoring all candidates with Stage 2), the two-stage pipeline delivers **identical top-tier accuracy (99.11% Macro F0.5)** at a **3x execution speedup** (0.02s vs 0.06s).
5. **Decision Threshold Selection:** Grid search identified **0.65** as the optimal threshold, achieving **Macro Precision = 99.00%**, **Macro Recall = 100.00%**, and **Macro F0.5 = 99.11%** with explicit singleton handling.
6. **Submission Verification:** The final test submission files (`outputs/matching_results.tsv` and `outputs/candidate_pairs.tsv`) were programmatically generated and verified by `utils/validate_submission.py`, satisfying all competition constraints without error.

---

## 2. Why Person 4 is Structurally Independent

In accordance with Section 3 of the challenge brief, Person 4 does not reproduce the architectures of the other three teammates:

| System | Owner | Core Technique | Structural Differentiator |
|---|---|---|---|
| **Person 1** | Classical ER | Single-pass weighted fuzzy/rule scoring | Classical heuristics |
| **Person 2** | Pair Classifier | Direct supervised binary classifier on pair features | Supervised ML model |
| **Person 3** | Semantic Matching | Dense embedding similarity search | Representation learning |
| **Person 4 (EntityResolve-X)** | **Independent Pipeline** | **Multi-pass retrieval + Two-stage cascading reranking** | **Cascaded retrieval filter & fine-grained evidence fusion** |

Person 4's pipeline operates **100% standalone** (`person1_scores = person2_scores = person3_scores = None`), while exposing optional pluggable loader interfaces in `src/ensemble.py` if teammate scores are provided in the future.

---

## 3. Data Processing & Conservative Normalization

Text normalization adheres strictly to closed-data fair play rules (zero external geocoding or external entity APIs):
- **Business Names:** Unicode NFKD normalization, conservative punctuation removal, tokenization ($\ge 2$ chars), character 3-grams, and regex extraction of international legal suffixes (`pvt ltd`, `inc`, `llc`, `corp`, `gmbh`, etc.). Crucially, if suffix stripping leaves an empty string (e.g. business named "Limited"), the original core name is preserved.
- **Addresses:** Street type abbreviation expansion (`rd` $\rightarrow$ `road`, `st` $\rightarrow$ `street`, `ste` $\rightarrow$ `suite`), numeric sequence extraction, and 4–6 digit postal code candidate identification.
- **Country:** Open-set string standardization (uppercase and trimmed). Never relies on hard-coded country lists, ensuring full generalization to unseen test countries (e.g. France, Germany, Japan).

---

## 4. Candidate Generation Empirical Evaluation

Candidate generation was evaluated on the held-out Source 1 validation split across all individual blocks and the multi-pass union:

| Strategy | Description | Candidate Recall | Avg Candidates / Query | Reduction Ratio | Runtime |
|---|---|---|---|---|---|
| **Block A** | Country + Name Tokens | **100.00%** | 3.14 | 99.42% | < 0.01s |
| **Block B** | Character 3-grams (Typo/Abbreviation Tolerant) | **100.00%** | 4.92 | 99.09% | < 0.01s |
| **Block C** | Informative Address Tokens | **100.00%** | 26.94 | 94.99% | < 0.01s |
| **Block D** | Numeric / Postal Evidence | **100.00%** | 1.16 | 99.78% | < 0.01s |
| **Block E** | Rare Tokens (IDF $\le 1\%$ corpus frequency) | **100.00%** | 2.16 | 99.60% | < 0.01s |
| **Block F (Union)** | **Multi-Pass Union + Adaptive Fallbacks** | **100.00%** | **28.56** | **94.69%** | **< 0.01s** |

*Takeaway:* Block F achieves an empirical **100% Candidate Recall ceiling**, ensuring downstream matching stages have access to all true matches while filtering out over **94.6%** of cartesian product noise.

---

## 5. Stage-1 Broad Filtering & Staging Comparison

### Stage 1 Efficiency
- **Stage 1 Filter:** Weighted Jaccard on Name Tokens (0.40), Name 3-grams (0.30), Address Tokens (0.20), Address Numbers (0.10).
- **Threshold:** `STAGE1_MIN_SCORE = 0.15`
- **Candidate Pool Reduction:** Average candidates pruned from **28.56** down to **3.02** per query.
- **Noise Pruned:** **89.4%** of non-matching pairs eliminated in **0.01 seconds**.
- **Recall Retention:** **100.00%** (zero true matches lost in broad filter).

### Single-Stage vs Two-Stage Empirical Comparison

| Architecture | Macro Precision | Macro Recall | Macro $F_{0.5}$ | False Positives | Runtime | Speedup |
|---|---|---|---|---|---|---|
| **Single-Stage Baseline** (Candidate Gen $\rightarrow$ Stage 2 Scorer) | 99.00% | 100.00% | **99.11%** | 1 | 0.06s | 1.0x |
| **Two-Stage Pipeline** (Candidate Gen $\rightarrow$ Stage 1 $\rightarrow$ Stage 2) | 99.00% | 100.00% | **99.11%** | 1 | **0.02s** | **3.0x** |

*Conclusion:* The two-stage design delivers identical top-tier accuracy while reducing computational load by 67%, validating the TRD architecture.

---

## 6. Decision Threshold Optimization

Sweeps conducted on validation data with Source 1 macro-averaging and explicit singleton handling:

| Threshold | Macro Precision | Macro Recall | Macro $F_{0.5}$ | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.50 | 53.67% | 84.00% | 56.06% | 54 | 84 | 0 |
| 0.55 | 63.70% | 86.00% | 65.70% | 54 | 47 | 0 |
| 0.60 | 92.00% | 98.00% | 92.64% | 54 | 8 | 0 |
| **0.65 (Optimal)** | **99.00%** | **100.00%** | **99.11%** | **54** | **1** | **0** |
| 0.70 | 96.00% | 95.00% | 95.67% | 49 | 0 | 5 |
| 0.75 | 82.00% | 82.00% | 82.00% | 39 | 0 | 15 |
| 0.80 | 34.00% | 32.00% | 33.33% | 3 | 0 | 51 |

- At thresholds $< 0.65$, false positives accumulate rapidly, lowering $F_{0.5}$.
- At thresholds $> 0.65$, subtle abbreviation and spelling variants are rejected, causing false negatives.
- **Threshold 0.65** is selected and frozen.

---

## 7. Error Analysis

Extracted in `reports/error_analysis.csv` at optimal threshold:
- **False Positives:** 1 case (`S1-00186` matched to `S3-00003`)
- **False Negatives:** 0 cases
- **Root Cause Diagnosis:** `numeric_conflict`
  - Query: `"Falcon Security Systems #186", "421 Oakland Avenue, San Francisco 16882", US`
  - Distractor Target: `"Falcon Security #6", "79 Oakland Ave, San Francisco 10222", US`
  - *Analysis:* The business names and street names matched closely, but the street numbers (`421` vs `79`) and unit numbers (`#186` vs `#6`) differed. Increasing the weight of numeric disagreement or requiring exact street number agreement on dense streets resolves such edge cases.

---

## 8. Teammate Comparison & Ensemble Status

As required by competition integrity rules (§18 & §24), values for systems not yet executed on this validation split remain strictly `N/A`:

| System | Candidate Recall | Precision | Recall | $F_{0.5}$ | False Positives | False Negatives | Runtime | Memory |
|---|---|---|---|---|---|---|---|---|
| **Person 1** (Classical) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Person 2** (Supervised) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Person 3** (Embedding) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **Person 4** (EntityResolve-X) | **1.0000** | **0.9900** | **1.0000** | **0.9911** | **1** | **0** | **0.02s** | **<45MB** |

*Ensemble experiments:* Documented in `src/ensemble.py`. If teammate outputs are supplied in the common format (`source1_entity_id`, `candidate_entity_id`, `score`), they can be blended via `combine_pair_scores()`.

---

## 9. Final Test Predictions & Submission Verification

Running `src/prediction.py` on test data produced:
1. `outputs/matching_results.tsv`:
   - Exact 83 Source 1 test entities represented.
   - 93 total predicted links.
   - 21 singletons correctly formatted with empty match lists.
2. `outputs/candidate_pairs.tsv`:
   - 2,407 candidates generated (average 29.0 per Source 1 entity).
3. `utils/validate_submission.py` verification output:
   - **All hard constraints verified:** zero self-matches, zero duplicate IDs, all predictions are subsets of candidate pairs, and all IDs exist in test Source 2/Source 3.

---

## 10. Reproducibility & Verification Commands

All results can be reproduced directly inside the workspace:

```bash
# 1. Run complete unit test suite (31 tests across 8 suites)
python -m pytest tests/ -v

# 2. Run blocking, staging, and threshold experiments
python -m src.run_experiments

# 3. Run test prediction with frozen pipeline
python -m src.prediction

# 4. Run standalone submission validator
python utils/validate_submission.py
```

---

## 11. System Limitations

1. **Extreme Name Corruption:** If both the name and address are truncated to 1-2 generic letters, adaptive blocking casts a broad country net, which may dilute ranking precision.
2. **Co-located Businesses:** Multiple unrelated businesses operating at the same commercial address require strong name dissimilarity weighting to avoid false positive merges.
3. **Open-Set Typo Tolerance:** Extremely noisy transliterated city/state names are currently matched via n-grams; if corruptions exceed 40% of characters, recall depends on postal code agreement.
