# Product Requirements Document (PRD) — Person 4 Entity Resolution System

## 1. Problem

Amazon ML Unstop Business Entity Resolution Challenge asks us to link noisy,
duplicated business records to a clean reference list. We are given three
sources:

- **Source 1** — a deduplicated set of reference businesses (the "canonical" list).
- **Source 2** — noisy business records (may contain 0, 1, or many records that
  refer to the same real business as a given Source 1 entity).
- **Source 3** — a second, independent set of noisy business records with the
  same structure as Source 2.

Each record has only four fields: `entity_id`, `business_name`,
`business_address`, `country`. There is no phone number, website, category,
or geocoded location — matching must be done from text alone.

For every Source 1 entity we must find **all** Source 2/Source 3 records that
refer to the same real-world business. A Source 1 entity can have zero, one,
or many correct matches.

## 2. Users / Team Stakeholders

- **Person 4 (this system)** — the author/maintainer of this independent
  pipeline, responsible for producing candidate pairs and final predictions,
  and for measuring how well they work.
- **Person 1, Person 2, Person 3** — teammates building alternative matching
  systems (classical fuzzy rules, supervised pair classifier, embedding
  similarity respectively). Their outputs may later be combined with Person
  4's outputs in an ensemble, but Person 4 must stand on its own.
- **Team lead / reviewers** — need to be able to read this documentation and
  reproduce results without needing to read all of the code first.

## 3. Objective

Build an independent, multi-stage retrieval-and-reranking system that:

1. Generates a high-recall candidate set of possible Source 2/Source 3 matches
   for every Source 1 entity, using several different blocking strategies.
2. Cheaply filters those candidates in a first pass (Stage 1).
3. Carefully reranks the smaller surviving set using more detailed evidence
   (Stage 2).
4. Converts scores into a final yes/no decision per candidate.
5. Aggregates decisions per Source 1 entity into a final submission.
6. Measures everything it does, honestly, against a held-out validation split.

## 4. Input Data

```
data/train/train_source1.tsv
data/train/train_source2.tsv
data/train/train_source3.tsv
data/train/train_ground_truth.tsv

data/test/test_source1.tsv
data/test/test_source2.tsv
data/test/test_source3.tsv
```

All files are tab-separated (`sep="\t"`), never comma-separated. Source
files have columns `entity_id, business_name, business_address, country`.
Ground truth has columns `source1_entity_id, matched_entity_ids` where
`matched_entity_ids` is a comma-separated list of Source 2/Source 3 IDs (or
empty for a singleton).

**As of the writing of this document, no data files have been placed in this
repository yet.** All modules below are designed to work with them once they
arrive, and are covered by unit tests using small synthetic fixtures in the
meantime (see `tests/`).

## 5. Expected Outputs

```
outputs/matching_results.tsv   -> source1_entity_id, matched_entity_ids
outputs/candidate_pairs.tsv    -> source1_entity_id, candidate_entity_ids
reports/experiments.csv        -> one row per experiment run
reports/system_comparison.csv  -> Person1 vs Person2 vs Person3 vs Person4
reports/error_analysis.csv     -> hard false positives / false negatives
reports/final_report.md        -> narrative summary of everything measured
```

## 6. Success Criteria

- The pipeline runs end-to-end on the provided train/test data with only
  `pd.read_csv(path, sep="\t")` used for loading — no external data sources.
- Candidate recall, precision, recall, and F0.5 are measured at the
  **Source-1-entity level** (not just pair level) and macro-averaged, exactly
  as the official challenge scores submissions.
- Every configuration choice (blocking method, Stage-1 filter, Stage-2
  features, decision threshold, ensemble weights) is backed by a row in
  `reports/experiments.csv` — not by intuition.
- `outputs/matching_results.tsv` and `outputs/candidate_pairs.tsv` pass all
  checks in `utils/validate_submission.py`.
- No metric is ever written into documentation before it has actually been
  computed. Unmeasured values are recorded as "Not yet measured".

## 7. Constraints

- **No external data or services**: no web search, no geocoding APIs, no
  commercial entity-resolution APIs, no external business databases.
- **Country is open-set**: never hard-code a list of allowed countries (the
  test set includes at least France in addition to the training countries).
- **Model size**: any ML/embedding model used must be ≤ 8B parameters and
  appropriately licensed.
- **Environment constraint (discovered during Phase 0 inspection)**: this
  container has no network access, so common fuzzy-matching libraries
  (`rapidfuzz`, `fuzzywuzzy`, `jellyfish`, `recordlinkage`, `unidecode`)
  cannot be installed. The system is built using only the standard library,
  `pandas`, `numpy`, and `scikit-learn`, which are already installed.
- **Independence from Person 1/2/3**: the pipeline must work with
  `person1_scores = person2_scores = person3_scores = None`. Teammate
  signals are optional inputs, integrated only after their contribution is
  measured on the same validation split.
- **Submission format rules**: exactly one output row per Source 1 test
  entity; predicted IDs must come from Source 2/3 test data, must not
  duplicate, must not include Source 1 IDs, and must be a subset of the
  generated candidate set; singleton entities get an empty
  `matched_entity_ids` value.

## 8. Evaluation Metric

Entity-level **F0.5**, computed per Source 1 entity and macro-averaged
across all Source 1 entities (see `docs/EVALUATION_PROTOCOL.md` for the
exact formula and edge-case handling, including singletons).

## 9. System Workflow

```
Raw TSVs
   -> Preprocessing (name/address/country normalization)
   -> Multi-pass candidate generation (several independent blocking passes)
   -> Candidate recall measurement
   -> Stage 1: cheap broad-retrieval scoring/filter
   -> Stage 2: detailed reranking
   -> Decision logic (threshold/rule chosen from measured validation results)
   -> Source-1 entity-level aggregation
   -> Evaluation (candidate recall, precision, recall, F0.5, errors)
   -> Output generation + validation
```

## 10. Functional Requirements

- FR1: Load and validate all TSV files, reporting row counts, duplicate IDs,
  and missing values, without silently modifying IDs.
- FR2: Produce normalized/tokenized representations of name, address, and
  country without hard-coding country values.
- FR3: Generate candidates via at least the six blocking strategies listed in
  `docs/TRD.md` (country+name tokens, character n-grams, address tokens,
  numeric evidence, rare tokens, multi-pass union), with adaptive fallback
  when name or address is weak/missing.
- FR4: Measure candidate recall, candidate count, reduction ratio, and
  runtime for every blocking configuration tested.
- FR5: Score candidates in two stages (broad filter, then detailed rerank)
  and measure the recall/precision impact of each stage separately.
- FR6: Convert stage-2 scores into discrete decisions using a threshold or
  rule selected from experiments, not guessed.
- FR7: Aggregate per-candidate decisions into one prediction list per Source
  1 entity, including correctly representing singletons.
- FR8: Compute entity-level precision/recall/F0.5, macro-averaged, with
  explicit singleton handling.
- FR9: Produce `reports/error_analysis.csv` categorizing hard FP/FN cases.
- FR10: Provide optional loaders for Person 1/2/3 scores in a common
  `source1_entity_id, candidate_entity_id, score` schema, without assuming
  they help, and support ensemble experiments once available.
- FR11: Produce and validate the two required output TSVs.
- FR12: Track every experiment in `reports/experiments.csv` with enough
  metadata (seed, configuration, metrics) to reproduce it.

## 11. Non-Functional Requirements

- **Reproducibility**: fixed random seeds, deterministic Source-1-level
  train/validation split, recorded configuration per experiment.
- **Readability**: small functions, type hints, docstrings/comments,
  beginner-friendly code (this system will be maintained by teammates who
  are not all ML specialists).
- **No hidden state**: no hard-coded paths, no hard-coded country lists, no
  unexplained magic thresholds.
- **Honesty over completeness**: an unmeasured number must never appear as
  if it were measured.
- **Portability**: the system should not depend on GPU access or any
  package that is not already available in this environment (see
  Constraints above).
