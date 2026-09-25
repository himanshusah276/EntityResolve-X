# Technical Requirements Document (TRD) — Person 4 Entity Resolution System

## 1. Architecture Overview

Person 4 is a **multi-stage retrieval-and-reranking system**. This is its key
structural difference from the other three team systems:

- Person 1 is a single-pass classical weighted fuzzy-rule scorer.
- Person 2 trains one supervised pair classifier directly on candidate pairs.
- Person 3 relies on a single embedding model + similarity search.
- **Person 4 instead runs multiple independent blocking passes to build
  candidates, then applies two separate, independently-designed scoring
  stages (a cheap broad filter, then a detailed reranker) before making a
  decision.** No single scoring function does all the work; the stages are
  measured separately so we know which one is actually contributing.

```
                     ┌─────────────────────┐
 Raw TSVs  ────────► │   data_loader.py     │
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │  preprocessing.py    │  (name / address / country
                     └─────────┬───────────┘   normalization + tokens)
                               ▼
                     ┌─────────────────────┐
                     │ candidate_generation │  (Blocks A-F, multi-pass union)
                     │        .py           │
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │  evaluation.py       │  candidate recall measurement
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │ advanced_matching.py │  Stage 1 (broad filter)
                     │                      │  Stage 2 (detailed rerank)
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │ ensemble.py          │  optional P1/P2/P3 blending
                     │ (optional interfaces)│  (only used if scores provided)
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │  prediction.py       │  decision logic + aggregation
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │  output.py           │  TSV generation + validation
                     └─────────┬───────────┘
                               ▼
                     ┌─────────────────────┐
                     │  error_analysis.py   │  FP/FN categorization
                     └─────────────────────┘
```

## 2. Modules

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | Load TSVs, validate columns, report row counts / duplicates / missing values, preserve raw IDs. |
| `src/preprocessing.py` | Build normalized name/address/country representations and tokens. No external normalization services. |
| `src/candidate_generation.py` | Multi-pass blocking (Blocks A–F), adaptive fallback rules, per-block metrics. |
| `src/advanced_matching.py` | Stage-1 broad scorer/filter and Stage-2 detailed reranker. Pure feature engineering + a simple, interpretable scorer (see §6). |
| `src/ensemble.py` | Optional loaders for Person 1/2/3 scores, score normalization, weighted/learned ensemble combination. |
| `src/evaluation.py` | Candidate recall, entity-level precision/recall/F0.5, macro-averaging, singleton handling. |
| `src/error_analysis.py` | Extract and categorize hard false positives / false negatives. |
| `src/prediction.py` | Decision logic (threshold or rule), Source-1 entity-level aggregation. |
| `src/output.py` | Write `matching_results.tsv` / `candidate_pairs.tsv`, run submission validation. |
| `src/config.py` | Central configuration object (paths, seed, thresholds, blocking parameters) — no magic numbers scattered through the codebase. |
| `utils/validate_submission.py` | Standalone validator, runnable from the command line, independent of the rest of the pipeline. |

## 3. Data Flow

1. `data_loader.py` reads the four training TSVs (and later the three test
   TSVs) into pandas DataFrames, validates them, and returns them alongside
   a small data-quality report (dict/DataFrame).
2. `preprocessing.py` takes each source DataFrame and adds normalized
   columns (`name_norm`, `name_tokens`, `name_ngrams`, `addr_norm`,
   `addr_tokens`, `addr_numeric_tokens`, `country_norm`, ...). It never
   mutates `entity_id`.
3. `candidate_generation.py` takes the preprocessed Source 1 DataFrame and
   the preprocessed Source 2 + Source 3 DataFrame(s), and returns a
   candidate-pairs table: `source1_entity_id, candidate_entity_id,
   candidate_source, block_name`. A pair can appear from multiple blocks;
   duplicates are unioned and the contributing block names retained for
   analysis.
4. `evaluation.py`'s `candidate_recall()` function compares the candidate
   table against ground truth (train only) to report how many true pairs
   were captured.
5. `advanced_matching.py` computes Stage-1 features/score for every
   candidate pair, filters out clearly-implausible ones, then computes
   Stage-2 features/score for the survivors.
6. `ensemble.py` optionally merges in Person 1/2/3 scores for the same
   pairs (left join on `source1_entity_id, candidate_entity_id`), missing
   values kept as NaN rather than imputed silently.
7. `prediction.py` applies the chosen decision rule to Stage-2 (or ensemble)
   scores, then groups by `source1_entity_id` to build the final match
   lists, explicitly handling entities with zero surviving candidates as
   singletons.
8. `output.py` writes the two TSVs and runs `utils/validate_submission.py`.
9. `evaluation.py` and `error_analysis.py` are used on the validation split
   (never on the unlabeled test set) to produce the metrics and error
   report that feed `reports/final_report.md`.

## 4. Preprocessing

Conservative, reversible transformations only (see PRD constraint on not
over-merging unrelated businesses):

- **Name**: lowercase, Unicode NFKD normalization, whitespace collapse,
  punctuation stripped to a small safe set, legal-suffix detection (e.g.
  "ltd", "inc", "pvt", "gmbh" — detected from the data's own vocabulary
  frequency, not a hard-coded universal list, since the note says never
  hard-code country-like open sets; we do keep a *small documented* list of
  extremely common legal suffixes as a heuristic aid, not as a filter that
  discards information), tokenization, character 3-grams, numeric-token
  extraction.
- **Address**: lowercase normalized string, tokenization, character
  n-grams, numeric tokens (candidate house numbers / PIN-like codes,
  detected by regex pattern, not assumed to always be meaningful), street
  component tokens (first/last token heuristics).
- **Country**: raw value preserved as-is; a normalized (lowercased,
  stripped) string version added alongside it. No enumeration of allowed
  countries anywhere in code.

## 5. Candidate Generation (Blocks A–F)

- **Block A — Country + name token**: group by `(country_norm, most
  informative name token)`.
- **Block B — Character n-gram signature**: group by shared rare character
  n-grams to tolerate typos/abbreviations.
- **Block C — Address tokens**: group by informative address tokens.
- **Block D — Numeric evidence**: group by shared numeric tokens (house
  numbers, PIN-like codes) when present.
- **Block E — Rare tokens**: tokens that are rare corpus-wide generate
  stronger (higher-precision) candidate groups.
- **Block F — Multi-pass union**: union of A–E, deduplicated, with block
  provenance retained per pair for analysis.
- **Adaptive fallback**: if a record's name token set is empty/weak, rely
  more on address-based blocks; if address is weak, rely more on
  name-based blocks; if both are weak, fall back to the broadest
  character-n-gram retrieval. These rules are implemented as explicit,
  documented conditionals — not hidden inside a single opaque score.

## 6. Stage-1 Matching (Broad Retrieval Filter)

Cheap, interpretable features computed per candidate pair:
normalized-name token overlap (Jaccard), character n-gram overlap,
address token overlap, numeric-token agreement, country agreement, and
exact informative-token match. Combined into a simple broad score (e.g. a
maximum or a simple weighted sum with weights fixed by experiment, not
trained), used only to drop pairs that are very unlikely to match, while
keeping the surviving set generous. This stage is deliberately *not* a
close reproduction of Person 1's fuzzy-rule system — it exists purely to
cut the candidate set down cheaply before Stage 2 does more expensive
comparisons, and its recall is always measured before its impact on
precision is even considered.

## 7. Stage-2 Reranking

More detailed, still-interpretable features per surviving pair: rare-token
overlap, character similarity (e.g. `difflib.SequenceMatcher` ratio),
length compatibility, abbreviation compatibility heuristics, address
numeric/component agreement, cross-field consistency (name/address
consistency, country agreement), plus optional external columns
(`person1_score`, `person2_probability`, `person3_similarity`) when
available. These features feed a **scikit-learn classifier** trained on the
training split's ground-truth labels (e.g. logistic regression or gradient
boosting via `sklearn.ensemble`), so the reranker itself is learned rather
than hand-weighted — this is what makes Stage 2 meaningfully different
from Person 1's fixed weighted rules, while still being architecturally
distinct from Person 2 (Person 2's classifier operates directly on all
candidate pairs; Person 4's classifier only reranks pairs that already
survived independent multi-pass blocking + Stage-1 filtering).

## 8. Optional Ensemble Interfaces

`src/ensemble.py` defines:

```python
load_person1_scores(path) -> DataFrame[source1_entity_id, candidate_entity_id, score]
load_person2_scores(path) -> DataFrame[source1_entity_id, candidate_entity_id, score]
load_person3_scores(path) -> DataFrame[source1_entity_id, candidate_entity_id, score]
```

Each loader validates the schema, keeps the raw score column untouched, and
adds a separate normalized column (e.g. min-max scaled to [0,1] within that
teammate's score distribution) so ensembling never silently changes the
teammate's raw numbers. All three loaders default to returning `None` if the
file does not exist yet, so the rest of the pipeline runs unaffected.

## 9. Evaluation Architecture

`src/evaluation.py` implements:

- `candidate_recall(candidates_df, ground_truth_df)`
- `entity_level_metrics(predictions_df, ground_truth_df)` returning per-entity
  precision/recall/F0.5, then macro-averaged
- explicit singleton handling (true=[] & pred=[] => full credit; true=[] &
  pred=[x] => false positive; true=[x] & pred=[] => false negative)

## 10. Output Generation

`src/output.py` writes both required TSVs with real tab characters (using
`DataFrame.to_csv(path, sep="\t", index=False)`), joins ID lists with commas,
and represents singleton match lists as an empty string, then calls
`utils/validate_submission.py` programmatically before declaring success.

## 11. Configuration

A single `src/config.py` dataclass holds: file paths, random seed, blocking
parameters (e.g. n-gram size, rare-token frequency cutoff), Stage-1 filter
parameters, Stage-2 model choice/hyperparameters, decision threshold, and
ensemble weights. Experiments override fields of this config rather than
editing constants scattered through the code.

## 12. Logging

Standard library `logging` module, one logger per module
(`logging.getLogger(__name__)`), INFO level by default, with row counts,
timings, and key metrics logged at each pipeline stage so a run can be
audited from the console output alone.

## 13. Reproducibility

- A single seed value drives: the Source-1-level train/validation split,
  any randomized model training (e.g. `sklearn` `random_state`), and any
  sampling used in error analysis.
- Every experiment run is written as one row to `reports/experiments.csv`
  with the seed and full configuration recorded (see
  `docs/EXPERIMENT_PLAN.md`).

## 14. Dependencies

Given the no-network constraint discovered in Phase 0, dependencies are
kept to what's already installed:

```
pandas
numpy
scikit-learn
```

Only the Python standard library (`re`, `unicodedata`, `difflib`,
`dataclasses`, `logging`, `json`, `csv`, `collections`) is used for text
normalization and similarity — no `rapidfuzz`/`jellyfish`/`unidecode`,
since they cannot be installed in this environment. This is recorded in
`requirements.txt` with a comment explaining the constraint, so a teammate
running this on a machine *with* internet access understands why those
"obvious" libraries were deliberately avoided rather than forgotten.
