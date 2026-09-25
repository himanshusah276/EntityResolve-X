# Integration Plan — Person 4 Entity Resolution System

This document explains how Person 1/2/3's outputs get combined with Person
4's system, and only that. It assumes `docs/EVALUATION_PROTOCOL.md` is used
to judge whether any combination actually helps.

## 1. Current Status

As of this document's creation (Phase 1, repository inspection), **no
Person 1, Person 2, or Person 3 output files exist in this repository**.
Every interface described below is designed to work when
`person1_scores = person2_scores = person3_scores = None`, and the full
Person 4 pipeline (candidate generation through final prediction) does not
require any of them to run.

## 2. Common Candidate-Pair Schema

All teammate outputs are expected in this shape (TSV or CSV, loader handles
both):

```
source1_entity_id    candidate_entity_id    score
```

- `source1_entity_id`: a Source 1 `entity_id`.
- `candidate_entity_id`: a Source 2 or Source 3 `entity_id`.
- `score`: a single float. Meaning differs by teammate (documented below)
  but the column name is always `score` once loaded internally.

## 3. How Each Teammate's Output Is Loaded

### Person 1 (classical fuzzy/weighted similarity)

```python
load_person1_scores(path: str) -> Optional[pd.DataFrame]
```

- Expects `score` to be their classical weighted-similarity score (typically
  already roughly in [0, 1], but not guaranteed).
- Returns `None` if `path` does not exist, with a logged warning — the
  pipeline continues without Person 1.

### Person 2 (supervised pair classifier)

```python
load_person2_scores(path: str) -> Optional[pd.DataFrame]
```

- Expects `score` to be a predicted probability in [0, 1].
- Returns `None` if `path` does not exist.

### Person 3 (embedding-based semantic matching)

```python
load_person3_scores(path: str) -> Optional[pd.DataFrame]
```

- Expects `score` to be an embedding similarity (e.g. cosine similarity),
  which may range differently depending on the embedding model (e.g.
  [-1, 1] or [0, 1]).
- Returns `None` if `path` does not exist.

## 4. Score Normalization

Because the three teammates' raw scores are not guaranteed to be on the
same scale, `src/ensemble.py` keeps two versions of every loaded score:

- `<name>_score_raw` — untouched, exactly as loaded.
- `<name>_score_norm` — min-max normalized to [0, 1] **using statistics
  computed only on the training split** (to avoid leakage from validation
  into the normalization statistics), then applied to both splits.

Person 4's own Stage-2 score is normalized the same way for a fair
comparison inside the ensemble.

## 5. Ensemble Experiments

See `docs/EXPERIMENT_PLAN.md` §5 for the exact list of combinations tested.
Two ensembling strategies are planned, tested separately:

1. **Weighted linear combination** with weights fit by logistic regression
   over the normalized scores as features (predicting the true label on the
   training split), evaluated on the validation split.
2. **Simple rule-based fallback** (e.g. "use teammate score only when
   Person 4's Stage-2 score is near the decision boundary") — tested only if
   the linear combination does not clearly help, to see if a narrower,
   more targeted use of teammate signal does better.

Weights are never hand-picked; they come from a reproducible fitting
procedure recorded with the seed used.

## 6. Final Integration Process

1. Load whichever of Person 1/2/3's outputs exist (each independently
   optional).
2. Join onto Person 4's Stage-2 candidate-pair table by
   `(source1_entity_id, candidate_entity_id)`; unmatched teammate rows are
   left as `NaN`, not imputed to 0 or any other value.
3. Run the ensemble experiments in `docs/EXPERIMENT_PLAN.md` §5 that are
   possible given which teammate outputs are actually present.
4. Compare every combination against the Person-4-alone baseline using the
   protocol in `docs/EVALUATION_PROTOCOL.md`.
5. Only adopt an ensemble configuration for the final frozen pipeline if it
   measurably improves validation F0.5 over Person 4 alone; otherwise the
   final configuration remains Person 4 alone, and this is stated plainly in
   `reports/final_report.md` rather than forcing an ensemble that doesn't
   help.
