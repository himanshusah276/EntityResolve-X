# Evaluation Protocol — Person 4 Entity Resolution System

This document is the single source of truth for how Person 4 (and, for
comparison, Person 1/2/3) is scored. Any evaluation code must implement
exactly what is written here.

## 1. Validation Split

The split is performed at the **Source 1 entity level**, never at the
individual candidate-pair level. This prevents leakage where records from
the same Source 1 entity appear in both train and validation.

```python
from sklearn.model_selection import train_test_split

source1_ids = train_source1_df["entity_id"].unique()
train_ids, val_ids = train_test_split(
    source1_ids, test_size=0.2, random_state=SEED
)
```

- `SEED` is a single integer defined once in `src/config.py` and recorded
  with every experiment.
- All Source 2/Source 3 candidate generation, scoring, and prediction for a
  given split only ever "sees" the Source 1 IDs assigned to that split.
- The split is deterministic: the same seed always reproduces the same
  `train_ids` / `val_ids`.
- Ground truth rows are filtered by `source1_entity_id` membership in the
  relevant split — never resampled independently.

## 2. Candidate Recall

```
candidate_recall =
    (number of true (source1_id, matched_id) pairs present in the
     generated candidate set)
    / (total number of true (source1_id, matched_id) pairs in ground truth)
```

Computed over the validation split's Source 1 entities only, using their
ground-truth matches. This measures whether candidate generation is even
capable of finding the right answer — it is a ceiling on downstream
performance, and is reported separately from final precision/recall/F0.5,
never confused with them.

## 3. Entity-Level Precision / Recall / F0.5

For each Source 1 entity `e` in the validation split:

```
true_matches(e)      = set of ground-truth matched IDs for e (may be empty)
predicted_matches(e) = set of predicted matched IDs for e (may be empty)

TP(e) = |true_matches(e) ∩ predicted_matches(e)|
FP(e) = |predicted_matches(e) - true_matches(e)|
FN(e) = |true_matches(e) - predicted_matches(e)|
```

**Singleton handling (explicit rule, no ambiguity):**

- If `true_matches(e)` is empty **and** `predicted_matches(e)` is empty ->
  entity gets full credit: `precision(e) = 1.0`, `recall(e) = 1.0`,
  `f0.5(e) = 1.0`.
- If `true_matches(e)` is empty **and** `predicted_matches(e)` is non-empty
  -> every predicted ID is a false positive: `precision(e) = 0.0`. Recall is
  undefined by the standard formula (0/0); by convention we set
  `recall(e) = 1.0` is **not** used here — instead we define
  `recall(e) = 0.0` is also incorrect since there was nothing to recall. The
  rule adopted: when `true_matches(e)` is empty, `recall(e) = 1.0` only if
  `predicted_matches(e)` is also empty (the case above); otherwise
  `recall(e)` is treated as **not applicable and excluded from the recall
  average for that entity**, while `precision(e) = 0.0` and `f0.5(e) = 0.0`
  still count (since precision is 0, F0.5 is 0 regardless of recall).
- If `true_matches(e)` is non-empty **and** `predicted_matches(e)` is empty
  -> `precision(e)` is **not applicable and excluded from the precision
  average for that entity** (there is no prediction to be precise about),
  `recall(e) = 0.0`, and `f0.5(e) = 0.0`.
- Otherwise (both non-empty): standard formulas.

```
precision(e) = TP(e) / (TP(e) + FP(e))   [when denominator > 0]
recall(e)    = TP(e) / (TP(e) + FN(e))   [when denominator > 0]
```

## 4. F0.5

```
F0.5(e) = 1.25 * precision(e) * recall(e)
          -----------------------------------
          0.25 * precision(e) + recall(e)
```

with `F0.5(e) = 0.0` whenever the denominator is 0 and it is not the
full-credit singleton case above.

## 5. Macro-Averaging

```
precision_macro = mean(precision(e) for e in validation entities where precision(e) is defined)
recall_macro    = mean(recall(e) for e in validation entities where recall(e) is defined)
f0.5_macro      = mean(f0.5(e) for e in validation entities)
```

`f0.5_macro` is always computed over **all** validation entities (singletons
included, per §17/§18 of the master brief), since F0.5(e) is always defined
(0.0 or the standard formula or 1.0 for the true-singleton case).

## 6. False Positives / False Negatives

Reported both per-entity (in `reports/error_analysis.csv`) and aggregated
(`sum(FP(e))`, `sum(FN(e))` across the validation split) alongside
precision/recall/F0.5 in `reports/experiments.csv`.

## 7. Runtime

Wall-clock seconds for: candidate generation, Stage 1, Stage 2, and total
end-to-end, measured with `time.perf_counter()` and recorded per experiment.

## 8. Memory

Peak resident memory during the run, measured with `tracemalloc` (Python
standard library, no extra dependency) or `resource.getrusage` on Linux,
recorded in MB per experiment. If measurement is not implemented yet for a
given experiment, record `N/A` — never a guessed number.

## 9. Comparison Methodology

`reports/system_comparison.csv` compares Person 1/2/3/4 using this exact
protocol, on the exact same validation split (same seed, same Source 1
entity IDs in the validation set) so the comparison is apples-to-apples.
Any teammate metric that cannot be recomputed under this protocol (e.g. if
only their final predictions are available, not raw scores) is recomputed
from their submitted predictions directly using the entity-level formulas
above — not taken from their own self-reported numbers, so all four systems
are judged the same way.
