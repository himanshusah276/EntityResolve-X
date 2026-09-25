# Experiment Plan — Person 4 Entity Resolution System

This document defines *what* experiments will be run and *how* they will be
recorded. It does not contain results — measured results live in
`reports/experiments.csv` and are only summarized in
`reports/final_report.md` once they exist. Nothing here should be read as a
prediction of which configuration will win.

## 1. Blocking Experiments

For each individual block (A–F, see `docs/TRD.md` §5) and for the union of
all blocks, measure on the training split:

- candidate count
- candidate recall
- reduction ratio (`1 - candidate_count / (n_source1 * n_candidates_pool)`)
- runtime

Naming convention: `block_<letter>` (e.g. `block_a`, `block_e`), and
`block_union_all` for the full union. Adaptive-blocking variants are named
`block_adaptive_v1`, `block_adaptive_v2`, etc.

## 2. Scoring Experiments

For Stage 1 (broad filter) and Stage 2 (reranker), measure separately:

- Stage-1-only: candidates -> Stage 1 -> final decision (skipping Stage 2),
  named `stage1_only_v<N>`.
- Stage-1 + Stage-2: candidates -> Stage 1 -> Stage 2 -> final decision,
  named `stage1_stage2_v<N>`.
- A "no staging" baseline: candidates -> single combined scorer -> decision,
  named `single_stage_baseline_v<N>`, used only as a reference point to
  justify (or reject) the two-stage design with actual numbers, per §15 of
  the master instructions.

## 3. Threshold Experiments

For the chosen scoring configuration, sweep decision thresholds/rules (e.g.
0.3, 0.4, 0.5, 0.6, 0.7, plus any non-threshold rule such as "top-k
candidates per entity" or "score margin above second-best"), recording for
each: precision, recall, F0.5, candidate_recall, false_positives,
false_negatives, prediction_count, zero_match_entities, runtime. Naming:
`threshold_<value>` or `rule_<description>`.

## 4. Reranking Experiments

Variants of the Stage-2 feature set / model choice (e.g. logistic
regression vs. gradient boosting, with vs. without rare-token overlap
feature, with vs. without cross-field consistency feature). Naming:
`stage2_<variant_description>_v<N>`.

## 5. Ensemble Experiments

Only run once at least one of Person 1/2/3's outputs exists. Planned
combinations (see `docs/INTEGRATION_PLAN.md` for schema details):

- `ensemble_p4_only` (baseline / control)
- `ensemble_p4_p1`
- `ensemble_p4_p2`
- `ensemble_p4_p3`
- `ensemble_p4_p1_p2`
- `ensemble_p4_p1_p3`
- `ensemble_p4_p2_p3`
- `ensemble_p4_p1_p2_p3`

Weights for each combination are fit on the training split (e.g. logistic
regression over the individual scores as features, or a small grid search
tuned by cross-validation), never hand-picked, and evaluated on the held-out
validation split.

## 6. Metrics Recorded for Every Experiment

```
experiment_id, date, git_commit_if_available, validation_seed,
blocking_configuration, stage1_configuration, stage2_configuration,
threshold, ensemble_configuration, precision, recall, f0.5,
candidate_recall, false_positives, false_negatives, runtime, memory, notes
```

## 7. Experiment Naming Convention

`<phase>_<short_description>_v<N>`, e.g. `blocking_block_e_v1`,
`stage2_gbdt_with_rare_tokens_v2`, `threshold_0.6_v1`,
`ensemble_p4_p2_v1`. Version numbers increment whenever a configuration is
re-run with a change.

## 8. Result Storage

- `reports/experiments.csv` — one row per run, append-only, human-readable.
- `reports/experiments.json` — same data in machine-readable form for
  programmatic comparison/plotting, generated alongside the CSV.

## 9. Reproducibility Requirements

- Every experiment run records the `validation_seed` used to create the
  Source-1-level train/validation split (see `docs/EVALUATION_PROTOCOL.md`).
- Every experiment run records the exact configuration used (ideally the
  full `src/config.py` dataclass serialized to JSON) so it can be replayed.
- Experiments are run through a single entry point (a `scripts/run_experiment.py`
  or equivalent, added when we reach Phase 10/13) that takes a configuration
  and appends a row to `reports/experiments.csv` — results are never
  hand-typed into the CSV.
- No experiment's numbers are written anywhere in documentation until the
  code has actually executed and produced them.
