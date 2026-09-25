"""
Person 2 — Candidate-Pair Scoring & Probability Generation Module
Loads trained supervised model (primary: LightGBM) and generates continuous match probabilities
for all candidate pairs. Enforces strict output integrity verification.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from person2.src.feature_engineering import PairFeatureExtractor

MODELS_DIR = PROJECT_ROOT / "person2" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "person2" / "outputs"
FEATURES_DIR = PROJECT_ROOT / "person2" / "features"


def load_model(model_name: str = "lightgbm"):
    """Loads a trained model artifact from person2/models/."""
    # Normalize model name for filename
    clean_name = model_name.lower().replace("_", "")
    model_path = MODELS_DIR / f"person2_{clean_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path.resolve()}")
    print(f"Loading model artifact from {model_path.resolve()}...")
    return joblib.load(model_path)


def verify_output_integrity(
    scores_df: pl.DataFrame,
    original_candidate_file: Path,
) -> Dict[str, bool]:
    """
    Strict integrity verification:
    - Exactly same candidate pairs as Person 1's input
    - No external entity IDs
    - All probabilities in [0.0, 1.0]
    - Zero NaN or Inf values
    - No accidental duplicate pairs
    """
    print("\n" + "=" * 65)
    print("RUNNING OUTPUT INTEGRITY VERIFICATION")
    print("=" * 65)

    n_rows = len(scores_df)
    probs = scores_df["match_probability"].to_numpy()

    # 1. Finite check
    has_nan = bool(np.isnan(probs).any())
    has_inf = bool(np.isinf(probs).any())
    assert not has_nan, "FAIL: Output contains NaN probabilities!"
    assert not has_inf, "FAIL: Output contains Infinite probabilities!"
    print("  [PASSED] Probabilities are strictly finite.")

    # 2. Probability bounds check [0, 1]
    min_prob = float(np.min(probs))
    max_prob = float(np.max(probs))
    assert min_prob >= 0.0, f"FAIL: Probability below 0 ({min_prob})"
    assert max_prob <= 1.0, f"FAIL: Probability above 1 ({max_prob})"
    print(f"  [PASSED] Probabilities strictly in [0.0, 1.0] (Min: {min_prob:.4f}, Max: {max_prob:.4f}).")

    # 3. Duplicate check
    n_unique_pairs = scores_df.select(["source1_entity_id", "candidate_entity_id"]).n_unique()
    assert n_rows == n_unique_pairs, f"FAIL: Duplicate pairs found! ({n_rows} vs {n_unique_pairs} unique)"
    print(f"  [PASSED] Zero duplicate pairs ({n_unique_pairs:,} unique pairs).")

    # 4. Check against Person 1 original candidate file
    orig_cand_df = pl.read_csv(original_candidate_file, separator="\t")
    orig_pairs = set()
    for row in orig_cand_df.iter_rows(named=True):
        s1 = str(row["source1_entity_id"]).strip()
        cand_str = str(row["candidate_entity_ids"] or "").strip()
        for c in cand_str.split(","):
            if c.strip():
                orig_pairs.add((s1, c.strip()))

    scored_pairs = set(zip(
        scores_df["source1_entity_id"].to_list(),
        scores_df["candidate_entity_id"].to_list(),
    ))

    # All scored pairs must originate from Person 1
    diff_pairs = scored_pairs - orig_pairs
    assert len(diff_pairs) == 0, f"FAIL: {len(diff_pairs)} external candidate pairs detected!"
    print(f"  [PASSED] 100% of scored pairs originate strictly from Person 1 candidates.")

    # Count parity
    assert len(scored_pairs) == len(orig_pairs), f"Count mismatch: {len(scored_pairs)} scored vs {len(orig_pairs)} in input"
    print(f"  [PASSED] Exact 1-to-1 candidate count match ({len(scored_pairs):,} pairs).")

    print("=" * 65)
    print("ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)

    return {
        "finite": True,
        "bounded": True,
        "unique": True,
        "exact_parity": True,
    }


def score_candidate_pairs(
    feature_parquet: Optional[Path] = None,
    candidate_file: Optional[Path] = None,
    model_name: str = "lightgbm",
    output_parquet: Optional[Path] = None,
) -> pl.DataFrame:
    """
    Loads features, scores with primary classifier, performs integrity checks,
    and writes person2_scores.parquet and person2_train_scores.parquet.
    """
    start_time = time.time()
    feat_path = feature_parquet or (FEATURES_DIR / "train_features.parquet")
    cand_file = candidate_file or (PROJECT_ROOT / "outputs" / "candidate_pairs.tsv")
    out_scores = output_parquet or (OUTPUTS_DIR / "person2_scores.parquet")
    out_train_scores = OUTPUTS_DIR / "person2_train_scores.parquet"

    print(f"Loading features from {feat_path.resolve()}...")
    df = pl.read_parquet(feat_path)

    # 1. Load model
    model = load_model(model_name)

    # 2. Extract feature matrix
    feature_cols = PairFeatureExtractor.FEATURE_COLUMNS
    X = df.select(feature_cols).to_numpy().astype(np.float32)

    # 3. Predict continuous match probability
    print(f"Generating match probabilities for {X.shape[0]:,} candidate pairs using {model_name}...")
    t0 = time.time()
    probabilities = model.predict_proba(X)[:, 1].astype(np.float64)
    scoring_time = time.time() - t0
    print(f"Scoring completed in {scoring_time:.3f} seconds ({len(probabilities) / max(0.001, scoring_time):,.0f} pairs/sec)!")

    # 4. Construct Person 2 score output DataFrame
    df_scores = pl.DataFrame({
        "source1_entity_id": df["source1_entity_id"],
        "candidate_entity_id": df["candidate_entity_id"],
        "match_probability": probabilities,
        "candidate_source": df["candidate_source"],
        "model_name": [model_name] * len(df),
    })

    # 5. Output Integrity Verification
    verify_output_integrity(df_scores, cand_file)

    # 6. Save primary deliverable: outputs/person2_scores.parquet
    print(f"\nSaving Person 2 main score output to {out_scores.resolve()}...")
    df_scores.select([
        "source1_entity_id",
        "candidate_entity_id",
        "match_probability",
    ]).write_parquet(out_scores, compression="zstd")

    # 7. Save training score output with ground-truth label: outputs/person2_train_scores.parquet
    if "label" in df.columns:
        print(f"Saving training diagnostic scores to {out_train_scores.resolve()}...")
        df_train_scores = pl.DataFrame({
            "source1_entity_id": df["source1_entity_id"],
            "candidate_entity_id": df["candidate_entity_id"],
            "label": df["label"],
            "match_probability": probabilities,
        })
        df_train_scores.write_parquet(out_train_scores, compression="zstd")

    elapsed = time.time() - start_time
    print(f"\nAll scoring artifacts saved successfully in {elapsed:.2f} seconds!")
    print(f"  person2_scores.parquet: {out_scores.resolve()} ({out_scores.stat().st_size / 1024:.1f} KB)")
    if out_train_scores.exists():
        print(f"  person2_train_scores.parquet: {out_train_scores.resolve()} ({out_train_scores.stat().st_size / 1024:.1f} KB)")

    return df_scores


if __name__ == "__main__":
    score_candidate_pairs(model_name="lightgbm")
