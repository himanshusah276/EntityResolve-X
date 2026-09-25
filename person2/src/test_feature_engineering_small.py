"""
Small-sample verification test for Person 2 Feature Engineering.
Validates candidate-pair construction, label assignment, TF-IDF fitting,
and numerical feature integrity on a sample subset.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from person2.src.feature_engineering import PairFeatureExtractor, normalize_text


def run_small_sample_test(max_s1: int = 40):
    print("=" * 65)
    print("RUNNING SMALL-SAMPLE FEATURE ENGINEERING VERIFICATION")
    print("=" * 65)

    # 1. Load Ground Truth
    gt_path = PROJECT_ROOT / "data" / "train" / "train_ground_truth.tsv"
    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_pairs = set()
    for row in gt_df.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"] or ""
        for m in matched_str.split(","):
            if m.strip():
                gt_pairs.add((s1, m.strip()))

    print(f"Loaded {len(gt_pairs)} total ground-truth true match pairs.")

    # 2. Load candidate pairs (sample)
    cand_path = PROJECT_ROOT / "outputs" / "candidate_pairs.tsv"
    cand_df = pl.read_csv(cand_path, separator="\t")

    pair_records = []
    count_s1 = 0
    for row in cand_df.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        cand_str = row["candidate_entity_ids"] or ""
        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        for c in cands:
            is_match = 1 if (s1, c) in gt_pairs else 0
            pair_records.append((s1, c, c.split("-")[0], is_match))
        count_s1 += 1
        if count_s1 >= max_s1:
            break

    print(f"Sampled {count_s1} Source 1 entities -> {len(pair_records)} candidate pairs.")
    pos_count = sum(r[3] for r in pair_records)
    neg_count = len(pair_records) - pos_count
    print(f"Positives: {pos_count} ({pos_count / len(pair_records):.2%}) | Negatives: {neg_count} ({neg_count / len(pair_records):.2%})")

    # 3. Load Source records for lookup
    s1_df = pl.read_csv(PROJECT_ROOT / "data" / "train" / "train_source1.tsv", separator="\t")
    s2_df = pl.read_csv(PROJECT_ROOT / "data" / "train" / "train_source2.tsv", separator="\t")
    s3_df = pl.read_csv(PROJECT_ROOT / "data" / "train" / "train_source3.tsv", separator="\t")

    s1_lookup = {r["entity_id"]: r for r in s1_df.iter_rows(named=True)}
    target_lookup = {r["entity_id"]: r for r in s2_df.iter_rows(named=True)}
    target_lookup.update({r["entity_id"]: r for r in s3_df.iter_rows(named=True)})

    # Build aligned lists
    s1_names, s1_addrs, s1_countries = [], [], []
    cand_names, cand_addrs, cand_countries = [], [], []
    labels = []

    for s1, cand, source, label in pair_records:
        rec1 = s1_lookup.get(s1, {})
        rec2 = target_lookup.get(cand, {})

        s1_names.append(normalize_text(rec1.get("business_name", "")))
        s1_addrs.append(normalize_text(rec1.get("business_address", "")))
        s1_countries.append(normalize_text(rec1.get("country", "")))

        cand_names.append(normalize_text(rec2.get("business_name", "")))
        cand_addrs.append(normalize_text(rec2.get("business_address", "")))
        cand_countries.append(normalize_text(rec2.get("country", "")))

        labels.append(label)

    # 4. Fit TF-IDF on corpus
    extractor = PairFeatureExtractor()
    all_names = list(set(s1_names + cand_names))
    all_addrs = list(set(s1_addrs + cand_addrs))
    extractor.fit_vectorizers(all_names, all_addrs)

    # 5. Extract batch features
    print("Extracting features for candidate pairs...")
    X = extractor.extract_features_batch(
        s1_names, s1_addrs, s1_countries,
        cand_names, cand_addrs, cand_countries
    )
    y = np.array(labels, dtype=np.int32)

    print(f"\nFeature matrix X shape: {X.shape}")
    print(f"Target labels y shape: {y.shape}")
    assert not np.isnan(X).any(), "NaN values found in feature matrix!"
    assert not np.isinf(X).any(), "Inf values found in feature matrix!"
    print("Verification PASSED: No NaN or Inf values detected.")

    # 6. Check feature correlations / contrasts between Positives and Negatives
    print("\nFeature Summary (Positives vs Negatives Mean Comparison):")
    col_names = extractor.FEATURE_COLUMNS
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    for idx, col in enumerate(col_names):
        pos_mean = X[pos_mask, idx].mean() if pos_mask.any() else 0.0
        neg_mean = X[neg_mask, idx].mean() if neg_mask.any() else 0.0
        diff = pos_mean - neg_mean
        print(f"  {col:<35} | Pos: {pos_mean:6.3f} | Neg: {neg_mean:6.3f} | Diff: {diff:+6.3f}")

    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING TEST COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_small_sample_test()
