"""
Person 2 — Training Pair and Label Construction Module
Explodes Person 1 candidate pairs, assigns ground-truth supervision labels,
computes pairwise tabular features, and persists intermediate Parquet feature tables.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from person2.src.feature_engineering import PairFeatureExtractor, normalize_text

DATA_DIR = PROJECT_ROOT / "data" / "train"
OUTPUTS_DIR = PROJECT_ROOT / "person2" / "outputs"
FEATURES_DIR = PROJECT_ROOT / "person2" / "features"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def parse_ground_truth(gt_path: Path) -> Set[Tuple[str, str]]:
    """
    Parses train_ground_truth.tsv into a set of (source1_id, matched_id) pairs.
    Handles comma-separated lists and singletons.
    """
    print(f"Loading Ground Truth from {gt_path.resolve()}...")
    gt_df = pl.read_csv(gt_path, separator="\t")
    gt_pairs: Set[Tuple[str, str]] = set()

    for row in gt_df.iter_rows(named=True):
        s1 = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"] or "").strip()
        if matched_str:
            for m in matched_str.split(","):
                m_clean = m.strip()
                if m_clean:
                    gt_pairs.add((s1, m_clean))

    print(f"Extracted {len(gt_pairs):,} true match pairs from ground truth.")
    return gt_pairs


def load_source_lookups() -> Dict[str, Dict[str, str]]:
    """
    Loads Source 1, Source 2, and Source 3 into a fast in-memory entity lookup dictionary.
    Normalizes names, addresses, and countries on load.
    """
    print("Loading entity records from Source 1, Source 2, and Source 3...")
    s1_path = DATA_DIR / "train_source1.tsv"
    s2_path = DATA_DIR / "train_source2.tsv"
    s3_path = DATA_DIR / "train_source3.tsv"

    entity_lookup: Dict[str, Dict[str, str]] = {}

    for path, prefix in [(s1_path, "S1"), (s2_path, "S2"), (s3_path, "S3")]:
        df = pl.read_csv(path, separator="\t")
        for row in df.iter_rows(named=True):
            eid = str(row["entity_id"]).strip()
            entity_lookup[eid] = {
                "name": normalize_text(row.get("business_name", "")),
                "address": normalize_text(row.get("business_address", "")),
                "country": normalize_text(row.get("country", "")),
            }

    print(f"Loaded {len(entity_lookup):,} total unique entities into lookup dictionary.")
    return entity_lookup


def build_labeled_candidate_table(
    candidate_file: Path,
    gt_pairs: Set[Tuple[str, str]],
    batch_size: int = 50000,
) -> pl.DataFrame:
    """
    Explodes candidate_pairs.tsv into pairwise rows, joins with entity text,
    and applies ground-truth labeling.
    """
    print(f"\nProcessing Person 1 candidate pairs from {candidate_file.resolve()}...")
    cand_df = pl.read_csv(candidate_file, separator="\t")

    pair_rows = []
    total_s1 = 0
    total_pairs = 0

    for row in cand_df.iter_rows(named=True):
        s1 = str(row["source1_entity_id"]).strip()
        cand_str = str(row["candidate_entity_ids"] or "").strip()
        if not cand_str:
            continue

        cands = [c.strip() for c in cand_str.split(",") if c.strip()]
        total_s1 += 1
        total_pairs += len(cands)

        for c in cands:
            label = 1 if (s1, c) in gt_pairs else 0
            src = c.split("-")[0] if "-" in c else "UNK"
            pair_rows.append({
                "source1_entity_id": s1,
                "candidate_entity_id": c,
                "candidate_source": src,
                "label": label,
            })

    df_pairs = pl.DataFrame(pair_rows)
    pos_count = df_pairs["label"].sum()
    neg_count = len(df_pairs) - pos_count
    pos_ratio = (pos_count / len(df_pairs)) if len(df_pairs) > 0 else 0.0

    print(f"Candidate table generated:")
    print(f"  Total Source 1 entities: {total_s1:,}")
    print(f"  Total candidate pairs: {len(df_pairs):,}")
    print(f"  Positive pairs (label=1): {pos_count:,} ({pos_ratio:.2%})")
    print(f"  Negative pairs (label=0): {neg_count:,} ({1.0 - pos_ratio:.2%})")

    return df_pairs


def generate_training_dataset(
    candidate_file: Optional[Path] = None,
    output_parquet: Optional[Path] = None,
) -> pl.DataFrame:
    """
    End-to-end training dataset generation:
    1. Loads ground truth & parses match pairs
    2. Explodes candidates and assigns labels
    3. Fits and saves TF-IDF vectorizers
    4. Computes 24 pairwise features
    5. Saves feature matrix to Parquet
    """
    start_time = time.time()
    c_path = candidate_file or (PROJECT_ROOT / "outputs" / "candidate_pairs.tsv")
    gt_path = DATA_DIR / "train_ground_truth.tsv"
    out_parquet = output_parquet or (FEATURES_DIR / "train_features.parquet")

    # 1. Ground truth
    gt_pairs = parse_ground_truth(gt_path)

    # 2. Entity lookups
    entity_lookup = load_source_lookups()

    # 3. Labeled candidate pairs
    df_pairs = build_labeled_candidate_table(c_path, gt_pairs)

    # 4. Aligned lists for feature extraction
    print("\nAligning entity text fields for pairwise feature extraction...")
    s1_names, s1_addrs, s1_countries = [], [], []
    c_names, c_addrs, c_countries = [], [], []

    for row in df_pairs.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        cand = row["candidate_entity_id"]

        e1 = entity_lookup.get(s1, {"name": "", "address": "", "country": ""})
        e2 = entity_lookup.get(cand, {"name": "", "address": "", "country": ""})

        s1_names.append(e1["name"])
        s1_addrs.append(e1["address"])
        s1_countries.append(e1["country"])

        c_names.append(e2["name"])
        c_addrs.append(e2["address"])
        c_countries.append(e2["country"])

    # 5. Fit TF-IDF on corpus
    extractor = PairFeatureExtractor()
    unique_names = list(set(s1_names + c_names))
    unique_addrs = list(set(s1_addrs + c_addrs))
    extractor.fit_vectorizers(unique_names, unique_addrs)

    # 6. Extract batch features
    print(f"Extracting {len(extractor.FEATURE_COLUMNS)} pair features for {len(df_pairs):,} pairs...")
    X = extractor.extract_features_batch(
        s1_names, s1_addrs, s1_countries,
        c_names, c_addrs, c_countries,
    )

    # Convert to Polars DataFrame with metadata and features
    feature_dict = {col: X[:, idx] for idx, col in enumerate(extractor.FEATURE_COLUMNS)}
    df_features = pl.DataFrame(feature_dict)

    df_complete = df_pairs.select([
        "source1_entity_id", "candidate_entity_id", "candidate_source", "label"
    ]).with_columns(df_features)

    # Save to Parquet
    print(f"Writing complete training feature dataset to {out_parquet.resolve()}...")
    df_complete.write_parquet(out_parquet, compression="zstd")

    # Save feature schema
    schema_rows = []
    for col in extractor.FEATURE_COLUMNS:
        schema_rows.append({
            "feature_name": col,
            "feature_type": "float32",
            "description": f"Engineered pairwise metric: {col}",
        })
    pl.DataFrame(schema_rows).write_csv(OUTPUTS_DIR / "person2_feature_schema.csv")

    elapsed = time.time() - start_time
    print(f"\nFeature generation completed in {elapsed:.2f} seconds!")
    print(f"Saved: {out_parquet.resolve()} ({out_parquet.stat().st_size / 1024:.1f} KB)")
    print(f"Saved: {OUTPUTS_DIR / 'person2_feature_schema.csv'}")

    return df_complete


if __name__ == "__main__":
    generate_training_dataset()
