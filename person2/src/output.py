"""
Person 2 — Output Exporter & Hand-off Manager
Prepares and validates all Person 2 deliverables for Person 4:
  - person2/outputs/person2_scores.parquet
  - person2/outputs/person2_scores.tsv (optional TSV mirror for flexible ingestion)
  - person2/outputs/person2_model_info.csv
  - person2/outputs/person2_feature_schema.csv
"""

import sys
from pathlib import Path
from typing import Optional

import polars as pl

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "person2" / "outputs"


def export_tsv_mirror(parquet_path: Optional[Path] = None, tsv_path: Optional[Path] = None):
    """Exports a TSV mirror of person2_scores for compatibility with teammates expecting TSV."""
    p_path = parquet_path or (OUTPUTS_DIR / "person2_scores.parquet")
    t_path = tsv_path or (OUTPUTS_DIR / "person2_scores.tsv")

    if not p_path.exists():
        print(f"Parquet score file not found at {p_path.resolve()}")
        return

    df = pl.read_parquet(p_path)
    df.write_csv(t_path, separator="\t")
    print(f"Exported TSV mirror to {t_path.resolve()} ({len(df):,} pairs).")


def print_handoff_summary():
    """Prints a structured summary of Person 2 deliverables for Person 4."""
    print("=" * 70)
    print("PERSON 2 DELIVERABLES & HANDOFF SUMMARY FOR PERSON 4")
    print("=" * 70)

    # 1. Main Scores
    scores_path = OUTPUTS_DIR / "person2_scores.parquet"
    if scores_path.exists():
        df = pl.read_parquet(scores_path)
        probs = df["match_probability"]
        print(f"\n1. Main Output (person2_scores.parquet):")
        print(f"   Path: {scores_path.resolve()}")
        print(f"   Total Scored Pairs: {len(df):,}")
        print(f"   Unique Source 1 Entities: {df['source1_entity_id'].n_unique():,}")
        print(f"   Unique Candidate Targets: {df['candidate_entity_id'].n_unique():,}")
        print(f"   Probability Range: [{probs.min():.4f}, {probs.max():.4f}]")
        print(f"   Sample Top Matches:")
        top_matches = df.sort("match_probability", descending=True).head(3)
        for r in top_matches.iter_rows(named=True):
            print(f"     {r['source1_entity_id']} <-> {r['candidate_entity_id']} : Prob = {r['match_probability']:.4f}")

    # 2. Model Info
    info_path = OUTPUTS_DIR / "person2_model_info.csv"
    if info_path.exists():
        print(f"\n2. Model Performance & Metadata (person2_model_info.csv):")
        df_info = pl.read_csv(info_path)
        for r in df_info.iter_rows(named=True):
            print(f"   * {r['model_name']:<20}: Train Time={r['training_runtime_seconds']:.3f}s, Score Time={r['scoring_runtime_seconds']:.3f}s")

    # 3. Feature Schema
    schema_path = OUTPUTS_DIR / "person2_feature_schema.csv"
    if schema_path.exists():
        df_schema = pl.read_csv(schema_path)
        print(f"\n3. Feature Schema (person2_feature_schema.csv):")
        print(f"   Total Features: {len(df_schema)}")
        print(f"   Features List: {', '.join(df_schema['feature_name'].to_list()[:6])} ... ({len(df_schema)-6} more)")

    print("\n" + "=" * 70)
    print("HANDOFF ARTIFACTS VERIFIED AND READY FOR PERSON 4")
    print("=" * 70)


if __name__ == "__main__":
    export_tsv_mirror()
    print_handoff_summary()
