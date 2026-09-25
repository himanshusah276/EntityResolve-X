"""
Unit tests for src/system_comparison.py
"""
import pytest
import pandas as pd
from pathlib import Path
from src.system_comparison import generate_system_comparison_table


def test_generate_system_comparison_table(tmp_path):
    p4_metrics = {
        "candidate_recall": 1.0,
        "macro_precision": 0.99,
        "macro_recall": 1.0,
        "macro_f0_5": 0.9911,
        "total_fp": 1,
        "total_fn": 0,
        "runtime_seconds": 0.02,
        "memory_mb": "45MB",
        "total_candidates": 1428,
        "total_predictions": 55,
    }

    out_file = tmp_path / "system_comparison.csv"
    generate_system_comparison_table(p4_metrics, output_path=out_file)

    assert out_file.is_file()
    df = pd.read_csv(out_file, keep_default_na=False, dtype=str)
    assert len(df) == 4
    # Person 1, 2, 3 should have N/A (unfabricated)
    p1_row = df[df["system"].str.contains("Person 1")].iloc[0]
    assert p1_row["f0.5"] == "N/A"
    assert p1_row["candidate_recall"] == "N/A"

    # Person 4 has measured metrics
    p4_row = df[df["system"].str.contains("Person 4")].iloc[0]
    assert p4_row["f0.5"] == "0.9911"
    assert p4_row["false_positives"] == "1"
