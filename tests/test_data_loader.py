"""
Unit tests for src/data_loader.py
"""
import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import (
    load_tsv,
    inspect_data_health,
    load_ground_truth_map,
    generate_synthetic_benchmark_dataset,
)


def test_load_tsv_and_schema_validation(tmp_path):
    tsv_file = tmp_path / "sample.tsv"
    tsv_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n001\tTest Corp\t123 Main St\tUS\n", encoding="utf-8")

    df = load_tsv(tsv_file, expected_columns=["entity_id", "business_name", "business_address", "country"])
    assert len(df) == 1
    # Check that ID string format is preserved (not converted to int 1)
    assert df.iloc[0]["entity_id"] == "001"

    # Schema failure test
    with pytest.raises(ValueError, match="Missing required columns"):
        load_tsv(tsv_file, expected_columns=["entity_id", "phone_number"])


def test_inspect_data_health():
    df = pd.DataFrame({
        "entity_id": ["E1", "E2", "E2", "E3"],
        "business_name": ["Corp A", "Corp B", "  ", "Corp C"],
        "business_address": ["Addr 1", "Addr 2", "Addr 3", ""],
        "country": ["US", "India", "US", "France"],
    })
    health = inspect_data_health(df, "test_health")
    assert health["row_count"] == 4
    assert health["unique_entity_ids"] == 3
    assert health["duplicate_entity_ids"] == 1
    assert health["empty_cell_counts"]["business_name"] == 1
    assert health["empty_cell_counts"]["business_address"] == 1


def test_load_ground_truth_map(tmp_path):
    gt_file = tmp_path / "ground_truth.tsv"
    gt_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-01\tS2-01,S3-01\n"
        "S1-02\t\n"
        "S1-03\tS2-05\n",
        encoding="utf-8",
    )
    gt_map = load_ground_truth_map(gt_file)
    assert gt_map["S1-01"] == ["S2-01", "S3-01"]
    assert gt_map["S1-02"] == []  # Singleton
    assert gt_map["S1-03"] == ["S2-05"]


def test_generate_synthetic_benchmark_dataset(tmp_path):
    out_dir = tmp_path / "synthetic_data"
    generate_synthetic_benchmark_dataset(out_dir, n_s1=20, seed=42)

    assert (out_dir / "train" / "train_source1.tsv").is_file()
    assert (out_dir / "train" / "train_source2.tsv").is_file()
    assert (out_dir / "train" / "train_source3.tsv").is_file()
    assert (out_dir / "train" / "train_ground_truth.tsv").is_file()
    assert (out_dir / "test" / "test_source1.tsv").is_file()

    # Load and check separation
    s1 = load_tsv(out_dir / "train" / "train_source1.tsv")
    assert len(s1) == 20
    assert "entity_id" in s1.columns
