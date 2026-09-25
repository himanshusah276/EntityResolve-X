"""
Unit tests for src/output.py and submission validation
"""
import pytest
import pandas as pd
from pathlib import Path
from src.output import (
    write_matching_results_tsv,
    write_candidate_pairs_tsv,
    validate_submission_files,
)


@pytest.fixture
def mock_test_env(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create test source files
    s1 = pd.DataFrame([
        {"entity_id": "S1-01", "business_name": "Acme Inc", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-02", "business_name": "Solo Corp", "business_address": "456 Oak Rd", "country": "US"},
    ])
    s2 = pd.DataFrame([
        {"entity_id": "S2-01", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"},
    ])
    s3 = pd.DataFrame([
        {"entity_id": "S3-01", "business_name": "Acme Tech", "business_address": "123 Main St", "country": "US"},
    ])

    s1_path = data_dir / "test_source1.tsv"
    s2_path = data_dir / "test_source2.tsv"
    s3_path = data_dir / "test_source3.tsv"

    s1.to_csv(s1_path, sep="\t", index=False)
    s2.to_csv(s2_path, sep="\t", index=False)
    s3.to_csv(s3_path, sep="\t", index=False)

    return s1_path, s2_path, s3_path, tmp_path


def test_write_and_validate_clean_submission(mock_test_env):
    s1_path, s2_path, s3_path, tmp_path = mock_test_env

    candidates = {
        "S1-01": ["S2-01", "S3-01"],
        "S1-02": ["S2-01"],  # distractor
    }
    predictions = {
        "S1-01": ["S2-01"],  # Valid match
        "S1-02": [],         # Valid singleton
    }
    all_s1 = ["S1-01", "S1-02"]

    match_path = tmp_path / "matching_results.tsv"
    cand_path = tmp_path / "candidate_pairs.tsv"

    write_matching_results_tsv(predictions, all_s1, output_path=match_path)
    write_candidate_pairs_tsv(candidates, all_s1, output_path=cand_path)

    # Validate
    summary = validate_submission_files(
        matching_results_path=match_path,
        candidate_pairs_path=cand_path,
        test_source1_path=s1_path,
        test_source2_path=s2_path,
        test_source3_path=s3_path,
    )
    assert summary["status"] == "PASSED"
    assert summary["singleton_entities"] == 1
    assert summary["total_predictions"] == 1


def test_validation_fails_on_candidate_not_in_pool(mock_test_env):
    s1_path, s2_path, s3_path, tmp_path = mock_test_env

    candidates = {"S1-01": ["S2-01"], "S1-02": []}
    # Prediction has S3-01 which is NOT in candidates for S1-01
    predictions = {"S1-01": ["S3-01"], "S1-02": []}
    all_s1 = ["S1-01", "S1-02"]

    match_path = tmp_path / "matching_results.tsv"
    cand_path = tmp_path / "candidate_pairs.tsv"

    write_matching_results_tsv(predictions, all_s1, output_path=match_path)
    write_candidate_pairs_tsv(candidates, all_s1, output_path=cand_path)

    with pytest.raises(ValueError, match="NOT in its candidate pair set"):
        validate_submission_files(
            matching_results_path=match_path,
            candidate_pairs_path=cand_path,
            test_source1_path=s1_path,
            test_source2_path=s2_path,
            test_source3_path=s3_path,
        )


def test_validation_fails_on_source1_id_in_prediction(mock_test_env):
    s1_path, s2_path, s3_path, tmp_path = mock_test_env

    candidates = {"S1-01": ["S1-01"], "S1-02": []}
    predictions = {"S1-01": ["S1-01"], "S1-02": []}
    all_s1 = ["S1-01", "S1-02"]

    match_path = tmp_path / "matching_results.tsv"
    cand_path = tmp_path / "candidate_pairs.tsv"

    write_matching_results_tsv(predictions, all_s1, output_path=match_path)
    write_candidate_pairs_tsv(candidates, all_s1, output_path=cand_path)

    with pytest.raises(ValueError, match="Source 1 ID .* is illegally included"):
        validate_submission_files(
            matching_results_path=match_path,
            candidate_pairs_path=cand_path,
            test_source1_path=s1_path,
            test_source2_path=s2_path,
            test_source3_path=s3_path,
        )
