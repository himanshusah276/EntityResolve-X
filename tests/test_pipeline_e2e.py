"""
End-to-end integration test verifying the complete Person 4 pipeline execution.
"""
import pytest
from pathlib import Path
from src.data_loader import generate_synthetic_benchmark_dataset
from src.prediction import run_test_prediction
from src.output import validate_submission_files
from utils.validate_submission import validate


def test_full_pipeline_e2e(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "outputs"

    # 1. Generate full dataset (train + test)
    generate_synthetic_benchmark_dataset(data_dir, n_s1=60, seed=99)

    # 2. Run prediction on test set
    pred_summary = run_test_prediction(
        test_dir=data_dir / "test",
        output_dir=out_dir,
    )

    assert pred_summary["total_source1_entities"] == 20
    assert Path(pred_summary["matching_tsv"]).is_file()
    assert Path(pred_summary["candidate_tsv"]).is_file()

    # 3. Validate outputs using internal validator
    report = validate_submission_files(
        matching_results_path=pred_summary["matching_tsv"],
        candidate_pairs_path=pred_summary["candidate_tsv"],
        test_source1_path=data_dir / "test" / "test_source1.tsv",
        test_source2_path=data_dir / "test" / "test_source2.tsv",
        test_source3_path=data_dir / "test" / "test_source3.tsv",
    )
    assert report["status"] == "PASSED"
    assert report["all_predictions_subset_of_candidates"] is True
    assert report["zero_source1_ids_in_predictions"] is True
    assert report["zero_duplicate_ids"] is True

    # 4. Validate outputs using standalone CLI validator function
    cli_passed = validate(
        matching_tsv=pred_summary["matching_tsv"],
        candidates_tsv=pred_summary["candidate_tsv"],
        test_s1_tsv=str(data_dir / "test" / "test_source1.tsv"),
        test_s2_tsv=str(data_dir / "test" / "test_source2.tsv"),
        test_s3_tsv=str(data_dir / "test" / "test_source3.tsv"),
    )
    assert cli_passed is True
