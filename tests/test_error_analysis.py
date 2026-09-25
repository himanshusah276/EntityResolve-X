"""
Unit tests for src/error_analysis.py
"""
import pytest
from src.preprocessing import preprocess_record
from src.error_analysis import (
    categorize_error,
    extract_error_records,
    generate_error_analysis_report,
)


def test_categorize_error():
    # 1. Country mismatch
    s1 = preprocess_record({"entity_id": "S1", "business_name": "Acme", "business_address": "Main", "country": "US"})
    c1 = preprocess_record({"entity_id": "C1", "business_name": "Acme", "business_address": "Main", "country": "FRANCE"})
    assert categorize_error(s1, c1, is_false_positive=True) == "country_mismatch"

    # 2. Numeric conflict
    s2 = preprocess_record({"entity_id": "S1", "business_name": "Acme Corp", "business_address": "100 Market St", "country": "US"})
    c2 = preprocess_record({"entity_id": "C1", "business_name": "Acme Corp", "business_address": "900 Market St", "country": "US"})
    assert categorize_error(s2, c2, is_false_positive=True) == "numeric_conflict"

    # 3. Shared address, different business
    s3 = preprocess_record({"entity_id": "S1", "business_name": "Sunrise Bakery", "business_address": "100 Market St", "country": "US"})
    c3 = preprocess_record({"entity_id": "C1", "business_name": "Midnight Law Office", "business_address": "100 Market St", "country": "US"})
    assert categorize_error(s3, c3, is_false_positive=True) == "shared_address"


def test_extract_and_generate_report(tmp_path):
    s1 = preprocess_record({"entity_id": "S1-1", "business_name": "Alpha Corp", "business_address": "123 Way", "country": "US"})
    cand_fp = preprocess_record({"entity_id": "S2-99", "business_name": "Beta Corp", "business_address": "123 Way", "country": "US"})
    cand_fn = preprocess_record({"entity_id": "S3-10", "business_name": "Alpha Corporation", "business_address": "123 Way", "country": "US"})

    targets = {cand_fp.entity_id: cand_fp, cand_fn.entity_id: cand_fn}
    gt = {"S1-1": ["S3-10"]}
    preds = {"S1-1": ["S2-99"]}  # S2-99 is FP, S3-10 is FN
    pair_scores = {("S1-1", "S2-99"): 0.72, ("S1-1", "S3-10"): 0.61}

    csv_path = tmp_path / "error_analysis.csv"
    df_err = generate_error_analysis_report(
        queries=[s1],
        targets_dict=targets,
        ground_truth=gt,
        predictions=preds,
        pair_scores_map=pair_scores,
        output_path=csv_path,
    )

    assert len(df_err) == 2
    assert set(df_err["true_label"]) == {0, 1}
    assert csv_path.is_file()
