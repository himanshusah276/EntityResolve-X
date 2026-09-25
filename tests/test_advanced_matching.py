"""
Unit tests for src/advanced_matching.py
"""
import pytest
from src.preprocessing import preprocess_record
from src.advanced_matching import (
    compute_stage1_broad_score,
    extract_stage2_features,
    compute_stage2_score,
    TwoStagePipeline,
)


@pytest.fixture
def query_record():
    return preprocess_record({
        "entity_id": "S1-001",
        "business_name": "Acme Technologies Inc",
        "business_address": "123 Main St, Austin 78701",
        "country": "US",
    })


@pytest.fixture
def matching_target():
    return preprocess_record({
        "entity_id": "S2-001",
        "business_name": "Acme Tech",
        "business_address": "123 Main Street, Austin 78701",
        "country": "US",
    })


@pytest.fixture
def distant_target():
    return preprocess_record({
        "entity_id": "S3-999",
        "business_name": "Totally Unrelated Bakery",
        "business_address": "999 Oak Way, Portland 97201",
        "country": "US",
    })


def test_stage1_broad_score(query_record, matching_target, distant_target):
    s1_match = compute_stage1_broad_score(query_record, matching_target)
    s1_dist = compute_stage1_broad_score(query_record, distant_target)
    assert s1_match > 0.35
    assert s1_dist < 0.10


def test_stage2_features_and_score(query_record, matching_target):
    feats = extract_stage2_features(query_record, matching_target)
    assert feats["same_country"] == 1.0
    assert feats["street_number_match"] == 1.0
    assert feats["postal_code_match"] == 1.0
    assert feats["name_partial_ratio"] == 1.0
    assert feats["name_token_sort_ratio"] > 0.65

    score = compute_stage2_score(feats)
    assert score > 0.65


def test_two_stage_pipeline(query_record, matching_target, distant_target):
    target_dict = {
        matching_target.entity_id: matching_target,
        distant_target.entity_id: distant_target,
    }
    pipeline = TwoStagePipeline(
        target_records=target_dict,
        stage1_threshold=0.15,
        stage2_threshold=0.60,
    )

    preds, scored = pipeline.predict_entity(
        query=query_record,
        initial_candidate_ids=[matching_target.entity_id, distant_target.entity_id],
    )

    # Distant target pruned at Stage 1, matching target predicted
    assert matching_target.entity_id in preds
    assert distant_target.entity_id not in preds
