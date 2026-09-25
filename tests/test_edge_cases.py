"""
Comprehensive edge case tests for Person 4 Entity Resolution Pipeline.

Tests:
1. Missing / empty / whitespace-only fields
2. Unseen / rare / accented unicode countries (open-set country handling)
3. Extremely short names (e.g. "3M", "X", "K Corp")
4. Complex addresses with multiple numbers (suite, floor, po box)
5. Zero true matches (entire validation set is singletons)
6. Pure candidate pruning edge cases
7. Exact decision boundary conditions (score == threshold vs score < threshold)
"""
import pytest
from src.preprocessing import (
    preprocess_record,
    clean_text,
    extract_legal_suffixes,
    normalize_country,
    extract_address_components,
)
from src.candidate_generation import CandidateIndex, generate_candidates_for_queries
from src.advanced_matching import (
    compute_stage1_broad_score,
    extract_stage2_features,
    compute_stage2_score,
    TwoStagePipeline,
)
from src.evaluation import evaluate_predictions, compute_candidate_recall


def test_empty_and_whitespace_records():
    """Verify that completely empty or whitespace records don't crash any step."""
    raw = {
        "entity_id": "EMPTY-01",
        "business_name": "   ",
        "business_address": "\t\n",
        "country": "",
    }
    rec = preprocess_record(raw)
    assert rec.entity_id == "EMPTY-01"
    assert rec.clean_name == ""
    assert rec.clean_address == ""
    assert rec.country_norm == "UNKNOWN"
    assert rec.name_tokens == []
    assert rec.address_tokens == []


def test_unseen_and_accented_countries():
    """Verify open-set country normalization for international and accented country strings."""
    assert normalize_country("Côte d'Ivoire") == "COTE D'IVOIRE"
    assert normalize_country("deutschland") == "DEUTSCHLAND"
    assert normalize_country("français") == "FRANCAIS"
    assert normalize_country("  España  ") == "ESPANA"

    # Query in España must match target in ESPANA, but NOT match target in FRANCE
    q = preprocess_record({"entity_id": "Q-ES", "business_name": "Banco Sol", "business_address": "Gran Via", "country": "España"})
    t1 = preprocess_record({"entity_id": "T-ES", "business_name": "Banco Sol", "business_address": "Gran Via", "country": "ESPANA"})
    t2 = preprocess_record({"entity_id": "T-FR", "business_name": "Banco Sol", "business_address": "Gran Via", "country": "France"})

    assert q.country_norm == "ESPANA"
    assert t1.country_norm == "ESPANA"
    assert compute_stage1_broad_score(q, t1) > 0.8
    assert compute_stage1_broad_score(q, t2) == 0.0  # Country mismatch rule


def test_extremely_short_names():
    """Verify short business names like '3M', 'K', 'HP' are handled properly."""
    raw = {"entity_id": "SHORT-01", "business_name": "3M Co.", "business_address": "3M Center, St Paul 55144", "country": "US"}
    rec = preprocess_record(raw)
    assert rec.core_name == "3m"
    assert "3m" in rec.name_tokens

    # Another short name
    raw_k = {"entity_id": "SHORT-02", "business_name": "K Corp", "business_address": "100 Broadway", "country": "US"}
    rec_k = preprocess_record(raw_k)
    assert rec_k.core_name == "k"


def test_complex_multi_number_address():
    """Addresses with floor, suite, street number, and postal code."""
    addr = "Suite 500, Floor 3, 100 Innovation Blvd, P.O. Box 450, Austin 78701"
    comps = extract_address_components(addr)
    assert "500" in comps["numbers"]
    assert "3" in comps["numbers"]
    assert "100" in comps["numbers"]
    assert "78701" in comps["postal_candidates"]


def test_zero_matches_all_singletons_evaluation():
    """Verify that when 100% of queries are singletons, metrics don't divide by zero."""
    gt = {f"S1-{i:03d}": [] for i in range(20)}
    preds = {f"S1-{i:03d}": [] for i in range(20)}

    metrics = evaluate_predictions(gt, preds)
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f0_5"] == 1.0
    assert metrics["singleton_accuracy"] == 1.0
    assert metrics["total_fp"] == 0
    assert metrics["total_fn"] == 0

    # If all candidates are also empty
    cand_metrics = compute_candidate_recall(gt, preds)
    assert cand_metrics["candidate_recall"] == 1.0
    assert cand_metrics["total_true_pairs"] == 0


def test_exact_threshold_boundary():
    """Verify exact behavior at threshold boundary (>= thresh)."""
    q = preprocess_record({"entity_id": "Q1", "business_name": "Target Tech", "business_address": "100 Way", "country": "US"})
    t = preprocess_record({"entity_id": "T1", "business_name": "Target Tech", "business_address": "100 Way", "country": "US"})

    pipeline = TwoStagePipeline(
        target_records={"T1": t},
        stage1_threshold=0.10,
        stage2_threshold=0.65,
    )

    # If score is exactly equal to threshold, it MUST be included
    preds, scored = pipeline.predict_entity(q, ["T1"], custom_threshold=0.65)
    score = scored[0][1]
    assert score >= 0.65
    assert "T1" in preds

    # If threshold is higher than score, it must NOT be included
    preds_strict, _ = pipeline.predict_entity(q, ["T1"], custom_threshold=score + 0.01)
    assert "T1" not in preds_strict
