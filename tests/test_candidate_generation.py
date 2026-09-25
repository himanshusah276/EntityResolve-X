"""
Unit tests for src/candidate_generation.py
"""
import pytest
from src.preprocessing import preprocess_record
from src.candidate_generation import CandidateIndex, generate_candidates_for_queries


@pytest.fixture
def sample_target_records():
    raw_targets = [
        # Target 1: Matches on name
        {"entity_id": "T1", "business_name": "Apex Global Solutions Inc", "business_address": "100 Market St, Austin 78701", "country": "US"},
        # Target 2: Matches on typo / abbreviation
        {"entity_id": "T2", "business_name": "Apx Glbl Sol", "business_address": "100 Market St, Austin 78701", "country": "US"},
        # Target 3: Matches on postal and street number
        {"entity_id": "T3", "business_name": "Completely Different Corp", "business_address": "100 Market St, Austin 78701", "country": "US"},
        # Target 4: Same name but different country (France) -> must NOT match query in US
        {"entity_id": "T4", "business_name": "Apex Global Solutions", "business_address": "Paris 75001", "country": "FRANCE"},
    ]
    return [preprocess_record(r) for r in raw_targets]


def test_block_a_name_tokens(sample_target_records):
    index = CandidateIndex(sample_target_records)
    query = preprocess_record({
        "entity_id": "Q1",
        "business_name": "Apex Global Corporation",
        "business_address": "999 Other Rd, Dallas 75001",
        "country": "US",
    })
    cands_a = index.retrieve_block_a(query)
    # Should retrieve T1 (shares "apex", "global")
    assert "T1" in cands_a
    # Must NOT retrieve T4 because country is FRANCE
    assert "T4" not in cands_a


def test_block_b_char_ngrams_typo(sample_target_records):
    index = CandidateIndex(sample_target_records)
    query = preprocess_record({
        "entity_id": "Q2",
        "business_name": "Apx Glbl Sol",  # Exact match for T2
        "business_address": "Different Addr",
        "country": "US",
    })
    cands_b = index.retrieve_block_b(query, min_common_ngrams=2)
    assert "T2" in cands_b


def test_block_d_numeric_postal(sample_target_records):
    index = CandidateIndex(sample_target_records)
    query = preprocess_record({
        "entity_id": "Q3",
        "business_name": "Unknown Entity",
        "business_address": "100 Market St, Austin 78701",
        "country": "US",
    })
    cands_d = index.retrieve_block_d(query)
    # T1, T2, T3 all have postal 78701 and street number 100
    assert "T1" in cands_d
    assert "T3" in cands_d


def test_multipass_union_adaptive(sample_target_records):
    index = CandidateIndex(sample_target_records)
    # Weak name query -> adaptive triggers address fallback
    weak_query = preprocess_record({
        "entity_id": "Q4",
        "business_name": "A",
        "business_address": "100 Market St, Austin 78701",
        "country": "US",
    })
    cands = index.retrieve_multipass_union(weak_query, max_candidates=10)
    assert "T1" in cands or "T3" in cands


def test_generate_candidates_for_queries(sample_target_records):
    index = CandidateIndex(sample_target_records)
    queries = [
        preprocess_record({
            "entity_id": "Q1",
            "business_name": "Apex Global Corp",
            "business_address": "Austin",
            "country": "US",
        })
    ]
    res = generate_candidates_for_queries(index, queries, blocking_method="block_union_all")
    assert "Q1" in res
    assert isinstance(res["Q1"], list)
