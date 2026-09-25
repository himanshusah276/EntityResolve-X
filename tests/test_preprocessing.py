"""
Unit tests for src/preprocessing.py
"""
import pytest
from src.preprocessing import (
    clean_text,
    extract_legal_suffixes,
    normalize_country,
    normalize_address_tokens,
    extract_address_components,
    generate_char_ngrams,
    preprocess_record,
)


def test_clean_text():
    assert clean_text("  ACME   Technologies, Inc.!  ") == "acme technologies inc"
    assert clean_text("Café & Co.") == "cafe co"
    assert clean_text("") == ""


def test_extract_legal_suffixes():
    core, suffixes = extract_legal_suffixes("ABC Technologies Pvt Ltd")
    assert core == "abc technologies"
    assert "pvt ltd" in suffixes

    # Conservative handling: do not reduce to empty
    core_alone, suffixes_alone = extract_legal_suffixes("Limited")
    assert core_alone == "limited"


def test_normalize_country():
    # Open set: preserves arbitrary country strings, never fails on unseen countries
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("  republic of korea  ") == "REPUBLIC OF KOREA"
    assert normalize_country("") == "UNKNOWN"
    assert normalize_country(None) == "UNKNOWN"


def test_normalize_address_tokens():
    tokens = normalize_address_tokens("123 Main St., MG Rd, Ste 400")
    assert "street" in tokens
    assert "road" in tokens
    assert "suite" in tokens


def test_extract_address_components():
    comps = extract_address_components("742 Evergreen Terrace, Springfield, 97477")
    assert comps["street_number"] == "742"
    assert "97477" in comps["postal_candidates"]
    assert "742" in comps["numbers"]


def test_generate_char_ngrams():
    ngrams = generate_char_ngrams("Acme", n=3)
    assert "acm" in ngrams
    assert "cme" in ngrams


def test_preprocess_record():
    raw = {
        "entity_id": "S1-001",
        "business_name": "Apex Global Solutions LLC",
        "business_address": "456 Market Blvd, Austin 78701",
        "country": "US",
    }
    rec = preprocess_record(raw)
    assert rec.entity_id == "S1-001"
    assert rec.country_norm == "US"
    assert rec.core_name == "apex global solutions"
    assert "llc" in rec.legal_suffixes
    assert "boulevard" in rec.address_tokens
    assert "78701" in rec.postal_candidates
    assert rec.street_number == "456"
