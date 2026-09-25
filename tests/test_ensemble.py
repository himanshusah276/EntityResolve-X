"""
Unit tests for src/ensemble.py
"""
import pytest
import pandas as pd
from pathlib import Path
from src.ensemble import (
    load_teammate_scores,
    build_teammate_lookup,
    combine_pair_scores,
    evaluate_ensemble_combinations,
)


def test_load_teammate_scores_missing():
    # Graceful degradation on missing file
    assert load_teammate_scores(None) is None
    assert load_teammate_scores("non_existent_file.tsv") is None


def test_load_teammate_scores_normalization(tmp_path):
    # Test normalization of scores
    tsv_file = tmp_path / "p3_scores.tsv"
    tsv_file.write_text(
        "source1_entity_id\tcandidate_entity_id\tscore\n"
        "S1-1\tS2-1\t0.85\n"
        "S1-1\tS2-2\t-0.50\n",
        encoding="utf-8",
    )
    df = load_teammate_scores(tsv_file, "Person 3")
    assert df is not None
    assert len(df) == 2
    # Check normalized scores are bounded in [0, 1]
    assert 0.0 <= df.iloc[0]["normalized_score"] <= 1.0
    assert 0.0 <= df.iloc[1]["normalized_score"] <= 1.0


def test_combine_pair_scores():
    # 1. No teammate score -> returns p4_score directly
    assert combine_pair_scores(0.75, None) == 0.75
    assert combine_pair_scores(0.75, {}) == 0.75

    # 2. Teammate score present -> weighted blend
    # w_p4=0.6, w_p1=0.15 -> total_w = 0.75 -> (0.80*0.60 + 0.60*0.15)/0.75 = (0.48 + 0.09)/0.75 = 0.57/0.75 = 0.76
    t_dict = {"person1": 0.60}
    blended = combine_pair_scores(0.80, t_dict)
    assert pytest.approx(blended, 0.01) == 0.76


def test_evaluate_ensemble_combinations():
    gt = {"S1-1": ["S2-1"]}
    p4_scored = {"S1-1": [("S2-1", 0.85), ("S2-2", 0.40)]}

    # When all teammates are None
    results = evaluate_ensemble_combinations(gt, p4_scored, p1_df=None, p2_df=None, p3_df=None)
    assert "P4_STANDALONE" in results
    assert results["P4_STANDALONE"]["macro_f0_5"] == 1.0
    assert "Not yet measured" in results["P4 + Person 1"]
    assert "Not yet measured" in results["P4 + Person 2"]
    assert "Not yet measured" in results["P4 + Person 3"]
