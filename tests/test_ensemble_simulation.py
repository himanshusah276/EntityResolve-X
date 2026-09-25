"""
Ensemble simulation test verifying teammate score integration and blending behavior.
"""
import pytest
import pandas as pd
from src.ensemble import (
    load_person1_scores,
    load_person2_scores,
    load_person3_scores,
    build_teammate_lookup,
    combine_pair_scores,
    evaluate_ensemble_combinations,
)


def test_ensemble_simulation_with_all_teammates(tmp_path):
    # 1. Create simulated score files in common schema
    p1_file = tmp_path / "p1_scores.tsv"
    p2_file = tmp_path / "p2_scores.tsv"
    p3_file = tmp_path / "p3_scores.tsv"

    p1_file.write_text("source1_entity_id\tcandidate_entity_id\tscore\nS1-1\tS2-1\t0.82\nS1-1\tS2-2\t0.30\n", encoding="utf-8")
    p2_file.write_text("source1_entity_id\tcandidate_entity_id\tscore\nS1-1\tS2-1\t0.91\nS1-1\tS2-2\t0.15\n", encoding="utf-8")
    p3_file.write_text("source1_entity_id\tcandidate_entity_id\tscore\nS1-1\tS2-1\t0.88\nS1-1\tS2-2\t-0.20\n", encoding="utf-8")

    # 2. Load via individual teammate loaders
    p1_df = load_person1_scores(p1_file)
    p2_df = load_person2_scores(p2_file)
    p3_df = load_person3_scores(p3_file)

    assert p1_df is not None
    assert p2_df is not None
    assert p3_df is not None

    # 3. Build lookup
    lookup = build_teammate_lookup(p1_df, p2_df, p3_df)
    assert ("S1-1", "S2-1") in lookup
    assert "person1" in lookup[("S1-1", "S2-1")]
    assert "person2" in lookup[("S1-1", "S2-1")]
    assert "person3" in lookup[("S1-1", "S2-1")]

    # 4. Check score combination
    p4_score = 0.85
    blended = combine_pair_scores(p4_score, lookup[("S1-1", "S2-1")])
    assert 0.80 <= blended <= 0.95

    # 5. Evaluate ensemble combinations
    gt = {"S1-1": ["S2-1"]}
    p4_scored = {"S1-1": [("S2-1", 0.85), ("S2-2", 0.25)]}

    res = evaluate_ensemble_combinations(
        ground_truth=gt,
        p4_scored_pairs=p4_scored,
        p1_df=p1_df,
        p2_df=p2_df,
        p3_df=p3_df,
        decision_threshold=0.65,
    )

    assert "P4_STANDALONE" in res
    assert "P4 + Person 1" in res
    assert "P4 + Person 2" in res
    assert "P4 + Person 3" in res
    assert "P4 + All Available Teammates" in res

    # Verify all evaluated combinations achieved valid Macro F0.5 metrics
    assert isinstance(res["P4 + Person 1"], dict)
    assert res["P4 + Person 1"]["macro_f0_5"] == 1.0
    assert res["P4 + All Available Teammates"]["macro_f0_5"] == 1.0
