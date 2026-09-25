"""
Unit tests for src/evaluation.py
"""
import pytest
from pathlib import Path
from src.evaluation import (
    create_source1_validation_split,
    compute_candidate_recall,
    compute_reduction_ratio,
    calculate_entity_f0_5,
    evaluate_predictions,
    record_experiment_result,
)


def test_create_source1_validation_split():
    s1_ids = [f"S1-{i:03d}" for i in range(100)]
    train_ids, val_ids = create_source1_validation_split(s1_ids, val_ratio=0.2, seed=42)

    assert len(train_ids) == 80
    assert len(val_ids) == 20
    # No leakage: sets are strictly disjoint
    assert set(train_ids).isdisjoint(set(val_ids))
    # Seed reproducibility: same seed gives identical lists
    t2, v2 = create_source1_validation_split(s1_ids, val_ratio=0.2, seed=42)
    assert train_ids == t2
    assert val_ids == v2


def test_candidate_recall_and_reduction_ratio():
    gt = {
        "S1-1": ["S2-10", "S3-20"],
        "S1-2": ["S2-30"],
        "S1-3": [],  # singleton
    }
    # 3 total true links
    cands = {
        "S1-1": ["S2-10"],          # 1 found, 1 missed
        "S1-2": ["S2-30", "S3-99"], # 1 found
        "S1-3": ["S2-50"],          # distractor
    }
    res = compute_candidate_recall(gt, cands)
    assert res["total_true_pairs"] == 3
    assert res["retrieved_true_pairs"] == 2
    assert pytest.approx(res["candidate_recall"], 0.01) == 2 / 3
    assert res["total_candidates"] == 4

    rr = compute_reduction_ratio(n_source1=10, n_targets_pool=100, total_candidates=50)
    assert pytest.approx(rr, 0.001) == 1.0 - (50 / 1000)


def test_evaluate_predictions_singleton_and_matching():
    gt = {
        "S1-1": ["S2-1"],           # regular match
        "S1-2": [],                 # singleton correct
        "S1-3": [],                 # singleton false positive
        "S1-4": ["S2-4"],           # false negative
    }
    preds = {
        "S1-1": ["S2-1"],           # Perfect: P=1, R=1, F0.5=1
        "S1-2": [],                 # Perfect singleton: P=1, R=1, F0.5=1
        "S1-3": ["S2-9"],           # Singleton FP: P=0, R=0, F0.5=0
        "S1-4": [],                 # Missed FN: P=0, R=0, F0.5=0
    }
    metrics = evaluate_predictions(gt, preds)
    # 2 entities have 1.0, 2 entities have 0.0 -> macro average = 0.5
    assert pytest.approx(metrics["macro_precision"], 0.01) == 0.50
    assert pytest.approx(metrics["macro_recall"], 0.01) == 0.50
    assert pytest.approx(metrics["macro_f0_5"], 0.01) == 0.50
    assert metrics["total_tp"] == 1
    assert metrics["total_fp"] == 1
    assert metrics["total_fn"] == 1
    assert pytest.approx(metrics["singleton_accuracy"], 0.01) == 0.50


def test_calculate_entity_f0_5():
    # If P=1.0 and R=1.0, F0.5=1.0
    assert pytest.approx(calculate_entity_f0_5(1.0, 1.0), 0.001) == 1.0
    # F0.5 weights precision higher than recall:
    # If P=1.0, R=0.5: F0.5 = 1.25 * 1 * 0.5 / (0.25 * 1 + 0.5) = 0.625 / 0.75 = 0.8333
    assert pytest.approx(calculate_entity_f0_5(1.0, 0.5), 0.001) == 0.8333
    # If P=0.5, R=1.0: F0.5 = 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1) = 0.625 / 1.125 = 0.5555
    assert pytest.approx(calculate_entity_f0_5(0.5, 1.0), 0.001) == 0.5555


def test_record_experiment_result(tmp_path):
    csv_file = tmp_path / "exp.csv"
    metrics = {"macro_f0_5": 0.85, "macro_precision": 0.90, "macro_recall": 0.80}
    record_experiment_result("EXP-01", "Stage 1 baseline", metrics, {"thresh": 0.5}, "test note", csv_path=csv_file)
    assert csv_file.is_file()
    content = csv_file.read_text(encoding="utf-8")
    assert "EXP-01" in content
    assert "0.8500" in content
