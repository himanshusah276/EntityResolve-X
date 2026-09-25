"""
Evaluation module for Person 4 Entity Resolution Pipeline.

Implements:
1. Source-1 entity level deterministic validation split (no leakage).
2. Candidate generation metrics (Candidate Recall, Reduction Ratio, Candidates per query).
3. Entity-level macro-averaged Precision, Recall, and F0.5 with explicit singleton handling.
4. Experiment logging utilities for reports/experiments.csv.
"""
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import config

logger = logging.getLogger(__name__)


def create_source1_validation_split(
    source1_ids: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Split Source 1 entity IDs into train and validation sets.
    Deterministic, seeded, operating strictly at the Source 1 entity level.
    """
    unique_ids = sorted(list(set(source1_ids)))
    if len(unique_ids) < 2:
        return unique_ids, []

    train_ids, val_ids = train_test_split(
        unique_ids,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )
    return sorted(train_ids), sorted(val_ids)


def compute_candidate_recall(
    ground_truth: Dict[str, List[str]],
    candidates: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Computes candidate generation metrics:
    - candidate_recall: (true pairs in candidates) / (total true pairs in ground truth)
    - total_candidates: sum of all candidates retrieved
    - avg_candidates_per_entity: average candidate pool size per S1 entity
    """
    total_true_pairs = 0
    retrieved_true_pairs = 0
    total_candidates = 0

    for s1_id, true_matches in ground_truth.items():
        total_true_pairs += len(true_matches)
        cand_set = set(candidates.get(s1_id, []))
        total_candidates += len(cand_set)

        for m in true_matches:
            if m in cand_set:
                retrieved_true_pairs += 1

    candidate_recall = (
        (retrieved_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    )
    num_queries = len(ground_truth) if len(ground_truth) > 0 else 1
    avg_candidates = total_candidates / num_queries

    return {
        "candidate_recall": candidate_recall,
        "retrieved_true_pairs": retrieved_true_pairs,
        "total_true_pairs": total_true_pairs,
        "total_candidates": total_candidates,
        "avg_candidates_per_entity": avg_candidates,
    }


def compute_reduction_ratio(
    n_source1: int,
    n_targets_pool: int,
    total_candidates: int,
) -> float:
    """
    Computes reduction ratio:
    1 - (total_candidates_generated / (n_source1 * n_targets_pool))
    """
    total_cartesian_pairs = n_source1 * n_targets_pool
    if total_cartesian_pairs == 0:
        return 0.0
    return 1.0 - (total_candidates / total_cartesian_pairs)


def calculate_entity_f0_5(p: float, r: float) -> float:
    """Calculates F0.5 score: 1.25 * P * R / (0.25 * P + R)."""
    denom = 0.25 * p + r
    if denom <= 0.0:
        return 0.0
    return (1.25 * p * r) / denom


def evaluate_predictions(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth using macro-averaged entity-level metrics.
    Explicit singleton rules:
    - true=[] & pred=[] => P=1.0, R=1.0, F0.5=1.0 (correct singleton)
    - true=[] & pred=[x] => P=0.0, R=0.0, F0.5=0.0 (false positive singleton)
    - true=[x] & pred=[] => P=0.0, R=0.0, F0.5=0.0 (missed true match)
    """
    precisions: List[float] = []
    recalls: List[float] = []
    f0_5_scores: List[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    singleton_correct = 0
    singleton_total = 0

    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        pred_set = set(predictions.get(s1_id, []))

        is_singleton = len(true_set) == 0
        if is_singleton:
            singleton_total += 1

        if is_singleton and len(pred_set) == 0:
            # Singleton entity correctly predicted with zero matches
            p = 1.0
            r = 1.0
            f0_5 = 1.0
            singleton_correct += 1
        elif is_singleton and len(pred_set) > 0:
            # Singleton entity incorrectly predicted with matches (False Positives)
            p = 0.0
            r = 0.0
            f0_5 = 0.0
            total_fp += len(pred_set)
        elif not is_singleton and len(pred_set) == 0:
            # Non-singleton entity missed completely (False Negatives)
            p = 0.0
            r = 0.0
            f0_5 = 0.0
            total_fn += len(true_set)
        else:
            tp = len(true_set.intersection(pred_set))
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            p = tp / len(pred_set) if len(pred_set) > 0 else 0.0
            r = tp / len(true_set) if len(true_set) > 0 else 0.0
            f0_5 = calculate_entity_f0_5(p, r)

        precisions.append(p)
        recalls.append(r)
        f0_5_scores.append(f0_5)

    n_entities = len(ground_truth) if len(ground_truth) > 0 else 1
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f0_5 = float(np.mean(f0_5_scores)) if f0_5_scores else 0.0
    singleton_acc = (
        (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f0_5": macro_f0_5,
        "singleton_accuracy": singleton_acc,
        "singleton_count": singleton_total,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "evaluated_entities": len(ground_truth),
    }


def record_experiment_result(
    experiment_id: str,
    stage_desc: str,
    metrics: Dict[str, Any],
    parameters: Dict[str, Any],
    notes: str = "",
    csv_path: Optional[Path] = None,
) -> None:
    """
    Appends an experiment record to reports/experiments.csv.
    """
    path = csv_path or config.EXPERIMENTS_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "timestamp",
        "stage_desc",
        "macro_f0_5",
        "macro_precision",
        "macro_recall",
        "candidate_recall",
        "singleton_accuracy",
        "total_tp",
        "total_fp",
        "total_fn",
        "avg_candidates",
        "parameters",
        "notes",
    ]

    file_exists = path.is_file()

    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "stage_desc": stage_desc,
        "macro_f0_5": f"{metrics.get('macro_f0_5', 0.0):.4f}",
        "macro_precision": f"{metrics.get('macro_precision', 0.0):.4f}",
        "macro_recall": f"{metrics.get('macro_recall', 0.0):.4f}",
        "candidate_recall": f"{metrics.get('candidate_recall', 0.0):.4f}",
        "singleton_accuracy": f"{metrics.get('singleton_accuracy', 0.0):.4f}",
        "total_tp": metrics.get("total_tp", 0),
        "total_fp": metrics.get("total_fp", 0),
        "total_fn": metrics.get("total_fn", 0),
        "avg_candidates": f"{metrics.get('avg_candidates_per_entity', 0.0):.2f}",
        "parameters": str(parameters),
        "notes": notes,
    }

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Recorded experiment {experiment_id} in {path}")
