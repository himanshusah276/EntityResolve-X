"""
Ensemble and Teammate Integration Module for Person 4 Pipeline.

Supports optional loading and calibration of teammate scores:
- Person 1 (Classical fuzzy / weighted score)
- Person 2 (Supervised classifier probability)
- Person 3 (Embedding semantic similarity)

Adheres strictly to constraints:
- Pipeline runs completely standalone when teammate scores are None.
- Validates the common schema: source1_entity_id, candidate_entity_id, score.
- Never assumes ensembling helps without empirical validation.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import config
from src.evaluation import evaluate_predictions

logger = logging.getLogger(__name__)

COMMON_SCHEMA = ["source1_entity_id", "candidate_entity_id", "score"]


def load_teammate_scores(
    file_path: Optional[Union[str, Path]],
    teammate_name: str = "Teammate",
) -> Optional[pd.DataFrame]:
    """
    Loads teammate scores file (TSV or CSV) in the common schema:
    source1_entity_id    candidate_entity_id    score
    Returns DataFrame with [source1_entity_id, candidate_entity_id, raw_score, normalized_score],
    or None if file is not provided or does not exist.
    """
    if file_path is None:
        logger.info(f"{teammate_name} scores path is None. Running standalone.")
        return None

    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"{teammate_name} score file not found at {path.resolve()}. Running standalone.")
        return None

    # Detect delimiter
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, sep=sep, dtype=str)
    except Exception as e:
        logger.error(f"Failed to read {teammate_name} scores from {path}: {e}")
        return None

    # Validate columns
    missing = [c for c in COMMON_SCHEMA if c not in df.columns]
    if missing:
        logger.error(f"{teammate_name} scores missing required columns: {missing}. Found: {list(df.columns)}")
        return None

    # Clean IDs and numeric scores
    df["source1_entity_id"] = df["source1_entity_id"].astype(str).str.strip()
    df["candidate_entity_id"] = df["candidate_entity_id"].astype(str).str.strip()
    df["raw_score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    # Score normalization
    raw = df["raw_score"].values
    min_val, max_val = float(np.min(raw)), float(np.max(raw))

    if min_val < 0.0:
        # e.g. cosine similarity in [-1, 1]
        df["normalized_score"] = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
    elif max_val > 1.0:
        # Min-max scale if scores exceed 1.0
        denom = (max_val - min_val) if (max_val - min_val) > 0 else 1.0
        df["normalized_score"] = np.clip((raw - min_val) / denom, 0.0, 1.0)
    else:
        df["normalized_score"] = np.clip(raw, 0.0, 1.0)

    logger.info(f"Loaded {len(df)} score records for {teammate_name} from {path.name}")
    return df[["source1_entity_id", "candidate_entity_id", "raw_score", "normalized_score"]]


def load_person1_scores(path: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
    """Loader for Person 1 (Classical Fuzzy Matching)."""
    return load_teammate_scores(path, "Person 1 (Classical)")


def load_person2_scores(path: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
    """Loader for Person 2 (Supervised Pair Classifier)."""
    return load_teammate_scores(path, "Person 2 (Supervised)")


def load_person3_scores(path: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
    """Loader for Person 3 (Embedding Semantic Matching)."""
    return load_teammate_scores(path, "Person 3 (Embedding)")


def build_teammate_lookup(
    p1_df: Optional[pd.DataFrame] = None,
    p2_df: Optional[pd.DataFrame] = None,
    p3_df: Optional[pd.DataFrame] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Builds nested map: (source1_id, candidate_id) -> {'person1': score, 'person2': score, 'person3': score}.
    """
    lookup: Dict[Tuple[str, str], Dict[str, float]] = {}

    for name, df in [("person1", p1_df), ("person2", p2_df), ("person3", p3_df)]:
        if df is not None:
            for _, row in df.iterrows():
                key = (row["source1_entity_id"], row["candidate_entity_id"])
                if key not in lookup:
                    lookup[key] = {}
                lookup[key][name] = float(row["normalized_score"])

    return lookup


def combine_pair_scores(
    p4_score: float,
    teammate_dict: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Combines Person 4 score with available teammate scores using normalized weighted averaging.
    Gracefully uses Person 4 score alone if no teammate scores are present.
    """
    if not teammate_dict:
        return p4_score

    w = weights or {"p4": 0.60, "person1": 0.15, "person2": 0.15, "person3": 0.10}
    active_weights = {"p4": w.get("p4", 0.60)}
    active_scores = {"p4": p4_score}

    for t_name in ["person1", "person2", "person3"]:
        if t_name in teammate_dict and not np.isnan(teammate_dict[t_name]):
            active_weights[t_name] = w.get(t_name, 0.10)
            active_scores[t_name] = teammate_dict[t_name]

    total_w = sum(active_weights.values())
    if total_w <= 0:
        return p4_score

    final_score = sum(active_scores[k] * active_weights[k] for k in active_scores) / total_w
    return float(np.clip(final_score, 0.0, 1.0))


def evaluate_ensemble_combinations(
    ground_truth: Dict[str, List[str]],
    p4_scored_pairs: Dict[str, List[Tuple[str, float]]],
    p1_df: Optional[pd.DataFrame] = None,
    p2_df: Optional[pd.DataFrame] = None,
    p3_df: Optional[pd.DataFrame] = None,
    decision_threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Evaluates P4 standalone vs available teammate combinations on the same validation split.
    If teammate scores are unavailable, explicitly records 'Not yet measured'.
    """
    results: Dict[str, Any] = {}

    # 1. P4 Standalone (Control)
    p4_preds = {
        q_id: [cid for cid, s in pairs if s >= decision_threshold]
        for q_id, pairs in p4_scored_pairs.items()
    }
    results["P4_STANDALONE"] = evaluate_predictions(ground_truth, p4_preds)

    lookup = build_teammate_lookup(p1_df, p2_df, p3_df)

    combos = [
        ("P4 + Person 1", bool(p1_df is not None)),
        ("P4 + Person 2", bool(p2_df is not None)),
        ("P4 + Person 3", bool(p3_df is not None)),
        ("P4 + All Available Teammates", bool(lookup)),
    ]

    for combo_name, is_available in combos:
        if not is_available:
            results[combo_name] = "Not yet measured (Teammate scores unavailable)"
        else:
            combo_preds = {}
            for q_id, pairs in p4_scored_pairs.items():
                matched_cids = []
                for cid, p4_s in pairs:
                    t_dict = lookup.get((q_id, cid))
                    blended = combine_pair_scores(p4_s, t_dict)
                    if blended >= decision_threshold:
                        matched_cids.append(cid)
                combo_preds[q_id] = matched_cids
            results[combo_name] = evaluate_predictions(ground_truth, combo_preds)

    return results
