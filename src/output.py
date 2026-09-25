"""
Output Generation and Validation Module for Person 4 Pipeline.

Generates:
1. outputs/matching_results.tsv (source1_entity_id, matched_entity_ids)
2. outputs/candidate_pairs.tsv (source1_entity_id, candidate_entity_ids)

Performs strict submission validation:
- Tab-separated formatting without pandas index
- Exact 1-to-1 match of Source 1 test entities
- Predictions must be a strict subset of candidate pairs
- Target IDs must exist in Source 2 or Source 3
- Zero self-matches (no Source 1 ID in predictions)
- Zero duplicate IDs within match lists
- Proper singleton representation (empty string)
"""
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import pandas as pd

from src.config import config
from src.data_loader import load_tsv

logger = logging.getLogger(__name__)


def write_matching_results_tsv(
    predictions: Dict[str, List[str]],
    all_source1_ids: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Writes predictions to matching_results.tsv using programmatic tab-delimited formatting.
    Ensures every Source 1 entity appears exactly once in the output.
    """
    path = Path(output_path or config.MATCHING_RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])

        for s1_id in all_source1_ids:
            matches = predictions.get(s1_id, [])
            matched_str = ",".join(matches) if matches else ""
            writer.writerow([s1_id, matched_str])

    logger.info(f"Successfully wrote matching results TSV to {path.resolve()}")
    return path


def write_candidate_pairs_tsv(
    candidates: Dict[str, List[str]],
    all_source1_ids: List[str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Writes candidates to candidate_pairs.tsv using programmatic tab-delimited formatting.
    Ensures every Source 1 entity appears exactly once in the output.
    """
    path = Path(output_path or config.CANDIDATE_PAIRS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id in all_source1_ids:
            cand_list = candidates.get(s1_id, [])
            cand_str = ",".join(cand_list) if cand_list else ""
            writer.writerow([s1_id, cand_str])

    logger.info(f"Successfully wrote candidate pairs TSV to {path.resolve()}")
    return path


def validate_submission_files(
    matching_results_path: Union[str, Path],
    candidate_pairs_path: Union[str, Path],
    test_source1_path: Union[str, Path],
    test_source2_path: Union[str, Path],
    test_source3_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Validates submission TSV files against all competition constraints:
    1. Schema & tab delimiters
    2. Exact row count & alignment with test Source 1
    3. Prediction subset of candidates
    4. Target ID validity (must exist in S2/S3, never S1)
    5. Deduplication
    """
    s1_df = load_tsv(test_source1_path, expected_columns=["entity_id"])
    s2_df = load_tsv(test_source2_path, expected_columns=["entity_id"])
    s3_df = load_tsv(test_source3_path, expected_columns=["entity_id"])

    s1_id_list = list(s1_df["entity_id"].astype(str).str.strip())
    s1_id_set = set(s1_id_list)
    s2_id_set = set(s2_df["entity_id"].astype(str).str.strip())
    s3_id_set = set(s3_df["entity_id"].astype(str).str.strip())
    valid_target_ids = s2_id_set.union(s3_id_set)

    # 1. Validate Candidate Pairs TSV
    cand_path = Path(candidate_pairs_path)
    if not cand_path.is_file():
        raise FileNotFoundError(f"Candidate pairs file not found: {cand_path}")

    cands_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
    if list(cands_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        raise ValueError(f"Invalid columns in candidate_pairs.tsv: {list(cands_df.columns)}")

    cand_s1_ids = list(cands_df["source1_entity_id"].astype(str).str.strip())
    if len(cand_s1_ids) != len(s1_id_list):
        raise ValueError(
            f"Candidate pairs row count mismatch: found {len(cand_s1_ids)}, expected {len(s1_id_list)}"
        )
    if cand_s1_ids != s1_id_list:
        raise ValueError("Candidate pairs source1_entity_id order or content does not match test_source1")

    # Map candidates
    cand_dict: Dict[str, Set[str]] = {}
    for _, row in cands_df.iterrows():
        eid = str(row["source1_entity_id"]).strip()
        c_str = str(row["candidate_entity_ids"]).strip()
        items = [x.strip() for x in c_str.split(",") if x.strip()]
        if len(items) != len(set(items)):
            raise ValueError(f"Duplicate candidate IDs found for entity {eid}")
        cand_dict[eid] = set(items)

    # 2. Validate Matching Results TSV
    match_path = Path(matching_results_path)
    if not match_path.is_file():
        raise FileNotFoundError(f"Matching results file not found: {match_path}")

    match_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)
    if list(match_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        raise ValueError(f"Invalid columns in matching_results.tsv: {list(match_df.columns)}")

    pred_s1_ids = list(match_df["source1_entity_id"].astype(str).str.strip())
    if len(pred_s1_ids) != len(s1_id_list):
        raise ValueError(
            f"Matching results row count mismatch: found {len(pred_s1_ids)}, expected {len(s1_id_list)}"
        )
    if pred_s1_ids != s1_id_list:
        raise ValueError("Matching results source1_entity_id order or content does not match test_source1")

    total_predictions = 0
    total_singletons = 0

    for _, row in match_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        m_str = str(row["matched_entity_ids"]).strip()
        matches = [x.strip() for x in m_str.split(",") if x.strip()]

        if not matches:
            total_singletons += 1
            continue

        # Check duplicates
        if len(matches) != len(set(matches)):
            raise ValueError(f"Duplicate predicted IDs found for entity {s1_id}: {matches}")

        # Check for self-match
        if s1_id in matches:
            raise ValueError(f"Source 1 ID {s1_id} is illegally included in predictions!")

        # Check valid target existence
        for target_id in matches:
            if target_id not in valid_target_ids:
                raise ValueError(
                    f"Predicted ID {target_id} for entity {s1_id} does NOT exist in test Source 2 or Source 3!"
                )
            if target_id in s1_id_set:
                raise ValueError(
                    f"Predicted ID {target_id} is a Source 1 entity ID, which is strictly prohibited!"
                )

        # Check subset of candidate pairs
        allowed_cands = cand_dict.get(s1_id, set())
        for target_id in matches:
            if target_id not in allowed_cands:
                raise ValueError(
                    f"Predicted ID {target_id} for entity {s1_id} is NOT in its candidate pair set!"
                )

        total_predictions += len(matches)

    validation_summary = {
        "status": "PASSED",
        "total_source1_entities": len(s1_id_list),
        "total_predictions": total_predictions,
        "singleton_entities": total_singletons,
        "non_singleton_entities": len(s1_id_list) - total_singletons,
        "all_predictions_subset_of_candidates": True,
        "zero_source1_ids_in_predictions": True,
        "zero_duplicate_ids": True,
    }

    logger.info(f"Validation successfully passed: {validation_summary}")
    return validation_summary
