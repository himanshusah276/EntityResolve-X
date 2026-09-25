"""
Error Analysis Module for Person 4 Entity Resolution Pipeline.

Extracts and categorizes hard False Positives (FP) and False Negatives (FN).
Saves results into reports/error_analysis.csv following the challenge schema:
source1_entity_id, candidate_entity_id, source1_name, candidate_name,
source1_address, candidate_address, source1_country, candidate_country,
prediction_score, true_label, error_type
"""
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz

from src.config import config
from src.preprocessing import ProcessedRecord

logger = logging.getLogger(__name__)

# Allowed error categories as specified in master brief §16
ERROR_CATEGORIES = {
    "similar_name",
    "shared_address",
    "abbreviation",
    "numeric_conflict",
    "country_mismatch",
    "generic_name",
    "partial_address",
    "spelling_noise",
    "missing_information",
    "other",
    "unknown",
}


def categorize_error(
    s1: ProcessedRecord,
    cand: ProcessedRecord,
    is_false_positive: bool,
) -> str:
    """
    Diagnoses error root cause using multi-field evidence.
    Never invents an explanation; uses 'unknown' if uncertain.
    """
    # 1. Country mismatch
    if s1.country_norm != cand.country_norm:
        return "country_mismatch"

    # 2. Missing information
    if not s1.clean_name or not cand.clean_name or not s1.clean_address or not cand.clean_address:
        return "missing_information"

    # 3. Numeric conflict (e.g. house number 102 vs 804 in same street)
    if (s1.street_number and cand.street_number) and (s1.street_number != cand.street_number):
        return "numeric_conflict"

    q_name = s1.core_name or s1.clean_name
    c_name = cand.core_name or cand.clean_name
    name_ratio = fuzz.ratio(q_name, c_name) / 100.0
    name_partial = fuzz.partial_ratio(q_name, c_name) / 100.0
    name_sort = fuzz.token_sort_ratio(q_name, c_name) / 100.0

    addr_sort = fuzz.token_sort_ratio(s1.clean_address, cand.clean_address) / 100.0

    # 4. Shared address with distinct business name (common false positive)
    if addr_sort >= 0.80 and name_sort < 0.50:
        return "shared_address"

    # 5. Similar name with different address (false positive or difficult negative)
    if name_sort >= 0.80 and addr_sort < 0.40:
        return "similar_name"

    # 6. Abbreviation
    if (name_partial >= 0.90 and name_ratio < 0.70) or any(
        len(t) <= 3 and t in c_name for t in s1.name_tokens
    ):
        return "abbreviation"

    # 7. Spelling noise
    if 0.65 <= name_ratio < 0.90:
        return "spelling_noise"

    # 8. Partial address
    if len(s1.address_tokens) <= 2 or len(cand.address_tokens) <= 2:
        return "partial_address"

    return "other"


def extract_error_records(
    queries: List[ProcessedRecord],
    targets_dict: Dict[str, ProcessedRecord],
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    pair_scores_map: Dict[Tuple[str, str], float],
) -> List[Dict[str, Any]]:
    """
    Identifies all false positive and false negative candidate pairs and categorizes them.
    """
    error_rows: List[Dict[str, Any]] = []

    for q in queries:
        s1_id = q.entity_id
        true_set = set(ground_truth.get(s1_id, []))
        pred_set = set(predictions.get(s1_id, []))

        # False Positives: predicted but not in ground truth
        fps = pred_set - true_set
        for cid in fps:
            cand = targets_dict.get(cid)
            if not cand:
                continue
            score = pair_scores_map.get((s1_id, cid), 0.0)
            err_type = categorize_error(q, cand, is_false_positive=True)
            error_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "source1_name": q.raw_name,
                "candidate_name": cand.raw_name,
                "source1_address": q.raw_address,
                "candidate_address": cand.raw_address,
                "source1_country": q.country_raw,
                "candidate_country": cand.country_raw,
                "prediction_score": f"{score:.4f}",
                "true_label": 0,
                "error_type": err_type,
            })

        # False Negatives: in ground truth but missed by predictions
        fns = true_set - pred_set
        for cid in fns:
            cand = targets_dict.get(cid)
            if not cand:
                continue
            score = pair_scores_map.get((s1_id, cid), 0.0)
            err_type = categorize_error(q, cand, is_false_positive=False)
            error_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "source1_name": q.raw_name,
                "candidate_name": cand.raw_name,
                "source1_address": q.raw_address,
                "candidate_address": cand.raw_address,
                "source1_country": q.country_raw,
                "candidate_country": cand.country_raw,
                "prediction_score": f"{score:.4f}",
                "true_label": 1,
                "error_type": err_type,
            })

    return error_rows


def generate_error_analysis_report(
    queries: List[ProcessedRecord],
    targets_dict: Dict[str, ProcessedRecord],
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    pair_scores_map: Dict[Tuple[str, str], float],
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates reports/error_analysis.csv and returns error DataFrame.
    """
    path = output_path or config.ERROR_ANALYSIS_CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    errors = extract_error_records(
        queries=queries,
        targets_dict=targets_dict,
        ground_truth=ground_truth,
        predictions=predictions,
        pair_scores_map=pair_scores_map,
    )

    fieldnames = [
        "source1_entity_id",
        "candidate_entity_id",
        "source1_name",
        "candidate_name",
        "source1_address",
        "candidate_address",
        "source1_country",
        "candidate_country",
        "prediction_score",
        "true_label",
        "error_type",
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    df_errors = pd.DataFrame(errors)
    logger.info(f"Generated error analysis report with {len(df_errors)} errors in {path}")
    return df_errors
