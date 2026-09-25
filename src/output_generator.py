"""
Output generation for matching_results.tsv and candidate_pairs.tsv complying with format requirements.
"""
import csv
from pathlib import Path
from typing import Dict, List


def write_candidate_pairs(
    candidate_dict: Dict[str, List[str]],
    output_path: Path
):
    """
    Saves candidate_pairs.tsv in the format:
    source1_entity_id\tcandidate_entity_ids
    where candidate_entity_ids are comma-separated.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for s1_id, cands in candidate_dict.items():
            cand_str = ",".join(cands)
            writer.writerow([s1_id, cand_str])


def write_matching_results(
    matching_dict: Dict[str, List[str]],
    output_path: Path
):
    """
    Saves matching_results.tsv in the format:
    source1_entity_id\tmatched_entity_ids
    where matched_entity_ids are comma-separated. Singletons have empty field.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id, matches in matching_dict.items():
            match_str = ",".join(matches)
            writer.writerow([s1_id, match_str])
