"""
Weighted scoring and thresholding decision logic for candidate pairs.
"""
from typing import Dict, List, Set, Tuple
from .config import config
from .similarity import (
    compute_name_similarity,
    compute_address_similarity,
    numeric_overlap_score,
)


class EntityMatcher:
    """
    Computes pairwise feature scores and predicts matches based on thresholding.
    """

    def __init__(
        self,
        w_name: float = config.W_NAME,
        w_addr: float = config.W_ADDRESS,
        w_num: float = config.W_NUMERIC,
        threshold: float = config.MATCH_THRESHOLD
    ):
        self.w_name = w_name
        self.w_addr = w_addr
        self.w_num = w_num
        self.threshold = threshold

    def compute_pair_score(
        self,
        s1_record: Dict,
        target_record: Dict
    ) -> float:
        """
        Calculate composite weighted score between Source 1 record and candidate.
        """
        # Name similarity
        name_sim = compute_name_similarity(
            s1_record["name_norm"],
            target_record["name_norm"],
            s1_record["name_core_tokens"],
            target_record["name_core_tokens"]
        )

        # Address similarity
        addr_sim = compute_address_similarity(
            s1_record["addr_norm"],
            target_record["addr_norm"],
            s1_record["addr_tokens"],
            target_record["addr_tokens"]
        )

        # Numeric score
        num_score = numeric_overlap_score(
            s1_record["numbers"],
            target_record["numbers"]
        )

        final_score = (
            self.w_name * name_sim +
            self.w_addr * addr_sim +
            self.w_num * num_score
        )
        return final_score

    def match_candidates(
        self,
        s1_record: Dict,
        candidates: List[Dict],
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Score all candidates and return those exceeding the threshold.
        """
        thresh = threshold if threshold is not None else self.threshold
        matched = []
        for cand in candidates:
            score = self.compute_pair_score(s1_record, cand)
            if score >= thresh:
                matched.append((cand["entity_id"], score))
        # Sort matches by score descending
        matched.sort(key=lambda x: x[1], reverse=True)
        return matched
