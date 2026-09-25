"""
Advanced Two-Stage Matching and Reranking Module for Person 4 Pipeline.

Architecture:
- Stage 1: Fast broad filter using set operations and token overlaps.
  Filters out distant non-matches with high recall retention.
- Stage 2: Detailed feature extraction and reranking using multi-scale string similarities,
  address component verification, numeric agreement, and cross-field consistency.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from rapidfuzz import fuzz

from src.preprocessing import ProcessedRecord

# Default feature weights for Stage 2 interpretable scoring
DEFAULT_STAGE2_WEIGHTS = {
    "name_fuzz_ratio": 0.15,
    "name_token_sort_ratio": 0.15,
    "name_partial_ratio": 0.15,
    "name_token_jaccard": 0.10,
    "core_name_exact": 0.05,
    "address_token_sort": 0.12,
    "address_partial_ratio": 0.08,
    "address_token_jaccard": 0.05,
    "street_number_match": 0.05,
    "postal_code_match": 0.05,
    "cross_field_agreement": 0.05,
}


def jaccard_similarity(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def compute_stage1_broad_score(query: ProcessedRecord, candidate: ProcessedRecord) -> float:
    """
    Lightweight Stage 1 scoring using pure set operations.
    Returns broad score in [0.0, 1.0].
    """
    # Country mismatch immediately disqualifies
    if query.country_norm != candidate.country_norm:
        return 0.0

    name_tok_sim = jaccard_similarity(set(query.name_tokens), set(candidate.name_tokens))
    name_ng_sim = jaccard_similarity(query.name_ngrams, candidate.name_ngrams)
    addr_tok_sim = jaccard_similarity(set(query.address_tokens), set(candidate.address_tokens))
    num_sim = jaccard_similarity(set(query.address_numbers), set(candidate.address_numbers))

    # Broad weighted combination
    score = (
        0.40 * name_tok_sim
        + 0.30 * name_ng_sim
        + 0.20 * addr_tok_sim
        + 0.10 * num_sim
    )
    return score


def extract_stage2_features(
    query: ProcessedRecord,
    candidate: ProcessedRecord,
    teammate_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Extracts comprehensive, interpretable feature vector for Stage 2 reranking:
    - Name similarities (fuzz ratio, token sort, partial ratio, token jaccard, exact core match)
    - Address similarities (token sort, partial ratio, token jaccard)
    - Numeric & postal evidence (street number match, postal code match)
    - Cross-field consistency (joint name and address confirmation)
    - Optional teammate scores (person1, person2, person3)
    """
    feats: Dict[str, float] = {}

    # Country agreement
    same_country = 1.0 if query.country_norm == candidate.country_norm else 0.0
    feats["same_country"] = same_country

    if same_country == 0.0:
        for k in DEFAULT_STAGE2_WEIGHTS:
            feats[k] = 0.0
        feats["stage1_score"] = 0.0
        return feats

    # 1. Name evidence (using core names with legal suffixes normalized)
    q_name = query.core_name or query.clean_name
    c_name = candidate.core_name or candidate.clean_name

    feats["name_fuzz_ratio"] = fuzz.ratio(q_name, c_name) / 100.0 if q_name and c_name else 0.0
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(q_name, c_name) / 100.0 if q_name and c_name else 0.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(q_name, c_name) / 100.0 if q_name and c_name else 0.0
    feats["name_token_jaccard"] = jaccard_similarity(set(query.name_tokens), set(candidate.name_tokens))
    feats["core_name_exact"] = 1.0 if query.core_name and (query.core_name == candidate.core_name) else 0.0

    # 2. Address evidence
    q_addr = query.clean_address
    c_addr = candidate.clean_address
    feats["address_token_sort"] = fuzz.token_sort_ratio(q_addr, c_addr) / 100.0 if q_addr and c_addr else 0.0
    feats["address_partial_ratio"] = fuzz.partial_ratio(q_addr, c_addr) / 100.0 if q_addr and c_addr else 0.0
    feats["address_token_jaccard"] = jaccard_similarity(set(query.address_tokens), set(candidate.address_tokens))

    # 3. Numeric & Postal Evidence
    # Street number agreement
    if query.street_number and candidate.street_number:
        feats["street_number_match"] = 1.0 if query.street_number == candidate.street_number else 0.0
    else:
        feats["street_number_match"] = 0.5  # Neutral / missing

    # Postal code agreement
    q_postal = set(query.postal_candidates)
    c_postal = set(candidate.postal_candidates)
    if q_postal and c_postal:
        feats["postal_code_match"] = 1.0 if bool(q_postal.intersection(c_postal)) else 0.0
    else:
        feats["postal_code_match"] = 0.5  # Neutral / missing

    # 4. Cross-Field Consistency
    # If both name and address have high agreement, confidence increases
    name_high = max(feats["name_token_sort_ratio"], feats["name_partial_ratio"]) >= 0.70
    addr_high = max(feats["address_token_sort"], feats["address_partial_ratio"]) >= 0.60
    feats["cross_field_agreement"] = 1.0 if (name_high and addr_high) else 0.0

    # Include Stage 1 score as an additional feature
    feats["stage1_score"] = compute_stage1_broad_score(query, candidate)

    # 5. Optional teammate scores
    if teammate_scores:
        feats["person1_score"] = teammate_scores.get("person1", np.nan)
        feats["person2_score"] = teammate_scores.get("person2", np.nan)
        feats["person3_score"] = teammate_scores.get("person3", np.nan)

    return feats


def compute_stage2_score(
    features: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Computes Stage 2 final composite score in [0.0, 1.0] from extracted features.
    """
    if features.get("same_country", 1.0) == 0.0:
        return 0.0

    w_dict = weights or DEFAULT_STAGE2_WEIGHTS
    total_w = sum(w_dict.values())
    weighted_sum = sum(features.get(k, 0.0) * w for k, w in w_dict.items())

    final_score = weighted_sum / total_w if total_w > 0 else 0.0
    return float(np.clip(final_score, 0.0, 1.0))


class TwoStagePipeline:
    """
    Coordinates Candidate Generation -> Stage 1 Filter -> Stage 2 Detailed Rerank.
    """

    def __init__(
        self,
        target_records: Dict[str, ProcessedRecord],
        stage1_threshold: float = 0.15,
        stage2_threshold: float = 0.65,
        stage2_weights: Optional[Dict[str, float]] = None,
    ):
        self.target_records = target_records
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.stage2_weights = stage2_weights or DEFAULT_STAGE2_WEIGHTS

    def filter_stage1(
        self,
        query: ProcessedRecord,
        candidate_ids: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Stage 1 Broad Filter:
        Retains candidates with stage1_score >= stage1_threshold.
        Returns list of (candidate_id, stage1_score).
        """
        survivors = []
        for cid in candidate_ids:
            cand = self.target_records.get(cid)
            if cand is None:
                continue
            s1_score = compute_stage1_broad_score(query, cand)
            if s1_score >= self.stage1_threshold:
                survivors.append((cid, s1_score))

        # Sort descending by Stage 1 score
        survivors.sort(key=lambda x: x[1], reverse=True)
        return survivors

    def rerank_stage2(
        self,
        query: ProcessedRecord,
        stage1_candidates: List[Tuple[str, float]],
        teammate_scores_map: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Stage 2 Detailed Reranking:
        Extracts rich features and computes fine-grained score.
        Returns list of (candidate_id, stage2_score, features_dict).
        """
        ranked = []
        for cid, _ in stage1_candidates:
            cand = self.target_records.get(cid)
            if cand is None:
                continue
            pair_teammate = teammate_scores_map.get((query.entity_id, cid)) if teammate_scores_map else None
            feats = extract_stage2_features(query, cand, pair_teammate)
            s2_score = compute_stage2_score(feats, self.stage2_weights)
            ranked.append((cid, s2_score, feats))

        # Sort descending by Stage 2 score
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def predict_entity(
        self,
        query: ProcessedRecord,
        initial_candidate_ids: List[str],
        custom_threshold: Optional[float] = None,
    ) -> Tuple[List[str], List[Tuple[str, float]]]:
        """
        Runs Stage 1 -> Stage 2 -> Threshold Decision for a single query entity.
        Returns:
        (predicted_matched_ids, all_scored_pairs [(cid, score)])
        """
        thresh = custom_threshold if custom_threshold is not None else self.stage2_threshold
        s1_survivors = self.filter_stage1(query, initial_candidate_ids)
        s2_ranked = self.rerank_stage2(query, s1_survivors)

        predicted_ids = [cid for cid, score, _ in s2_ranked if score >= thresh]
        scored_pairs = [(cid, score) for cid, score, _ in s2_ranked]

        return predicted_ids, scored_pairs
