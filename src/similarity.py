"""
String and token similarity metrics using rapidfuzz and set overlaps.
"""
from typing import Set, List
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Jaccard similarity between two token lists."""
    if not tokens1 or not tokens2:
        return 0.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """Overlap coefficient: intersection divided by min length (handles containment)."""
    if not tokens1 or not tokens2:
        return 0.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    intersection = len(s1 & s2)
    min_len = min(len(s1), len(s2))
    return intersection / min_len if min_len > 0 else 0.0


def numeric_overlap_score(nums1: Set[str], nums2: Set[str]) -> float:
    """Score matching between numbers extracted from addresses or names."""
    if not nums1 and not nums2:
        return 0.5  # Neutral when neither has numbers
    if not nums1 or not nums2:
        return 0.2  # Slight penalty if one has numbers and other doesn't
    intersection = len(nums1 & nums2)
    if intersection > 0:
        return 1.0
    return 0.0


def compute_name_similarity(name1: str, name2: str, core1: List[str], core2: List[str]) -> float:
    """
    Combined name similarity using token ratio and token sort ratio.
    """
    if not name1 or not name2:
        return 0.0
    if name1 == name2:
        return 1.0
        
    # Levenshtein ratio on cleaned string
    lev_ratio = fuzz.ratio(name1, name2) / 100.0
    # Token sort ratio to tolerate word transposition
    sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
    # Jaccard on core tokens
    jaccard = token_jaccard(core1, core2)
    # Overlap on core tokens
    overlap = token_overlap_ratio(core1, core2)
    
    return 0.35 * lev_ratio + 0.35 * sort_ratio + 0.15 * jaccard + 0.15 * overlap


def compute_address_similarity(addr1: str, addr2: str, tokens1: List[str], tokens2: List[str]) -> float:
    """
    Combined address similarity handling partial and missing addresses.
    """
    if not addr1 and not addr2:
        return 0.5  # Both empty
    if not addr1 or not addr2:
        return 0.2  # One missing
    if addr1 == addr2:
        return 1.0
        
    sort_ratio = fuzz.token_set_ratio(addr1, addr2) / 100.0
    jaccard = token_jaccard(tokens1, tokens2)
    overlap = token_overlap_ratio(tokens1, tokens2)
    
    return 0.50 * sort_ratio + 0.20 * jaccard + 0.30 * overlap
