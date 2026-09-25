"""
Person 2 — Supervised Feature Engineering Module
Generates deterministic, pairwise tabular features from candidate entity pairs.

Features Implemented:
- Name:
    name_jaccard, name_levenshtein, name_tfidf_cosine, name_token_overlap,
    name_character_similarity, name_exact_match, name_shared_token_count,
    name_length_ratio, name_length_difference
- Address:
    address_jaccard, address_levenshtein, address_tfidf_cosine, address_token_overlap,
    address_character_similarity, address_exact_match, address_shared_token_count,
    address_shared_numeric_token_count, address_numeric_token_overlap,
    address_length_ratio, address_length_difference
- Global / Cross-Field:
    country_match, total_shared_token_count, shared_numeric_token_count,
    numeric_token_overlap
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure project root and paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "person2" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Text Normalization & Helper Utilities
# ---------------------------------------------------------------------------

def normalize_text(text: Optional[str]) -> str:
    """Normalize text: lowercase, strip accents, collapse spaces."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_numbers(text: str) -> Set[str]:
    """Extract numeric tokens (house numbers, pin codes) stripped of leading zeros."""
    if not text:
        return set()
    nums = re.findall(r"\b\d+\b", text)
    return {n.lstrip("0") or "0" for n in nums}


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from normalized text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def fast_levenshtein_ratio(s1: str, s2: str) -> float:
    """Computes Levenshtein ratio in [0, 1] using standard DP or rapidfuzz if available."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    # Quick length heuristic
    if abs(len1 - len2) / max(len1, len2) > 0.7:
        return 0.0

    # Two-row DP buffer for memory efficiency
    prev_row = list(range(len2 + 1))
    curr_row = [0] * (len2 + 1)

    for i, c1 in enumerate(s1):
        curr_row[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row[j + 1] = min(insertions, deletions, substitutions)
        prev_row, curr_row = curr_row, prev_row

    dist = prev_row[len2]
    return 1.0 - (dist / max(len1, len2))


# ---------------------------------------------------------------------------
# Feature Engineering Class
# ---------------------------------------------------------------------------

class PairFeatureExtractor:
    """
    Computes pairwise feature vectors between Source 1 records and candidate target records.
    Fits and manages reusable sparse TF-IDF models.
    """

    FEATURE_COLUMNS = [
        # Name features
        "name_jaccard",
        "name_levenshtein",
        "name_tfidf_cosine",
        "name_token_overlap",
        "name_character_similarity",
        "name_exact_match",
        "name_shared_token_count",
        "name_length_ratio",
        "name_length_difference",
        # Address features
        "address_jaccard",
        "address_levenshtein",
        "address_tfidf_cosine",
        "address_token_overlap",
        "address_character_similarity",
        "address_exact_match",
        "address_shared_token_count",
        "address_shared_numeric_token_count",
        "address_numeric_token_overlap",
        "address_length_ratio",
        "address_length_difference",
        # Global features
        "country_match",
        "total_shared_token_count",
        "shared_numeric_token_count",
        "numeric_token_overlap",
    ]

    def __init__(self):
        self.name_vectorizer: Optional[TfidfVectorizer] = None
        self.addr_vectorizer: Optional[TfidfVectorizer] = None

    def fit_vectorizers(self, corpus_names: List[str], corpus_addrs: List[str]):
        """Fits sparse TF-IDF vectorizers on training corpus and persists them."""
        print("Fitting Name TF-IDF vectorizer (word + char-wb)...")
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            max_features=50000,
            dtype=np.float32,
        )
        self.name_vectorizer.fit(corpus_names)

        print("Fitting Address TF-IDF vectorizer (word + char-wb)...")
        self.addr_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=2,
            max_features=50000,
            dtype=np.float32,
        )
        self.addr_vectorizer.fit(corpus_addrs)

        # Save to disk
        joblib.dump(self.name_vectorizer, FEATURES_DIR / "name_tfidf.joblib")
        joblib.dump(self.addr_vectorizer, FEATURES_DIR / "address_tfidf.joblib")
        print(f"Saved fitted TF-IDF models to {FEATURES_DIR.resolve()}")

    def load_vectorizers(self):
        """Loads fitted TF-IDF vectorizers from disk."""
        name_path = FEATURES_DIR / "name_tfidf.joblib"
        addr_path = FEATURES_DIR / "address_tfidf.joblib"
        if not name_path.exists() or not addr_path.exists():
            raise FileNotFoundError("Fitted TF-IDF vectorizers not found. Fit them first.")
        self.name_vectorizer = joblib.load(name_path)
        self.addr_vectorizer = joblib.load(addr_path)

    def extract_features_batch(
        self,
        s1_names: List[str],
        s1_addrs: List[str],
        s1_countries: List[str],
        cand_names: List[str],
        cand_addrs: List[str],
        cand_countries: List[str],
    ) -> np.ndarray:
        """
        Computes pairwise numerical features for aligned parallel lists of entity fields.
        Returns a 2D float32 numpy array of shape (N, len(FEATURE_COLUMNS)).
        """
        n_samples = len(s1_names)
        if n_samples == 0:
            return np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float32)

        # Ensure vectorizers are loaded
        if self.name_vectorizer is None or self.addr_vectorizer is None:
            self.load_vectorizers()

        # 1. Compute Sparse TF-IDF Cosine Similarity
        # Normalize vectors for L2 cosine dot product
        s1_name_vecs = self.name_vectorizer.transform(s1_names)
        c_name_vecs = self.name_vectorizer.transform(cand_names)
        name_tfidf_cos = np.array(s1_name_vecs.multiply(c_name_vecs).sum(axis=1)).ravel()

        s1_addr_vecs = self.addr_vectorizer.transform(s1_addrs)
        c_addr_vecs = self.addr_vectorizer.transform(cand_addrs)
        addr_tfidf_cos = np.array(s1_addr_vecs.multiply(c_addr_vecs).sum(axis=1)).ravel()

        # Preallocate feature matrix
        X = np.zeros((n_samples, len(self.FEATURE_COLUMNS)), dtype=np.float32)

        # 2. Compute token, string, and numeric metrics row-by-row
        for i in range(n_samples):
            n1, n2 = s1_names[i], cand_names[i]
            a1, a2 = s1_addrs[i], cand_addrs[i]
            c1, c2 = s1_countries[i], cand_countries[i]

            # Tokens
            t_n1 = set(n1.split())
            t_n2 = set(n2.split())
            t_a1 = set(a1.split())
            t_a2 = set(a2.split())

            # Numbers
            nums1 = extract_numbers(a1 + " " + n1)
            nums2 = extract_numbers(a2 + " " + n2)

            # --- NAME FEATURES ---
            # Jaccard
            inter_n = len(t_n1 & t_n2)
            union_n = len(t_n1 | t_n2)
            name_jaccard = (inter_n / union_n) if union_n > 0 else 0.0

            # Token overlap
            min_n = min(len(t_n1), len(t_n2))
            name_token_overlap = (inter_n / min_n) if min_n > 0 else 0.0

            # Levenshtein
            name_lev = fast_levenshtein_ratio(n1, n2)

            # Character 3-gram similarity
            ng1 = get_char_ngrams(n1, 3)
            ng2 = get_char_ngrams(n2, 3)
            inter_ng = len(ng1 & ng2)
            union_ng = len(ng1 | ng2)
            name_char_sim = (inter_ng / union_ng) if union_ng > 0 else 0.0

            # Exact match, lengths
            name_exact = 1.0 if (n1 and n1 == n2) else 0.0
            l1, l2 = len(n1), len(n2)
            name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0
            name_len_diff = float(abs(l1 - l2))

            # --- ADDRESS FEATURES ---
            # Jaccard
            inter_a = len(t_a1 & t_a2)
            union_a = len(t_a1 | t_a2)
            addr_jaccard = (inter_a / union_a) if union_a > 0 else 0.0

            # Token overlap
            min_a = min(len(t_a1), len(t_a2))
            addr_token_overlap = (inter_a / min_a) if min_a > 0 else 0.0

            # Levenshtein
            addr_lev = fast_levenshtein_ratio(a1, a2)

            # Character 3-gram similarity
            ang1 = get_char_ngrams(a1, 3)
            ang2 = get_char_ngrams(a2, 3)
            inter_ang = len(ang1 & ang2)
            union_ang = len(ang1 | ang2)
            addr_char_sim = (inter_ang / union_ang) if union_ang > 0 else 0.0

            # Address exact match, lengths
            addr_exact = 1.0 if (a1 and a1 == a2) else 0.0
            al1, al2 = len(a1), len(a2)
            addr_len_ratio = (min(al1, al2) / max(al1, al2)) if max(al1, al2) > 0 else 1.0
            addr_len_diff = float(abs(al1 - al2))

            # Address numeric token features
            addr_nums1 = extract_numbers(a1)
            addr_nums2 = extract_numbers(a2)
            inter_addr_nums = len(addr_nums1 & addr_nums2)
            min_addr_nums = min(len(addr_nums1), len(addr_nums2))
            addr_num_overlap = (inter_addr_nums / min_addr_nums) if min_addr_nums > 0 else (0.5 if not addr_nums1 and not addr_nums2 else 0.0)

            # --- GLOBAL / CROSS-FIELD FEATURES ---
            country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
            total_shared_tokens = float(inter_n + inter_a)
            shared_nums_count = float(len(nums1 & nums2))
            min_all_nums = min(len(nums1), len(nums2))
            all_num_overlap = (shared_nums_count / min_all_nums) if min_all_nums > 0 else (0.5 if not nums1 and not nums2 else 0.0)

            # Pack row into feature matrix
            X[i, :] = [
                name_jaccard,
                name_lev,
                name_tfidf_cos[i],
                name_token_overlap,
                name_char_sim,
                name_exact,
                float(inter_n),
                name_len_ratio,
                name_len_diff,
                addr_jaccard,
                addr_lev,
                addr_tfidf_cos[i],
                addr_token_overlap,
                addr_char_sim,
                addr_exact,
                float(inter_a),
                float(inter_addr_nums),
                addr_num_overlap,
                addr_len_ratio,
                addr_len_diff,
                country_match,
                total_shared_tokens,
                shared_nums_count,
                all_num_overlap,
            ]

        return X
