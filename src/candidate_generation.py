"""
Multi-Pass Candidate Generation Module for Person 4 Entity Resolution Pipeline.

Implements:
- Block A: Country + Important Name Tokens
- Block B: Character n-grams (spelling/abbreviation tolerant)
- Block C: Informative Address Tokens
- Block D: Numeric & Postal Evidence (house numbers, postal codes)
- Block E: Rare Tokens (IDF-weighted inverted index)
- Block F: Multi-Pass Union (Recall-optimized)
- Adaptive Blocking: Dynamic fallbacks for weak/missing names or addresses.
"""
from collections import Counter, defaultdict
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from src.preprocessing import ProcessedRecord

logger = logging.getLogger(__name__)

# Stopwords and ultra-generic words to skip for high-frequency token blocking
GENERIC_STOPWORDS = {
    "the", "and", "or", "of", "in", "at", "to", "for", "a", "an", "is",
    "services", "solutions", "enterprise", "enterprises", "global", "group",
    "international", "national", "tech", "technology", "technologies",
    "india", "usa", "us", "uk", "france", "germany",
    "road", "street", "avenue", "lane", "drive", "court", "cross", "main",
    "floor", "suite", "building", "near", "opp", "opposite", "behind",
}


class CandidateIndex:
    """
    Multi-pass inverted index built from target records (Source 2 and Source 3).
    Supports isolated per-block retrieval and unified adaptive retrieval.
    """

    def __init__(self, target_records: List[ProcessedRecord]):
        self.target_records: Dict[str, ProcessedRecord] = {r.entity_id: r for r in target_records}
        self.total_targets = len(target_records)

        # Inverted index mappings: (country, key) -> Set[target_id]
        self.idx_name_tokens: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.idx_name_ngrams: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.idx_addr_tokens: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.idx_postal_codes: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.idx_street_numbers: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.idx_rare_tokens: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # Inverted index for fallback: country -> Set[target_id]
        self.idx_country: Dict[str, Set[str]] = defaultdict(set)

        # Compute document frequencies for rare-token blocking (Block E)
        self.doc_freq_name: Counter = Counter()
        self.doc_freq_addr: Counter = Counter()

        self._build_indexes(target_records)

    def _build_indexes(self, records: List[ProcessedRecord]) -> None:
        """Populates all inverted indexes and calculates token document frequencies."""
        for r in records:
            c = r.country_norm
            self.idx_country[c].add(r.entity_id)

            # Document frequencies
            for tok in set(r.name_tokens):
                self.doc_freq_name[tok] += 1
            for tok in set(r.address_tokens):
                self.doc_freq_addr[tok] += 1

        # Rare token threshold: appears in <= 10 records or <= 1% of corpus
        rare_max_count = max(2, int(0.01 * self.total_targets)) if self.total_targets > 100 else 10

        for r in records:
            c = r.country_norm
            eid = r.entity_id

            # Block A: Name tokens (excluding generic stopwords)
            for tok in r.name_tokens:
                if len(tok) >= 3 and tok not in GENERIC_STOPWORDS:
                    self.idx_name_tokens[(c, tok)].add(eid)

            # Block B: Name 3-grams
            for ng in r.name_ngrams:
                self.idx_name_ngrams[(c, ng)].add(eid)

            # Block C: Informative Address tokens
            for tok in r.address_tokens:
                if len(tok) >= 3 and tok not in GENERIC_STOPWORDS:
                    self.idx_addr_tokens[(c, tok)].add(eid)

            # Block D: Numeric/Postal evidence
            for post in r.postal_candidates:
                self.idx_postal_codes[(c, post)].add(eid)
            if r.street_number:
                self.idx_street_numbers[(c, r.street_number)].add(eid)

            # Block E: Rare tokens
            for tok in r.name_tokens:
                if self.doc_freq_name[tok] <= rare_max_count and tok not in GENERIC_STOPWORDS:
                    self.idx_rare_tokens[(c, tok)].add(eid)
            for tok in r.address_tokens:
                if self.doc_freq_addr[tok] <= rare_max_count and tok not in GENERIC_STOPWORDS:
                    self.idx_rare_tokens[(c, tok)].add(eid)

        logger.info(
            f"Built CandidateIndex with {len(records)} targets. "
            f"Name keys={len(self.idx_name_tokens)}, Ngram keys={len(self.idx_name_ngrams)}, "
            f"Addr keys={len(self.idx_addr_tokens)}, Postal keys={len(self.idx_postal_codes)}"
        )

    def retrieve_block_a(self, query: ProcessedRecord) -> Set[str]:
        """Block A: Country + Important Name Token."""
        c = query.country_norm
        candidates: Set[str] = set()
        for tok in query.name_tokens:
            if len(tok) >= 3 and tok not in GENERIC_STOPWORDS:
                candidates.update(self.idx_name_tokens.get((c, tok), ()))
        return candidates

    def retrieve_block_b(self, query: ProcessedRecord, min_common_ngrams: int = 2) -> Set[str]:
        """
        Block B: Character n-grams.
        Requires at least `min_common_ngrams` shared trigrams to avoid explosive candidate sets.
        """
        c = query.country_norm
        ngram_counts: Counter = Counter()
        for ng in query.name_ngrams:
            for eid in self.idx_name_ngrams.get((c, ng), ()):
                ngram_counts[eid] += 1

        return {eid for eid, count in ngram_counts.items() if count >= min_common_ngrams}

    def retrieve_block_c(self, query: ProcessedRecord) -> Set[str]:
        """Block C: Informative Address Tokens."""
        c = query.country_norm
        candidates: Set[str] = set()
        for tok in query.address_tokens:
            if len(tok) >= 3 and tok not in GENERIC_STOPWORDS:
                candidates.update(self.idx_addr_tokens.get((c, tok), ()))
        return candidates

    def retrieve_block_d(self, query: ProcessedRecord) -> Set[str]:
        """Block D: Numeric / Address Evidence (postal code OR street number agreement)."""
        c = query.country_norm
        candidates: Set[str] = set()

        for post in query.postal_candidates:
            candidates.update(self.idx_postal_codes.get((c, post), ()))

        if query.street_number:
            candidates.update(self.idx_street_numbers.get((c, query.street_number), ()))

        return candidates

    def retrieve_block_e(self, query: ProcessedRecord) -> Set[str]:
        """Block E: Rare business-name and address tokens."""
        c = query.country_norm
        candidates: Set[str] = set()
        for tok in query.name_tokens + query.address_tokens:
            candidates.update(self.idx_rare_tokens.get((c, tok), ()))
        return candidates

    def retrieve_multipass_union(
        self,
        query: ProcessedRecord,
        max_candidates: int = 80,
        enable_adaptive: bool = True,
    ) -> List[str]:
        """
        Block F: Multi-Pass Union + Adaptive Blocking.
        Combines Blocks A, B, C, D, E.
        Applies adaptive fallbacks if name or address is weak or missing.
        Scores candidate frequencies across blocks to select top candidates up to max_candidates.
        """
        c = query.country_norm
        # Track how many blocks retrieved each candidate as an initial relevance signal
        candidate_block_votes: Counter = Counter()

        # Check for weak name / weak address
        is_name_weak = len(query.name_tokens) <= 1 or len(query.clean_name) <= 3
        is_addr_weak = len(query.address_tokens) <= 1 or len(query.clean_address) <= 4

        # Adaptive strategy selection
        if enable_adaptive and is_name_weak and not is_addr_weak:
            # Name-weak fallback: rely more heavily on address, postal, and rare address tokens
            for eid in self.retrieve_block_c(query):
                candidate_block_votes[eid] += 2
            for eid in self.retrieve_block_d(query):
                candidate_block_votes[eid] += 3
            for eid in self.retrieve_block_e(query):
                candidate_block_votes[eid] += 2
            # Light name ngrams
            for eid in self.retrieve_block_b(query, min_common_ngrams=1):
                candidate_block_votes[eid] += 1

        elif enable_adaptive and is_addr_weak and not is_name_weak:
            # Address-weak fallback: rely heavily on name tokens, n-grams, and rare name tokens
            for eid in self.retrieve_block_a(query):
                candidate_block_votes[eid] += 3
            for eid in self.retrieve_block_b(query, min_common_ngrams=2):
                candidate_block_votes[eid] += 2
            for eid in self.retrieve_block_e(query):
                candidate_block_votes[eid] += 2

        elif enable_adaptive and is_name_weak and is_addr_weak:
            # Both weak: cast broad net across country n-grams and any token present
            for eid in self.retrieve_block_b(query, min_common_ngrams=1):
                candidate_block_votes[eid] += 1
            for eid in self.retrieve_block_c(query):
                candidate_block_votes[eid] += 1
            for eid in self.retrieve_block_d(query):
                candidate_block_votes[eid] += 1
            # If still empty, pull first few targets in same country
            if not candidate_block_votes and self.idx_country.get(c):
                for eid in list(self.idx_country[c])[:max_candidates]:
                    candidate_block_votes[eid] += 1

        else:
            # Standard multi-pass union across all 5 blocks
            for eid in self.retrieve_block_a(query):
                candidate_block_votes[eid] += 2
            for eid in self.retrieve_block_b(query, min_common_ngrams=2):
                candidate_block_votes[eid] += 1
            for eid in self.retrieve_block_c(query):
                candidate_block_votes[eid] += 1
            for eid in self.retrieve_block_d(query):
                candidate_block_votes[eid] += 1
            for eid in self.retrieve_block_e(query):
                candidate_block_votes[eid] += 2

        if not candidate_block_votes:
            return []

        # Sort by vote count descending, take up to max_candidates
        sorted_candidates = [
            eid for eid, _ in candidate_block_votes.most_common(max_candidates)
        ]
        return sorted_candidates


def generate_candidates_for_queries(
    index: CandidateIndex,
    query_records: List[ProcessedRecord],
    blocking_method: str = "block_union_all",
    max_candidates: int = 80,
) -> Dict[str, List[str]]:
    """
    Generates candidates for a list of query records using the specified blocking method:
    - 'block_a': Block A (name tokens)
    - 'block_b': Block B (character n-grams)
    - 'block_c': Block C (address tokens)
    - 'block_d': Block D (numeric/postal)
    - 'block_e': Block E (rare tokens)
    - 'block_union_all': Block F (multi-pass union with adaptive blocking)
    """
    results: Dict[str, List[str]] = {}

    for q in query_records:
        if blocking_method == "block_a":
            cands = list(index.retrieve_block_a(q))[:max_candidates]
        elif blocking_method == "block_b":
            cands = list(index.retrieve_block_b(q))[:max_candidates]
        elif blocking_method == "block_c":
            cands = list(index.retrieve_block_c(q))[:max_candidates]
        elif blocking_method == "block_d":
            cands = list(index.retrieve_block_d(q))[:max_candidates]
        elif blocking_method == "block_e":
            cands = list(index.retrieve_block_e(q))[:max_candidates]
        elif blocking_method == "block_union_all":
            cands = index.retrieve_multipass_union(q, max_candidates=max_candidates, enable_adaptive=True)
        else:
            raise ValueError(f"Unknown blocking method: {blocking_method}")

        results[q.entity_id] = sorted(cands)

    return results
