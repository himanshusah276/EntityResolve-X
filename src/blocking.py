"""Multi-strategy inverted index blocking for candidate generation.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from .preprocessing import normalize_business_name, normalize_address, normalize_country


class InvertedIndexBlocker:
    """
    Builds inverted index over Source 2 and Source 3 records partitioned by country.
    Supports multi-strategy candidate retrieval:
      - Block 1: Exact country partitioning
      - Block 2: Informative name tokens
      - Block 3: Name character 3-grams / prefix tokens
      - Block 4: Key address tokens
      - Block 5: Address numbers (e.g. house number, pin codes)
    """    
    def __init__(self, max_token_freq: int = 5000):
        # country -> token -> list of target_entity_ids
        self.name_token_index = defaultdict(lambda: defaultdict(list))
        self.addr_token_index = defaultdict(lambda: defaultdict(list))
        self.num_index = defaultdict(lambda: defaultdict(list))
        self.max_token_freq = max_token_freq

    def index_target_record(
        self,
        entity_id: str,
        country: str,
        name_tokens: List[str],
        addr_tokens: List[str],
        numbers: Set[str]
    ):
        country_norm = normalize_country(country)
        
        # Index name tokens (length >= 3)
        for t in name_tokens:
            if len(t) >= 3:
                self.name_token_index[country_norm][t].append(entity_id)
                
        # Index key address tokens (length >= 4)
        for t in addr_tokens:
            if len(t) >= 4 and not t.isdigit():
                self.addr_token_index[country_norm][t].append(entity_id)
                
        # Index numbers
        for num in numbers:
            if len(num) >= 2:
                self.num_index[country_norm][num].append(entity_id)

    def retrieve_candidates_union(
        self,
        country: str,
        name_tokens: List[str],
        addr_tokens: List[str],
        numbers: Set[str],
        max_candidates: int = 60
    ) -> List[str]:
        """
        Retrieves candidates by taking the UNION of matches from multiple blocks,
        ranked by frequency of blocking hits.
        """
        country_norm = normalize_country(country)
        hit_counts = defaultdict(int)

        # Strategy 1: Name tokens
        for t in name_tokens:
            if len(t) >= 3 and t in self.name_token_index[country_norm]:
                posting = self.name_token_index[country_norm][t]
                if len(posting) <= self.max_token_freq:
                    for eid in posting:
                        hit_counts[eid] += 3  # Higher weight for name token match

        # Strategy 2: Address tokens
        for t in addr_tokens:
            if len(t) >= 4 and not t.isdigit() and t in self.addr_token_index[country_norm]:
                posting = self.addr_token_index[country_norm][t]
                if len(posting) <= self.max_token_freq:
                    for eid in posting:
                        hit_counts[eid] += 1        # Strategy 3: Number overlap (building numbers, postal codes)
        for num in numbers:
            if len(num) >= 2 and num in self.num_index[country_norm]:
                posting = self.num_index[country_norm][num]
                if len(posting) <= self.max_token_freq:
                    for eid in posting:
                        hit_counts[eid] += 2
            return []

        # Sort candidate IDs by hit count descending
        sorted_candidates = sorted(hit_counts.keys(), key=lambda eid: hit_counts[eid], reverse=True)
        return sorted_candidates[:max_candidates]
