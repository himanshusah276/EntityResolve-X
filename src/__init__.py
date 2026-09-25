"""
Person 1: Classical Entity Resolution + Fuzzy Matching Pipeline.
"""
from .config import config
from .preprocessing import normalize_business_name, normalize_address, normalize_country
from .blocking import InvertedIndexBlocker
from .similarity import compute_name_similarity, compute_address_similarity
from .matcher import EntityMatcher
from .evaluation import evaluate_macro_metrics, evaluate_candidate_recall
from .output_generator import write_candidate_pairs, write_matching_results
