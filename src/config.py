"""
Global configuration for Person 1: Classical Entity Resolution + Fuzzy Matching Pipeline.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR: Path = BASE_DIR / "dataset"
    OUTPUT_DIR: Path = BASE_DIR / "business_entity_resolution" / "outputs"
    
    # Input file paths
    TRAIN_S1_PATH: Path = DATASET_DIR / "train" / "train_source1.tsv"
    TRAIN_S2_PATH: Path = DATASET_DIR / "train" / "train_source2.tsv"
    TRAIN_S3_PATH: Path = DATASET_DIR / "train" / "train_source3.tsv"
    TRAIN_GT_PATH: Path = DATASET_DIR / "train" / "train_ground_truth.tsv"
    
    TEST_S1_PATH: Path = DATASET_DIR / "test" / "test_source1.tsv"
    TEST_S2_PATH: Path = DATASET_DIR / "test" / "test_source2.tsv"
    TEST_S3_PATH: Path = DATASET_DIR / "test" / "test_source3.tsv"
    
    # Output file paths
    CANDIDATE_OUTPUT_PATH: Path = OUTPUT_DIR / "candidate_pairs.tsv"
    MATCHING_OUTPUT_PATH: Path = OUTPUT_DIR / "matching_results.tsv"
    
    # Blocking parameters
    BLOCKING_MAX_CANDIDATES_PER_S1: int = 50
    MIN_TOKEN_LEN: int = 3
    
    # Weights for similarity scoring
    # final_score = w_name * name_score + w_addr * addr_score + w_num * num_score
    W_NAME: float = 0.55
    W_ADDRESS: float = 0.35
    W_NUMERIC: float = 0.10
    
    # Decision threshold for matching (tuned on validation set)
    MATCH_THRESHOLD: float = 0.72


config = Config()
