"""
Configuration settings for Person 4 Independent Entity Resolution Pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # Base paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    TRAIN_DIR: Path = DATA_DIR / "train"
    TEST_DIR: Path = DATA_DIR / "test"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIXTURES_DIR: Path = PROJECT_ROOT / "tests" / "fixtures"

    # Dataset file paths
    TRAIN_S1_PATH: Path = TRAIN_DIR / "train_source1.tsv"
    TRAIN_S2_PATH: Path = TRAIN_DIR / "train_source2.tsv"
    TRAIN_S3_PATH: Path = TRAIN_DIR / "train_source3.tsv"
    TRAIN_GT_PATH: Path = TRAIN_DIR / "train_ground_truth.tsv"

    TEST_S1_PATH: Path = TEST_DIR / "test_source1.tsv"
    TEST_S2_PATH: Path = TEST_DIR / "test_source2.tsv"
    TEST_S3_PATH: Path = TEST_DIR / "test_source3.tsv"

    # Output file paths
    MATCHING_RESULTS_PATH: Path = OUTPUTS_DIR / "matching_results.tsv"
    CANDIDATE_PAIRS_PATH: Path = OUTPUTS_DIR / "candidate_pairs.tsv"

    # Reports
    EXPERIMENTS_CSV_PATH: Path = REPORTS_DIR / "experiments.csv"
    SYSTEM_COMPARISON_CSV_PATH: Path = REPORTS_DIR / "system_comparison.csv"
    ERROR_ANALYSIS_CSV_PATH: Path = REPORTS_DIR / "error_analysis.csv"

    # FROZEN CONFIGURATION (Phase 15 Checkpoint - Locked based on empirical validation)
    # Frozen Seed
    SEED: int = 42

    # Validation split ratio
    VAL_RATIO: float = 0.2

    # Expected column schemas
    SOURCE_COLUMNS: List[str] = field(
        default_factory=lambda: ["entity_id", "business_name", "business_address", "country"]
    )
    GROUND_TRUTH_COLUMNS: List[str] = field(
        default_factory=lambda: ["source1_entity_id", "matched_entity_ids"]
    )

    # Candidate generation parameters
    BLOCKING_STRATEGY: str = "block_union_all"
    MAX_CANDIDATES_PER_SOURCE1: int = 80
    MIN_TOKEN_LENGTH: int = 3
    NGRAM_N: int = 3

    # Stage-1 and Stage-2 thresholds
    STAGE1_MIN_SCORE: float = 0.15
    STAGE2_DECISION_THRESHOLD: float = 0.65

    # Frozen Stage 2 weights
    STAGE2_WEIGHTS: dict = field(
        default_factory=lambda: {
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
    )


# Default singleton instance
config = Config()
