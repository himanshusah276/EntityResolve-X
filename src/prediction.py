"""Main prediction runner to generate test competition outputs."""
import time
from .config import (
    TEST_S1, TEST_S2, TEST_S3,
    CANDIDATE_OUTPUT_PATH, MATCHING_OUTPUT_PATH,
    MATCH_THRESHOLD, WEIGHT_NAME, WEIGHT_ADDRESS, WEIGHT_NUMERIC
)
from .data_loader import load_tsv_records, Record
from .matcher import EntityMatcher
from .output_generator import save_candidate_pairs, save_matching_results


def run_test_inference(limit_s1: int = None, limit_targets: int = None):
    start_time = time.time()
    print("=" * 70)
    print("STARTING PERSON 1 PREDICTION PIPELINE ON TEST DATA")
    print(f"Decision Threshold: {MATCH_THRESHOLD}")
    print(f"Weights: Name={WEIGHT_NAME}, Address={WEIGHT_ADDRESS}, Numeric={WEIGHT_NUMERIC}")
    print("=" * 70)

    # 1. Load Test Target Records (Source 2 and Source 3)
    print(f"Loading test target records from:\n  {TEST_S2}\n  {TEST_S3}")
    s2_records = load_tsv_records(TEST_S2, limit=limit_targets)
    s3_records = load_tsv_records(TEST_S3, limit=limit_targets)
    all_targets = s2_records + s3_records
    print(f"Loaded {len(s2_records)} S2 records, {len(s3_records)} S3 records (Total: {len(all_targets)})")

    # 2. Initialize Matcher and Index Targets
    matcher = EntityMatcher(
        threshold=MATCH_THRESHOLD,
        w_name=WEIGHT_NAME,
        w_addr=WEIGHT_ADDRESS,
        w_num=WEIGHT_NUMERIC
    )
    matcher.index_target_records(all_targets)

    # 3. Load Test Source 1 Queries
    print(f"\nLoading test query records from:\n  {TEST_S1}")
    s1_records = load_tsv_records(TEST_S1, limit=limit_s1)
    print(f"Loaded {len(s1_records)} S1 test query records.")

    # 4. Generate Candidates and Predictions
    print("\nMatching queries against indexed targets...")
    candidates_map, matches_map = matcher.match_queries(s1_records)

    # 5. Save Outputs
    print(f"\nSaving candidate pairs to: {CANDIDATE_OUTPUT_PATH}")
    save_candidate_pairs(candidates_map, CANDIDATE_OUTPUT_PATH)

    print(f"Saving matching results to: {MATCHING_OUTPUT_PATH}")
    save_matching_results(matches_map, MATCHING_OUTPUT_PATH)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Prediction pipeline completed in {elapsed:.2f} seconds!")
    print(f"Total S1 entities processed: {len(s1_records)}")
    singletons = sum(1 for m in matches_map.values() if not m)
    matched_s1 = len(s1_records) - singletons
    total_matches = sum(len(m) for m in matches_map.values())
    print(f"Entities with predicted matches: {matched_s1} ({matched_s1/len(s1_records)*100:.2f}%)")
    print(f"Singletons (no match): {singletons} ({singletons/len(s1_records)*100:.2f}%)")
    print(f"Total matches generated: {total_matches}")
    print("=" * 70)


if __name__ == "__main__":
    run_test_inference()
