"""
Prediction Pipeline for Person 4 Entity Resolution System.

Executes the frozen two-stage pipeline on test data:
1. Loads test_source1, test_source2, test_source3.
2. Applies conservative multi-scale preprocessing.
3. Builds multi-pass candidate index over test target pool (S2 + S3).
4. Generates Block F candidate sets (multi-pass union with adaptive fallbacks).
5. Filters candidates using Stage 1 broad filter.
6. Reranks survivors using Stage 2 multi-feature scorer.
7. Applies frozen decision threshold (0.65).
8. Exports matching_results.tsv and candidate_pairs.tsv.
9. Runs automated submission validation.
"""
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import config
from src.data_loader import load_test_data
from src.preprocessing import preprocess_dataframe, ProcessedRecord
from src.candidate_generation import CandidateIndex, generate_candidates_for_queries
from src.advanced_matching import TwoStagePipeline
from src.output import (
    write_matching_results_tsv,
    write_candidate_pairs_tsv,
    validate_submission_files,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_test_prediction(
    test_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end test prediction with the frozen pipeline configuration.
    """
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING TEST PREDICTION PIPELINE (FROZEN CONFIGURATION)")
    logger.info("=" * 70)

    # 1. Load test data
    t_dir = test_dir or config.TEST_DIR
    s1_df, s2_df, s3_df, health = load_test_data(t_dir)
    all_s1_ids = list(s1_df["entity_id"].astype(str).str.strip())

    logger.info(f"Loaded test entities: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}")

    # 2. Preprocess records
    logger.info("Preprocessing test query and target records...")
    s1_records = preprocess_dataframe(s1_df)
    s2_records = preprocess_dataframe(s2_df)
    s3_records = preprocess_dataframe(s3_df)

    target_records = s2_records + s3_records
    targets_dict = {r.entity_id: r for r in target_records}
    logger.info(f"Target pool size: {len(target_records)} records")

    # 3. Build candidate index & retrieve candidates
    logger.info("Building multi-pass candidate index...")
    index = CandidateIndex(target_records)

    logger.info(f"Generating candidates using frozen strategy: {config.BLOCKING_STRATEGY}...")
    candidates = generate_candidates_for_queries(
        index=index,
        query_records=s1_records,
        blocking_method=config.BLOCKING_STRATEGY,
        max_candidates=config.MAX_CANDIDATES_PER_SOURCE1,
    )
    total_candidates = sum(len(c) for c in candidates.values())
    avg_cands = total_candidates / len(s1_records) if s1_records else 0.0
    logger.info(f"Generated {total_candidates} candidates (Average {avg_cands:.1f} per S1 entity)")

    # 4. Two-Stage Matching & Decision Logic
    logger.info(
        f"Running Two-Stage Pipeline (Stage 1 min={config.STAGE1_MIN_SCORE}, "
        f"Stage 2 thresh={config.STAGE2_DECISION_THRESHOLD})..."
    )
    pipeline = TwoStagePipeline(
        target_records=targets_dict,
        stage1_threshold=config.STAGE1_MIN_SCORE,
        stage2_threshold=config.STAGE2_DECISION_THRESHOLD,
        stage2_weights=config.STAGE2_WEIGHTS,
    )

    predictions: Dict[str, List[str]] = {}
    scored_pairs_all: Dict[str, List[tuple]] = {}

    for q in s1_records:
        c_ids = candidates.get(q.entity_id, [])
        preds, scored_pairs = pipeline.predict_entity(q, c_ids)
        predictions[q.entity_id] = preds
        scored_pairs_all[q.entity_id] = scored_pairs

    total_predicted = sum(len(p) for p in predictions.values())
    singletons = sum(1 for p in predictions.values() if len(p) == 0)
    logger.info(
        f"Inference complete: Total matches={total_predicted}, "
        f"Singletons={singletons}, Non-singletons={len(s1_records) - singletons}"
    )

    # 5. Export TSVs
    out_dir = output_dir or config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    match_path = out_dir / "matching_results.tsv"
    cand_path = out_dir / "candidate_pairs.tsv"

    write_matching_results_tsv(predictions, all_s1_ids, output_path=match_path)
    write_candidate_pairs_tsv(candidates, all_s1_ids, output_path=cand_path)

    # 6. Automated Validation
    logger.info("Running automated submission validation...")
    val_report = validate_submission_files(
        matching_results_path=match_path,
        candidate_pairs_path=cand_path,
        test_source1_path=t_dir / "test_source1.tsv",
        test_source2_path=t_dir / "test_source2.tsv",
        test_source3_path=t_dir / "test_source3.tsv",
    )

    elapsed = time.time() - start_time
    logger.info(f"Test prediction and validation completed successfully in {elapsed:.2f} seconds.")

    return {
        "matching_tsv": str(match_path),
        "candidate_tsv": str(cand_path),
        "total_source1_entities": len(s1_records),
        "total_predictions": total_predicted,
        "singleton_count": singletons,
        "runtime_seconds": elapsed,
        "validation_report": val_report,
    }


if __name__ == "__main__":
    run_test_prediction()
