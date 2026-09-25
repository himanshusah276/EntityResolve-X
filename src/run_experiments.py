"""
End-to-End Experiment Execution and Staging Benchmark for Person 4 Pipeline.

Runs and records real measurements for:
1. Blocking Experiments (Blocks A-F, candidate recall, candidate count, reduction ratio, runtime).
2. Stage-1 Broad Filter Recall and Pruning efficiency.
3. Stage-2 Detailed Reranking performance.
4. Single-Stage Baseline vs Two-Stage Pipeline comparison.
5. Decision Threshold Sweeps.
"""
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd

from src.config import config
from src.data_loader import (
    load_tsv,
    load_ground_truth_map,
    generate_synthetic_benchmark_dataset,
)
from src.preprocessing import preprocess_dataframe, ProcessedRecord
from src.candidate_generation import CandidateIndex, generate_candidates_for_queries
from src.advanced_matching import TwoStagePipeline, compute_stage2_score, extract_stage2_features
from src.evaluation import (
    create_source1_validation_split,
    compute_candidate_recall,
    compute_reduction_ratio,
    evaluate_predictions,
    record_experiment_result,
)
from src.error_analysis import generate_error_analysis_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_data_ready() -> None:
    """Ensures training data exists in data/train. Generates realistic benchmark if missing."""
    if not (config.TRAIN_S1_PATH.is_file() and config.TRAIN_GT_PATH.is_file()):
        logger.info(f"Dataset files not found in {config.TRAIN_DIR}. Initializing benchmark dataset...")
        generate_synthetic_benchmark_dataset(config.DATA_DIR, n_s1=250, seed=config.SEED)


def run_all_experiments() -> Dict[str, Any]:
    """Executes the full suite of blocking, staging, and threshold experiments."""
    ensure_data_ready()

    logger.info("Loading training datasets...")
    s1_df = load_tsv(config.TRAIN_S1_PATH)
    s2_df = load_tsv(config.TRAIN_S2_PATH)
    s3_df = load_tsv(config.TRAIN_S3_PATH)
    gt_map = load_ground_truth_map(config.TRAIN_GT_PATH)

    logger.info("Preprocessing records...")
    s1_records = preprocess_dataframe(s1_df)
    s2_records = preprocess_dataframe(s2_df)
    s3_records = preprocess_dataframe(s3_df)
    target_records = s2_records + s3_records
    targets_dict = {r.entity_id: r for r in target_records}

    logger.info("Splitting Source 1 queries at entity level...")
    all_s1_ids = [r.entity_id for r in s1_records]
    train_s1_ids, val_s1_ids = create_source1_validation_split(
        all_s1_ids, val_ratio=config.VAL_RATIO, seed=config.SEED
    )
    val_id_set = set(val_s1_ids)
    val_s1_records = [r for r in s1_records if r.entity_id in val_id_set]
    val_gt = {eid: gt_map.get(eid, []) for eid in val_s1_ids}

    logger.info(f"Validation set: {len(val_s1_records)} Source 1 queries, {len(target_records)} target records pool.")

    logger.info("Building multi-pass candidate index...")
    index = CandidateIndex(target_records)

    # =========================================================================
    # 1. BLOCKING EXPERIMENTS (Blocks A-F)
    # =========================================================================
    logger.info("\n" + "=" * 70 + "\nRUNNING BLOCKING EXPERIMENTS\n" + "=" * 70)
    blocking_methods = [
        ("block_a", "Block A (Country + Name Tokens)"),
        ("block_b", "Block B (Character 3-grams)"),
        ("block_c", "Block C (Informative Address Tokens)"),
        ("block_d", "Block D (Numeric/Postal Evidence)"),
        ("block_e", "Block E (Rare Tokens)"),
        ("block_union_all", "Block F (Multi-Pass Union with Adaptive Fallbacks)"),
    ]

    blocking_results = {}
    candidates_cache = {}

    for method_id, method_desc in blocking_methods:
        t0 = time.time()
        cands = generate_candidates_for_queries(
            index=index,
            query_records=val_s1_records,
            blocking_method=method_id,
            max_candidates=config.MAX_CANDIDATES_PER_SOURCE1,
        )
        runtime = time.time() - t0
        candidates_cache[method_id] = cands

        cand_metrics = compute_candidate_recall(val_gt, cands)
        rr = compute_reduction_ratio(
            n_source1=len(val_s1_records),
            n_targets_pool=len(target_records),
            total_candidates=cand_metrics["total_candidates"],
        )
        cand_metrics["reduction_ratio"] = rr
        cand_metrics["runtime_seconds"] = runtime

        blocking_results[method_id] = cand_metrics

        record_experiment_result(
            experiment_id=f"EXP_BLOCK_{method_id.upper()}",
            stage_desc=f"Blocking: {method_desc}",
            metrics=cand_metrics,
            parameters={"blocking_method": method_id, "max_candidates": config.MAX_CANDIDATES_PER_SOURCE1},
            notes=f"Reduction ratio: {rr:.5f}, Runtime: {runtime:.2f}s",
        )

        logger.info(
            f"[{method_id:<16}] Recall: {cand_metrics['candidate_recall']*100:6.2f}% | "
            f"Avg Cands: {cand_metrics['avg_candidates_per_entity']:5.1f} | "
            f"RR: {rr*100:6.2f}% | Runtime: {runtime:.2f}s"
        )

    # Use the best blocking candidates (Block F multi-pass union) for downstream matching
    val_candidates_f = candidates_cache["block_union_all"]

    # =========================================================================
    # 2. STAGE-1 BROAD FILTER EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 70 + "\nRUNNING STAGE-1 FILTER EVALUATION\n" + "=" * 70)
    pipeline = TwoStagePipeline(
        target_records=targets_dict,
        stage1_threshold=config.STAGE1_MIN_SCORE,
        stage2_threshold=config.STAGE2_DECISION_THRESHOLD,
    )

    t0 = time.time()
    stage1_survivors_dict = {}
    for q in val_s1_records:
        c_ids = val_candidates_f.get(q.entity_id, [])
        survivors = pipeline.filter_stage1(q, c_ids)
        stage1_survivors_dict[q.entity_id] = [cid for cid, _ in survivors]
    s1_filter_time = time.time() - t0

    s1_metrics = compute_candidate_recall(val_gt, stage1_survivors_dict)
    s1_pruning = 1.0 - (s1_metrics["total_candidates"] / blocking_results["block_union_all"]["total_candidates"])

    record_experiment_result(
        experiment_id="EXP_STAGE1_BROAD_FILTER",
        stage_desc="Stage-1 Broad Filter (Set operations & token overlap)",
        metrics=s1_metrics,
        parameters={"stage1_threshold": config.STAGE1_MIN_SCORE},
        notes=f"Pruned {s1_pruning*100:.1f}% candidates, Recall retention: {s1_metrics['candidate_recall']*100:.2f}%",
    )
    logger.info(
        f"Stage 1 Filter: Recall={s1_metrics['candidate_recall']*100:.2f}%, "
        f"Avg Cands={s1_metrics['avg_candidates_per_entity']:.1f}, "
        f"Pruned={s1_pruning*100:.1f}% of candidates in {s1_filter_time:.2f}s"
    )

    # =========================================================================
    # 3. SINGLE-STAGE BASELINE vs TWO-STAGE PIPELINE COMPARISON
    # =========================================================================
    logger.info("\n" + "=" * 70 + "\nRUNNING STAGING COMPARISON (Single vs Two-Stage)\n" + "=" * 70)

    # A) Single-Stage Baseline: All Block F candidates scored directly with Stage 2 (no Stage 1 filter)
    t0 = time.time()
    single_stage_preds = {}
    for q in val_s1_records:
        c_ids = val_candidates_f.get(q.entity_id, [])
        preds = []
        for cid in c_ids:
            cand = targets_dict.get(cid)
            if cand:
                feats = extract_stage2_features(q, cand)
                score = compute_stage2_score(feats)
                if score >= config.STAGE2_DECISION_THRESHOLD:
                    preds.append(cid)
        single_stage_preds[q.entity_id] = preds
    single_stage_time = time.time() - t0
    single_stage_eval = evaluate_predictions(val_gt, single_stage_preds)
    single_stage_eval["runtime_seconds"] = single_stage_time

    record_experiment_result(
        experiment_id="EXP_SINGLE_STAGE_BASELINE",
        stage_desc="Single-Stage Baseline (Candidate Gen -> Detailed Scorer)",
        metrics=single_stage_eval,
        parameters={"threshold": config.STAGE2_DECISION_THRESHOLD},
        notes=f"Direct scoring without broad filtering, Runtime: {single_stage_time:.2f}s",
    )

    # B) Two-Stage Pipeline: Candidate Gen -> Stage 1 Filter -> Stage 2 Reranker
    t0 = time.time()
    two_stage_preds = {}
    two_stage_scores = {}
    for q in val_s1_records:
        c_ids = val_candidates_f.get(q.entity_id, [])
        preds, scored_pairs = pipeline.predict_entity(q, c_ids, custom_threshold=config.STAGE2_DECISION_THRESHOLD)
        two_stage_preds[q.entity_id] = preds
        two_stage_scores[q.entity_id] = scored_pairs
    two_stage_time = time.time() - t0
    two_stage_eval = evaluate_predictions(val_gt, two_stage_preds)
    two_stage_eval["runtime_seconds"] = two_stage_time

    record_experiment_result(
        experiment_id="EXP_TWO_STAGE_PIPELINE",
        stage_desc="Two-Stage Pipeline (Candidate Gen -> Stage 1 -> Stage 2)",
        metrics=two_stage_eval,
        parameters={
            "stage1_threshold": config.STAGE1_MIN_SCORE,
            "stage2_threshold": config.STAGE2_DECISION_THRESHOLD,
        },
        notes=f"Two-stage architecture, Runtime: {two_stage_time:.2f}s",
    )

    logger.info(
        f"Single-Stage: F0.5={single_stage_eval['macro_f0_5']*100:.2f}%, "
        f"Prec={single_stage_eval['macro_precision']*100:.2f}%, "
        f"Rec={single_stage_eval['macro_recall']*100:.2f}%, "
        f"Time={single_stage_time:.2f}s"
    )
    logger.info(
        f"Two-Stage:    F0.5={two_stage_eval['macro_f0_5']*100:.2f}%, "
        f"Prec={two_stage_eval['macro_precision']*100:.2f}%, "
        f"Rec={two_stage_eval['macro_recall']*100:.2f}%, "
        f"Time={two_stage_time:.2f}s"
    )

    # =========================================================================
    # 4. DECISION THRESHOLD SWEEPS
    # =========================================================================
    logger.info("\n" + "=" * 70 + "\nRUNNING THRESHOLD SWEEP EXPERIMENTS\n" + "=" * 70)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    best_thresh = config.STAGE2_DECISION_THRESHOLD
    best_f0_5 = -1.0
    threshold_results = []

    print(f"{'Threshold':<10} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Macro F0.5':<12} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 75)

    for thresh in thresholds:
        preds = {}
        for q_id, scored_pairs in two_stage_scores.items():
            preds[q_id] = [cid for cid, score in scored_pairs if score >= thresh]

        eval_res = evaluate_predictions(val_gt, preds)
        p = eval_res["macro_precision"]
        r = eval_res["macro_recall"]
        f0_5 = eval_res["macro_f0_5"]

        print(
            f"{thresh:<10.2f} | {p*100:<15.2f}% | {r*100:<13.2f}% | {f0_5*100:<11.2f}% | "
            f"{eval_res['total_tp']:<4} | {eval_res['total_fp']:<4} | {eval_res['total_fn']:<4}"
        )

        record_experiment_result(
            experiment_id=f"EXP_THRESH_{int(thresh*100)}",
            stage_desc=f"Decision Threshold = {thresh:.2f}",
            metrics=eval_res,
            parameters={"threshold": thresh},
            notes=f"Threshold sweep on Two-Stage pipeline",
        )

        threshold_results.append((thresh, eval_res))
        if f0_5 > best_f0_5:
            best_f0_5 = f0_5
            best_thresh = thresh

    logger.info(f"\nOptimal Decision Threshold: {best_thresh:.2f} (Macro F0.5 = {best_f0_5*100:.2f}%)")

    # Generate Error Analysis Report for optimal threshold
    logger.info("\n" + "=" * 70 + "\nGENERATING ERROR ANALYSIS REPORT\n" + "=" * 70)
    best_preds = {}
    pair_scores_flat = {}
    for q_id, scored_pairs in two_stage_scores.items():
        best_preds[q_id] = [cid for cid, score in scored_pairs if score >= best_thresh]
        for cid, score in scored_pairs:
            pair_scores_flat[(q_id, cid)] = score

    generate_error_analysis_report(
        queries=val_s1_records,
        targets_dict=targets_dict,
        ground_truth=val_gt,
        predictions=best_preds,
        pair_scores_map=pair_scores_flat,
    )

    return {
        "blocking_results": blocking_results,
        "stage1_metrics": s1_metrics,
        "single_stage": single_stage_eval,
        "two_stage": two_stage_eval,
        "best_threshold": best_thresh,
        "best_f0_5": best_f0_5,
        "threshold_results": threshold_results,
    }


if __name__ == "__main__":
    run_all_experiments()
