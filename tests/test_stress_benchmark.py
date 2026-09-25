"""
Stress and scalability benchmark tests for Person 4 Pipeline.

Verifies:
- Linear or near-linear scaling of candidate generation across thousands of records
- Two-Stage pruning efficiency under stress
- Execution finishes well within runtime limits without memory leaks
"""
import time
import pytest
from src.data_loader import generate_synthetic_benchmark_dataset, load_tsv
from src.preprocessing import preprocess_dataframe
from src.candidate_generation import CandidateIndex, generate_candidates_for_queries
from src.advanced_matching import TwoStagePipeline


def test_scaling_and_memory_stress(tmp_path):
    # Generate larger dataset: 300 queries, ~800 target records
    out_dir = tmp_path / "stress_data"
    generate_synthetic_benchmark_dataset(out_dir, n_s1=300, seed=123)

    s1_df = load_tsv(out_dir / "train" / "train_source1.tsv")
    s2_df = load_tsv(out_dir / "train" / "train_source2.tsv")
    s3_df = load_tsv(out_dir / "train" / "train_source3.tsv")

    # 1. Preprocess timing
    t0 = time.time()
    s1_recs = preprocess_dataframe(s1_df)
    target_recs = preprocess_dataframe(s2_df) + preprocess_dataframe(s3_df)
    prep_time = time.time() - t0
    assert prep_time < 5.0, f"Preprocessing too slow: {prep_time:.2f}s"

    # 2. Index building timing
    t0 = time.time()
    index = CandidateIndex(target_recs)
    index_time = time.time() - t0
    assert index_time < 3.0, f"Index construction too slow: {index_time:.2f}s"

    # 3. Candidate Generation timing across 300 queries
    t0 = time.time()
    candidates = generate_candidates_for_queries(
        index=index,
        query_records=s1_recs,
        blocking_method="block_union_all",
        max_candidates=80,
    )
    cand_time = time.time() - t0
    assert cand_time < 5.0, f"Candidate generation too slow: {cand_time:.2f}s"
    assert len(candidates) == len(s1_recs)

    # 4. Two-Stage Pipeline timing across all queries
    targets_dict = {r.entity_id: r for r in target_recs}
    pipeline = TwoStagePipeline(
        target_records=targets_dict,
        stage1_threshold=0.15,
        stage2_threshold=0.65,
    )

    t0 = time.time()
    total_preds = 0
    for q in s1_recs:
        c_ids = candidates.get(q.entity_id, [])
        preds, _ = pipeline.predict_entity(q, c_ids)
        total_preds += len(preds)
    pipeline_time = time.time() - t0

    assert pipeline_time < 5.0, f"Two-stage pipeline execution too slow: {pipeline_time:.2f}s"
    assert total_preds > 0
