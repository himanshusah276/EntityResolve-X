"""Validation and threshold tuning script using training data."""
import time
from pathlib import Path
from business_entity_resolution.src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    WEIGHT_NAME, WEIGHT_ADDRESS, WEIGHT_NUMERIC
)
from business_entity_resolution.src.data_loader import load_tsv_records, load_ground_truth
from business_entity_resolution.src.matcher import EntityMatcher
from business_entity_resolution.src.evaluation import (
    evaluate_predictions, evaluate_candidate_recall, calculate_entity_metrics
)
from business_entity_resolution.src.similarity import compute_pair_similarity
from business_entity_resolution.src.preprocessing import preprocess_record


def run_validation(sample_size: int = 5000):
    print("=" * 75)
    print(f"RUNNING VALIDATION & TUNING EXPERIMENT (Sample S1: {sample_size})")
    print("=" * 75)

    start_t = time.time()

    # 1. Load Ground Truth
    print("Loading ground truth...")
    full_gt = load_ground_truth(TRAIN_GT)

    # 2. Select validation S1 sample
    print(f"Loading first {sample_size} Source 1 records for validation...")
    val_s1 = load_tsv_records(TRAIN_S1, limit=sample_size)
    val_s1_dict = {r.entity_id: r for r in val_s1}
    val_gt = {r.entity_id: full_gt.get(r.entity_id, []) for r in val_s1}

    # Find which target IDs are actually needed to evaluate recall
    needed_gt_targets = set()
    for targets in val_gt.values():
        needed_gt_targets.update(targets)
    print(f"Ground truth matches in this sample: {len(needed_gt_targets)}")

    # 3. Load Target Records (S2 + S3)
    # To test realistically, load a large pool (e.g. 50,000 S2 + 50,000 S3 + ensure GT targets are included)
    print("Loading target records pool (50,000 S2 + 50,000 S3)...")
    s2_pool = load_tsv_records(TRAIN_S2, limit=50000)
    s3_pool = load_tsv_records(TRAIN_S3, limit=50000)
    target_pool_dict = {r.entity_id: r for r in (s2_pool + s3_pool)}

    # Ensure missing ground-truth target records are added into the pool so recall can be faithfully tested
    missing_targets = needed_gt_targets - set(target_pool_dict.keys())
    if missing_targets:
        print(f"Fetching {len(missing_targets)} ground truth target records from full datasets...")
        for path in [TRAIN_S2, TRAIN_S3]:
            records = load_tsv_records(path)
            for r in records:
                if r.entity_id in missing_targets:
                    target_pool_dict[r.entity_id] = r
                    missing_targets.remove(r.entity_id)
            if not missing_targets:
                break

    all_targets = list(target_pool_dict.values())
    print(f"Total target records indexed: {len(all_targets)}")

    # 4. Build Index
    matcher = EntityMatcher(
        threshold=0.80,
        w_name=WEIGHT_NAME,
        w_addr=WEIGHT_ADDRESS,
        w_num=WEIGHT_NUMERIC
    )
    matcher.index_target_records(all_targets)

    # 5. Candidate Generation
    print("Generating candidates for validation S1 queries...")
    for q in val_s1:
        preprocess_record(q)

    val_candidates = {}
    for q in val_s1:
        if q.country in matcher.country_indexes:
            cands = matcher.country_indexes[q.country].retrieve_candidates(q, max_candidates_per_query=60)
            val_candidates[q.entity_id] = sorted(list(cands))
        else:
            val_candidates[q.entity_id] = []

    cand_recall = evaluate_candidate_recall(val_gt, val_candidates)
    avg_cands = sum(len(c) for c in val_candidates.values()) / len(val_candidates)
    print(f"\n[BLOCKING EVALUATION]")
    print(f"Candidate Recall (Upper Bound): {cand_recall * 100:.2f}%")
    print(f"Average Candidates per S1 Query: {avg_cands:.2f}")

    # 6. Precompute candidate pair similarities for fast grid evaluation
    print("\nPrecomputing candidate pair similarity scores...")
    pair_scores = []  # list of (s1_id, cand_id, score)
    for q in val_s1:
        s1_id = q.entity_id
        for cid in val_candidates[s1_id]:
            cand_rec = matcher.target_records.get(cid)
            if cand_rec:
                score = compute_pair_similarity(
                    q, cand_rec,
                    w_name=WEIGHT_NAME,
                    w_addr=WEIGHT_ADDRESS,
                    w_num=WEIGHT_NUMERIC
                )
                pair_scores.append((s1_id, cid, score))

    # 7. Evaluate over a grid of thresholds
    thresholds = [0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    print("\n" + "=" * 65)
    print(f"{'Threshold':<10} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Macro F0.5':<12}")
    print("-" * 65)

    best_thresh = 0.82
    best_f0_5 = -1.0
    best_metrics = None

    for thresh in thresholds:
        preds = {s1_id: [] for s1_id in val_s1_dict}
        for s1_id, cid, score in pair_scores:
            if score >= thresh:
                preds[s1_id].append(cid)

        metrics = evaluate_predictions(val_gt, preds)
        p = metrics["macro_precision"]
        r = metrics["macro_recall"]
        f0_5 = metrics["macro_f0_5"]

        print(f"{thresh:<10.2f} | {p*100:<15.2f}% | {r*100:<13.2f}% | {f0_5*100:<11.2f}%")

        if f0_5 > best_f0_5:
            best_f0_5 = f0_5
            best_thresh = thresh
            best_metrics = metrics

    print("=" * 65)
    print(f"Optimal Threshold: {best_thresh} (Macro F0.5 = {best_f0_5*100:.2f}%)")
    print(f"Singleton Accuracy: {best_metrics['singleton_accuracy']*100:.2f}%")
    print(f"Validation finished in {time.time() - start_t:.2f} seconds.")


if __name__ == "__main__":
    run_validation(sample_size=3000)
