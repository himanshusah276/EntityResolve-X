"""Fast validation experiment reading target records on demand."""
import time
import csv
from business_entity_resolution.src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    WEIGHT_NAME, WEIGHT_ADDRESS, WEIGHT_NUMERIC
)
from business_entity_resolution.src.data_loader import Record
from business_entity_resolution.src.matcher import EntityMatcher
from business_entity_resolution.src.evaluation import (
    evaluate_predictions, evaluate_candidate_recall
)
from business_entity_resolution.src.similarity import compute_pair_similarity
from business_entity_resolution.src.preprocessing import preprocess_record


def run_fast_validation(sample_size=1000):
    start_t = time.time()
    print("=" * 70)
    print(f"RUNNING STREAMING VALIDATION EXPERIMENT (S1 sample = {sample_size})")
    print("=" * 70)

    # 1. Load S1 query sample
    s1_records = []
    val_gt = {}
    with open(TRAIN_S1, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i >= sample_size: break
            s1_records.append(Record(row[0], row[1], row[2], row[3]))

    s1_ids = {r.entity_id for r in s1_records}

    # 2. Load ground truth for these S1 queries
    needed_targets = set()
    with open(TRAIN_GT, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if row[0] in s1_ids:
                matches = [m.strip() for m in row[1].split(',') if m.strip()] if len(row) > 1 and row[1].strip() else []
                val_gt[row[0]] = matches
                needed_targets.update(matches)

    print(f"Loaded {len(s1_records)} S1 queries with {len(needed_targets)} true matches.")

    # 3. Load targets: sample 30,000 S2 and 30,000 S3 + needed_targets
    targets_dict = {}
    found_needed = 0

    print("Loading target records pool...")
    for path, prefix in [(TRAIN_S2, 'S2'), (TRAIN_S3, 'S3')]:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            count = 0
            for row in reader:
                eid, name, addr, country = row[0], row[1], row[2], row[3]
                is_needed = eid in needed_targets
                if is_needed:
                    found_needed += 1
                if count < 30000 or is_needed:
                    targets_dict[eid] = Record(eid, name, addr, country)
                    count += 1
                if count >= 30000 and found_needed >= len(needed_targets):
                    break

    target_records = list(targets_dict.values())
    print(f"Total target records indexed: {len(target_records)}")

    # 4. Build index
    matcher = EntityMatcher(
        threshold=0.82,
        w_name=WEIGHT_NAME,
        w_addr=WEIGHT_ADDRESS,
        w_num=WEIGHT_NUMERIC
    )
    matcher.index_target_records(target_records)

    # 5. Candidate Generation
    print("Generating candidates for S1 queries...")
    for q in s1_records:
        preprocess_record(q)

    val_candidates = {}
    for q in s1_records:
        if q.country in matcher.country_indexes:
            cands = matcher.country_indexes[q.country].retrieve_candidates(q, max_candidates_per_query=60)
            val_candidates[q.entity_id] = sorted(list(cands))
        else:
            val_candidates[q.entity_id] = []

    cand_recall = evaluate_candidate_recall(val_gt, val_candidates)
    avg_cands = sum(len(c) for c in val_candidates.values()) / len(val_candidates)
    print("\n[BLOCKING PERFORMANCE]")
    print(f"Candidate Recall (Upper Bound): {cand_recall * 100:.2f}%")
    print(f"Average Candidates per S1 Query: {avg_cands:.2f}")

    # 6. Precompute candidate pair similarities for grid threshold search
    print("\nPrecomputing similarities for candidate pairs...")
    pair_scores = []
    for q in s1_records:
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

    # 7. Grid search thresholds
    thresholds = [0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
    print("\n" + "=" * 65)
    print(f"{'Threshold':<10} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Macro F0.5':<12}")
    print("-" * 65)

    best_thresh = 0.82
    best_f0_5 = -1.0
    best_metrics = None

    for thresh in thresholds:
        preds = {r.entity_id: [] for r in s1_records}
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
    print(f"Macro Precision at optimal: {best_metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall at optimal: {best_metrics['macro_recall']*100:.2f}%")
    print(f"Singleton Accuracy: {best_metrics['singleton_accuracy']*100:.2f}%")
    print(f"Total time: {time.time() - start_t:.2f} seconds.")

if __name__ == '__main__':
    run_fast_validation()
