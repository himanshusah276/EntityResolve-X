"""
Validation script to tune weights, thresholds, and benchmark performance.
"""
import sys
import time
from typing import Dict, List
from .config import config
from .data_loader import stream_tsv_records, load_ground_truth
from .preprocessing import normalize_business_name, normalize_address, normalize_country
from .blocking import InvertedIndexBlocker
from .similarity import compute_name_similarity, compute_address_similarity, numeric_overlap_score
from .matcher import EntityMatcher
from .evaluation import evaluate_macro_metrics, evaluate_candidate_recall


def preprocess_record(record: Dict[str, str]) -> Dict:
    name_norm, name_core, name_tokens = normalize_business_name(record.get("business_name", ""))
    addr_norm, addr_tokens, numbers = normalize_address(record.get("business_address", ""))
    country_norm = normalize_country(record.get("country", ""))
    
    return {
        "entity_id": record["entity_id"],
        "name_norm": name_norm,
        "name_core": name_core,
        "name_core_tokens": name_tokens,
        "addr_norm": addr_norm,
        "addr_tokens": addr_tokens,
        "numbers": numbers,
        "country": country_norm,
        "orig_name": record.get("business_name", ""),
        "orig_addr": record.get("business_address", ""),
    }


def run_validation(sample_size: int = 5000):
    print(f"=== Running Person 1 Validation on {sample_size} Source 1 Entities ===")
    start_time = time.time()
    
    # 1. Load ground truth for sample S1 entities
    print("Loading Ground Truth...")
    gt_all = load_ground_truth(str(config.TRAIN_GT_PATH), limit=sample_size)
    s1_ids_sample = set(gt_all.keys())
    print(f"Loaded {len(s1_ids_sample)} S1 entities for validation.")
    
    # 2. Load and preprocess Source 1 records
    print("Loading and preprocessing S1 records...")
    s1_records = {}
    for rec in stream_tsv_records(str(config.TRAIN_S1_PATH)):
        if rec["entity_id"] in s1_ids_sample:
            s1_records[rec["entity_id"]] = preprocess_record(rec)
            if len(s1_records) == len(s1_ids_sample):
                break
                
    # 3. Index target records from S2 and S3
    print("Building Inverted Index for S2 and S3...")
    blocker = InvertedIndexBlocker(max_token_freq=3000)
    target_records = {}
    
    # Stream a subset of S2 and S3 relevant for validation, plus broader background
    # Collect all true match IDs to ensure they are available for candidate recall testing
    true_target_ids = set(m for matches in gt_all.values() for m in matches)
    print(f"Total true target matches in validation set: {len(true_target_ids)}")
    
    # Stream S2
    s2_indexed = 0
    for rec in stream_tsv_records(str(config.TRAIN_S2_PATH), limit=60000):
        prep = preprocess_record(rec)
        target_records[prep["entity_id"]] = prep
        blocker.index_target_record(
            prep["entity_id"], prep["country"],
            prep["name_core_tokens"], prep["addr_tokens"], prep["numbers"]
        )
        s2_indexed += 1
        
    # Stream S3
    s3_indexed = 0
    for rec in stream_tsv_records(str(config.TRAIN_S3_PATH), limit=60000):
        prep = preprocess_record(rec)
        target_records[prep["entity_id"]] = prep
        blocker.index_target_record(
            prep["entity_id"], prep["country"],
            prep["name_core_tokens"], prep["addr_tokens"], prep["numbers"]
        )
        s3_indexed += 1
        
    print(f"Indexed {s2_indexed} S2 records and {s3_indexed} S3 records into multi-strategy blocks.")
    
    # 4. Generate candidates for each S1 entity
    print("Generating candidates via multi-strategy union blocking...")
    candidates_dict = {}
    total_candidates = 0
    for s1_id, s1_rec in s1_records.items():
        cands = blocker.retrieve_candidates_union(
            s1_rec["country"],
            s1_rec["name_core_tokens"],
            s1_rec["addr_tokens"],
            s1_rec["numbers"],
            max_candidates=config.BLOCKING_MAX_CANDIDATES_PER_S1
        )
        candidates_dict[s1_id] = cands
        total_candidates += len(cands)
        
    avg_cands = total_candidates / len(s1_records) if s1_records else 0
    print(f"Generated candidate pairs. Average candidates per S1: {avg_cands:.2f}")
    
    cand_recall = evaluate_candidate_recall(gt_all, candidates_dict)
    print(f"Candidate Recall on available target records: {cand_recall*100:.2f}%")
    
    # 5. Tune Thresholds
    print("\n--- Tuning Matching Threshold on Validation Set ---")
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]
    matcher = EntityMatcher()
    
    best_thresh = 0.70
    best_f05 = 0.0
    best_metrics = None
    
    for th in thresholds:
        preds = {}
        for s1_id, s1_rec in s1_records.items():
            cand_eids = candidates_dict.get(s1_id, [])
            cand_objs = [target_records[eid] for eid in cand_eids if eid in target_records]
            matches = matcher.match_candidates(s1_rec, cand_objs, threshold=th)
            preds[s1_id] = [m[0] for m in matches]
            
        metrics = evaluate_macro_metrics(gt_all, preds)
        print(f"Th: {th:.2f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F0.5: {metrics['f05']:.4f} | Singleton Acc: {metrics['singleton_accuracy']:.4f}")
        
        if metrics["f05"] > best_f05:
            best_f05 = metrics["f05"]
            best_thresh = th
            best_metrics = metrics
            
    print(f"\n>>> Optimal Threshold: {best_thresh} with Validation F0.5: {best_f05:.4f} <<<")
    elapsed = time.time() - start_time
    print(f"Validation completed in {elapsed:.2f}s")


if __name__ == "__main__":
    run_validation(sample_size=3000)
