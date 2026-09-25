"""
Standalone Submission Validator for Amazon ML Unstop Business Entity Resolution Challenge.

Can be run directly from CLI:
    python utils/validate_submission.py --matching outputs/matching_results.tsv --candidates outputs/candidate_pairs.tsv

Validates:
1. File format (tab-separated TSV, correct column names)
2. Every Source 1 test entity appears exactly once
3. No duplicate predicted IDs per entity
4. No Source 1 IDs in predictions
5. Every predicted ID exists in test Source 2 or Source 3
6. Every prediction is a strict subset of the candidate set
7. Singletons have empty matched_entity_ids
"""
import argparse
import sys
from pathlib import Path
import pandas as pd


def validate(
    matching_tsv: str,
    candidates_tsv: str,
    test_s1_tsv: str,
    test_s2_tsv: str,
    test_s3_tsv: str,
) -> bool:
    print("=" * 60)
    print("RUNNING SUBMISSION VALIDATION")
    print("=" * 60)

    # 1. Load test ground entity IDs
    for p, label in [(test_s1_tsv, "Test S1"), (test_s2_tsv, "Test S2"), (test_s3_tsv, "Test S3")]:
        if not Path(p).is_file():
            print(f"[ERROR] {label} file not found: {p}")
            return False

    df_s1 = pd.read_csv(test_s1_tsv, sep="\t", dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(test_s2_tsv, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(test_s3_tsv, sep="\t", dtype=str, keep_default_na=False)

    s1_ids = list(df_s1["entity_id"].astype(str).str.strip())
    s1_set = set(s1_ids)
    s2_set = set(df_s2["entity_id"].astype(str).str.strip())
    s3_set = set(df_s3["entity_id"].astype(str).str.strip())
    valid_targets = s2_set | s3_set

    print(f"Loaded reference entities: S1={len(s1_ids)}, S2={len(s2_set)}, S3={len(s3_set)}")

    # 2. Check Candidate Pairs TSV
    if not Path(candidates_tsv).is_file():
        print(f"[ERROR] Candidate pairs file not found: {candidates_tsv}")
        return False

    df_cand = pd.read_csv(candidates_tsv, sep="\t", dtype=str, keep_default_na=False)
    expected_cand_cols = ["source1_entity_id", "candidate_entity_ids"]
    if list(df_cand.columns) != expected_cand_cols:
        print(f"[ERROR] Candidate TSV columns invalid. Expected {expected_cand_cols}, got {list(df_cand.columns)}")
        return False

    cand_ids = list(df_cand["source1_entity_id"].astype(str).str.strip())
    if len(cand_ids) != len(s1_ids):
        print(f"[ERROR] Candidate TSV row count mismatch. Expected {len(s1_ids)}, got {len(cand_ids)}")
        return False
    if cand_ids != s1_ids:
        print("[ERROR] Candidate TSV source1_entity_id list or order differs from test_source1.tsv")
        return False

    cand_map = {}
    for _, row in df_cand.iterrows():
        eid = str(row["source1_entity_id"]).strip()
        raw_cands = str(row["candidate_entity_ids"]).strip()
        clist = [c.strip() for c in raw_cands.split(",") if c.strip()]
        if len(clist) != len(set(clist)):
            print(f"[ERROR] Duplicate candidates for entity {eid}")
            return False
        cand_map[eid] = set(clist)

    print(f"[OK] Candidate pairs TSV passed structure checks ({len(df_cand)} entities).")

    # 3. Check Matching Results TSV
    if not Path(matching_tsv).is_file():
        print(f"[ERROR] Matching results file not found: {matching_tsv}")
        return False

    df_match = pd.read_csv(matching_tsv, sep="\t", dtype=str, keep_default_na=False)
    expected_match_cols = ["source1_entity_id", "matched_entity_ids"]
    if list(df_match.columns) != expected_match_cols:
        print(f"[ERROR] Matching TSV columns invalid. Expected {expected_match_cols}, got {list(df_match.columns)}")
        return False

    match_ids = list(df_match["source1_entity_id"].astype(str).str.strip())
    if len(match_ids) != len(s1_ids):
        print(f"[ERROR] Matching TSV row count mismatch. Expected {len(s1_ids)}, got {len(match_ids)}")
        return False
    if match_ids != s1_ids:
        print("[ERROR] Matching TSV source1_entity_id list or order differs from test_source1.tsv")
        return False

    total_preds = 0
    total_singletons = 0

    for _, row in df_match.iterrows():
        eid = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matched_entity_ids"]).strip()
        mlist = [m.strip() for m in raw_matches.split(",") if m.strip()]

        if not mlist:
            total_singletons += 1
            continue

        # Check duplicates
        if len(mlist) != len(set(mlist)):
            print(f"[ERROR] Duplicate predictions for entity {eid}: {mlist}")
            return False

        # Check no self-matching
        if eid in mlist:
            print(f"[ERROR] Source 1 entity {eid} found in its own match list!")
            return False

        # Check validity & candidate subset
        allowed_candidates = cand_map.get(eid, set())
        for target in mlist:
            if target not in valid_targets:
                print(f"[ERROR] Target {target} for {eid} does NOT exist in test S2 or S3!")
                return False
            if target in s1_set:
                print(f"[ERROR] Target {target} is a Source 1 entity ID, which is strictly prohibited!")
                return False
            if target not in allowed_candidates:
                print(f"[ERROR] Target {target} for {eid} is NOT in candidate pair list!")
                return False

        total_preds += len(mlist)

    print(f"[OK] Matching results TSV passed all constraint checks.")
    print(f"Summary: Total S1 Entities={len(s1_ids)}, Total Matches={total_preds}, Singletons={total_singletons}")
    print("=" * 60)
    print("SUBMISSION VERIFICATION: PASSED (ALL HARD CONSTRAINTS MET)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate submission TSVs for entity resolution challenge")
    parser.add_argument("--matching", default="outputs/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidates", default="outputs/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-s1", default="data/test/test_source1.tsv", help="Path to test_source1.tsv")
    parser.add_argument("--test-s2", default="data/test/test_source2.tsv", help="Path to test_source2.tsv")
    parser.add_argument("--test-s3", default="data/test/test_source3.tsv", help="Path to test_source3.tsv")

    args = parser.parse_args()
    success = validate(
        matching_tsv=args.matching,
        candidates_tsv=args.candidates,
        test_s1_tsv=args.test_s1,
        test_s2_tsv=args.test_s2,
        test_s3_tsv=args.test_s3,
    )
    sys.exit(0 if success else 1)
