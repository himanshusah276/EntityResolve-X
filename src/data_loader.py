"""
Data loader module for Person 4 Entity Resolution Pipeline.

Loads TSVs using pandas (tab-separated), validates schema, preserves IDs,
reports data health metrics, and parses ground truth mappings.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_tsv(
    file_path: Union[str, Path],
    expected_columns: Optional[List[str]] = None,
    allow_missing_cols: bool = False,
) -> pd.DataFrame:
    """
    Load a tab-separated file (.tsv) into a pandas DataFrame.
    Preserves all IDs as exact string representations without numeric conversion.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    # All columns must be read as strings to preserve IDs and postal codes without truncation
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )

    # Validate schema if expected columns are specified
    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing and not allow_missing_cols:
            raise ValueError(
                f"Schema validation failed for {path.name}. Missing required columns: {missing}. Found: {list(df.columns)}"
            )

    return df


def inspect_data_health(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Generate a data health report: row count, unique IDs, duplicate count, missing values.
    """
    report: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
    }

    if "entity_id" in df.columns:
        n_unique = df["entity_id"].nunique()
        report["unique_entity_ids"] = n_unique
        report["duplicate_entity_ids"] = len(df) - n_unique
    elif "source1_entity_id" in df.columns:
        n_unique = df["source1_entity_id"].nunique()
        report["unique_source1_entity_ids"] = n_unique
        report["duplicate_source1_entity_ids"] = len(df) - n_unique

    # Count empty or whitespace-only cells
    empty_counts = {}
    for col in df.columns:
        empty_counts[col] = int((df[col].str.strip() == "").sum())
    report["empty_cell_counts"] = empty_counts

    if "country" in df.columns:
        report["country_counts"] = df["country"].value_counts().to_dict()

    return report


def load_ground_truth_map(
    file_path: Union[str, Path]
) -> Dict[str, List[str]]:
    """
    Loads ground truth mapping: source1_entity_id -> list of matched_entity_ids.
    Explicitly handles singletons (maps to empty list []).
    """
    df = load_tsv(file_path, expected_columns=config.GROUND_TRUTH_COLUMNS)
    gt_map: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        if not matched_str:
            gt_map[s1_id] = []
        else:
            # Comma-separated target IDs
            gt_map[s1_id] = [m.strip() for m in matched_str.split(",") if m.strip()]

    return gt_map


def load_training_data(
    train_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]], Dict[str, Any]]:
    """
    Loads train_source1, train_source2, train_source3, and train_ground_truth.
    Returns (s1_df, s2_df, s3_df, gt_map, health_summary).
    """
    base = Path(train_dir) if train_dir else config.TRAIN_DIR
    s1_path = base / "train_source1.tsv"
    s2_path = base / "train_source2.tsv"
    s3_path = base / "train_source3.tsv"
    gt_path = base / "train_ground_truth.tsv"

    s1_df = load_tsv(s1_path, expected_columns=config.SOURCE_COLUMNS)
    s2_df = load_tsv(s2_path, expected_columns=config.SOURCE_COLUMNS)
    s3_df = load_tsv(s3_path, expected_columns=config.SOURCE_COLUMNS)
    gt_map = load_ground_truth_map(gt_path)

    health_summary = {
        "source1": inspect_data_health(s1_df, "train_source1"),
        "source2": inspect_data_health(s2_df, "train_source2"),
        "source3": inspect_data_health(s3_df, "train_source3"),
        "ground_truth_entities": len(gt_map),
        "total_true_links": sum(len(m) for m in gt_map.values()),
        "singletons_count": sum(1 for m in gt_map.values() if len(m) == 0),
    }

    logger.info(
        f"Loaded training data: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}, "
        f"GT={len(gt_map)} (Links={health_summary['total_true_links']}, Singletons={health_summary['singletons_count']})"
    )

    return s1_df, s2_df, s3_df, gt_map, health_summary


def load_test_data(
    test_dir: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads test_source1, test_source2, and test_source3.
    """
    base = Path(test_dir) if test_dir else config.TEST_DIR
    s1_path = base / "test_source1.tsv"
    s2_path = base / "test_source2.tsv"
    s3_path = base / "test_source3.tsv"

    s1_df = load_tsv(s1_path, expected_columns=config.SOURCE_COLUMNS)
    s2_df = load_tsv(s2_path, expected_columns=config.SOURCE_COLUMNS)
    s3_df = load_tsv(s3_path, expected_columns=config.SOURCE_COLUMNS)

    health_summary = {
        "test_source1": inspect_data_health(s1_df, "test_source1"),
        "test_source2": inspect_data_health(s2_df, "test_source2"),
        "test_source3": inspect_data_health(s3_df, "test_source3"),
    }

    logger.info(f"Loaded test data: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}")
    return s1_df, s2_df, s3_df, health_summary


def generate_synthetic_benchmark_dataset(
    output_dir: Union[str, Path],
    n_s1: int = 150,
    seed: int = 42,
) -> None:
    """
    Generates a realistic synthetic benchmark dataset matching the exact schema and
    variations of the competition for testing and verification:
    - Multiple countries (US, India, France, Germany, etc. - open set)
    - Realistic business names with abbreviations (Pvt Ltd, Inc, Corp, Tech, Solutions)
    - Address variations (typos, abbreviations like Rd / Road, St / Street, Ave / Avenue)
    - Numeric tokens (house numbers, PIN/zip codes)
    - Singletons (no matches), 1 match, 2 matches, 3 matches
    - Both train and test partitions
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    train_dir = out_path / "train"
    test_dir = out_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    countries = ["US", "India", "France", "Germany", "Japan", "UK", "Canada"]
    base_names = [
        ("Acme Technologies", "Acme Tech"),
        ("Apex Global Logistics", "Apex Logistics Ltd"),
        ("Beacon Financial Services", "Beacon Finance Corp"),
        ("Crestview Medical Clinic", "Crestview Health Clinic"),
        ("Delta Precision Engineering", "Delta Precision Eng"),
        ("Echo Horizon Media", "Echo Media Group"),
        ("Falcon Security Systems", "Falcon Security"),
        ("Genesis BioPharm Labs", "Genesis BioPharma"),
        ("Harbor Point Retail", "Harbor Point Stores"),
        ("Ironclad Industrial Solutions", "Ironclad Ind"),
        ("Jupiter Data Systems", "Jupiter Data"),
        ("Keystone Software Labs", "Keystone Soft"),
        ("Lighthouse Creative Studio", "Lighthouse Studio"),
        ("Matrix Cloud Services", "Matrix Cloud"),
        ("Nexus Energy Solutions", "Nexus Energy"),
        ("Orion Aerospace Dynamics", "Orion Aero"),
        ("Pioneer Robotics Corp", "Pioneer Robotics"),
        ("Quantum Dynamics Inc", "Quantum Dyn"),
        ("Radiant Solar Energy", "Radiant Solar"),
        ("Summit Wealth Partners", "Summit Wealth"),
    ]

    street_types = [("Street", "St."), ("Road", "Rd"), ("Avenue", "Ave"), ("Boulevard", "Blvd")]
    cities_by_country = {
        "US": ["New York", "San Francisco", "Austin", "Seattle", "Chicago"],
        "India": ["Bengaluru", "Mumbai", "Hyderabad", "Delhi", "Pune"],
        "France": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"],
        "Germany": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Stuttgart"],
        "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama", "Nagoya"],
        "UK": ["London", "Manchester", "Birmingham", "Edinburgh", "Bristol"],
        "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
    }

    def make_address(country, num, street_name, st_type, zip_code):
        city = rng.choice(cities_by_country.get(country, ["Metropolis"]))
        return f"{num} {street_name} {st_type}, {city} {zip_code}"

    s1_rows, s2_rows, s3_rows = [], [], []
    gt_rows = []

    s2_counter = 1
    s3_counter = 1

    for i in range(1, n_s1 + 1):
        s1_id = f"S1-{i:05d}"
        country = rng.choice(countries)
        base_pair = base_names[i % len(base_names)]
        name_s1 = f"{base_pair[0]} #{i}"
        st_name = f"Oakland" if i % 2 == 0 else "Victoria"
        st_t1, st_t2 = rng.choice(street_types)
        zip_code = f"{10000 + (i * 37) % 89999}"
        house_num = f"{(i * 13) % 999 + 1}"
        addr_s1 = make_address(country, house_num, st_name, st_t1, zip_code)

        s1_rows.append({
            "entity_id": s1_id,
            "business_name": name_s1,
            "business_address": addr_s1,
            "country": country,
        })

        # Match profile: 25% singletons, 40% 1 match, 25% 2 matches, 10% 3 matches
        match_roll = rng.random()
        matched_ids = []

        if match_roll > 0.25:
            # At least 1 match
            num_matches = 1 if match_roll < 0.65 else (2 if match_roll < 0.90 else 3)
            for m in range(num_matches):
                in_s2 = (m % 2 == 0)
                # Introduce realistic noise
                noisy_name = f"{base_pair[1]} #{i}"
                noisy_addr = make_address(country, house_num, st_name, st_t2, zip_code)

                if in_s2:
                    cid = f"S2-{s2_counter:05d}"
                    s2_counter += 1
                    s2_rows.append({
                        "entity_id": cid,
                        "business_name": noisy_name,
                        "business_address": noisy_addr,
                        "country": country,
                    })
                else:
                    cid = f"S3-{s3_counter:05d}"
                    s3_counter += 1
                    s3_rows.append({
                        "entity_id": cid,
                        "business_name": noisy_name,
                        "business_address": noisy_addr,
                        "country": country,
                    })
                matched_ids.append(cid)

        gt_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids),
        })

    # Add distractor noisy records to S2 and S3 (unmatched entities)
    for k in range(n_s1 // 2):
        country = rng.choice(countries)
        d_name = f"Distractor Entity Group {k}"
        d_addr = f"{100 + k} Industrial Way, Sector {k}, 560001"
        cid_s2 = f"S2-{s2_counter:05d}"
        s2_counter += 1
        s2_rows.append({
            "entity_id": cid_s2,
            "business_name": d_name,
            "business_address": d_addr,
            "country": country,
        })
        cid_s3 = f"S3-{s3_counter:05d}"
        s3_counter += 1
        s3_rows.append({
            "entity_id": cid_s3,
            "business_name": f"{d_name} Subsidiary",
            "business_address": d_addr,
            "country": country,
        })

    # Save train files
    pd.DataFrame(s1_rows).to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows).to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows).to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(gt_rows).to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Save test files (first 40% of queries as test benchmark)
    n_test = max(20, n_s1 // 3)
    pd.DataFrame(s1_rows[:n_test]).to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    pd.DataFrame(s2_rows).to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    pd.DataFrame(s3_rows).to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    logger.info(f"Generated synthetic benchmark dataset in {output_dir}")
