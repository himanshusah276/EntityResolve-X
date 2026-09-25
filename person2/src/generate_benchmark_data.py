"""
Benchmark Data Generator for Person 2 Pipeline Development
Generates realistic preprocessed entity records, candidate pairs from Person 1's blocking,
and ground-truth pairs for supervised model training and evaluation.
"""

import os
import random
from pathlib import Path
import polars as pl

# Reproducible seed
random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Sample base entities
COMPANIES = [
    ("Amazon Web Services LLC", "410 Terry Ave N, Seattle, WA 98109", "US"),
    ("Microsoft Corporation", "One Microsoft Way, Redmond, WA 98052", "US"),
    ("Google LLC", "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "US"),
    ("Apple Inc.", "One Apple Park Way, Cupertino, CA 95014", "US"),
    ("Meta Platforms Inc", "1 Hacker Way, Menlo Park, CA 94025", "US"),
    ("Walmart Inc.", "702 SW 8th St, Bentonville, AR 72716", "US"),
    ("Target Corporation", "1000 Nicollet Mall, Minneapolis, MN 55403", "US"),
    ("Siemens AG", "Werner-von-Siemens-Strasse 1, 80333 Munich", "DE"),
    ("SAP SE", "Dietmar-Hopp-Allee 16, 69190 Walldorf", "DE"),
    ("Sony Group Corporation", "1-7-1 Konan, Minato-ku, Tokyo 108-0075", "JP"),
    ("Samsung Electronics Co Ltd", "129 Samsung-ro, Yeongtong-gu, Suwon-si, Gyeonggi-do", "KR"),
    ("TCS Limited", "TCS House, Raveline Street, Fort, Mumbai 400001", "IN"),
    ("Infosys Limited", "Electronics City, Hosur Road, Bengaluru 560100", "IN"),
    ("Reliance Industries Limited", "Maker Chambers IV, 222 Nariman Point, Mumbai 400021", "IN"),
    ("Alibaba Group Holding Ltd", "969 West Wen Yi Road, Yu Hang District, Hangzhou 311121", "CN"),
    ("Tencent Holdings Ltd", "Tencent Binhai Towers, No. 33 Haitian 2nd Road, Shenzhen", "CN"),
    ("BMW Group", "Petuelring 130, 80788 Munich", "DE"),
    ("Toyota Motor Corporation", "1 Toyota-cho, Toyota City, Aichi Prefecture 471-8571", "JP"),
    ("Barclays PLC", "1 Churchill Place, London E14 5HP", "GB"),
    ("HSBC Holdings plc", "8 Canada Square, London E14 5HQ", "GB"),
    ("TotalEnergies SE", "2 Place Jean Millier, 92400 Courbevoie", "FR"),
    ("LVMH Moet Hennessy Louis Vuitton", "22 Avenue Montaigne, 75008 Paris", "FR"),
    ("Nestle S.A.", "Avenue Nestle 55, 1800 Vevey", "CH"),
    ("Novartis AG", "Lichtstrasse 35, 4056 Basel", "CH"),
    ("AstraZeneca PLC", "1 Francis Crick Avenue, Cambridge CB2 0AA", "GB"),
]

def mutate_name(name: str) -> str:
    mutations = [
        lambda s: s.lower(),
        lambda s: s.upper(),
        lambda s: s.replace("Inc.", "Inc").replace("LLC", "Ltd").replace("Corporation", "Corp"),
        lambda s: s.replace(" Co ", " Company ").replace(" Ltd", " Limited"),
        lambda s: s.replace(" Group", "").replace(" Holding", ""),
        lambda s: s + " Int.",
        lambda s: s.split()[0] if len(s.split()) > 1 else s,
    ]
    return random.choice(mutations)(name)

def mutate_address(addr: str) -> str:
    mutations = [
        lambda s: s.replace("Ave", "Avenue").replace("Pkwy", "Parkway").replace("St", "Street"),
        lambda s: s.replace("Rd", "Road").replace("Blvd", "Boulevard"),
        lambda s: s.lower(),
        lambda s: " ".join(s.split()[:-1]) if len(s.split()) > 2 else s,
        lambda s: s.replace(",", ""),
        lambda s: s.replace("Suite", "Ste").replace("Floor", "Fl"),
    ]
    return random.choice(mutations)(addr)

def generate_benchmark_dataset(num_source1: int = 1500, candidates_per_s1: int = 6):
    print(f"Generating benchmark dataset with {num_source1} Source 1 entities...")
    
    s1_records = []
    s2_records = []
    s3_records = []
    candidate_pairs = []
    ground_truth_pairs = []

    s2_id_counter = 100000000
    s3_id_counter = 500000000

    for i in range(num_source1):
        s1_id = f"S1-{700000000 + i}"
        base_name, base_addr, base_country = random.choice(COMPANIES)
        suffix = f" Branch {i % 100}" if (i % 3 == 0) else ""
        s1_name = f"{base_name}{suffix}"
        s1_addr = f"{100 + (i % 900)} {base_addr}"
        s1_records.append({
            "entity_id": s1_id,
            "business_name": s1_name,
            "business_address": s1_addr,
            "country": base_country
        })

        # Generate candidates for this S1 entity (some true matches, some hard negative candidates from blocking)
        num_pos = random.choice([1, 2])
        num_neg = candidates_per_s1 - num_pos

        # Positive matches (S2 or S3)
        for _ in range(num_pos):
            source = random.choice(["S2", "S3"])
            if source == "S2":
                s2_id_counter += 1
                cand_id = f"S2-{s2_id_counter}"
                cand_name = mutate_name(s1_name)
                cand_addr = mutate_address(s1_addr)
                s2_records.append({
                    "entity_id": cand_id,
                    "business_name": cand_name,
                    "business_address": cand_addr,
                    "country": base_country
                })
            else:
                s3_id_counter += 1
                cand_id = f"S3-{s3_id_counter}"
                cand_name = mutate_name(s1_name)
                cand_addr = mutate_address(s1_addr)
                s3_records.append({
                    "entity_id": cand_id,
                    "business_name": cand_name,
                    "business_address": cand_addr,
                    "country": base_country
                })

            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": source
            })
            ground_truth_pairs.append((s1_id, cand_id))

        # Negative matches (candidates produced by blocking that share name tokens or country, but are distinct entities)
        for _ in range(num_neg):
            source = random.choice(["S2", "S3"])
            neg_base_name, neg_base_addr, neg_base_country = random.choice(COMPANIES)
            # Maybe same country or shared token to simulate blocking candidate
            if random.random() < 0.5:
                # Share first token of S1 name + different business
                token = s1_name.split()[0]
                neg_name = f"{token} Services & Solutions {random.randint(1, 99)}"
            else:
                neg_name = f"{neg_base_name} {random.randint(10, 99)}"
            
            neg_addr = f"{random.randint(10, 999)} {neg_base_addr}"
            neg_country = base_country if random.random() < 0.7 else neg_base_country

            if source == "S2":
                s2_id_counter += 1
                cand_id = f"S2-{s2_id_counter}"
                s2_records.append({
                    "entity_id": cand_id,
                    "business_name": neg_name,
                    "business_address": neg_addr,
                    "country": neg_country
                })
            else:
                s3_id_counter += 1
                cand_id = f"S3-{s3_id_counter}"
                s3_records.append({
                    "entity_id": cand_id,
                    "business_name": neg_name,
                    "business_address": neg_addr,
                    "country": neg_country
                })

            candidate_pairs.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "candidate_source": source
            })

    # Shuffle candidates to mimic real blocking output
    random.shuffle(candidate_pairs)

    # Save to Parquet using Polars
    print("Writing source1_clean.parquet...")
    pl.DataFrame(s1_records).write_parquet(DATA_DIR / "source1_clean.parquet")

    print("Writing source2_clean.parquet...")
    pl.DataFrame(s2_records).write_parquet(DATA_DIR / "source2_clean.parquet")

    print("Writing source3_clean.parquet...")
    pl.DataFrame(s3_records).write_parquet(DATA_DIR / "source3_clean.parquet")

    print("Writing candidates.parquet...")
    cand_df = pl.DataFrame(candidate_pairs)
    cand_df.write_parquet(DATA_DIR / "candidates.parquet")

    # Save train_ground_truth.tsv
    print("Writing train_ground_truth.tsv...")
    with open(DATA_DIR / "train_ground_truth.tsv", "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_id\n")
        for s1, cand in ground_truth_pairs:
            f.write(f"{s1}\t{cand}\n")

    print("\nDataset generation completed successfully!")
    print(f"Source 1 Entities: {len(s1_records):,}")
    print(f"Source 2 Entities: {len(s2_records):,}")
    print(f"Source 3 Entities: {len(s3_records):,}")
    print(f"Total Candidate Pairs: {len(cand_df):,}")
    print(f"Ground Truth True Matches: {len(ground_truth_pairs):,}")
    print(f"Negative Candidate Pairs: {len(cand_df) - len(ground_truth_pairs):,}")
    print(f"Positive Ratio: {len(ground_truth_pairs) / len(cand_df):.2%}")

if __name__ == "__main__":
    generate_benchmark_dataset()
