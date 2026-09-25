# EntityResolve-X

> Advanced two-stage Business Entity Resolution system for matching noisy business records across multiple data sources.

## 📌 Overview

**EntityResolve-X** is an independent machine learning pipeline developed for the **Amazon ML Unstop Business Entity Resolution Challenge**.

The system identifies records from **Source 2 and Source 3** that refer to the same real-world business as a given **Source 1** entity.

Instead of relying on a single matching technique, EntityResolve-X uses a **two-stage matching architecture**:

```text
Raw Business Records
        ↓
Preprocessing
        ↓
Advanced Multi-Pass Candidate Generation
        ↓
Candidate Recall Evaluation
        ↓
Stage-1 Broad Matching
        ↓
Stage-2 Detailed Reranking
        ↓
Confidence / Decision Layer
        ↓
Source-1 Entity Aggregation
        ↓
F0.5 Evaluation
        ↓
Final Predictions
```

The system is designed to be independently useful while also supporting optional integration with other team matching systems.

---

## 🎯 Problem

Business data collected from different sources often contains inconsistent information.

For example:

```text
Source 1:
ABC Technologies Pvt Ltd
MG Road, Bangalore

Source 2:
ABC Technology Private Limited
M.G. Road Bengaluru

Source 3:
ABC Tech
Near MG Road, Bangalore
```

These records may refer to the same business even though their:

- names differ
- addresses differ
- abbreviations differ
- punctuation differs
- components may be missing
- spelling may contain noise

The goal is to determine which records represent the same real-world entity.

---

## 🚀 Key Features

### 1. Advanced Candidate Generation

Multiple blocking strategies are used to generate high-recall candidate pairs.

Possible strategies include:

- Name-token blocking
- Character n-gram blocking
- Address-token blocking
- Rare-token blocking
- Numeric/address-component blocking
- Multi-pass blocking
- Adaptive blocking

Candidate generation is evaluated separately so retrieval errors can be distinguished from matching errors.

### 2. Two-Stage Matching

#### Stage 1 — Broad Matching

A lightweight scoring stage removes clearly unlikely candidates while preserving high recall.

Possible evidence:

- token overlap
- character overlap
- address evidence
- numeric agreement
- country agreement

#### Stage 2 — Detailed Reranking

The remaining candidates are analyzed using more detailed evidence:

- business-name similarity
- address similarity
- token evidence
- numeric agreement
- country agreement
- name/address consistency
- optional teammate model scores

### 3. Entity-Level Evaluation

For every Source 1 entity:

1. Generate candidates
2. Score candidates
3. Generate predictions
4. Compare with ground truth
5. Calculate precision
6. Calculate recall
7. Calculate F0.5

Singleton entities are also included.

---

## 📊 Evaluation Metrics

The primary metric is **F0.5**:

```text
F0.5 =
1.25 × Precision × Recall
-------------------------
0.25 × Precision + Recall
```

The system also reports:

- Precision
- Recall
- F0.5
- Candidate Recall
- False Positives
- False Negatives
- Number of Predictions
- Number of Candidates
- Runtime
- Memory usage

### Candidate Recall

```text
Candidate Recall =
True matching pairs found in candidates
----------------------------------------
Total true matching pairs
```

This helps determine whether an error comes from candidate generation or final matching.

---

## 🧠 Why a Two-Stage System?

Comparing every Source 1 record against every Source 2 and Source 3 record can be computationally expensive.

Instead:

```text
All possible pairs
      ↓
Candidate generation
      ↓
Smaller candidate set
      ↓
Stage 1 filtering
      ↓
Reduced candidate set
      ↓
Detailed reranking
```

This allows the system to maintain broad retrieval while applying more detailed analysis only to promising candidates.

---

## 🔬 Experimental Framework

The project is designed around measured experiments rather than assumptions.

Candidate-generation experiments compare:

- Candidate count
- Candidate recall
- Reduction ratio
- Runtime

Matching experiments compare:

- Precision
- Recall
- F0.5
- Candidate recall
- False positives
- False negatives
- Runtime

Threshold experiments evaluate multiple decision configurations.

Experiment results are stored in:

```text
reports/experiments.csv
```

---

## 👥 Team System Integration

EntityResolve-X is designed as an independent system but supports optional integration with other team approaches.

| System | Main Approach |
|---|---|
| Person 1 | Classical fuzzy/rule-based matching |
| Person 2 | Supervised pair classification |
| Person 3 | Embedding-based semantic matching |
| Person 4 / EntityResolve-X | Multi-pass retrieval + two-stage reranking |

The outputs from other systems are treated as **optional signals** and are not required to run the independent Person 4 pipeline.

---

## 🔌 Optional Ensemble Integration

If outputs from other systems are available, they can be supplied using a common format:

```text
source1_entity_id    candidate_entity_id    score
S1-00001             S2-00047               0.91
S1-00001             S3-00012               0.74
```

Possible experiments include:

```text
Person 4
Person 4 + Person 1
Person 4 + Person 2
Person 4 + Person 3
Person 4 + Person 1 + Person 2 + Person 3
```

An ensemble is **not assumed to be better**. Each combination must be evaluated on the same validation split.

---

## 📁 Project Structure

```text
EntityResolve-X/
│
├── data/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── candidate_generation.py
│   ├── advanced_matching.py
│   ├── ensemble.py
│   ├── evaluation.py
│   ├── error_analysis.py
│   ├── prediction.py
│   └── output.py
│
├── models/
│
├── outputs/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── reports/
│   ├── experiments.csv
│   ├── error_analysis.csv
│   └── final_report.md
│
├── docs/
│   ├── PRD.md
│   ├── TRD.md
│   ├── EXPERIMENT_PLAN.md
│   ├── EVALUATION_PROTOCOL.md
│   └── INTEGRATION_PLAN.md
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## 📂 Dataset Format

All datasets are tab-separated files.

### Source files

```text
entity_id
business_name
business_address
country
```

### Ground Truth

```text
source1_entity_id
matched_entity_ids
```

Always load TSV files using:

```python
import pandas as pd

df = pd.read_csv("file.tsv", sep="\t")
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd EntityResolve-X
```

Create a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

The exact commands will follow the final implementation.

Typical development commands are:

### Load and inspect data

```bash
python src/data_loader.py
```

### Generate candidates

```bash
python src/candidate_generation.py
```

### Run matching

```bash
python src/advanced_matching.py
```

### Run evaluation

```bash
python src/evaluation.py
```

### Generate predictions

```bash
python src/prediction.py
```

The final README should be updated if the executable interface changes during development.

---

## 📤 Output

### `matching_results.tsv`

```text
source1_entity_id    matched_entity_ids
```

Example:

```text
S1-00001    S2-00047,S2-00193,S3-00812
S1-00002    S3-00004
S1-00003
```

### `candidate_pairs.tsv`

```text
source1_entity_id    candidate_entity_ids
```

Example:

```text
S1-00001    S2-00047,S2-00193,S3-00812,S3-00999
S1-00002    S3-00004
S1-00003
```

Every predicted match must exist in the candidate set.

---

## 🔍 Error Analysis

The project maintains structured error analysis.

Possible error categories include:

- Similar business names
- Shared addresses
- Abbreviations
- Numeric conflicts
- Country mismatch
- Generic business names
- Partial addresses
- Spelling noise
- Missing information

The purpose is to understand why the system makes mistakes and use those findings to guide further experiments.

---

## 🧪 Reproducibility

Experiments should record:

- Validation split
- Random seed
- Blocking configuration
- Stage-1 configuration
- Stage-2 configuration
- Threshold
- Ensemble configuration
- Evaluation metrics
- Runtime

No final configuration should be selected solely from intuition.

---

## 🔐 Data & Fair-Play

The pipeline is designed to operate using only challenge-provided data.

It must not use:

- External business databases
- Internet business searches
- Geocoding services
- Commercial entity-resolution APIs
- Government business registries
- External business datasets
- External business information

---

## ⚠️ Limitations

Potential difficult cases include:

- Extremely short business names
- Missing addresses
- Generic business names
- Heavy spelling corruption
- Multiple businesses sharing similar addresses
- Missing country information
- Highly incomplete records

Candidate generation can also limit maximum achievable recall if the correct match is never retrieved.

---

## 🛠️ Development Philosophy

The project follows:

```text
Build
  ↓
Measure
  ↓
Analyze
  ↓
Improve
  ↓
Validate
  ↓
Freeze
```

The goal is to make decisions using measured validation evidence rather than assumptions.

---

## 📌 Status

**Development Stage:** Active

Current development focus:

- Data loading
- Preprocessing
- Advanced candidate generation
- Candidate recall evaluation
- Two-stage matching
- Entity-level F0.5 evaluation
- Error analysis
- Optional team-system integration

Final performance numbers will be added after experiments are executed.

---

## 👨‍💻 Team

Developed as part of the **Amazon ML Unstop Business Entity Resolution Challenge** team.

**Person 4 — Advanced Matching & Evaluation System**

---

## 📄 License

Add the appropriate project license and verify the licenses of all dependencies and models used in the final implementation.
