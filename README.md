# EntityResolve-X: Two-Stage Business Entity Resolution Pipeline

> **Person 4 Independent ML Pipeline** for the **Amazon ML Unstop Business Entity Resolution Challenge**.
> Multi-pass candidate generation followed by two-stage broad filtering and fine-grained reranking.

---

## 📌 Executive Overview

**EntityResolve-X** links noisy, duplicate business records from **Source 2** and **Source 3** to a deduplicated reference list of canonical businesses in **Source 1**.

Unlike single-pass fuzzy scorers or black-box pair classifiers, EntityResolve-X employs an **independent two-stage cascading architecture**:

```text
Raw Business Records (Source 1, Source 2, Source 3)
                     ↓
Conservative Preprocessing (Legal Suffix Extraction, Address Normalization, N-grams)
                     ↓
Advanced Multi-Pass Blocking (Blocks A-E + Block F Union + Adaptive Fallbacks)
                     ↓
Stage 1: Broad Filter (Fast set-theoretic Jaccard overlap, prunes 89.4% noise in 0.01s)
                     ↓
Stage 2: Detailed Reranker (Multi-scale string similarity, postal & street number agreement)
                     ↓
Decision Logic (Empirically tuned threshold = 0.65)
                     ↓
Entity-Level Aggregation & Macro F0.5 Evaluation
                     ↓
Submission Files (outputs/matching_results.tsv & outputs/candidate_pairs.tsv)
```

---

## 🚀 Key Measured Results

All metrics measured on the held-out Source 1 validation split:

- **Candidate Recall Ceiling:** **100.00%** (Block F union retrieves all true matches).
- **Candidate Pool Reduction:** Average candidates pruned from **28.6** to **3.0** per query.
- **Stage 1 Pruning Rate:** **89.4%** of candidate noise eliminated in **0.01 seconds**.
- **Macro Precision:** **99.00%**
- **Macro Recall:** **100.00%**
- **Macro $F_{0.5}$:** **99.11%** (optimal threshold = 0.65)
- **Staging Speedup:** Two-Stage pipeline is **3x faster** than the single-stage baseline (0.02s vs 0.06s).

---

## 📁 Repository Structure

```text
EntityResolve-X/
│
├── data/
│   ├── train/                  # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                   # test_source1.tsv, test_source2.tsv, test_source3.tsv
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Frozen configuration & hyperparameters
│   ├── data_loader.py          # TSV loading, schema validation, data health reporting
│   ├── preprocessing.py        # Normalization, legal suffix stripping, address expansion, n-grams
│   ├── candidate_generation.py # Blocks A–F multi-pass inverted indexes + adaptive fallbacks
│   ├── advanced_matching.py    # Stage 1 broad filter & Stage 2 detailed reranker
│   ├── ensemble.py             # Optional teammate score loaders & blending logic
│   ├── evaluation.py           # Source 1 validation split, Candidate Recall, Macro F0.5
│   ├── error_analysis.py       # Root-cause error categorization & CSV export
│   ├── prediction.py           # Test prediction orchestrator
│   ├── run_experiments.py      # End-to-end experiment runner
│   └── output.py               # TSV generation & submission validation logic
│
├── utils/
│   └── validate_submission.py  # Standalone CLI submission validator
│
├── outputs/
│   ├── matching_results.tsv    # Final predictions (source1_entity_id \t matched_entity_ids)
│   └── candidate_pairs.tsv     # Candidate pools (source1_entity_id \t candidate_entity_ids)
│
├── reports/
│   ├── experiments.csv         # Experiment tracking log across all blocks and thresholds
│   ├── error_analysis.csv      # Categorized false positives and false negatives
│   ├── system_comparison.csv   # System comparison table across Person 1–4
│   └── final_report.md         # Comprehensive engineering and evaluation report
│
├── docs/
│   ├── PRD.md                  # Product Requirements Document
│   ├── TRD.md                  # Technical Requirements Document
│   ├── EXPERIMENT_PLAN.md      # Experiment execution plan
│   ├── EVALUATION_PROTOCOL.md  # Official scoring protocol & singleton rules
│   └── INTEGRATION_PLAN.md     # Teammate integration & schema contracts
│
├── tests/                      # Automated unit test suite (31 tests across 8 modules)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Minimal verified dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

1. Clone repository and navigate to root:
   ```bash
   git clone https://github.com/himanshusah276/EntityResolve-X.git
   cd EntityResolve-X
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

### 1. Run All Automated Unit Tests (31 Passing Tests)
```bash
python -m pytest tests/ -v
```

### 2. Run Experiments & Staging Benchmarks
Executes candidate generation comparisons (Blocks A–F), Stage 1 broad filtering, Stage 2 reranking, and threshold sweeps:
```bash
python -m src.run_experiments
```
Results are saved to `reports/experiments.csv` and `reports/error_analysis.csv`.

### 3. Run Final Test Prediction Pipeline
Runs the frozen two-stage pipeline on `data/test/`:
```bash
python -m src.prediction
```
Generates `outputs/matching_results.tsv` and `outputs/candidate_pairs.tsv`.

### 4. Run Standalone Submission Validator
```bash
python utils/validate_submission.py
```

---

## 🔒 Frozen Pipeline Configuration

Locked in `src/config.py` based on empirical validation:
- **Seed:** `42`
- **Validation Split:** 80% Train, 20% Held-out Validation (Source 1 entity-level split)
- **Blocking Strategy:** `block_union_all` (Multi-pass union + adaptive fallbacks)
- **Max Candidates:** `80` per Source 1 entity
- **Stage 1 Minimum Broad Score:** `0.15`
- **Stage 2 Decision Threshold:** `0.65`
- **Ensemble Setting:** Pure Standalone mode (`person1_scores = person2_scores = person3_scores = None`)

---

## 👥 Teammate Integration (Optional)

If teammates supply score files in the common format:
```tsv
source1_entity_id	candidate_entity_id	score
```
They can be integrated using `src/ensemble.py`:
```python
from src.ensemble import load_person1_scores, evaluate_ensemble_combinations

p1_df = load_person1_scores("path/to/person1_scores.tsv")
```
Ensembles are evaluated only on the identical validation split without assuming improvement over standalone P4.

---

## 📄 License & Fair Play Compliance

- **Fair Play:** Operates exclusively on provided challenge datasets; zero web requests, zero commercial APIs, zero external geocoding.
- **Licenses:** All dependencies (`rapidfuzz`, `scikit-learn`, `pandas`, `scipy`, `pytest`) are permissively licensed (MIT / BSD).
