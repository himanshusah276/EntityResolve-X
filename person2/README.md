# Person 2 — Supervised Pair-Classification Entity Resolution System

> **Team Role:** Person 2 of 4 (Amazon ML Unstop Business Entity Resolution Challenge)  
> **Specialization:** Supervised Pair Classification, Feature Engineering, Large-Scale Tabular ML (LightGBM)

---

## 1. Team Responsibility & Strict Boundaries

| Role | Primary Responsibility | Boundary / Contract |
| :--- | :--- | :--- |
| **Person 1** | Classical Preprocessing & Blocking | Produces candidate pairs (`candidate_pairs.tsv`) |
| **Person 2 (You)** | **Supervised Pair-Classification System** | **Consumes Person 1 candidates, creates pairwise features, trains supervised classifiers, and outputs continuous match probabilities.** |
| **Person 3** | Semantic Embedding / Vector Model | Independent vector similarity model |
| **Person 4** | Official Validation, Ensembling & Submission | Threshold tuning, F0.5 evaluation, model selection |

> [!IMPORTANT]
> - Person 2 does **NOT** redo preprocessing, normalization, or blocking.
> - Person 2 does **NOT** invent arbitrary negatives; negative examples are strictly derived from Person 1 candidate pairs that do not exist in ground truth.
> - Person 2 does **NOT** select a final threshold or declare a competition winner; all scores are continuous probabilities $[0, 1]$ handed off to Person 4.

---

## 2. Pipeline Architecture

```text
Person 1 Candidate Pairs (outputs/candidate_pairs.tsv)
                        ↓
Ground Truth Label Assignment (data/train/train_ground_truth.tsv)
                        ↓
Pairwise Feature Engineering (24 numerical features: Name, Address, Numeric, TF-IDF)
                        ↓
Supervised Classifiers:
    1. Logistic Regression (Baseline)
    2. Random Forest
    3. LightGBM (Primary High-Performance Candidate)
                        ↓
Candidate-Level Match Probabilities (outputs/person2_scores.parquet)
                        ↓
Person 4 Handoff & Validation Layer
```

---

## 3. Engineered Pair Features (24 Dimensions)

All features are deterministic, numerical, and reusable:

### Name Features (9)
- `name_jaccard`: Token-set Jaccard similarity
- `name_levenshtein`: Normalized string Levenshtein distance ratio
- `name_tfidf_cosine`: Cosine similarity of character n-gram TF-IDF vectors
- `name_token_overlap`: Token overlap coefficient ($\frac{|A \cap B|}{\min(|A|, |B|)}$)
- `name_character_similarity`: Character 3-gram Jaccard similarity
- `name_exact_match`: Binary indicator (1 if normalized strings are identical, else 0)
- `name_shared_token_count`: Integer count of shared tokens
- `name_length_ratio`: Ratio of shorter name to longer name
- `name_length_difference`: Absolute difference in character lengths

### Address Features (11)
- `address_jaccard`: Token-set Jaccard similarity
- `address_levenshtein`: Normalized Levenshtein ratio
- `address_tfidf_cosine`: Cosine similarity of address character n-gram TF-IDF vectors
- `address_token_overlap`: Address token overlap ratio
- `address_character_similarity`: Character 3-gram similarity
- `address_exact_match`: Binary indicator (1 if identical, else 0)
- `address_shared_token_count`: Shared non-numeric address tokens
- `address_shared_numeric_token_count`: Shared street/PIN/house numbers
- `address_numeric_token_overlap`: Numeric token overlap ratio
- `address_length_ratio`: Length ratio
- `address_length_difference`: Absolute character difference

### Global & Categorical Features (4)
- `country_match`: Binary indicator (1 if country matches, else 0)
- `total_shared_token_count`: Combined shared tokens across name and address
- `shared_numeric_token_count`: Total shared numeric tokens across the entity
- `numeric_token_overlap`: Global numeric overlap coefficient

---

## 4. Supervised Models & Performance Summary

Models trained with fixed reproducible random seeds (`seed=42`) and class imbalance compensation:

| Model | Class Weighting | Training Runtime | Scoring Throughput | Artifact Path |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | `balanced` | 0.016 s | > 1,000,000 pairs/sec | `person2/models/person2_logisticregression.joblib` |
| **Random Forest** | `balanced` | 0.169 s | ~86,000 pairs/sec | `person2/models/person2_randomforest.joblib` |
| **LightGBM** | `scale_pos_weight` | 0.036 s | ~340,000 pairs/sec | `person2/models/person2_lightgbm.joblib` |

---

## 5. Artifacts and Person 4 Deliverables

### Main Deliverables
1. **Primary Score File:**
   - **Path:** [`person2/outputs/person2_scores.parquet`](file:///d:/Amazon%20Project/person2/outputs/person2_scores.parquet)
   - **Columns:** `source1_entity_id`, `candidate_entity_id`, `match_probability`
   - **Integrity:** Exact 1-to-1 parity with Person 1's candidate pairs; zero duplicates, zero NaNs, probabilities strictly bounded in $[0.0, 1.0]$.
2. **TSV Mirror (Optional for Flexible Teammate Loading):**
   - **Path:** [`person2/outputs/person2_scores.tsv`](file:///d:/Amazon%20Project/person2/outputs/person2_scores.tsv)
3. **Model Information & Metadata:**
   - **Path:** [`person2/outputs/person2_model_info.csv`](file:///d:/Amazon%20Project/person2/outputs/person2_model_info.csv)
4. **Feature Schema Specification:**
   - **Path:** [`person2/outputs/person2_feature_schema.csv`](file:///d:/Amazon%20Project/person2/outputs/person2_feature_schema.csv)
5. **Trained Models & Preprocessors:**
   - `person2/models/person2_lightgbm.joblib`
   - `person2/models/person2_randomforest.joblib`
   - `person2/models/person2_logisticregression.joblib`
   - `person2/models/feature_schema.json`
   - `person2/features/name_tfidf.joblib`
   - `person2/features/address_tfidf.joblib`

---

## 6. How to Reproduce & Score New Candidate Pairs

To score candidate pairs:
```python
import joblib
import polars as pl
from person2.src.feature_engineering import PairFeatureExtractor

# 1. Load precomputed features (or extract from candidate pairs)
df = pl.read_parquet("person2/features/train_features.parquet")
feature_cols = PairFeatureExtractor.FEATURE_COLUMNS
X = df.select(feature_cols).to_numpy()

# 2. Load trained model
model = joblib.load("person2/models/person2_lightgbm.joblib")

# 3. Generate probabilities
probabilities = model.predict_proba(X)[:, 1]
df_scores = pl.DataFrame({
    "source1_entity_id": df["source1_entity_id"],
    "candidate_entity_id": df["candidate_entity_id"],
    "match_probability": probabilities,
})
```
