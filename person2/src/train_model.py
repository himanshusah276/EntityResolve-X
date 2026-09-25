"""
Person 2 — Supervised Model Training Module
Trains supervised pairwise classification models:
  1. Logistic Regression (Baseline)
  2. Random Forest
  3. LightGBM (Primary candidate)

Saves trained model artifacts to person2/models/ and logs model metadata to person2/outputs/person2_model_info.csv.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from person2.src.feature_engineering import PairFeatureExtractor

MODELS_DIR = PROJECT_ROOT / "person2" / "models"
OUTPUTS_DIR = PROJECT_ROOT / "person2" / "outputs"
FEATURES_DIR = PROJECT_ROOT / "person2" / "features"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_features(parquet_path: Path) -> Tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Loads feature matrix X and labels y from the precomputed Parquet file."""
    print(f"Loading training features from {parquet_path.resolve()}...")
    df = pl.read_parquet(parquet_path)

    feature_cols = PairFeatureExtractor.FEATURE_COLUMNS
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    y = df["label"].to_numpy().astype(np.int32)

    print(f"Loaded {X.shape[0]:,} rows with {X.shape[1]} features.")
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    print(f"Class distribution: Positives={pos_count:,} ({pos_count / len(y):.2%}), Negatives={neg_count:,} ({neg_count / len(y):.2%})")

    return X, y, df


def train_models(
    feature_parquet: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains Logistic Regression, Random Forest, and LightGBM models.
    Persists models and records training/scoring metadata.
    """
    feat_path = feature_parquet or (FEATURES_DIR / "train_features.parquet")
    X, y, df_metadata = load_training_features(feat_path)

    n_samples, n_features = X.shape
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))

    # Save feature schema metadata
    schema_info = {
        "feature_names": PairFeatureExtractor.FEATURE_COLUMNS,
        "n_features": n_features,
        "seed": seed,
    }
    with open(MODELS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=2)

    # Calculate class weight ratio for imbalance handling
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }

    model_info_records = []
    trained_artifacts = {}

    print("\n" + "=" * 65)
    print("STARTING SUPERVISED MODEL TRAINING")
    print("=" * 65)

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        clf.fit(X, y)
        train_runtime = time.time() - t0

        # Scoring latency benchmark
        t_score_start = time.time()
        probs = clf.predict_proba(X)[:, 1]
        scoring_runtime = time.time() - t_score_start

        # Internal diagnostics (STRICTLY INTERNAL DEVELOPMENT ONLY)
        internal_roc_auc = float(roc_auc_score(y, probs))
        internal_pr_auc = float(average_precision_score(y, probs))

        print(f"  Training Runtime: {train_runtime:.3f} s")
        print(f"  Scoring Runtime ({n_samples:,} rows): {scoring_runtime:.3f} s")
        print(f"  [INTERNAL_ONLY] Diagnostic ROC-AUC: {internal_roc_auc:.4f}")
        print(f"  [INTERNAL_ONLY] Diagnostic PR-AUC:  {internal_pr_auc:.4f}")

        # Save model artifact
        artifact_filename = f"person2_{name.lower()}.joblib"
        artifact_path = MODELS_DIR / artifact_filename
        joblib.dump(clf, artifact_path)
        trained_artifacts[name] = artifact_path
        print(f"  Saved artifact: {artifact_path.resolve()}")

        # Record metadata
        approx_memory_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
        model_info_records.append({
            "model_name": name,
            "number_of_features": n_features,
            "training_rows": n_samples,
            "positive_rows": pos_count,
            "negative_rows": neg_count,
            "training_runtime_seconds": round(train_runtime, 4),
            "scoring_runtime_seconds": round(scoring_runtime, 4),
            "approximate_memory_mb": round(approx_memory_mb, 2),
            "internal_diagnostic_roc_auc": round(internal_roc_auc, 4),
            "internal_diagnostic_pr_auc": round(internal_pr_auc, 4),
            "artifact_path": str(artifact_path.resolve()),
        })

    # Save model info CSV
    df_info = pl.DataFrame(model_info_records)
    info_csv = OUTPUTS_DIR / "person2_model_info.csv"
    df_info.write_csv(info_csv)
    print(f"\nModel metadata table saved to {info_csv.resolve()}")

    return {
        "artifacts": trained_artifacts,
        "metadata_table": df_info,
        "n_samples": n_samples,
        "n_features": n_features,
    }


if __name__ == "__main__":
    train_models()
