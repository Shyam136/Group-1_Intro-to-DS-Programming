# model.py
"""
Model training, evaluation, and prediction helpers.

Functions:
- prepare_data(df, feature_cols=None, target_col='is_hit')
- train_baselines(df, save_dir='models', random_state=42)
- load_artifacts(save_dir='models')
- predict_proba_for_pair(movie_row_a, movie_row_b, artifacts)
- evaluate_models(X, y, artifacts)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str] | None = None,
    target_col: str = "is_hit",
    hit_threshold: float | None = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Prepare features and target.
    - If 'target_col' not present, create a binary 'is_hit' target:
        is_hit = adjusted_gross > median(adjusted_gross)  (or hit_threshold if provided)
    - feature_cols: list of columns to use. If None, auto-select numeric and some categoricals.

    Returns (X, y, metadata) where metadata contains feature lists.
    """
    df = df.copy()

    # Ensure adjusted gross exists
    if "adjusted_gross" not in df.columns and "gross" in df.columns:
        logger.warning("No 'adjusted_gross' column: attempting to use 'gross' as fallback.")
        df["adjusted_gross"] = df["gross"]

    # Create target if not present
    if target_col not in df.columns:
        if hit_threshold is None:
            hit_threshold = df["adjusted_gross"].median()
        df[target_col] = (df["adjusted_gross"] > hit_threshold).astype(int)

    # Default feature selection: numeric columns except target + a handful of categoricals if present
    if feature_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # remove fields we don't want
        numeric = [c for c in numeric if c not in (target_col, "adjusted_gross", "gross")]
        # choose some categorical columns
        possible_cat = [c for c in ["genre", "rating", "decade", "primary_genre"] if c in df.columns]
        feature_cols = numeric + possible_cat

    # Filter to rows without NaN in feature cols and target (we keep machine-friendly pipeline to impute later if needed)
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    metadata = {
        "feature_cols": feature_cols,
        "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": [c for c in feature_cols if c not in X.select_dtypes(include=[np.number]).columns.tolist()],
    }

    return X, y, metadata


def _build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """
    Build preprocessing + classifier pipeline. We return only preprocessing here and will
    wrap with different classifiers later.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor


def train_baselines(
    df: pd.DataFrame,
    save_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """
    Train LogisticRegression and RandomForest baselines, evaluate, and save artifacts.
    Returns a dict with metadata, scores, and paths.
    Artifacts saved: pipeline, models, and a metadata joblib file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    preprocessor = _build_pipeline(meta["numeric_features"], meta["categorical_features"])

    # Logistic Regression pipeline
    log_clf = Pipeline(steps=[("preproc", preprocessor), ("clf", LogisticRegression(max_iter=500, random_state=random_state))])
    rf_clf = Pipeline(steps=[("preproc", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state))])

    logger.info("Training Logistic Regression...")
    log_clf.fit(X_train, y_train)

    logger.info("Training Random Forest...")
    rf_clf.fit(X_train, y_train)

    # Evaluate
    def eval_pipeline(pipe, X_t, y_t):
        preds = pipe.predict(X_t)
        probs = pipe.predict_proba(X_t)[:, 1] if hasattr(pipe, "predict_proba") else None
        return {
            "accuracy": accuracy_score(y_t, preds),
            "precision": precision_score(y_t, preds, zero_division=0),
            "recall": recall_score(y_t, preds, zero_division=0),
            "f1": f1_score(y_t, preds, zero_division=0),
            "report": classification_report(y_t, preds, zero_division=0),
            "preds": preds,
            "probs": probs,
        }

    scores_log = eval_pipeline(log_clf, X_test, y_test)
    scores_rf = eval_pipeline(rf_clf, X_test, y_test)

    # Save artifacts
    log_path = save_dir / "logistic_pipeline.joblib"
    rf_path = save_dir / "rf_pipeline.joblib"
    meta_path = save_dir / "metadata.joblib"

    joblib.dump(log_clf, log_path)
    joblib.dump(rf_clf, rf_path)
    joblib.dump(meta, meta_path)

    results = {
        "logistic": {"path": str(log_path), "scores": scores_log},
        "random_forest": {"path": str(rf_path), "scores": scores_rf},
        "meta": {"path": str(meta_path), "content": meta},
    }

    logger.info("Saved models to %s", save_dir)
    return results


def load_artifacts(save_dir: str = "models") -> Dict:
    """Load saved models + metadata. Returns dict with keys: logistic, random_forest, meta"""
    save_dir = Path(save_dir)
    logistic = joblib.load(save_dir / "logistic_pipeline.joblib")
    rf = joblib.load(save_dir / "rf_pipeline.joblib")
    meta = joblib.load(save_dir / "metadata.joblib")
    return {"logistic": logistic, "random_forest": rf, "meta": meta}


def predict_proba_for_pair(
    movie_row_a: pd.Series, movie_row_b: pd.Series, artifacts: Dict, model_name: str = "random_forest"
) -> Dict[str, float]:
    """
    Given two movie rows (Series with feature columns), return a dict:
    { "movie_a_name": prob, "movie_b_name": prob, "winner": "A"|"B", "probabilities": (pa, pb) }
    - artifacts is the result of load_artifacts()
    - movie_row_* should include the same features used in training.
    """
    model = artifacts[model_name]
    meta = artifacts["meta"]

    feature_cols = meta.get("feature_cols", None)
    if feature_cols is None:
        raise RuntimeError("Feature metadata missing in artifacts.")

    # Build single-row dataframes
    Xa = pd.DataFrame([movie_row_a[feature_cols]])
    Xb = pd.DataFrame([movie_row_b[feature_cols]])

    # Predict probabilities for class 1 (is_hit)
    pa = model.predict_proba(Xa)[:, 1][0] if hasattr(model, "predict_proba") else model.predict(Xa)[0]
    pb = model.predict_proba(Xb)[:, 1][0] if hasattr(model, "predict_proba") else model.predict(Xb)[0]

    winner = "A" if pa > pb else ("B" if pb > pa else "Tie")

    return {
        "movie_a_prob": float(pa),
        "movie_b_prob": float(pb),
        "winner": winner,
        "movie_a_name": movie_row_a.get("title", "Movie A"),
        "movie_b_name": movie_row_b.get("title", "Movie B"),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series, artifacts: Dict) -> Dict[str, Dict]:
    """
    Produce evaluation metrics on provided X,y using loaded artifacts.
    Returns dict with metrics for both models.
    """
    out = {}
    for key in ("logistic", "random_forest"):
        pipe = artifacts[key]
        preds = pipe.predict(X)
        probs = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None
        out[key] = {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "report": classification_report(y, preds, zero_division=0),
        }
    return out