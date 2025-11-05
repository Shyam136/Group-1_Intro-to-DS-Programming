"""
model.py

Train, evaluate, and save baseline models for movie gross comparison.
Requires existing prepare_features(df) in apputil.py
Saves artifacts to ./models/
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Import prepare_features from your apputil. Adjust import path if needed.
from apputil import prepare_features  # <- ensure apputil.py is in same package/root


MODEL_DIR = Path("models") if "Path" in globals() else __import__("pathlib").Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _fit_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit model defensively and return fitted estimator."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Return common classification metrics. For AUC, require predict_proba support."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    # AUC if predict_proba available and y_test has both classes
    try:
        if len(np.unique(y_test)) == 2:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["auc"] = float(roc_auc_score(y_test, proba))
        else:
            metrics["auc"] = float("nan")
    except Exception:
        metrics["auc"] = float("nan")
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Dict]:
    """
    Prepare features using apputil.prepare_features, train two models,
    evaluate them on a holdout set, save models and metadata to models/.

    Returns:
        results: {
            "logistic": {"model_path": str, "metrics": {...}, "feature_cols":[...]},
            "random_forest": {...}
        }
    """
    # Build features
    X, y, feature_cols = prepare_features(df)

    # drop rows with missing values in X or y
    df_xy = pd.concat([X, y.rename("_target")], axis=1)
    use = df_xy.dropna()
    if use.empty:
        raise ValueError("No usable rows after dropping NA — check your processed dataset")

    X_use = use[feature_cols]
    y_use = use["_target"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_use, y_use, test_size=test_size, random_state=random_state, stratify=y_use
    )

    results: Dict[str, Dict] = {}

    # Logistic Regression baseline
    log = LogisticRegression(max_iter=500)
    log = _fit_model(log, X_tr, y_tr)
    log_metrics = evaluate_model(log, X_te, y_te)
    log_path = MODEL_DIR / "model_logistic.joblib"
    joblib.dump({"model": log, "feature_cols": feature_cols}, log_path)
    results["logistic"] = {
        "model_path": str(log_path),
        "metrics": log_metrics,
        "feature_cols": feature_cols,
    }

    # Random Forest baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf = _fit_model(rf, X_tr, y_tr)
    rf_metrics = evaluate_model(rf, X_te, y_te)
    rf_path = MODEL_DIR / "model_random_forest.joblib"
    joblib.dump({"model": rf, "feature_cols": feature_cols}, rf_path)
    results["random_forest"] = {
        "model_path": str(rf_path),
        "metrics": rf_metrics,
        "feature_cols": feature_cols,
    }

    # Optionally save a small JSON/CSV summary of metrics
    try:
        import json

        with open(MODEL_DIR / "metrics_summary.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    return results


def load_saved_model(path: str):
    """Load a saved joblib model container (dict with model and feature_cols)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


if __name__ == "__main__":
    # convenience CLI: train on processed dataset if present
    print("Training models on processed data (expects data/processed/movies_model.csv)...")
    from apputil import load_processed

    df_proc = load_processed()
    res = train_and_evaluate(df_proc)
    print("Done. Results:")
    for name, detail in res.items():
        print(name, detail["metrics"])