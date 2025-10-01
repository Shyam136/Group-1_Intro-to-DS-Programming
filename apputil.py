"""
apputil.py — Baseline model + helpers for the Streamlit app.

Public functions:
- load_data()
- train_baseline(df=None)
- predict_gross(movie_a, movie_b, df=None, model=None)
- generate_insights(df=None)
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# -----------------------
# Paths / constants
# -----------------------
PROCESSED_PATHS = (
    "data/processed/movies_model.csv",
    "./data/processed/movies_model.csv",
    "../data/processed/movies_model.csv",
)

TARGET_COL = "gross_higher"   # 1 if “this movie wins” (pairing scheme below), else 0
ID_COL = "title"              # Neville’s cleaned “movie title” column (adjust if different)

# -----------------------
# Loading
# -----------------------
def load_processed() -> pd.DataFrame:
    """
    Load the final, model-ready dataset exported by data cleaning.
    Expected at data/processed/movies_model.csv.
    """
    for p in PROCESSED_PATHS:
        if Path(p).is_file():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("Could not find data/processed/movies_model.csv")

    # Basic hygiene
    # Ensure numeric types (adjust column names if your schema differs)
    for col in ["budget_adj", "gross_adj", "runtime", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Example engineered column (Neville may already supply these)
    if "decade" not in df.columns and "year" in df.columns:
        df["decade"] = (df["year"] // 10) * 10

    return df


# -----------------------
# Feature prep
# -----------------------
CATEGORICALS = ["genre", "rating", "decade"]  # adjust to match your clean schema
NUMERICS     = ["budget_adj", "runtime"]      # adjust as needed

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build X, y for a single-record model (predict probability that
    a given movie is a “winner” vs. others).
    NOTE: For the app’s head-to-head comparison, we’ll score each movie,
    then pick the higher score as the predicted winner.
    """
    if TARGET_COL not in df.columns:
        # If Neville didn’t create a label, make a weak proxy for baseline:
        # mark a movie as “winner” if its gross/budget ratio is in top 50% within-year.
        if "gross_adj" in df.columns and "budget_adj" in df.columns:
            ratio = df["gross_adj"] / (df["budget_adj"].replace(0, np.nan))
            df["tmp_ratio"] = ratio
            by_year_median = df.groupby(df["year"])["tmp_ratio"].transform("median")
            df[TARGET_COL] = (df["tmp_ratio"] > by_year_median).astype("int")
        else:
            raise ValueError("Missing target and cannot derive proxy label.")

    # One-hot categoricals
    X_cat = pd.get_dummies(df[CATEGORICALS], dummy_encoding="onehot", drop_first=False) if CATEGORICALS else pd.DataFrame(index=df.index)
    # Newer pandas uses `dtype_backend`; this keeps it simple.
    X_num = df[NUMERICS].copy() if NUMERICS else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET_COL].astype(int)

    # keep feature order for later
    feature_cols = X.columns.tolist()
    return X, y, feature_cols


# -----------------------
# Training (baseline)
# -----------------------
def train_baseline(df: pd.DataFrame, random_state: int = 42) -> Tuple[LogisticRegression, List[str], float]:
    """
    Train a simple Logistic Regression baseline.
    Returns: model, feature_cols, AUC (holdout)
    """
    X, y, feature_cols = prepare_features(df)
    # Drop rows with NA in X or y
    use = pd.concat([X, y], axis=1).dropna()
    X, y = use[feature_cols], use[TARGET_COL]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Logistic Regression (lbfgs handles multi/dense well)
    model = LogisticRegression(max_iter=200, n_jobs=None)
    model.fit(X_tr, y_tr)

    # quick holdout AUC for sanity
    yhat = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, yhat)
    return model, feature_cols, auc


# -----------------------
# Head-to-head prediction
# -----------------------
def _row_to_features(row: pd.Series, feature_cols: List[str]) -> pd.DataFrame:
    """Convert a single movie row to the same feature columns used in training."""
    # Build small frames then align columns
    cat_df = pd.get_dummies(
        pd.DataFrame({k: [row.get(k, np.nan)] for k in CATEGORICALS}),
        dummy_encoding="onehot", drop_first=False
    )
    num_df = pd.DataFrame({k: [row.get(k, np.nan)] for k in NUMERICS})

    x = pd.concat([num_df, cat_df], axis=1)
    # Add any missing columns and order
    for c in feature_cols:
        if c not in x.columns:
            x[c] = 0
    x = x[feature_cols]
    return x


def predict_gross(movie_a: str, movie_b: str,
                  df: pd.DataFrame,
                  model: LogisticRegression,
                  feature_cols: List[str]) -> Dict:
    """
    Score each selected movie with the baseline classifier.
    Return winner + probabilities + any warnings.
    """
    out = {"ok": True, "warnings": [], "movie_a": movie_a, "movie_b": movie_b}

    # find rows (handle duplicates by picking the most recent)
    rows_a = df[df[ID_COL].str.casefold() == movie_a.casefold()]
    rows_b = df[df[ID_COL].str.casefold() == movie_b.casefold()]

    if rows_a.empty or rows_b.empty:
        out["ok"] = False
        out["warnings"].append("One or both movies not found in processed data.")
        return out

    ra = rows_a.sort_values("year", ascending=False).iloc[0]
    rb = rows_b.sort_values("year", ascending=False).iloc[0]

    # minimal missing-value checks
    missing_a = [c for c in (NUMERICS + CATEGORICALS) if pd.isna(ra.get(c))]
    missing_b = [c for c in (NUMERICS + CATEGORICALS) if pd.isna(rb.get(c))]
    if missing_a:
        out["warnings"].append(f"{movie_a}: missing {missing_a}")
    if missing_b:
        out["warnings"].append(f"{movie_b}: missing {missing_b}")

    xa = _row_to_features(ra, feature_cols)
    xb = _row_to_features(rb, feature_cols)

    pa = float(model.predict_proba(xa)[:, 1][0])
    pb = float(model.predict_proba(xb)[:, 1][0])

    out["proba_a"] = pa
    out["proba_b"] = pb
    out["predicted_winner"] = movie_a if pa >= pb else movie_b
    return out


# -----------------------
# Quick insights
# -----------------------
def generate_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a few simple, user-facing insights (top genres by median adj. gross).
    """
    if "genre" not in df.columns or "gross_adj" not in df.columns:
        return pd.DataFrame()

    g = (
        df.dropna(subset=["genre", "gross_adj"])
          .groupby("genre")["gross_adj"]
          .median()
          .sort_values(ascending=False)
          .reset_index(name="median_gross_adj")
    )
    return g