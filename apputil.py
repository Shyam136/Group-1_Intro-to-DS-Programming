"""
apputil.py — Baseline model + helpers for the Streamlit app.

Public functions:
- load_data()
- train_baseline(df=None)
- predict_gross(movie_a, movie_b, df=None, model=None)
- generate_insights(df=None)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


# ---------------------------
# Data loading
# ---------------------------
_DEFAULT_PATHS: List[str] = [
    "data/processed/movies_model.csv",
    "./data/processed/movies_model.csv",
    "../data/processed/movies_model.csv",
    "data/processed/movies_clean.csv",         # accepted fallback
    "./data/processed/movies_clean.csv",
    "../data/processed/movies_clean.csv",
]


def load_data(paths: Optional[List[str]] = None) -> pd.DataFrame:
    """Load the processed/model-ready dataset with a helpful error."""
    paths = paths or _DEFAULT_PATHS
    for p in paths:
        fp = Path(p)
        if fp.is_file():
            df = pd.read_csv(fp)
            # light normalization of expected columns
            # you can adjust these column names if Neville’s file differs
            rename_map = {
                "title": "title",
                "Title": "title",
                "name": "title",
                "genre": "genre",
                "Genre": "genre",
                "rating": "rating",
                "Rating": "rating",
                "runtime": "runtime",
                "Runtime": "runtime",
                "year": "year",
                "Year": "year",
                "budget_adj": "budget_adj",
                "Budget_Adjusted": "budget_adj",
                "gross_adj": "gross_adj",
                "Domestic_Gross_Adjusted": "gross_adj",
                "domestic_gross_adj": "gross_adj",
            }
            cols = {c: rename_map.get(c, c) for c in df.columns}
            df = df.rename(columns=cols)
            # ensure expected minimal columns exist
            required = {"title", "genre", "rating", "runtime", "year", "budget_adj", "gross_adj"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise KeyError(
                    f"Dataset loaded from '{p}' but missing columns: {missing}. "
                    "Please align column names in cleaning notebook."
                )
            return df

    tried = ", ".join(paths)
    raise FileNotFoundError(
        "Processed dataset not found. Expected at one of: "
        f"{tried} . Make sure Neville committed data/processed/movies_model.csv"
    )


# ---------------------------
# Baseline model
# ---------------------------
@dataclass
class BaselineModel:
    pipeline: Pipeline
    feature_cols: List[str]
    target_col: str = "gross_adj"


# Keep a simple in-process cache to avoid retraining in Streamlit reruns
_MODEL_CACHE: Optional[BaselineModel] = None


def _split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Pick columns for a simple baseline."""
    target = "gross_adj"
    cat_cols = [c for c in ["genre", "rating"] if c in df.columns]
    num_cols = [c for c in ["runtime", "year", "budget_adj"] if c in df.columns]
    X = df[cat_cols + num_cols].copy()
    y = df[target].copy()
    return X, y, cat_cols, num_cols


def train_baseline(df: Optional[pd.DataFrame] = None) -> BaselineModel:
    """
    Train a baseline regression model to predict adjusted domestic gross.
    Pipeline: OneHotEncoder(cats) + StandardScaler(nums) -> LinearRegression.
    """
    global _MODEL_CACHE
    if df is None:
        df = load_data()

    X, y, cat_cols, num_cols = _split_features(df)

    # Preprocess: OneHot for categoricals; scale numerics (in-pipeline, avoids leakage)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", LinearRegression()),
        ]
    )

    # simple NA handling: drop rows missing the target; model will ignore unseen cats at predict
    train_mask = y.notna() & X.notna().all(axis=1)
    pipe.fit(X.loc[train_mask], y.loc[train_mask])

    _MODEL_CACHE = BaselineModel(pipeline=pipe, feature_cols=cat_cols + num_cols)
    return _MODEL_CACHE


def _select_by_title(df: pd.DataFrame, title: str) -> pd.Series:
    """Return a single row by exact title match; if duplicates, take first."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Title must be a non-empty string.")
    hits = df[df["title"].str.lower() == title.strip().lower()]
    if hits.empty:
        raise LookupError(f"Movie not found: '{title}'")
    return hits.iloc[0]


def predict_gross(
    movie_a: str,
    movie_b: str,
    df: Optional[pd.DataFrame] = None,
    model: Optional[BaselineModel] = None,
) -> Dict[str, Any]:
    """
    Predict which movie will have higher adjusted domestic gross.
    Returns a dict with predictions and a 'winner' key.
    Includes friendly error messages for Streamlit UI.

    Example result:
    {
      "movie_a": {"title": "...", "pred": 123456789.0},
      "movie_b": {"title": "...", "pred": 987654321.0},
      "winner": "Movie B",
      "notes": "baseline-linear"
    }
    """
    # lazy-load df/model
    if df is None:
        df = load_data()
    if model is None:
        model = _MODEL_CACHE or train_baseline(df)

    # fetch rows
    try:
        row_a = _select_by_title(df, movie_a)
        row_b = _select_by_title(df, movie_b)
    except Exception as e:
        return {"error": str(e)}

    # build prediction frames with the model feature columns
    X_cols = model.feature_cols
    Xa = row_a[X_cols].to_frame().T
    Xb = row_b[X_cols].to_frame().T

    # if any required predictors are missing, return a helpful message
    if Xa.isna().any().any() or Xb.isna().any().any():
        return {
            "error": "Missing data for one or both titles. "
                     "Please pick movies with complete genre/rating/runtime/year/budget."
        }

    # predict
    pa = float(model.pipeline.predict(Xa)[0])
    pb = float(model.pipeline.predict(Xb)[0])

    winner = movie_a if pa >= pb else movie_b

    return {
        "movie_a": {"title": movie_a, "pred": pa},
        "movie_b": {"title": movie_b, "pred": pb},
        "winner": winner,
        "notes": "baseline-linear",
    }


# ---------------------------
# Insights for the app
# ---------------------------
def generate_insights(df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """
    Quick EDA tables for the app:
    - avg gross by genre
    - avg gross by rating
    - trend: median gross by decade
    """
    if df is None:
        df = load_data()

    out: Dict[str, pd.DataFrame] = {}

    if {"genre", "gross_adj"}.issubset(df.columns):
        out["by_genre"] = (
            df.groupby("genre", dropna=True)["gross_adj"]
              .mean()
              .sort_values(ascending=False)
              .reset_index(name="avg_gross_adj")
        )

    if {"rating", "gross_adj"}.issubset(df.columns):
        out["by_rating"] = (
            df.groupby("rating", dropna=True)["gross_adj"]
              .mean()
              .sort_values(ascending=False)
              .reset_index(name="avg_gross_adj")
        )

    if {"year", "gross_adj"}.issubset(df.columns):
        decade = (df["year"] // 10) * 10
        out["by_decade"] = (
            df.assign(decade=decade)
              .groupby("decade", dropna=True)["gross_adj"]
              .median()
              .reset_index(name="median_gross_adj")
              .sort_values("decade")
        )

    return out


# ---------------------------
# Tiny self-check (won't run in Streamlit)
# ---------------------------
if __name__ == "__main__":
    try:
        _df = load_data()
        _m = train_baseline(_df)
        demo = predict_gross(movie_a=_df.iloc[0]["title"], movie_b=_df.iloc[1]["title"], df=_df, model=_m)
        print("Demo prediction:", demo)
        print("Insights:", list(generate_insights(_df).keys()))
    except Exception as exc:
        print("Self-check error:", exc)