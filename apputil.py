"""
apputil.py — Baseline model + helpers for the Streamlit app.

Public functions:
- load_data()
- load_processed()
- prepare_features(), prepare_features_for_regression()
- train_baseline(), train_improved()
- predict_gross()
- generate_insights()
- load_model_artifact(), predict_with_saved_model()   <- NEW
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

# Data handling
import pandas as pd
import numpy as np
import joblib

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# -----------------------
# Paths / constants
# -----------------------
PROCESSED_PATHS = (
    "Data/processed/movies_model.csv",
    "./Data/processed/movies_model.csv",
    "../Data/processed/movies_model.csv",
)

TARGET_COL = "gross_higher"   # 1 if “this movie wins” (pairing scheme below), else 0
ID_COL = "name"              # Neville’s cleaned “movie title” column (adjust if different)

# -----------------------
# Loading
# -----------------------
def load_data() -> pd.DataFrame:
    """
    Load the raw movie dataset.
    Attempts to load processed model-ready data first (load_processed).
    Falls back to raw CSV paths under Data/raw/.
    """
    try:
        return load_processed()
    except FileNotFoundError:
        raw_paths = [
            "Data/raw/movies.csv",
            "./Data/raw/movies.csv",
            "../Data/raw/movies.csv"
        ]
        for p in raw_paths:
            if Path(p).is_file():
                return pd.read_csv(p)
        raise FileNotFoundError("Could not find movies data file (raw or processed).")


def load_processed() -> pd.DataFrame:
    """
    Load the final, model-ready dataset exported by data cleaning.
    Expected at Data/processed/movies_model.csv (or similar).
    """
    for p in PROCESSED_PATHS:
        if Path(p).is_file():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("Could not find Data/processed/movies_model.csv")

    # Basic hygiene
    for col in ["budget_adj", "gross_adj", "gross", "runtime", "year", "score", "votes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Example engineered column (Neville may already supply these)
    if "decade" not in df.columns and "year" in df.columns:
        df["decade"] = (df["year"] // 10) * 10

    return df


# -----------------------
# Feature prep
# -----------------------
# Initialize with defaults — these are adjusted at train-* time if needed
CATEGORICALS = ["genre", "rating", "decade"]
NUMERICS = ["runtime"]  # baseline; will be expanded in training helpers


def _get_available_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return available numeric and categorical columns that we will use."""
    available_numerics = [col for col in NUMERICS if col in df.columns]
    available_categoricals = [col for col in CATEGORICALS if col in df.columns]

    if not available_numerics:
        potential_numerics = ["budget_adj", "gross_adj", "budget", "gross", "runtime", "year", "score", "votes"]
        available_numerics = [col for col in potential_numerics if col in df.columns]

    return available_numerics, available_categoricals


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for a classification baseline.
    Target is TARGET_COL (1 if above median gross else 0) or a derived proxy.
    Returns (X, y, feature_cols).
    """
    df = df.copy()

    available_numerics, available_categoricals = _get_available_columns(df)

    # Create target if missing
    if TARGET_COL not in df.columns:
        if "gross" in df.columns:
            median_gross = df["gross"].median()
            df[TARGET_COL] = (df["gross"] > median_gross).astype(int)
        else:
            # fallback synthetic target for small/demo sets
            df[TARGET_COL] = 0
            if len(df) > 1:
                df.loc[df.sample(frac=0.5, random_state=42).index, TARGET_COL] = 1

    # Clean numeric and categorical fields
    for col in available_numerics + available_categoricals:
        if col in df.columns:
            if col in available_numerics:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if not df[col].isnull().all():
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("unknown").astype(str)

    # One-hot encode categoricals
    if available_categoricals:
        X_cat = pd.get_dummies(df[available_categoricals], drop_first=False)
    else:
        X_cat = pd.DataFrame(index=df.index)

    X_num = df[available_numerics].copy() if available_numerics else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)

    # Ensure at least one feature
    if X.shape[1] == 0:
        X["dummy_feature"] = 1

    y = df[TARGET_COL].astype(int)

    # Make sure y has at least two classes (simple flip if necessary)
    if y.nunique() == 1 and len(y) > 1:
        y.iloc[0] = 1 - y.iloc[0]

    feature_cols = X.columns.tolist()
    return X, y, feature_cols


def prepare_features_for_regression(df: pd.DataFrame, exclude_genre: bool = False) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for regression (predicting gross).
    
    Args:
        df: Input DataFrame containing the data
        exclude_genre: If True, exclude the 'genre' feature to reduce bias
        
    Returns:
        Tuple of (X, y, feature_cols)
    """
    df = df.copy()

    available_numerics, available_categoricals = _get_available_columns(df)
    if exclude_genre and 'genre' in available_categoricals:
        available_categoricals.remove('genre')

    if "gross" not in df.columns:
        raise ValueError("'gross' column not found in the dataset for regression target")

    # Clean features
    for col in available_numerics + available_categoricals:
        if col in df.columns:
            if col in available_numerics:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if not df[col].isnull().all():
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("unknown").astype(str)

    # Start with numeric features
    X = df[available_numerics].copy() if available_numerics else pd.DataFrame(index=df.index)

    # Add categorical one-hot columns (prefix optional)
    for cat_col in available_categoricals:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)

    if X.empty:
        X["dummy"] = 1

    y = pd.to_numeric(df["gross"], errors="coerce")
    return X, y, X.columns.tolist()


# -----------------------
# Training (baseline)
# -----------------------
def train_baseline(df: pd.DataFrame, random_state: int = 42, exclude_genre: bool = False):
    """
    Train a baseline regression model to predict gross values.
    Returns: (model, feature_cols, r2)
    """
    global NUMERICS, CATEGORICALS
    original_numerics = NUMERICS.copy()
    original_categoricals = CATEGORICALS.copy()

    try:
        NUMERICS = ["runtime", "budget_adj", "year"]
        CATEGORICALS = ["genre", "rating"]

        X, y, feature_cols = prepare_features_for_regression(df, exclude_genre=exclude_genre)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=random_state,
            n_jobs=-1,
            max_depth=5,
            min_samples_split=10
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        return model, feature_cols, r2
    finally:
        NUMERICS = original_numerics
        CATEGORICALS = original_categoricals


def train_improved(df: pd.DataFrame, random_state: int = 42, exclude_genre: bool = False):
    """
    Train a stronger regression model using more features.
    Returns: (model, feature_cols, r2)
    """
    global NUMERICS, CATEGORICALS
    original_numerics = NUMERICS.copy()
    original_categoricals = CATEGORICALS.copy()

    try:
        NUMERICS = ["runtime", "budget_adj", "year", "score", "votes", "gross"]
        CATEGORICALS = ["genre", "rating", "country", "director"]

        if "budget_adj" in df.columns and "year" in df.columns:
            df["budget_year_ratio"] = df["budget_adj"] / (df["year"] - df["year"].min() + 1)
            NUMERICS.append("budget_year_ratio")

        X, y, feature_cols = prepare_features_for_regression(df, exclude_genre=exclude_genre)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return model, feature_cols, r2
    finally:
        NUMERICS = original_numerics
        CATEGORICALS = original_categoricals


# -----------------------
# Head-to-head prediction
# -----------------------
def _row_to_features(row: pd.Series, feature_cols: List[str]) -> pd.DataFrame:
    """Convert a single movie row to the same feature columns used in training."""

    # Initialize with zeros
    x = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Numeric candidates to try map from row to features
    numeric_candidates = ['runtime', 'year', 'gross', 'budget_adj', 'budget', 'score', 'votes']
    for col in numeric_candidates:
        if col in feature_cols and col in row and pd.notna(row[col]):
            try:
                x[col] = float(row[col])
            except Exception:
                x[col] = 0.0

    # Categorical one-hot columns named like "genre_Action" or "rating_PG-13"
    for col_type in ['genre', 'rating', 'decade', 'country', 'director']:
        if col_type in row and pd.notna(row[col_type]):
            encoded_col = f"{col_type}_{row[col_type]}"
            if encoded_col in feature_cols:
                x[encoded_col] = 1

    return x


def predict_gross(movie_a: str, movie_b: str,
                  df: pd.DataFrame,
                  model: Any,
                  feature_cols: List[str]) -> Dict[str, Any]:
    """
    Predict which movie will have a higher gross using a regression model.
    Returns a dictionary with predictions, actuals (if available), warnings, and winner name.
    """
    out: Dict[str, Any] = {
        "ok": True,
        "warnings": [],
        "gross_a": None,
        "gross_b": None,
        "predicted_gross_a": None,
        "predicted_gross_b": None,
        "predicted_winner": None
    }

    try:
        # find rows (case-insensitive)
        rows_a = df[df[ID_COL].str.casefold() == movie_a.casefold()]
        rows_b = df[df[ID_COL].str.casefold() == movie_b.casefold()]

        if rows_a.empty or rows_b.empty:
            missing = []
            if rows_a.empty:
                missing.append(movie_a)
            if rows_b.empty:
                missing.append(movie_b)
            out["ok"] = False
            out["warnings"].append(f"Movie(s) not found: {', '.join(missing)}")
            return out

        ra = rows_a.sort_values("year", ascending=False).iloc[0]
        rb = rows_b.sort_values("year", ascending=False).iloc[0]

        # Build feature vectors
        xa = _row_to_features(ra, feature_cols)[feature_cols]
        xb = _row_to_features(rb, feature_cols)[feature_cols]

        # Predict
        try:
            pred_gross_a = float(model.predict(xa)[0])
            pred_gross_b = float(model.predict(xb)[0])
        except Exception as e:
            out["ok"] = False
            out["warnings"].append(f"Model prediction failed: {e}")
            return out

        out["predicted_gross_a"] = pred_gross_a
        out["predicted_gross_b"] = pred_gross_b
        out["gross_a"] = ra.get("gross", None)
        out["gross_b"] = rb.get("gross", None)

        if pred_gross_a > pred_gross_b:
            out["predicted_winner"] = movie_a
        elif pred_gross_b > pred_gross_a:
            out["predicted_winner"] = movie_b
        else:
            out["predicted_winner"] = "Tie"

        return out

    except Exception as e:
        out["ok"] = False
        out["warnings"].append(f"Error during prediction: {e}")
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


# -----------------------
# Model artifact helpers (NEW)
# -----------------------
def load_model_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a joblib model artifact saved by model.py or similar.
    Expected artifact format: dict with keys 'model' and 'feature_cols'.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    artifact = joblib.load(str(p))

    if not isinstance(artifact, dict):
        raise ValueError("Model artifact should be a dict with keys 'model' and 'feature_cols'")

    if "model" not in artifact or "feature_cols" not in artifact:
        raise ValueError("Model artifact missing required keys: 'model' and 'feature_cols'")

    return artifact


def predict_with_saved_model(movie_a: str, movie_b: str, df: pd.DataFrame, model_artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience wrapper that accepts a loaded artifact (with 'model' and 'feature_cols')
    and calls the existing predict_gross function.
    """
    model = model_artifact.get("model")
    feature_cols = model_artifact.get("feature_cols")
    if model is None or feature_cols is None:
        raise ValueError("Invalid model_artifact: must contain 'model' and 'feature_cols'")

    return predict_gross(movie_a=movie_a, movie_b=movie_b, df=df, model=model, feature_cols=feature_cols)