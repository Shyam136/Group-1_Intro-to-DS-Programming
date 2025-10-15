# apputil.py — Baseline model + helpers for the Streamlit app.
"""
Public functions:
- load_data()
- train_baseline_and_save(df=None, save_dir="models")
- predict_gross(movie_a, movie_b, df=None, model=None, feature_cols=None, model_name="random_forest")
- generate_insights(df=None)
- load_artifacts(save_dir="models")
- save_artifacts_from_training(results, save_dir="models")
"""

from __future__ import annotations
<<<<<<< HEAD
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Data handling
=======

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import joblib
>>>>>>> ddb021d (feat: baseline model pipeline & Streamlit integration)
import pandas as pd
import numpy as np

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Try to import model.py helpers (optional)
try:
    from model import train_baselines, load_artifacts as model_load_artifacts  # type: ignore
except Exception:
    train_baselines = None  # type: ignore
    model_load_artifacts = None  # type: ignore


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
    This is a placeholder - update with your actual data loading logic.
    """
    try:
        # Try to load from the processed data first
        return load_processed()
    except FileNotFoundError:
        # Fallback to raw data if processed not available
        raw_paths = [
            "Data/raw/movies.csv",
            "./Data/raw/movies.csv",
            "../Data/raw/movies.csv"
        ]
        for p in raw_paths:
            if Path(p).is_file():
                return pd.read_csv(p)
        raise FileNotFoundError("Could not find movies data file")

def load_processed() -> pd.DataFrame:
    """
    Load the final, model-ready dataset exported by data cleaning.
    Tries a few standard paths; raises FileNotFoundError if not found.
    """
    for p in PROCESSED_PATHS:
        if Path(p).is_file():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("Could not find data/processed/movies_model.csv")

    # Basic hygiene: coerce numeric columns if present
    for col in ["budget_adj", "gross_adj", "runtime", "year", "budget", "gross"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure decade exists
    if "decade" not in df.columns and "year" in df.columns:
        # integer decades like 1980, 1990, ...
        df["decade"] = (df["year"] // 10) * 10

    return df


# public convenience alias
def load_data() -> pd.DataFrame:
    """Alias for load_processed() — kept for the app's expected API."""
    return load_processed()


# -----------------------
# Feature prep
# -----------------------
<<<<<<< HEAD
# Initialize with empty lists, will be updated based on actual data
CATEGORICALS = ["genre", "rating", "decade"]
NUMERICS = ["runtime"]  # Start with just runtime which is more commonly available

def _get_available_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return available numeric and categorical columns"""
    available_numerics = [col for col in NUMERICS if col in df.columns]
    available_categoricals = [col for col in CATEGORICALS if col in df.columns]
    
    # If we don't have any numeric features, try to find some
    if not available_numerics:
        potential_numerics = ["budget", "gross", "runtime", "year", "score", "votes"]
        available_numerics = [col for col in potential_numerics if col in df.columns]
    
    return available_numerics, available_categoricals

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for gross prediction.
    Target is 1 if the movie's gross is above median, 0 otherwise.
    """
    df = df.copy()
    
    # Get available columns
    available_numerics, available_categoricals = _get_available_columns(df)
    
    # If target column doesn't exist, create it based on gross
    if TARGET_COL not in df.columns and 'gross' in df.columns:
        # Mark movies with above-median gross as 1, others as 0
        median_gross = df['gross'].median()
        df[TARGET_COL] = (df['gross'] > median_gross).astype(int)
    elif TARGET_COL not in df.columns:
        # If we don't have gross data, create a dummy target
        df[TARGET_COL] = 0
        if len(df) > 1:
            # Make half the movies "winners" for balance
            df.loc[df.sample(frac=0.5, random_state=42).index, TARGET_COL] = 1

    # Handle missing values in features
    for col in available_numerics + available_categoricals:
        if col in df.columns:
            if col in available_numerics:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
            else:  # Categorical
                df[col] = df[col].fillna("unknown").astype(str)

    # One-hot encode categoricals (only if we have any)
    if available_categoricals:
        X_cat = pd.get_dummies(df[available_categoricals], drop_first=False)
    else:
        X_cat = pd.DataFrame(index=df.index)
        
    # Get numeric features (only if we have any)
    if available_numerics:
        X_num = df[available_numerics].copy()
    else:
        X_num = pd.DataFrame(index=df.index)
=======
# default features — adjust to match your cleaned schema
CATEGORICALS = ["genre", "rating", "decade"]
NUMERICS = ["budget_adj", "runtime"]

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build X, y and feature_cols for training.
    If TARGET_COL not present, derive a proxy label using within-year ratio median.
    Returns: (X, y, feature_cols)
    """
    df = df.copy()

    if TARGET_COL not in df.columns:
        # derive a proxy label if possible
        if "gross_adj" in df.columns and "budget_adj" in df.columns and "year" in df.columns:
            ratio = df["gross_adj"] / df["budget_adj"].replace(0, np.nan)
            df["tmp_ratio"] = ratio
            by_year_median = df.groupby(df["year"])["tmp_ratio"].transform("median")
            df[TARGET_COL] = (df["tmp_ratio"] > by_year_median).astype(int)
        else:
            raise ValueError("Missing target and cannot derive proxy label (need gross_adj, budget_adj, year).")

    # safe handle columns not present
    cat_cols = [c for c in CATEGORICALS if c in df.columns]
    num_cols = [c for c in NUMERICS if c in df.columns]

    # one-hot encode categoricals (drop_first=False so columns are stable)
    X_cat = pd.get_dummies(df[cat_cols].fillna("<<NA>>"), drop_first=False) if cat_cols else pd.DataFrame(index=df.index)
    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
>>>>>>> ddb021d (feat: baseline model pipeline & Streamlit integration)

    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET_COL].astype(int)

<<<<<<< HEAD
    # Ensure we have at least some features
    if X.shape[1] == 0:
        X["dummy_feature"] = 1  # Add a dummy feature if none exist
        
    # Ensure we have a balanced target if possible
    if y.nunique() == 1 and len(y) > 1:
        y.iloc[0] = 1 - y.iloc[0]  # Flip one label to ensure two classes

=======
>>>>>>> ddb021d (feat: baseline model pipeline & Streamlit integration)
    feature_cols = X.columns.tolist()
    return X, y, feature_cols

def prepare_features_for_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for gross prediction (regression).
    Target is the gross value.
    """
    df = df.copy()
    
    # Get available columns
    available_numerics, available_categoricals = _get_available_columns(df)
    
    # Ensure we have a target
    if 'gross' not in df.columns:
        raise ValueError("'gross' column not found in the dataset")
    
    # Handle missing values in features
    for col in available_numerics + available_categoricals:
        if col in df.columns:
            if col in available_numerics:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
            else:  # Categorical
                df[col] = df[col].fillna("unknown").astype(str)
    
    # Select features
    X = df[available_numerics].copy()
    
    # Add one-hot encoded categorical variables
    for cat_col in available_categoricals:
        if cat_col in df.columns:
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
    
    # Ensure we have at least one feature
    if X.empty:
        X['dummy'] = 1  # Add a constant feature if no other features exist
    
    # Get target (gross)
    y = df['gross'].copy()
    
    return X, y, X.columns.tolist()


# -----------------------
# Training (baseline)
# -----------------------
def train_baseline(df: pd.DataFrame, random_state: int = 42):
    """
<<<<<<< HEAD
    Train a regression model to predict gross values directly.
    Returns: model, feature_cols, R2 score (holdout)
    """
    
    # Prepare features and target
    df_clean = df.copy()
    X, y, feature_cols = prepare_features_for_regression(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Create and train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    # Calculate RMSE manually for compatibility with older scikit-learn
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print some diagnostics
    print(f"Training completed with {len(feature_cols)} features")
    print(f"R2 Score: {r2:.3f}, RMSE: ${rmse:,.2f}")
    
    return model, feature_cols, r2
=======
    Train a simple Logistic Regression baseline.
    Returns: (fitted_model, feature_cols, auc_score).
    Note: this trains a plain sklearn LogisticRegression on one-hot / numeric features.
    """
    X, y, feature_cols = prepare_features(df)

    # drop rows with NA in either X or y
    use = pd.concat([X, y], axis=1).dropna()
    if use.shape[0] == 0:
        raise RuntimeError("No rows available for training after dropna; check cleaned dataset for missing values.")

    X_use = use[feature_cols]
    y_use = use[TARGET_COL]

    X_tr, X_te, y_tr, y_te = train_test_split(X_use, y_use, test_size=0.2, random_state=random_state, stratify=y_use)

    model = LogisticRegression(max_iter=200)
    model.fit(X_tr, y_tr)

    yhat = model.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, yhat))
    return model, feature_cols, auc
>>>>>>> ddb021d (feat: baseline model pipeline & Streamlit integration)


# -----------------------
# Integration with model.py (optional)
# -----------------------
def train_baseline_and_save(df: Optional[pd.DataFrame] = None, save_dir: str = "models") -> Dict[str, Any]:
    """
    Higher-level helper: if a model.py with train_baselines exists, call that (recommended).
    Otherwise train a local logistic baseline and save artifacts.
    Returns a dict summarizing saved artifact paths and scores.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_data()

    # Prefer the more advanced training function in model.py if available
    if callable(train_baselines):
        results = train_baselines(df, save_dir=save_dir)
        # results expected to be the train_baselines return dict; leave as-is
        return results

    # fallback: local logistic baseline, saved via joblib
    model, feature_cols, auc = train_baseline(df)
    model_path = save_path / "logistic_baseline.joblib"
    meta_path = save_path / "metadata.joblib"
    joblib.dump(model, model_path)
    joblib.dump({"feature_cols": feature_cols}, meta_path)
    return {"logistic": {"path": str(model_path), "auc": auc}, "meta": {"path": str(meta_path)}}


def load_artifacts(save_dir: str = "models") -> Dict[str, Any]:
    """
    Load saved model artifacts from `save_dir`.
    Expected artifacts:
      - logistic_pipeline.joblib (if trained by model.py) or logistic_baseline.joblib
      - rf_pipeline.joblib (optional)
      - metadata.joblib
    Returns a dict containing loaded objects.
    """
    save_dir = Path(save_dir)
    if model_load_artifacts is not None:
        # prefer model.py loader if available
        try:
            return model_load_artifacts(save_dir)
        except Exception:
            pass

    artifacts: Dict[str, Any] = {}
    # Try common file names:
    candidates = {
        "logistic": ["logistic_pipeline.joblib", "logistic_baseline.joblib", "logistic_pipeline.pkl"],
        "random_forest": ["rf_pipeline.joblib", "rf_pipeline.pkl", "random_forest.joblib"],
        "meta": ["metadata.joblib", "metadata.pkl"],
    }
    for key, names in candidates.items():
        for n in names:
            p = save_dir / n
            if p.exists():
                try:
                    artifacts[key] = joblib.load(p)
                    break
                except Exception:
                    continue
    return artifacts


# -----------------------
# Head-to-head prediction
# -----------------------
def _row_to_features(row: pd.Series, feature_cols: List[str]) -> pd.DataFrame:
<<<<<<< HEAD
    """Convert a single movie row to the same feature columns used in training."""
    # Debug: Show available columns in the row
    print(f"\n=== Processing: {row.get(ID_COL, 'Unknown')} ===")
    print("Available columns in row:", [col for col in row.index if pd.notna(row[col])])
    
    # Initialize with all feature columns set to 0
    x = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # Get numeric features (those that aren't one-hot encoded)
    numeric_cols = [col for col in ['runtime', 'year', 'gross', 'budget', 'score', 'votes'] 
                   if col in feature_cols]
    
    # Handle numeric features
    for col in numeric_cols:
        if col in row and not pd.isna(row[col]):
            try:
                x[col] = float(row[col])
            except (ValueError, TypeError):
                x[col] = 0.0
    
    # Handle categorical features (genre, rating, decade)
    for col_type in ['genre', 'rating', 'decade']:
        if col_type in row and not pd.isna(row[col_type]):
            # Create the one-hot encoded column name
            encoded_col = f"{col_type}_{row[col_type]}"
            if encoded_col in feature_cols:
                x[encoded_col] = 1
    
    # Debug: Show what features were set
    non_zero = x.loc[:, (x != 0).any()].to_dict('records')
    print(f"Features set: {non_zero[0] if non_zero else 'None'}")
    
=======
    """
    Convert a single movie row to a DataFrame with the same feature columns used for training.
    Missing columns will be filled with zeros.
    """
    # build the numeric and categorical dicts from known lists (if present on row)
    cat_dict = {k: row.get(k, np.nan) for k in CATEGORICALS if k in row.index}
    num_dict = {k: row.get(k, np.nan) for k in NUMERICS if k in row.index}

    cat_df = pd.get_dummies(pd.DataFrame([cat_dict]).fillna("<<NA>>"), drop_first=False) if cat_dict else pd.DataFrame(index=[0])
    num_df = pd.DataFrame([num_dict]) if num_dict else pd.DataFrame(index=[0])

    x = pd.concat([num_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    # ensure all training cols present
    for c in feature_cols:
        if c not in x.columns:
            x[c] = 0
    x = x[feature_cols]
>>>>>>> ddb021d (feat: baseline model pipeline & Streamlit integration)
    return x


def predict_gross(movie_a: str, movie_b: str,
<<<<<<< HEAD
                  df: pd.DataFrame,
                  model: Any,
                  feature_cols: List[str]) -> Dict:
    """
    Predict which movie will have a higher gross.
    Returns:
        dict: Contains prediction results and any warnings
    """
    print("\n=== Starting Prediction ===")
    print(f"Comparing: {movie_a} vs {movie_b}")
    
    # Initialize output
    out = {
        "ok": True,
        "warnings": [],
        "gross_a": 0,
        "gross_b": 0,
        "predicted_gross_a": 0,
        "predicted_gross_b": 0,
        "predicted_winner": "Tie"
    }

    try:
        # Find movie rows (case-insensitive match on movie name)
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

        # Get most recent version if there are duplicates
        ra = rows_a.sort_values("year", ascending=False).iloc[0]
        rb = rows_b.sort_values("year", ascending=False).iloc[0]

        print("\n=== Movie A Data ===")
        print(ra[['name', 'year', 'genre', 'rating', 'score', 'gross', 'budget']].to_dict())
        
        print("\n=== Movie B Data ===")
        print(rb[['name', 'year', 'genre', 'rating', 'score', 'gross', 'budget']].to_dict())

        # Get predicted gross values
        xa = _row_to_features(ra, feature_cols)[feature_cols]
        xb = _row_to_features(rb, feature_cols)[feature_cols]
        
        print("\n=== Feature Vectors ===")
        print("Movie A features (non-zero):", xa.loc[:, (xa != 0).any()].to_dict('records')[0])
        print("Movie B features (non-zero):", xb.loc[:, (xb != 0).any()].to_dict('records')[0])
        
        # Get predictions from model
        try:
            # Get predicted gross values
            pred_gross_a = model.predict(xa)[0]
            pred_gross_b = model.predict(xb)[0]
            
            # Store predictions
            out["predicted_gross_a"] = pred_gross_a
            out["predicted_gross_b"] = pred_gross_b
            
            # Use actual gross if available, otherwise use predicted
            actual_gross_a = ra.get('gross', pred_gross_a)
            actual_gross_b = rb.get('gross', pred_gross_b)
            
            out["gross_a"] = actual_gross_a
            out["gross_b"] = actual_gross_b
            
            # Determine winner
            if actual_gross_a > actual_gross_b:
                out["predicted_winner"] = movie_a
            elif actual_gross_b > actual_gross_a:
                out["predicted_winner"] = movie_b
            else:
                out["predicted_winner"] = "Tie"
            
            print("\n=== Predicted Gross Values ===")
            print(f"{movie_a}: ${pred_gross_a:,.2f} (actual: ${actual_gross_a:,.2f} if available)")
            print(f"{movie_b}: ${pred_gross_b:,.2f} (actual: ${actual_gross_b:,.2f} if available)")
            print(f"\n=== Predicted Winner ===")
            print(f"{out['predicted_winner']} is predicted to have higher gross")
            
        except Exception as e:
            out["warnings"].append(f"Prediction warning: {str(e)}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        out["ok"] = False
        out["warnings"].append(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
=======
                  df: Optional[pd.DataFrame] = None,
                  model: Optional[Any] = None,
                  feature_cols: Optional[List[str]] = None,
                  save_dir: str = "models",
                  model_name: str = "random_forest") -> Dict[str, Any]:
    """
    Score each selected movie and return a dictionary with results:
    - ok: bool
    - warnings: list[str]
    - proba_a, proba_b: float (0..1) or None if error
    - predicted_winner: title or "Tie"
    - error: optional error message

    movie_a/movie_b: movie titles (case-insensitive) OR integer index.
    If `model` is not provided, we will try to load artifacts from `save_dir`.
    """
    out: Dict[str, Any] = {"ok": True, "warnings": [], "movie_a": movie_a, "movie_b": movie_b}

    if df is None:
        try:
            df = load_data()
        except FileNotFoundError as e:
            return {"ok": False, "error": str(e)}

    # find rows: accept title match or integer index
    def _find_row(key):
        # integer index?
        try:
            if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
                idx = int(key)
                if idx in df.index:
                    return df.loc[idx]
        except Exception:
            pass

        # title match (case-insensitive)
        if ID_COL in df.columns and isinstance(key, str):
            matches = df[df[ID_COL].astype(str).str.casefold() == key.casefold()]
            if not matches.empty:
                return matches.sort_values("year", ascending=False).iloc[0]
        # last resort: if string present in title
        if ID_COL in df.columns and isinstance(key, str):
            contains = df[df[ID_COL].astype(str).str.casefold().str.contains(key.casefold())]
            if not contains.empty:
                return contains.sort_values("year", ascending=False).iloc[0]
        return None

    ra = _find_row(movie_a)
    rb = _find_row(movie_b)

    if ra is None or rb is None:
        out["ok"] = False
        out["warnings"].append("One or both movies not found in processed data.")
        out["error"] = "movie_not_found"
        return out

    # load model artifacts if needed
    artifacts = {}
    if model is None:
        artifacts = load_artifacts(save_dir)
        # pick model
        model = artifacts.get(model_name) or artifacts.get("random_forest") or artifacts.get("logistic")
        meta = artifacts.get("meta") or artifacts.get("metadata") or {}
        feature_cols = feature_cols or meta.get("feature_cols")
    else:
        # model provided; require feature_cols argument
        feature_cols = feature_cols or []

    if model is None:
        out["ok"] = False
        out["error"] = "no_model_available"
        out["warnings"].append("No trained model found in models/ and no model passed to function.")
        return out

    if not feature_cols:
        # attempt to infer feature columns from sklearn objects
        if hasattr(model, "named_steps"):
            # pipeline: assume preprocessor drops into final numeric feature vector length,
            # but we cannot easily reconstruct column names—require metadata.
            out["warnings"].append("Feature columns not supplied; predictions may be unreliable.")
        # but proceed with empty list -> will fill zeros for requested columns below
        feature_cols = feature_cols or []

    # check for missing required features in selected rows
    missing_a = [c for c in (NUMERICS + CATEGORICALS) if (c in ra.index and pd.isna(ra.get(c)))]
    missing_b = [c for c in (NUMERICS + CATEGORICALS) if (c in rb.index and pd.isna(rb.get(c)))]
    if missing_a:
        out["warnings"].append(f"{movie_a}: missing {missing_a}")
    if missing_b:
        out["warnings"].append(f"{movie_b}: missing {missing_b}")

    # build feature rows consistent with training feature list
    if not feature_cols:
        # try a reasonable fallback: use numeric + one-hot of available categoricals
        # This fallback produces columns but may not match training exactly
        tmp_cat = [c for c in CATEGORICALS if c in ra.index or c in rb.index]
        tmp_num = [c for c in NUMERICS if c in ra.index or c in rb.index]
        # create temporary feature list from the union of columns present
        xr = _row_to_features(ra, [])
        xb = _row_to_features(rb, [])
        # union the columns
        feature_cols = sorted(set(xr.columns.tolist()) | set(xb.columns.tolist()))

    xa = _row_to_features(ra, feature_cols)
    xb = _row_to_features(rb, feature_cols)

    # ensure model supports predict_proba; if not, use predict (0/1) as fallback
    try:
        pa = float(model.predict_proba(xa)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(xa)[0])
        pb = float(model.predict_proba(xb)[:, 1][0]) if hasattr(model, "predict_proba") else float(model.predict(xb)[0])
    except Exception as e:
        out["ok"] = False
        out["error"] = f"prediction_error: {e}"
        out["warnings"].append("Model failed to produce probabilities for the selected movies.")
        return out

    out["proba_a"] = pa
    out["proba_b"] = pb
    out["predicted_winner"] = ra.get(ID_COL, "Movie A") if pa >= pb else rb.get(ID_COL, "Movie B")
>>>>>>> ddb021d (feat: baseline model pipeline & Streamlit integration)
    return out


# -----------------------
# Quick insights
# -----------------------
def generate_insights(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Return a few simple, user-facing insights (top genres by median adj. gross).
    """
    if df is None:
        try:
            df = load_data()
        except FileNotFoundError:
            return pd.DataFrame()

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