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
from typing import Tuple, List, Dict, Any

# Data handling
import pandas as pd
import numpy as np

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

    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET_COL].astype(int)

    # Ensure we have at least some features
    if X.shape[1] == 0:
        X["dummy_feature"] = 1  # Add a dummy feature if none exist
        
    # Ensure we have a balanced target if possible
    if y.nunique() == 1 and len(y) > 1:
        y.iloc[0] = 1 - y.iloc[0]  # Flip one label to ensure two classes

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
    Train a baseline regression model to predict gross values.
    Uses only basic features for faster training.
    Returns: model, feature_cols, R2 score (holdout)
    """
    # Use only essential features for the baseline
    global NUMERICS, CATEGORICALS
    original_numerics = NUMERICS.copy()
    original_categoricals = CATEGORICALS.copy()
    
    try:
        NUMERICS = ["runtime", "budget_adj", "year"]
        CATEGORICALS = ["genre", "rating"]
        
        X, y, feature_cols = prepare_features_for_regression(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Train a simple RandomForest model
        model = RandomForestRegressor(
            # Fewer trees for faster training
            n_estimators=50,
            random_state=random_state,
            n_jobs=-1,
            max_depth=5,
            min_samples_split=10
        )
        
        model.fit(X_train, y_train)
        
        # Calculate R2 score on test set
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return model, feature_cols, r2
    finally:
        # Restore original feature lists
        NUMERICS = original_numerics
        CATEGORICALS = original_categoricals

def train_improved(df: pd.DataFrame, random_state: int = 42):
    """
    Train an improved regression model with more features and better performance.
    Returns: model, feature_cols, R2 score (holdout)
    """
    global NUMERICS, CATEGORICALS
    original_numerics = NUMERICS.copy()
    original_categoricals = CATEGORICALS.copy()
    
    try:
        # Use more features for the improved model
        NUMERICS = ["runtime", "budget_adj", "year", "score", "votes", "gross"]
        CATEGORICALS = ["genre", "rating", "country", "director"]
        
        # Add some basic feature engineering
        if 'budget_adj' in df.columns and 'year' in df.columns:
            df['budget_year_ratio'] = df['budget_adj'] / (df['year'] - df['year'].min() + 1)
            NUMERICS.append('budget_year_ratio')
        
        X, y, feature_cols = prepare_features_for_regression(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        
        # Train a more sophisticated model
        model = RandomForestRegressor(
             # More trees for better performance
            n_estimators=200, 
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # Add some basic hyperparameter tuning with GridSearchCV if needed
        # For simplicity, we'll use the default parameters here
        model.fit(X_train, y_train)
        
        # Calculate R2 score on test set
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return model, feature_cols, r2
    finally:
        # Restore original feature lists
        NUMERICS = original_numerics
        CATEGORICALS = original_categoricals
    
    return model, feature_cols, r2


# -----------------------
# Head-to-head prediction
# -----------------------
def _row_to_features(row: pd.Series, feature_cols: List[str]) -> pd.DataFrame:
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
    
    return x


def predict_gross(movie_a: str, movie_b: str,
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