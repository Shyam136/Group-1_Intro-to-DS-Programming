"""
apputil.py — utilities for the Streamlit app

Scaffold functions:
- load_data(...)         -> dict-like payload with at least a "title" list
- predict_gross(a, b, _) -> lightweight, deterministic placeholder model
- generate_insights(df)  -> small summary used on the landing page

Design goals:
- Zero external deps beyond pandas/numpy.
- Works without the dataset; upgrades automatically when a CSV is present.
- No Streamlit imports; optional caching works with or without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Optional cache decorator: uses st.cache_data if present, else lru_cache
# ---------------------------------------------------------------------
def _optional_cache(func):
    try:
        import streamlit as st  # noqa: WPS433 (import inside function)
        return st.cache_data(show_spinner=False)(func)
    except Exception:
        return lru_cache(maxsize=1)(func)


# ---------------------------------------------------------------------
# Data model (handy for future expansion, tests, and IDE hints)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class AppData:
    """Container for core fields we care about."""
    df: pd.DataFrame
    titles: Tuple[str, ...]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
@_optional_cache
def load_data(source: Optional[str | Path] = None) -> AppData:
    """
    Load the movie dataset, returning an AppData object.

    Parameters
    ----------
    source : str | Path | None
        Optional CSV path/URL. If None, tries common local paths:
        - ./data/movies.csv
        - ./movies.csv
        If nothing is found, returns a tiny demo dataset.

    Returns
    -------
    AppData
        - df: pandas DataFrame (may be demo data)
        - titles: tuple of unique movie titles (sorted)
    """
    # 1) Resolve a source CSV if provided or discover one locally.
    csv_candidates: Iterable[Path | str] = (
        [source] if source is not None else ["./data/movies.csv", "./movies.csv"]
    )

    df: Optional[pd.DataFrame] = None
    last_err: Optional[Exception] = None

    for candidate in csv_candidates:
        try:
            if candidate is None:
                continue
            p = Path(candidate)
            if (p.is_file()) or (isinstance(candidate, str) and candidate.startswith(("http://", "https://"))):
                df = pd.read_csv(candidate)
                break
        except Exception as exc:
            last_err = exc

    # 2) If no real data available, provide a minimal demo frame.
    if df is None:
        # Demo set (safe to ship; keeps the app functional)
        demo_titles = ["The Dark Knight", "Inception", "Interstellar", "The Matrix"]
        df = pd.DataFrame(
            {
                "title": demo_titles,
                # If you later have real columns like 'budget' or 'gross',
                # the predict_gross function will automatically use them.
                "year": [2008, 2010, 2014, 1999],
                "gross": [534.9, 292.6, 188.0, 171.5],  # USD (placeholder)
                "budget": [185.0, 160.0, 165.0, 63.0],  # USD (placeholder)
            }
        )

    # 3) Light, defensive cleaning (non-destructive).
    if "title" not in df.columns:
        # Make a best-effort title from any name-like column.
        for c in ("movie", "name"):
            if c in df.columns:
                df = df.rename(columns={c: "title"})
                break
    df["title"] = df["title"].astype(str).str.strip()

    # Normalize numeric columns if present
    for col in ("gross", "domestic_gross", "budget", "runtime", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    titles = tuple(sorted(t for t in df["title"].dropna().unique() if t))

    return AppData(df=df, titles=titles)


def predict_gross(
    movie_a: str,
    movie_b: str,
    data: Optional[AppData] = None,
) -> Dict[str, object]:
    """
    Predict which of two movies will have higher domestic gross (placeholder).

    Logic (tiered, deterministic):
    1) If a 'domestic_gross' column exists, use it directly.
    2) Else if 'gross' exists, use that.
    3) Else compute a simple heuristic score from (budget if available,
       otherwise a title-based stable hash).

    Parameters
    ----------
    movie_a, movie_b : str
        Titles chosen by the user (must exist in data.titles).
    data : AppData | None
        Output of load_data(); if None, this function will call load_data().

    Returns
    -------
    dict
        {
          "winner": <title>,
          "confidence": <float in 0..1 (placeholder)>,
          "scores": {title: score, ...},
          "notes": <brief explanation string>
        }
    """
    if data is None:
        data = load_data()

    df = data.df.copy()

    # Filter to the two candidates
    subset = df[df["title"].isin([movie_a, movie_b])].copy()
    if subset.empty or subset["title"].nunique() < 2:
        raise ValueError("Both selected titles must exist and be distinct.")

    # Choose the best available signal
    signal_col = None
    for candidate in ("domestic_gross", "domestic", "gross"):
        if candidate in subset.columns:
            signal_col = candidate
            break

    scores: Dict[str, float] = {}

    if signal_col is not None:
        # Use the numeric signal directly
        agg = (
            subset.groupby("title")[signal_col]
            .mean()  # if duplicates exist, average them
            .to_dict()
        )
        scores = {movie_a: float(agg.get(movie_a, np.nan)),
                  movie_b: float(agg.get(movie_b, np.nan))}
        notes = f"Used column '{signal_col}' as the prediction signal."
    else:
        # Heuristic fallback:
        # Prefer budget if available, else a stable title-based score.
        def fallback_score(row) -> float:
            if "budget" in row and pd.notnull(row["budget"]):
                return float(row["budget"])
            # Stable, deterministic pseudo-score from the title
            return float(abs(hash(row["title"])) % 10_000)

        subset["score"] = subset.apply(fallback_score, axis=1)
        agg = subset.groupby("title")["score"].mean().to_dict()
        scores = {movie_a: float(agg.get(movie_a, 0.0)),
                  movie_b: float(agg.get(movie_b, 0.0))}
        notes = (
            "No explicit gross columns found. "
            "Used budget if present, otherwise a deterministic title-based score."
        )

    # Pick the winner and a placeholder “confidence”
    a_score, b_score = scores[movie_a], scores[movie_b]
    if np.isnan(a_score) and np.isnan(b_score):
        raise ValueError("Insufficient numeric data to compare the two titles.")

    if (not np.isnan(a_score)) and (np.isnan(b_score)):
        winner, confidence = movie_a, 0.75
    elif (not np.isnan(b_score)) and (np.isnan(a_score)):
        winner, confidence = movie_b, 0.75
    else:
        if a_score == b_score:
            winner, confidence = (movie_a, 0.5)  # deterministic tie-breaker
        else:
            winner = movie_a if a_score > b_score else movie_b
            # naive confidence: sigmoid-like based on relative gap
            denom = max(a_score, b_score)
            gap = abs(a_score - b_score) / (denom if denom else 1.0)
            confidence = float(min(0.95, max(0.55, 0.55 + gap / 2)))

    return {
        "winner": winner,
        "confidence": confidence,
        "scores": scores,
        "notes": notes,
    }


def generate_insights(data: Optional[AppData] = None) -> Dict[str, object]:
    """
    Produce quick, safe summary stats for the landing page.

    Parameters
    ----------
    data : AppData | None
        Output of load_data(); if None, this function will call load_data().

    Returns
    -------
    dict
        {
          "rows": <int>,
          "features": <int>,
          "years": "<min–max or TBD>",
          "has_domestic_gross": <bool>,
          "has_gross": <bool>,
        }
    """
    if data is None:
        data = load_data()

    df = data.df
    rows, features = int(df.shape[0]), int(df.shape[1])

    years = "TBD"
    if "year" in df.columns:
        yr_min = pd.to_numeric(df["year"], errors="coerce").min()
        yr_max = pd.to_numeric(df["year"], errors="coerce").max()
        if pd.notnull(yr_min) and pd.notnull(yr_max):
            years = f"{int(yr_min)}–{int(yr_max)}"

    return {
        "rows": rows,
        "features": features,
        "years": years,
        "has_domestic_gross": "domestic_gross" in df.columns,
        "has_gross": "gross" in df.columns,
    }


# ---------------------------------------------------------------------
# (Optional) Quick local checks (won't run in Streamlit)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()  # demo or local CSV
    assert len(data.titles) >= 2
    res = predict_gross(data.titles[0], data.titles[1], data)
    assert "winner" in res and "confidence" in res
    info = generate_insights(data)
    assert {"rows", "features", "years"}.issubset(info.keys())
    print("apputil self-checks passed.")