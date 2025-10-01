# app.py
"""
Streamlit entry point: 'Which Movie Will Gross More?'

Scaffold only — wires to apputil stubs and provides a clean UX for:
- picking two movies
- running a prediction
- showing placeholder insights

To be replaced later with real data, models, and visuals.
"""

from __future__ import annotations

import streamlit as st
import joblib
import pandas as pd
from typing import Any, Iterable

# Import helpers from apputil.py
from apputil import load_processed, train_baseline, load_data, predict_gross, generate_insights

# ---------- Page setup ----------
st.set_page_config(page_title="Which Movie Will Gross More?", layout="wide")

with st.sidebar:
    st.header("About")
    st.write(
        "Compare two films based on release-time features to predict which "
        "is likely to have the higher **inflation-adjusted domestic gross**."
    )
    st.caption("Dataset: Movie Industry (Daniel Grijalva, Kaggle).")

st.title("Which Movie Will Gross More?")

# ---------- Data loading ----------
@st.cache_data(show_spinner=True)
def _get_data() -> pd.DataFrame:
    # load_data() is defined in apputil.py
    df = load_data()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("Loaded dataset is empty or not a DataFrame.")
    return df

error_box = st.empty()
try:
    df = _get_data()
except Exception as e:
    error_box.error(f"Failed to load data: {e}")
    st.stop()

# Figure out the title column name robustly
def _first_present(cols: Iterable[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None

title_col = _first_present(["title", "Title", "movie_title", "Movie", "name", "Name"])
if title_col is None:
    error_box.error(
        "Could not find a movie title column in the dataset. "
        "Please add a 'title' (or similar) column."
    )
    st.stop()

movie_list = (
    df[title_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique()
)
movie_list = sorted(movie_list.tolist())

st.write(
    "Pick two movies to compare. If one has missing key features, "
    "the app will tell you what needs attention."
)

# ---------- UI: two selectors ----------
col1, col2 = st.columns(2)
with col1:
    movie_a = st.selectbox("Movie A", movie_list, index=0 if movie_list else None)
with col2:
    default_b = 1 if len(movie_list) > 1 else 0
    movie_b = st.selectbox("Movie B", movie_list, index=default_b if movie_list else None)

st.divider()

# ---------- Predict ----------
@st.cache_resource  # cache model across reruns/users (Streamlit guidance)
def get_model_bundle():
    df = load_processed()
    model, feature_cols, auc = train_baseline(df)  # quick baseline retrain
    return df, model, feature_cols, auc

if st.button("Predict"):
    df, model, feature_cols, auc = get_model_bundle()
    res = predict_gross(movie_a, movie_b, df, model, feature_cols)

    if not res["ok"]:
        st.error("Oops — " + "; ".join(res["warnings"]))
    else:
        st.success(f"Predicted winner: **{res['predicted_winner']}**")
        st.caption(f"Baseline AUC (holdout): {auc:.3f}")
        with st.expander("Details"):
            st.write(res)

st.divider()

# ---------- Insights ----------
st.subheader("Insights")
st.caption("Quick diagnostics and exploratory summaries based on the current dataset.")

try:
    insights = generate_insights(df)  # from apputil.py
    if insights is None:
        st.write("No insights available yet.")
    elif isinstance(insights, (list, tuple)):
        for i, item in enumerate(insights, start=1):
            # Support Plotly figs, Altair charts, Matplotlib figs, or plain strings
            if hasattr(item, "to_dict") and hasattr(item, "to_json"):
                st.plotly_chart(item, use_container_width=True)
            elif hasattr(item, "mark_point") or getattr(item, "_class_name", "") == "Chart":
                st.altair_chart(item, use_container_width=True)  # if you ever use Altair
            elif str(type(item)).startswith("<class 'matplotlib"):
                st.pyplot(item)
            elif isinstance(item, str):
                st.write(item)
            else:
                st.write(item)
    else:
        # Single object fallback
        if hasattr(insights, "to_dict") and hasattr(insights, "to_json"):
            st.plotly_chart(insights, use_container_width=True)
        else:
            st.write(insights)
except Exception as e:
    st.info(f"(Insights pending) {e}")

# ---------- Footer ----------
st.divider()
st.caption("Built with Streamlit • Group 1 • Intro to DS Programming")
