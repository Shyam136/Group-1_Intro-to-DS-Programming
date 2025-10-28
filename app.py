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
from apputil import (
    load_processed, 
    train_baseline, 
    train_improved,
    load_data, 
    predict_gross, 
    generate_insights
)

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

# ---------- Model Selection ----------
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Baseline (Faster)", "Improved (More Accurate)"],
    help="Baseline is faster but less accurate. Improved uses more features but may be slower."
)

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
def get_model_bundle(use_improved: bool = False):
    df = load_processed()
    if use_improved:
        model, feature_cols, score = train_improved(df)
    else:
        model, feature_cols, score = train_baseline(df)
    return df, model, feature_cols, score

if st.button("Predict"):
    use_improved = model_type == "Improved (More Accurate)"
    df, model, feature_cols, score = get_model_bundle(use_improved=use_improved)
    res = predict_gross(movie_a, movie_b, df, model, feature_cols)

    if not res["ok"]:
        st.error("Oops — " + "; ".join(res["warnings"]))
    else:
        # Get predicted and actual gross values
        pred_gross_a = res.get('predicted_gross_a', 0)
        pred_gross_b = res.get('predicted_gross_b', 0)
        actual_gross_a = res.get('gross_a', 0)
        actual_gross_b = res.get('gross_b', 0)
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        
        # Add custom CSS for larger delta value and arrow
        st.markdown("""
        <style>
            [data-testid="stMetricDelta"] div {
                font-size: 1.2rem !important;
                font-weight: bold !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Helper function to get delta style
        def get_delta_style(actual, predicted):
            if actual == 0 or predicted == 0:
                return "normal"
            return "inverse" if predicted < actual else "normal"
        
        with col1:
            # Get delta style for movie A
            delta_color_a = get_delta_style(actual_gross_a, pred_gross_a)
            delta_text_a = f"Predicted: ${pred_gross_a:,.2f}" if pred_gross_a > 0 else None
            
            # Show the metric with actual as main value and prediction as delta
            st.metric(
                movie_a,
                f"Actual: ${actual_gross_a:,.2f}" if actual_gross_a > 0 else "Actual: N/A",
                delta=delta_text_a,
                delta_color=delta_color_a
            )
        
        with col2:
            # Get delta style for movie B
            delta_color_b = get_delta_style(actual_gross_b, pred_gross_b)
            delta_text_b = f"Predicted: ${pred_gross_b:,.2f}" if pred_gross_b > 0 else None
            
            # Show the metric with actual as main value and prediction as delta
            st.metric(
                movie_b,
                f"Actual: ${actual_gross_b:,.2f}" if actual_gross_b > 0 else "Actual: N/A",
                delta=delta_text_b,
                delta_color=delta_color_b
            )
        
        # Show winner
        if res['predicted_winner'] != "Tie":
            st.success(f"🎬 **{res['predicted_winner']}** is predicted to have higher gross!")
        else:
            st.info("🤝 It's a tie! Both movies are predicted to have similar gross.")
        
        # Show any warnings in a collapsible section
        if res.get('warnings'):
            with st.expander("⚠️ Note"):
                st.warning("\n".join(res['warnings']))

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
