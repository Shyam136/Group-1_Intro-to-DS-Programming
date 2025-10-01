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

# Import your module stubs (Shyam's file)
try:
    from apputil import load_data, predict_gross, generate_insights
except Exception:
    # Fallback no-op stubs so the page still runs before apputil exists.
    def load_data():
        # TODO: replace with real loader; keep names unique and sorted
        return {"title": [
            "The Dark Knight", "Inception", "Interstellar", "The Matrix"
        ]}

    def predict_gross(movie_a: str, movie_b: str):
        # TODO: replace with real prediction tuple (winner, details)
        winner = movie_a if len(movie_a) >= len(movie_b) else movie_b
        return {
            "winner": winner,
            "confidence": 0.50,
            "notes": "Placeholder model — for scaffold only."
        }

    def generate_insights(_data):
        # TODO: replace with real insight dict or DataFrame
        return {"rows": 7600, "features": 15, "years": "1980–2020 (approx.)"}


# --- Page setup ---
st.set_page_config(
    page_title="Which Movie Will Gross More?",
    page_icon="🎬",
    layout="wide",
    menu_items={
        "About": "Prototype app to compare two films and predict which "
                 "will gross more domestically."
    },
)

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.write(
        "Compare two films using metadata and a simple predictive model. "
        "This is a work-in-progress scaffold."
    )
    st.caption(
        "Dataset: Movie Industry (Daniel Grijalvas – Kaggle). "
        "Exact cleaning/adjustments will be documented in the README."
    )

    with st.expander("What’s coming next"):
        st.markdown(
            "- Data cleaning & inflation adjustments\n"
            "- Feature engineering (genre, budget, release window, etc.)\n"
            "- Model training & evaluation\n"
            "- Model cards & limitation notes"
        )

# --- Title & intro ---
st.title("Which Movie Will Gross More?")

st.write(
    "Pick two movies below and run a placeholder prediction. "
    "We’ll swap in the real model once the data pipeline is ready."
)

# --- Dataset quick facts ---
with st.container():
    st.subheader("About the dataset")
    info = generate_insights(None)
    st.markdown(
        f"- Approx. rows: **{info.get('rows', 'TBD')}**  \n"
        f"- Features: **{info.get('features', 'TBD')}**  \n"
        f"- Coverage: **{info.get('years', 'TBD')}**  \n"
        "Fields typically include: title, genre, rating, budget, gross, "
        "runtime, release year."
    )

st.divider()

# --- Load titles once (swap to @st.cache_data inside apputil) ---
data = load_data()
titles = sorted(set(data.get("title", [])))  # keep unique & sorted
if not titles:
    st.error("No titles available yet. Please load data in `apputil.load_data()`.")
    st.stop()

# --- Input form ---
st.subheader("Select two movies to compare")
with st.form("compare_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        movie_a = st.selectbox("Movie A", titles, index=0, key="movie_a")
    with col2:
        movie_b = st.selectbox("Movie B", titles, index=1, key="movie_b")

    invalid = (movie_a == movie_b)
    if invalid:
        st.warning("Please choose two *different* movies.")

    submitted = st.form_submit_button(
        "Predict",
        disabled=invalid,
        use_container_width=True
    )

# --- Results area ---
result_col, meta_col = st.columns([2, 1], gap="large")

if submitted:
    result = predict_gross(movie_a, movie_b)
    error = result.get("error")
    
    if error:
        with result_col:
            st.error(f"{error}")
            if result.get("errorMessage"):
                st.info(result["errorMessage"])
    else:
        winner = result.get("winner")
        confidence = result.get("confidence", None)
        error_message = result.get("errorMessage", "")

        with result_col:
            if winner:
                st.success(f"**Predicted winner:** {winner}")
                if confidence > 0:
                    st.caption(f"Confidence: {confidence:.0%}")
                if error_message:
                    st.warning(error_message)
            else:
                st.warning("Could not determine a winner.")
                if error_message:
                    st.info(error_message)

        with meta_col:
            if winner:
                st.markdown("**Why this result?**")
                st.write(
                    "Feature contributions and charts will appear here once the "
                    "real model is integrated (e.g., budget, release year, genre)."
                )
else:
    with result_col:
        st.info("Choose two movies and click **Predict** to see the result.")
