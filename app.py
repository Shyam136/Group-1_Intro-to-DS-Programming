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
import pandas as pd
from typing import Iterable

# Import helpers from apputil.py
from apputil import (
    load_processed, 
    train_baseline, 
    train_improved,
    load_data, 
    predict_gross, 
    generate_insights,
    ID_COL,
    _row_to_features,
)

tab1, tab2 = st.tabs(["Main App", "MVP Features"])

# ---------- Page setup ----------

with tab1:
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
        from datetime import datetime
        
        df = load_processed()
        if use_improved:
            model, feature_cols, score = train_improved(df)
            model_type = "improved"
        else:
            model, feature_cols, score = train_baseline(df)
            model_type = "baseline"
            
        # Add timestamp for when the model was trained
        train_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return df, model, feature_cols, score, train_timestamp, model_type

    if st.button("Predict"):
        use_improved = model_type == "Improved (More Accurate)"
        df, model, feature_cols, score, train_timestamp, model_type_name = get_model_bundle(use_improved=use_improved)
        
        # Display model info
        st.sidebar.subheader("Model Information")
        st.sidebar.metric("Model Type", model_type_name.capitalize())
        st.sidebar.metric("R² Score", f"{score:.3f}")
        st.sidebar.caption(f"Last trained: {train_timestamp}")
        
        # Get movie years for display
        def get_movie_year(movie_name):
            movie_row = df[df[title_col].str.casefold() == movie_name.casefold()].iloc[0]
            return int(movie_row['year']) if 'year' in movie_row and pd.notna(movie_row['year']) else None
            
        year_a = get_movie_year(movie_a)
        year_b = get_movie_year(movie_b)
        
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
            
            # Add custom CSS for better styling
            st.markdown("""
            <style>
                .stMarkdown h3 {
                    margin-top: 0;
                }
                .stMarkdown p {
                    margin-bottom: 0.5rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            def get_prediction_style(actual, predicted, value):
                """
                Returns a tuple of (formatted_text, color, bg_color) based on the comparison
                between actual and predicted values.
                """
                if predicted > actual:
                    # green background
                    return f"↑ ${value:,.2f}", "white", "green" 
                elif predicted < actual:
                    # darkred background
                    return f"↓ ${value:,.2f}", "white" , "darkred" 
                else:
                    # gray background
                    return f"→ ${value:,.2f}", "white", "gray"  
            
            with col1:
                # Get prediction style for movie A
                delta_text_a, text_color_a, bg_color_a = get_prediction_style(actual_gross_a, pred_gross_a, pred_gross_a)
                
                # Format the actual gross value
                actual_display = f"${actual_gross_a:,.2f}" if actual_gross_a > 0 else "N/A"
                year_display = f" ({year_a})"
                
                # Show the metric with actual as main value and prediction with arrow
                st.markdown(f"""
                <div style="margin-bottom: 1.5rem;">
                    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.5rem;">{movie_a}{year_display}</h3>
                    <p style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: 500;">
                        Actual: {actual_display}
                    </p>
                    <div style="
                        display: inline-block;
                        background-color: {bg_color_a};
                        color: {text_color_a};
                        padding: 0.75rem 1.25rem;
                        border-radius: 2rem;
                        font-size: 1.8rem;
                        font-weight: bold;
                        margin: 0.5rem 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        {delta_text_a if delta_text_a else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Get prediction style for movie B
                delta_text_b, text_color_b, bg_color_b = get_prediction_style(actual_gross_b, pred_gross_b, pred_gross_b)
                
                # Format the actual gross value
                actual_display = f"${actual_gross_b:,.2f}" if actual_gross_b > 0 else "N/A"
                year_display = f" ({year_b})"
                
                # Show the metric with actual as main value and prediction with arrow
                st.markdown(f"""
                <div style="margin-bottom: 1.5rem;">
                    <h3 style="margin: 0 0 0.5rem 0; font-size: 1.5rem;">{movie_b}{year_display}</h3>
                    <p style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: 500;">
                        Actual: {actual_display}
                    </p>
                    <div style="
                        display: inline-block;
                        background-color: {bg_color_b};
                        color: {text_color_b};
                        padding: 0.75rem 1.25rem;
                        border-radius: 2rem;
                        font-size: 1.8rem;
                        font-weight: bold;
                        margin: 0.5rem 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        {delta_text_b if delta_text_b else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            def calculate_confidence(model, xa, xb):
                """Calculate confidence based on model's predictions across all trees"""
                # Convert to numpy arrays to avoid feature name warnings
                xa_array = xa.values.reshape(1, -1)
                xb_array = xb.values.reshape(1, -1)
                
                # Get predictions from all trees for both movies
                preds_a = [tree.predict(xa_array)[0] for tree in model.estimators_]
                preds_b = [tree.predict(xb_array)[0] for tree in model.estimators_]
                
                # Count how often movie A is predicted to gross more than B
                a_wins = sum(1 for a, b in zip(preds_a, preds_b) if a > b)
                confidence = (a_wins / len(preds_a)) * 100
                
                # Return confidence between 50-100% (since we don't care which is higher, just how certain)
                return max(min(confidence, 95), 100 - confidence)

            # Calculate confidence using model's predictions
            try:
                # Get the feature vectors for both movies
                xa = _row_to_features(df[df[ID_COL].str.casefold() == movie_a.casefold()].iloc[0], feature_cols)[feature_cols]
                xb = _row_to_features(df[df[ID_COL].str.casefold() == movie_b.casefold()].iloc[0], feature_cols)[feature_cols]
                confidence = calculate_confidence(model, xa, xb)
            except Exception as e:
                print(f"Error calculating confidence: {e}")
                #fallback value
                confidence = 60
            
            # Show winner with confidence indicator
            if res['predicted_winner'] != "Tie":
                # Color code based on confidence level
                if confidence >= 75:
                    # Green for high confidence
                    color = "#2ecc71"  
                    confidence_text = "High confidence"
                elif confidence >= 60:
                    # Orange for moderate confidence
                    color = "#f39c12"  
                    confidence_text = "Moderate confidence"
                else:
                    # Red for low confidence
                    color = "#e74c3c"  
                    confidence_text = "Low confidence"
                
                # Display confidence bar and text
                st.markdown(f"""
                <div style='margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>Prediction Confidence:</span>
                        <span style='font-weight: bold; color: {color}'>{confidence_text} ({confidence:.0f}%)</span>
                    </div>
                    <div style='height: 8px; background: #ecf0f1; border-radius: 4px; overflow: hidden;'>
                        <div style='height: 100%; width: {confidence}%; background: {color}; border-radius: 4px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.success(f"🎬 **{res['predicted_winner']}** is predicted to have higher gross!")
            else:
                st.info("🤝 It's a tie! Both movies are predicted to have similar gross.")
                
                # Show neutral confidence indicator for ties
                st.markdown(f"""
                <div style='margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>Prediction Confidence:</span>
                        <span style='font-weight: bold; color: #7f8c8d'>Neutral (50%)</span>
                    </div>
                    <div style='height: 8px; background: #ecf0f1; border-radius: 4px; overflow: hidden;'>
                        <div style='height: 100%; width: 50%; background: #7f8c8d; border-radius: 4px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
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

with tab2:
    st.subheader("Current Progress")
    st.markdown("""
    ##### Our group has completed the baseline and improved models, integrated them with the Streamlit app, and successfully tested predictions between movies.
    ##### The models are now saving correctly, and results can be visualized in the app.
    """)
    
    st.subheader("Next Steps")
    st.markdown("""
    #### Model Optimization
    - Continue tuning the Random Forest and Gradient Boosting models to improve prediction accuracy and consistency.

    #### App Optimization
    - Streamline the Streamlit interface - cleaner layout, faster load time, and clearer result visuals.

    #### Feature Insights
    - Add feature importance and genre-based performance charts to help users understand why the model predicts certain outcomes.

    #### Automation (Undecided)
    - Implement a small Python script to generate weekly summary reports of model metrics.
    """)

    st.subheader("Roadblocks")
    st.markdown("""
    - Fixing a module import issue in the Jupyter notebook "ModuleNotFoundError: model"
    - Optimizing the model's performance to improve prediction accuracy and consistency.
    - We found that our models aren't as accurate as expected
    - Some UI changes on the streamlit app.
    """)
# ---------- Footer ----------
st.divider()
st.caption("Built with Streamlit • Group 1 • Intro to DS Programming")
