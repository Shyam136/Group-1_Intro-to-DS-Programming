# app.py
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Which Movie Will Gross More?",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("About")
    st.write(
        """
        Comparing two films based on metadata.
        """
    )
    st.caption("Dataset: Movie Industry Dataset (Daniel Grijalvas – Kaggle).")

# Title
st.title("Which Movie Will Gross More?")
st.write(
    "Select two movies below. The model will do the rest."
)

# About the Dataset section
st.subheader("About the Dataset")
st.write(
    """
    - Source: **Movie Industry Dataset** (Daniel Grijalvas – Kaggle).  
    - Time Range: Movies released between **1980–2020**.  
    - Size: ~7,600 movies, ~15 features.  
    - Features include:  
        • Genre  
        • Rating  
        • Budget (inflation-adjusted by us)  
        • Gross revenue (inflation-adjusted us)  
        • Runtime  
        • Release year  
    """
)

st.divider()

# Dropdown Menu
st.subheader("Select Two Movies to Compare and Let the Model Do the Rest")
placeholder_list = ["The Dark Knight", "Inception", "Interstellar", "The Matrix"]
col1, col2 = st.columns(2)
with col1:
    movie_a = st.selectbox("Movie A", placeholder_list, key="movie_a")
with col2:
    movie_b = st.selectbox("Movie B", placeholder_list, key="movie_b")

# Predict button
if st.button("Predict"):
    st.info("Prediction output placeholder")
