# app.py
import streamlit as st
import crawling_content
import preprocessing
import visualization
import evaluation

st.set_page_config(
    page_title="ESG Prediction App",
    page_icon="📊",
    layout="wide"
)

# Sidebar for navigation
st.sidebar.title("Navigasi")
menu_selection = st.sidebar.radio(
    "Pilih Menu:",
    ["Crawling Content", "Preprocessing", "Visualization", "Evaluation"]
)

# Display content based on menu selection
if menu_selection == "Crawling Content":
    crawling_content.app()
elif menu_selection == "Preprocessing":
    preprocessing.app()
elif menu_selection == "Visualization":
    visualization.app()
elif menu_selection == "Evaluation":
    evaluation.app()

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Prediksi ESG")
