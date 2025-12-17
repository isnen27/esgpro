# app.py
import streamlit as st

st.set_page_config(
    page_title="ESG Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("Selamat Datang di Aplikasi Prediksi ESG")
st.write("""
Aplikasi ini dirancang untuk membantu Anda dalam proses prediksi Environmental, Social, and Governance (ESG).
Gunakan menu di sidebar untuk menjelajahi berbagai fitur:
- **Crawling Content:** Untuk mengumpulkan data dari sumber web.
- **Preprocessing:** Untuk membersihkan dan menyiapkan data.
- **Visualization:** Untuk menganalisis data secara visual.
- **Evaluation:** Untuk mengevaluasi performa model prediksi.
""")

st.markdown("---")
st.info("Pilih menu di sidebar untuk memulai!")

# Kamu bisa menambahkan konten lain untuk halaman utama di sini
