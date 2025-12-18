import streamlit as st

st.set_page_config(
    page_title="ESG Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("Selamat Datang di Aplikasi Prediksi ESG 🚀")
st.write("""
Aplikasi ini dirancang untuk membantu Anda dalam proses prediksi Environmental, Social, and Governance (ESG). 🌿🌍🤝
Gunakan menu di sidebar untuk menjelajahi berbagai fitur:
- **Crawling Content:** Untuk mengumpulkan data dari sumber web. 🕸️ Saat ini tersedia untuk situs Kompas.com, Tribunnews.com, dan Detik.com. 📰
- **Analysis:** Untuk menganalisis artikel, meliputi ringkasan berita, identifikasi entitas dalam berita, dan Knowledge Graph dari berita. 🔍🧠
- **Recommendation:** Untuk memberikan ringkasan terkait analisis sentimen berita dan rekomendasi strategis yang dapat dilakukan manajemen. ✨📈
""")

st.markdown("---")
st.info("Pilih menu di sidebar untuk memulai! 👈")
