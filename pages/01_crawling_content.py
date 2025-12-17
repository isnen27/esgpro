# pages/01_crawling_content.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util

# --- Model Caching ---
# Menggunakan st.cache_resource untuk memuat model AI sekali saja
# agar tidak dimuat ulang setiap kali ada interaksi di Streamlit.
@st.cache_resource
def load_esg_model():
    """Loads the SentenceTransformer model for ESG semantic analysis."""
    with st.spinner("Memuat model AI untuk analisis semantik... Ini mungkin membutuhkan waktu beberapa saat."):
        # Pastikan folder cache ada
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        model = SentenceTransformer("indobenchmark/indobert-base-p2", cache_folder=cache_dir)
    return model

# --- Konten Utama Aplikasi Streamlit ---
st.title("Crawling Content & Prediksi Tema ESG")
st.write("Di sini Anda dapat memilih website, menginput URL, dan memprediksi tema ESG pada judul berita menggunakan AI Semantik.")

# --- 1. Pilih Website ---
st.subheader("1. Pilih Website Sumber")
website_choice = st.selectbox(
    "Pilih website yang akan di-crawl:",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

# --- 2. Input URL ---
st.subheader("2. Input URL untuk Analisis")
# Contoh URL yang relevan dengan format yang Anda berikan
default_url = "https://www.kompas.com/global/read/2023/10/26/123456789/peran-indonesia-dalam-penanganan-perubahan-iklim"
if website_choice == "Tribunnews.com":
    default_url = "https://www.tribunnews.com/bisnis/2024/01/15/perusahaan-ini-fokus-pada-energi-terbarukan-untuk-masa-depan"
elif website_choice == "Detik.com":
    default_url = "https://www.detik.com/finance/berita-ekonomi-bisnis/d-7000000/pemerintah-dorong-penerapan-tata-kelola-perusahaan-yang-baik"

url_input = st.text_input(f"Masukkan URL dari {website_choice} untuk dianalisis:", default_url)

# --- Placeholder untuk crawling aktual ---
def simulate_crawling(url):
    """
    Fungsi dummy untuk mensimulasikan hasil crawling.
    Ganti dengan logika crawling Anda yang sebenarnya.
    Seharusnya mengembalikan DataFrame dengan kolom 'link'.
    """
    st.info(f"Simulasi crawling dari URL: {url}")
    # Untuk demonstrasi, kita hanya membuat DataFrame dengan URL input
    return pd.DataFrame({"link": [url]})

# Tombol untuk memulai proses analisis
if st.button("Mulai Prediksi Tema ESG"):
    if not url_input:
        st.error("URL tidak boleh kosong. Silakan masukkan URL.")
        st.stop()

    # Simulasi crawling untuk mendapatkan DataFrame awal
    df = simulate_crawling(url_input)

    st.subheader("3. Ekstraksi Judul dari URL")
    with st.spinner("Mengekstrak judul dari URL..."):
        df["url_lengkap"] = df["link"].apply(lambda x: "https:" + x if isinstance(x, str) and x.startswith("//") else x)
        # Regex ini spesifik untuk struktur URL seperti Kompas.com/Detik.com.
        # Mungkin perlu disesuaikan untuk Tribunnews.com atau struktur lain.
        df["judul"] = df["url_lengkap"].str.extract(r"/\d{4}/\d{2}/\d{2}/[^/]+/([^/]+)")[0]
        df["judul"] = df["judul"].fillna("").astype(str).str.replace('-', ' ') # Ganti tanda hubung untuk pencocokan keyword yang lebih baik
    st.success("Judul berhasil diekstrak.")
    st.dataframe(df[['link', 'judul']])

    # --- Analisis Semantik (AI) ---
    st.subheader("4. Prediksi Tema ESG Menggunakan AI Semantik")
    model = load_esg_model() # Muat model yang di-cache

    themes = {
        "Environment": ["environmental sustainability", "climate change", "renewable energy", "carbon neutral", "lingkungan", "iklim", "emisi", "daur ulang"],
        "Social": ["social responsibility", "community development", "human rights", "workplace equality", "sosial", "masyarakat", "karyawan", "pendidikan"],
        "Governance": ["corporate governance", "anti corruption", "ethical management", "transparency", "tata kelola", "etika", "transparansi", "akuntabilitas"]
    }

    with st.spinner("Menghitung embedding tema ESG..."):
        theme_embeddings = {k: model.encode(v, convert_to_tensor=True) for k, v in themes.items()}
    st.success("Embedding tema ESG selesai.")

    with st.spinner("Menghitung embedding judul berita dan analisis kemiripan..."):
        # Filter judul yang kosong atau bukan string untuk mencegah error
        valid_titles_mask = df["judul"].astype(str).str.len() > 0
        valid_titles = df.loc[valid_titles_mask, "judul"].astype(str).tolist()

        if not valid_titles:
            st.warning("Tidak ada judul berita yang valid untuk analisis semantik.")
            df["esg_similarity"] = 0.0
            df["Prediksi_AI"] = "Non-ESG"
        else:
            judul_embeddings = model.encode(valid_titles, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
            
            # Inisialisasi skor dengan nol untuk semua baris DataFrame
            scores_env_full = np.zeros(len(df))
            scores_soc_full = np.zeros(len(df))
            scores_gov_full = np.zeros(len(df))

            # Hitung skor hanya untuk judul yang valid
            scores_env_valid = util.cos_sim(judul_embeddings, theme_embeddings["Environment"]).max(dim=1).values.cpu().numpy()
            scores_soc_valid = util.cos_sim(judul_embeddings, theme_embeddings["Social"]).max(dim=1).values.cpu().numpy()
            scores_gov_valid = util.cos_sim(judul_embeddings, theme_embeddings["Governance"]).max(dim=1).values.cpu().numpy()

            # Masukkan skor kembali ke DataFrame asli pada posisi yang benar
            scores_env_full[valid_titles_mask] = scores_env_valid
            scores_soc_full[valid_titles_mask] = scores_soc_valid
            scores_gov_full[valid_titles_mask] = scores_gov_valid

            df["esg_similarity"] = np.max([scores_env_full, scores_soc_full, scores_gov_full], axis=0)
            best_idx = np.argmax([scores_env_full, scores_soc_full, scores_gov_full], axis=0)
            df["Prediksi_AI"] = np.select(
                [best_idx == 0, best_idx == 1, best_idx == 2],
                ["Environment", "Social", "Governance"],
                default="Non-ESG"
            )
    st.success("Analisis semantik selesai.")
    st.dataframe(df[['judul', 'esg_similarity', 'Prediksi_AI']])


    # --- Terapkan Threshold Otomatis ---
    st.subheader("5. Filter Prediksi dengan Threshold Kemiripan")
    threshold = st.slider("Pilih nilai threshold kemiripan (skor di bawah ini akan dianggap 'Non-ESG'):", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    
    with st.spinner("Menerapkan threshold kemiripan..."):
        df["Tema_ESG_Final"] = df["Prediksi_AI"] # Mulai dengan prediksi AI
        # Ubah menjadi 'Non-ESG' jika skor kemiripan di bawah threshold
        df.loc[df["esg_similarity"] < threshold, "Tema_ESG_Final"] = "Non-ESG"
    st.success("Threshold kemiripan diterapkan.")
    st.dataframe(df[['judul', 'esg_similarity', 'Prediksi_AI', 'Tema_ESG_Final']])

    st.subheader("Hasil Akhir Prediksi Tema ESG")
    st.dataframe(df[['link', 'judul', 'Tema_ESG_Final', 'esg_similarity']])

    # Opsi untuk mengunduh hasil
    csv_output = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Unduh Hasil Prediksi (CSV)",
        data=csv_output,
        file_name="esg_predicted_content.csv",
        mime="text/csv",
    )

st.markdown("---")
st.info("Catatan: Fungsi crawling web aktual belum diimplementasikan. Data di atas adalah hasil simulasi dari URL input. Anda dapat memperluas daftar tema dalam kode untuk meningkatkan akurasi.")
