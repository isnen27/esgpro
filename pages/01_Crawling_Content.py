import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet as wn

# --- 1. Inisialisasi Resource & Model (Caching untuk Cloud) ---
@st.cache_resource
def load_resources():
    # Download NLTK data yang diperlukan
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Memuat model embedding (pilihan asmud/nomic-embed-indonesian efisien untuk RAM Cloud)
    model = SentenceTransformer("asmud/nomic-embed-indonesian", trust_remote_code=True)
    return model

model_ai = load_resources()

# --- 2. Daftar Kata Dasar ESG (Reproduced Keywords) ---
env_base = [
    "environment", "environmental", "green", "eco", "ekologis", "berkelanjutan", "keberlanjutan",
    "climate", "iklim", "perubahan iklim", "global warming", "pemanasan global", "emisi", "karbon",
    "net zero", "dekarbonisasi", "energi terbarukan", "energi bersih", "energi hijau", "hemat energi",
    "efisiensi energi", "energi alternatif", "hidrogen", "mobil listrik", "EV", "transportasi hijau",
    "sampah", "limbah", "polusi", "plastik", "daur ulang", "recycle", "circular economy", "ekonomi sirkular",
    "konservasi", "hutan", "reforestasi", "biodiversity", "keanekaragaman hayati", "flora", "fauna",
    "urban farming", "pertanian berkelanjutan", "produksi hijau", "rantai pasok hijau", "industri hijau",
    "lingkungan hidup", "mitigasi iklim", "adaptasi iklim", "energi surya", "geotermal"
]

soc_base = [
    "social", "sosial", "csr", "tanggung jawab sosial", "community", "masyarakat", "pemberdayaan masyarakat",
    "pemberdayaan perempuan", "human rights", "hak asasi manusia", "karyawan", "diversity", "kesetaraan",
    "gender equality", "empowerment", "pendidikan", "kesehatan", "keselamatan kerja", "well-being",
    "kesehatan mental", "donasi", "filantropi", "keamanan kerja", "human capital", "pembangunan manusia",
    "komunitas", "pengentasan kemiskinan", "lapangan kerja", "inklusifitas", "upah layak", "perlindungan konsumen",
    "kesejahteraan masyarakat", "partisipasi masyarakat", "komunikasi sosial", "hubungan industrial",
    "akses pendidikan", "perempuan dan anak", "pemberdayaan difabel", "UMKM", "ekonomi lokal", "volunteering",
    "philanthropy", "stakeholder engagement", "fair labor", "equal opportunity", "community development"
]

gov_base = [
    "governance", "tata kelola", "good corporate governance", "governansi", "kepatuhan", "regulasi",
    "transparansi", "akuntabilitas", "etika", "integritas", "kode etik", "etika bisnis", "audit",
    "dewan komisaris", "dewan direksi", "pengawasan", "manajemen risiko", "anti korupsi", "whistleblowing",
    "laporan keberlanjutan", "annual report", "disclosure", "stakeholder", "shareholder", "responsible management",
    "kepemimpinan beretika", "integritas bisnis", "pengendalian internal", "audit internal", "komite risiko",
    "AML", "KYC", "due diligence", "strategi tata kelola", "transparansi keuangan", "kepatuhan hukum",
    "sistem pelaporan pelanggaran", "good governance", "akuntabilitas publik", "independent board", "risk committee"
]

# --- 3. Fungsi Screening ESG ---
def perform_esg_screening(judul, model, threshold=0.45):
    j = str(judul).lower()
    
    # Klasifikasi Cepat via Keyword
    if any(k in j for k in env_base): cat_keyword = "Environment"
    elif any(k in j for k in soc_base): cat_keyword = "Social"
    elif any(k in j for k in gov_base): cat_keyword = "Governance"
    else: cat_keyword = "Non-ESG"
    
    # Analisis Semantik (AI)
    themes = {
        "Environment": ["environmental sustainability", "climate change", "renewable energy", "carbon neutral"],
        "Social": ["social responsibility", "community development", "human rights", "workplace equality"],
        "Governance": ["corporate governance", "anti corruption", "ethical management", "transparency"]
    }
    
    judul_emb = model.encode(judul, convert_to_tensor=True)
    
    max_score = 0
    prediksi_ai = "Non-ESG"
    
    for theme, keywords in themes.items():
        theme_emb = model.encode(keywords, convert_to_tensor=True)
        score = util.cos_sim(judul_emb, theme_emb).max().item()
        if score > max_score:
            max_score = score
            prediksi_ai = theme
            
    # Final Decision (Menggabungkan Keyword & AI Threshold)
    if cat_keyword == "Non-ESG" and max_score < threshold:
        return "Non-ESG", max_score
    elif cat_keyword != "Non-ESG":
        return cat_keyword, max_score
    else:
        return prediksi_ai, max_score

# --- 4. Fungsi Crawling (Source Asli) ---
@st.cache_data(ttl=3600)
def crawl_kompas(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'} [cite: 2]
        req = requests.get(url, headers=headers, timeout=10) [cite: 3]
        req.raise_for_status()
        soup = BeautifulSoup(req.text, 'lxml')
        
        title_tag = soup.find('h1', class_='read__title') or soup.find('h1') [cite: 4]
        crawled_title = title_tag.get_text(strip=True) if title_tag else 'Tidak ada judul'
        
        date_tag = soup.find('div', class_='read__time') or soup.find('time') [cite: 4]
        crawled_date = date_tag.get_text(strip=True) if date_tag else 'Tidak ada tanggal'
        
        content_div = soup.find('div', class_='read__content') [cite: 5]
        full_text = ""
        if content_div:
            paragraphs = content_div.find_all('p') [cite: 5]
            full_text = ' '.join(p.get_text(strip=True) for p in paragraphs) if paragraphs else content_div.get_text(strip=True) [cite: 6]
        
        # Bersihkan teks dari pola tidak diinginkan [cite: 9, 10, 11]
        unwanted = [r'Baca juga :.*', r'Penulis :.*', r'Editor :.*', r'Otomatis Mode Gelap.*']
        for pattern in unwanted:
            full_text = re.sub(pattern, '', full_text, flags=re.DOTALL | re.IGNORECASE).strip()
            
        return {'crawled_title': crawled_title, 'crawled_date': crawled_date, 'crawled_content': full_text}
    except Exception as e:
        return {'crawled_title': 'Gagal diakses', 'crawled_date': 'Gagal', 'crawled_content': str(e)}

@st.cache_data(ttl=3600)
def crawl_tribunnews(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=10) [cite: 15]
        soup = BeautifulSoup(req.text, 'lxml')
        title = soup.find('h1') [cite: 15, 16]
        crawled_title = title.get_text(strip=True) if title else 'Tidak ada judul'
        date = soup.find('time') [cite: 17]
        crawled_date = date.get_text(strip=True) if date else 'Tidak ada tanggal'
        content_div = soup.find('div', class_='side-article txt-article multi-fontsize') [cite: 18]
        full_text = content_div.get_text(separator=' ', strip=True) if content_div else ''
        return {'crawled_title': crawled_title, 'crawled_date': crawled_date, 'crawled_content': full_text}
    except Exception: return {'crawled_title': 'Gagal', 'crawled_date': '', 'crawled_content': ''}

@st.cache_data(ttl=3600)
def crawl_detik(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = requests.get(url, headers=headers, timeout=10) [cite: 25]
        soup = BeautifulSoup(req.text, 'lxml')
        title = soup.find('h1', class_='detail__title') [cite: 25, 26]
        crawled_title = title.get_text(strip=True) if title else 'Tidak ada judul'
        date = soup.find('div', class_='detail__date') [cite: 27]
        crawled_date = date.get_text(strip=True) if date else 'Tidak ada tanggal'
        content_div = soup.find('div', class_='detail__body-text itp_bodycontent') [cite: 28]
        full_text = content_div.get_text(strip=True) if content_div else ''
        return {'crawled_title': crawled_title, 'crawled_date': crawled_date, 'crawled_content': full_text}
    except Exception: return {'crawled_title': 'Gagal', 'crawled_date': '', 'crawled_content': ''}

# --- 5. Tampilan Utama Streamlit ---
st.title("Crawling Content dari Berita Online")
st.write("Pilih website, masukkan URL artikel, dan dapatkan analisis screening ESG otomatis.")

# Bagian 1: Pilih Website [cite: 33]
st.subheader("1. Pilih Website Sumber")
website_choice = st.selectbox(
    "Pilih website yang akan di-crawl:",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

# Bagian 2: Input URL Artikel [cite: 33]
st.subheader("2. Input URL Artikel")
if website_choice == "Kompas.com":
    default_url = "https://www.kompas.com/global/read/2023/10/26/123456789/peran-indonesia-dalam-penanganan-perubahan-iklim"
elif website_choice == "Tribunnews.com":
    default_url = "https://www.tribunnews.com/bisnis/2024/01/15/perusahaan-ini-fokus-pada-energi-terbarukan-untuk-masa-depan"
else:
    default_url = "https://www.detik.com/finance/berita-ekonomi-bisnis/d-7000000/pemerintah-dorong-penerapan-tata-kelola-perusahaan-yang-baik"

url_input = st.text_input(f"Masukkan URL artikel dari {website_choice}:", default_url)

# Tombol Eksekusi
if st.button("Mulai Crawling & Screening"):
    if not url_input:
        st.error("URL tidak boleh kosong.")
    else:
        with st.spinner(f"Sedang melakukan crawling dan screening ESG..."):
            # Proses Crawling [cite: 34]
            if website_choice == "Kompas.com":
                crawled_data = crawl_kompas(url_input)
            elif website_choice == "Tribunnews.com":
                crawled_data = crawl_tribunnews(url_input)
            else:
                crawled_data = crawl_detik(url_input)
            
            # Verifikasi Hasil Crawling
            if crawled_data and "Gagal" not in crawled_data['crawled_title']:
                # Proses Screening ESG (Semantik)
                esg_category, sim_score = perform_esg_screening(crawled_data['crawled_title'], model_ai)
                
                if esg_category == "Non-ESG":
                    st.error(f"⚠️ **Screening Gagal:** Artikel dideteksi sebagai **Non-ESG** (Skor: {sim_score:.2f}). Proses tidak dilanjutkan.")
                else:
                    st.success(f"✅ **Lolos Screening:** Kategori **{esg_category}** (Skor: {sim_score:.2f})")
                    
                    # Simpan data ke session_state [cite: 37]
                    st.session_state.crawled_content = crawled_data['crawled_content']
                    st.session_state.crawled_title = crawled_data['crawled_title']
                    st.session_state.crawled_date = crawled_data['crawled_date']
                    st.session_state.esg_category = esg_category
                    
                    # Tampilkan Hasil [cite: 36]
                    st.subheader("3. Hasil Crawling")
                    st.write(f"**Judul:** {crawled_data['crawled_title']}")
                    st.write(f"**Tanggal:** {crawled_data['crawled_date']}")
                    st.markdown("---")
                    st.subheader("Isi Artikel:")
                    st.write(crawled_data['crawled_content'])
                    st.success("Konten berhasil di-crawl dan siap dianalisis di halaman berikutnya!")
            else:
                st.error("Gagal mendapatkan konten. Silakan periksa kembali URL.")

st.markdown("---")
st.info("Catatan: Fungsi crawling bergantung pada struktur HTML. Perubahan website dapat mempengaruhi hasil.") [cite: 38]
