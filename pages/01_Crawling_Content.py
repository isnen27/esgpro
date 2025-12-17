import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet as wn
import nltk

# --- Inisialisasi NLTK & Model (Caching untuk Cloud) ---
@st.cache_resource
def prepare_resources():
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # Menggunakan model efisien untuk RAM Streamlit Cloud
    model = SentenceTransformer("asmud/nomic-embed-indonesian", trust_remote_code=True)
    return model

model_ai = prepare_resources()

# --- 1. Definisi Kata Dasar ESG ---
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

# --- 2. Fungsi Screening ESG Semantik ---
def perform_esg_screening(judul, model, threshold=0.45):
    judul_lower = str(judul).lower()
    
    # A. Klasifikasi Cepat (Keyword Match)
    if any(k in judul_lower for k in env_base): category = "Environment"
    elif any(k in judul_lower for k in soc_base): category = "Social"
    elif any(k in judul_lower for k in gov_base): category = "Governance"
    else: category = "Non-ESG"
    
    # B. Analisis Semantik AI (Hanya jika Non-ESG atau untuk verifikasi)
    themes = {
        "Environment": ["environmental sustainability", "climate change", "renewable energy", "carbon neutral"],
        "Social": ["social responsibility", "community development", "human rights", "workplace equality"],
        "Governance": ["corporate governance", "anti corruption", "ethical management", "transparency"]
    }
    
    judul_emb = model.encode(judul, convert_to_tensor=True)
    
    max_sim = 0
    best_theme = "Non-ESG"
    
    for theme_name, keywords in themes.items():
        theme_emb = model.encode(keywords, convert_to_tensor=True)
        score = util.cos_sim(judul_emb, theme_emb).max().item()
        if score > max_sim:
            max_sim = score
            best_theme = theme_name
            
    # Gabungkan Logika (Thresholding)
    if category == "Non-ESG" and max_sim >= threshold:
        category = best_theme
        
    return category, max_sim

# --- 3. Fungsi Crawling (Original Source) ---
@st.cache_data(ttl=3600)
def crawl_kompas(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'} [cite: 2]
        req = requests.get(url, headers=headers, timeout=10) [cite: 3]
        req.raise_for_status()
        soup = BeautifulSoup(req.text, 'lxml')
        
        title_tag = soup.find('h1', class_='read__title') or soup.find('h1') [cite: 4]
        crawled_title = title_tag.get_text(strip=True) if title_tag else 'Tidak ada judul' [cite: 4]
        
        date_tag = soup.find('div', class_='read__time') or soup.find('time') [cite: 4]
        crawled_date = date_tag.get_text(strip=True) if date_tag else 'Tidak ada tanggal' [cite: 4]
        
        main_content_div = soup.find('div', class_='read__content') [cite: 5]
        full_text = ""
        if main_content_div:
            paragraphs = main_content_div.find_all('p') [cite: 5]
            full_text = ' '.join(p.get_text(strip=True) for p in paragraphs) [cite: 6]
        
        # Pembersihan teks (Regex)
        unwanted = [r'Baca juga :.*', r'Penulis :.*', r'Editor :.*', r'Otomatis Mode Gelap.*'] [cite: 9, 10]
        for pattern in unwanted:
            full_text = re.sub(pattern, '', full_text, flags=re.DOTALL | re.IGNORECASE).strip() [cite: 10, 11]
            
        return {'crawled_title': crawled_title, 'crawled_date': crawled_date, 'crawled_content': full_text}
    except Exception as e:
        return {'crawled_title': f'Error: {e}', 'crawled_date': '', 'crawled_content': ''}

# (Fungsi crawl_tribunnews dan crawl_detik mengikuti pola yang sama dari source asli Anda) [cite: 15, 25]
@st.cache_data(ttl=3600)
def crawl_tribunnews(url):
    try:
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10) [cite: 15]
        soup = BeautifulSoup(req.text, 'lxml')
        title = soup.find('h1')
        crawled_title = title.get_text(strip=True) if title else 'Tidak ada judul' [cite: 16]
        date = soup.find('time')
        crawled_date = date.get_text(strip=True) if date else 'Tidak ada tanggal' [cite: 17]
        content_div = soup.find('div', class_='side-article txt-article multi-fontsize') [cite: 18]
        full_text = content_div.get_text(separator=' ', strip=True) if content_div else '' [cite: 18]
        return {'crawled_title': crawled_title, 'crawled_date': crawled_date, 'crawled_content': full_text}
    except Exception as e: return {'crawled_title': 'Gagal', 'crawled_date': '', 'crawled_content': str(e)}

@st.cache_data(ttl=3600)
def crawl_detik(url):
    try:
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10) [cite: 25]
        soup = BeautifulSoup(req.text, 'lxml')
        title = soup.find('h1', class_='detail__title') [cite: 25]
        crawled_title = title.get_text(strip=True) if title else 'Tidak ada judul' [cite: 26]
        date = soup.find('div', class_='detail__date') [cite: 27]
        crawled_date = date.get_text(strip=True) if date else 'Tidak ada tanggal' [cite: 27]
        content_div = soup.find('div', class_='detail__body-text itp_bodycontent') [cite: 28]
        full_text = content_div.get_text(strip=True) if content_div else '' [cite: 28]
        return {'crawled_title': crawled_title, 'crawled_date': crawled_date, 'crawled_content': full_text}
    except Exception as e: return {'crawled_title': 'Gagal', 'crawled_date': '', 'crawled_content': str(e)}

# --- 4. Konten Utama Streamlit ---
st.title("Crawling Content dari Berita Online")
st.write("Pilih website, masukkan URL artikel, dan sistem akan melakukan screening ESG secara otomatis.")

# --- Bagian Input Langsung ---
st.subheader("1. Pilih Website Sumber") [cite: 33]
website_choice = st.selectbox(
    "Pilih website yang akan di-crawl:",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

st.subheader("2. Input URL Artikel")
if website_choice == "Kompas.com":
    default_url = "https://www.kompas.com/global/read/2023/10/26/123456789/peran-indonesia-dalam-penanganan-perubahan-iklim"
elif website_choice == "Tribunnews.com":
    default_url = "https://www.tribunnews.com/bisnis/2024/01/15/perusahaan-ini-fokus-pada-energi-terbarukan-untuk-masa-depan"
else:
    default_url = "https://www.detik.com/finance/berita-ekonomi-bisnis/d-7000000/pemerintah-dorong-penerapan-tata-kelola-perusahaan-yang-baik"

url_input = st.text_input(f"Masukkan URL artikel dari {website_choice}:", default_url)

# --- Tombol Eksekusi ---
if st.button("Mulai Crawling & Screening"):
    if not url_input:
        st.error("URL tidak boleh kosong.")
    else:
        with st.spinner(f"Sedang memproses dari {website_choice}..."):
            # A. Crawling
            if website_choice == "Kompas.com": data = crawl_kompas(url_input)
            elif website_choice == "Tribunnews.com": data = crawl_tribunnews(url_input)
            else: data = crawl_detik(url_input)
            
            if data and data['crawled_title'] != "Gagal":
                # B. Screening ESG (Pilar Inti)
                esg_cat, score = perform_esg_screening(data['crawled_title'], model_ai)
                
                if esg_cat == "Non-ESG":
                    st.error(f"⚠️ **Screening Gagal:** Artikel ini dikategorikan sebagai **Non-ESG** (Similitude: {score:.2f}). Proses tidak dilanjutkan.")
                else:
                    st.success(f"✅ **Screening Berhasil:** Artikel terdeteksi bertema **{esg_cat}** (Similitude: {score:.2f})")
                    
                    # C. Tampilkan Hasil & Simpan Session [cite: 37]
                    st.subheader("3. Hasil Crawling") [cite: 36]
                    st.write(f"**Judul:** {data['crawled_title']}")
                    st.write(f"**Tanggal:** {data['crawled_date']}")
                    
                    st.session_state.crawled_content = data['crawled_content']
                    st.session_state.crawled_title = data['crawled_title']
                    st.session_state.esg_category = esg_cat
                    
                    with st.expander("Klik untuk melihat isi artikel"):
                        st.write(data['crawled_content'])
                    
                    st.info("Konten siap dianalisis di halaman 'Analysis'.")
            else:
                st.error("Gagal melakukan crawling. Silakan periksa URL.")

st.markdown("---")
st.info("Catatan: Screening menggunakan pendekatan semantik AI terhadap judul artikel.") [cite: 38]
