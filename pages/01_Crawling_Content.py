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

# Download WordNet data (diperlukan untuk fungsi perluasan kata)
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# --- Inisialisasi Model & Caching ---
@st.cache_resource
def load_esg_model():
    # Menggunakan model yang lebih ringan untuk efisiensi RAM di Cloud
    model = SentenceTransformer("asmud/nomic-embed-indonesian", trust_remote_code=True)
    return model

# --- Daftar Dasar ESG (Base Keywords) ---
env_base = ["environment", "green", "iklim", "emisi", "karbon", "energi terbarukan", "limbah", "hutan"] # Dipersingkat untuk contoh
soc_base = ["social", "sosial", "masyarakat", "hak asasi", "karyawan", "gender", "kesehatan"]
gov_base = ["governance", "tata kelola", "kepatuhan", "transparansi", "etika", "audit", "korupsi"]

# --- Fungsi Pendukung Screening (Reproduced) ---
def morphology_variants(word):
    variants = {word}
    if word.startswith("ber"): variants.add(word.replace("ber", "ke", 1))
    if word.startswith("ke"): variants.add(word.replace("ber", "ke", 1))
    if word.endswith("an"): variants.add(word[:-2])
    else: variants.add(word + "an")
    return list(variants)

@st.cache_data
def get_all_keywords():
    # Sederhanakan perluasan kata untuk demo agar tidak membebani startup
    # Di produksi, Anda bisa menjalankan expand_category() di sini
    return env_base, soc_base, gov_base

# --- Fungsi Screening ESG ---
def perform_esg_screening(text, title, model):
    # 1. Klasifikasi Cepat (Keyword Match)
    env_k, soc_k, gov_k = get_all_keywords()
    combined_text = (title + " " + text).lower()
    
    category = "Non-ESG"
    if any(k in combined_text for k in env_k): category = "Environment"
    elif any(k in combined_text for k in soc_k): category = "Social"
    elif any(k in combined_text for k in gov_k): category = "Governance"
    
    # 2. Analisis Semantik (AI)
    themes = {
        "Environment": ["environmental sustainability", "climate change", "renewable energy"],
        "Social": ["social responsibility", "human rights", "workplace equality"],
        "Governance": ["corporate governance", "anti corruption", "transparency"]
    }
    
    # Encode input & themes
    input_emb = model.encode(title, convert_to_tensor=True)
    
    scores = {}
    for label, phrases in themes.items():
        theme_emb = model.encode(phrases, convert_to_tensor=True)
        score = util.cos_sim(input_emb, theme_emb).max().item()
        scores[label] = score
    
    best_ai_category = max(scores, key=scores.get)
    max_score = scores[best_ai_category]
    
    # Terapkan Threshold
    threshold = 0.45
    if category == "Non-ESG" and max_score >= threshold:
        category = best_ai_category
        
    return category, max_score

# --- (Fungsi crawl_kompas, crawl_tribunnews, crawl_detik tetap sama seperti file Anda) ---
# [Masukkan fungsi crawling dari file asli Anda di sini]

# --- Konten Utama ---
st.title("ESG News Crawler & Screener")

model = load_esg_model()

# Bagian 1 & 2 (Pilih Website & Input URL) tetap sama...
# [Gunakan bagian input dari file asli Anda]

if st.button("Mulai Proses"):
    if url_input:
        with st.spinner("Melakukan Crawling dan Screening ESG..."):
            # 1. Crawling
            if website_choice == "Kompas.com": data = crawl_kompas(url_input)
            elif website_choice == "Tribunnews.com": data = crawl_tribunnews(url_input)
            else: data = crawl_detik(url_input)
            
            if "Gagal" not in data['crawled_title']:
                # 2. Screening ESG
                esg_category, score = perform_esg_screening(
                    data['crawled_content'], 
                    data['crawled_title'], 
                    model
                )
                
                # 3. Logika Kelanjutan
                if esg_category == "Non-ESG":
                    st.warning(f"Artikel ini dideteksi sebagai **Non-ESG** (Skor: {score:.2f}). Proses dihentikan.")
                else:
                    st.success(f"Artikel lolos screening! Kategori: **{esg_category}** (Skor: {score:.2f})")
                    
                    # Simpan ke session state
                    st.session_state.crawled_content = data['crawled_content']
                    st.session_state.esg_category = esg_category
                    
                    # Tampilkan hasil
                    st.subheader("Hasil Crawling")
                    st.write(f"**Judul:** {data['crawled_title']}")
                    st.expander("Lihat Isi").write(data['crawled_content'])
            else:
                st.error("Gagal melakukan crawling.")
