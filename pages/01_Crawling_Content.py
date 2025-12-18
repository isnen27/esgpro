import streamlit as st
import numpy as np
import torch
import os
import requests
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="ESG Screening Tool")

# =========================================================
# SESSION STATE
# =========================================================
for key in ["crawled_data", "final_esg_category", "crawled_url"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================================================
# ESG KEYWORDS (BASE ONLY – RINGAN)
# =========================================================
ENV = ["iklim", "lingkungan", "emisi", "energi", "karbon", "hutan", "sampah"]
SOC = ["sosial", "masyarakat", "pendidikan", "kesehatan", "karyawan"]
GOV = ["tata kelola", "transparansi", "audit", "korupsi", "etika"]

def classify_keyword(title):
    t = title.lower()
    if any(k in t for k in ENV):
        return "Environment"
    if any(k in t for k in SOC):
        return "Social"
    if any(k in t for k in GOV):
        return "Governance"
    return "Non-ESG"

# =========================================================
# SEMANTIC MODEL (LITE)
# =========================================================
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer(
        "indobenchmark/indobert-lite-p1",
        device="cpu"
    )
semantic_model = load_semantic_model()

THEMES = {
    "Environment": ["climate change", "renewable energy"],
    "Social": ["social responsibility", "human rights"],
    "Governance": ["corporate governance", "anti corruption"]
}

THEME_EMB = {
    k: semantic_model.encode(v, convert_to_tensor=True)
    for k, v in THEMES.items()
}

def classify_semantic(title):
    emb = semantic_model.encode(title, convert_to_tensor=True).unsqueeze(0)
    scores = {
        k: util.cos_sim(emb, v).max().item()
        for k, v in THEME_EMB.items()
    }
    best = max(scores, key=scores.get)
    return best, scores[best]

# =========================================================
# SIMPLE CRAWLER (KOMPAS / DETIK / TRIBUN)
# =========================================================
@st.cache_data(ttl=3600)
def crawl_generic(url):
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.find("h1")
    title = title.get_text(strip=True) if title else "Tidak ada judul"

    text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "crawled_title": title,
        "crawled_date": "N/A",
        "crawled_content": text[:5000]  # HARD LIMIT
    }

# =========================================================
# UI
# =========================================================
st.title("ESG Screening Tool")
url = st.text_input("Masukkan URL Artikel")

if st.button("Crawl & Screening"):
    data = crawl_generic(url)
    st.session_state.crawled_data = data
    st.session_state.crawled_url = url

    kw = classify_keyword(data["crawled_title"])
    sem, score = classify_semantic(data["crawled_title"])

    final = kw
    if kw == "Non-ESG" and score >= 0.45:
        final = sem

    st.session_state.final_esg_category = final

    st.markdown(f"### **Kategori ESG: {final}**")
    st.write(data["crawled_title"])
