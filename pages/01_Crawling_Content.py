import streamlit as st
import requests
import re
from bs4 import BeautifulSoup

st.set_page_config(page_title="Crawling Artikel", layout="wide")
st.title("📰 Crawling & Analisis Awal Artikel")

# =====================================================
# KEYWORD THEME ESG (BAHASA INDONESIA)
# =====================================================

ESG_KEYWORDS = {
    "Environment": [
        "lingkungan", "emisi", "polusi", "perubahan iklim",
        "energi terbarukan", "pemanasan global", "sampah"
    ],
    "Social": [
        "masyarakat", "karyawan", "kesehatan", "keselamatan",
        "pendidikan", "hak asasi", "kesejahteraan"
    ],
    "Governance": [
        "tata kelola", "transparansi", "korupsi",
        "kepatuhan", "regulasi", "etika", "manajemen"
    ]
}

# =====================================================
# FUNGSI ANALISIS TEMA ESG
# =====================================================

def analyze_esg_theme(text: str):
    text = text.lower()
    scores = {}

    for theme, keywords in ESG_KEYWORDS.items():
        score = sum(text.count(k) for k in keywords)
        scores[theme] = score

    dominant = max(scores, key=scores.get)
    return scores, dominant

# =====================================================
# FUNGSI CRAWLING
# =====================================================

def crawl_article(url: str, source: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = requests.get(url, headers=headers, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title = soup.find("h1") or soup.find("title")
        date = soup.find("time") or soup.find("div")

        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        content = re.sub(r"Baca juga.*", "", content, flags=re.I)
        content = re.sub(r"\s+", " ", content).strip()

        return {
            "crawled_title": title.get_text(strip=True) if title else "Tidak ada judul",
            "crawled_date": date.get_text(strip=True) if date else "Tidak ada tanggal",
            "crawled_content": content or "Tidak ada konten"
        }

    except Exception as e:
        return {
            "crawled_title": "Gagal diakses",
            "crawled_date": "Gagal diakses",
            "crawled_content": f"Gagal crawling: {e}"
        }

# =====================================================
# UI
# =====================================================

source = st.selectbox(
    "Pilih Sumber Berita",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

url = st.text_input("Masukkan URL Artikel")

if st.button("🚀 Crawl & Analisis"):
    if not url:
        st.warning("Masukkan URL terlebih dahulu.")
        st.stop()

    data = crawl_article(url, source)

    # ==========================
    # ANALISIS TEMA ESG
    # ==========================
    esg_scores, dominant_esg = analyze_esg_theme(data["crawled_content"])

    # ==========================
    # SIMPAN KE SESSION STATE
    # ==========================
    st.session_state.crawled_data = {
        **data,
        "esg_themes": esg_scores,
        "dominant_esg": dominant_esg
    }
    st.session_state.crawled_url = url

    # ==========================
    # DISPLAY
    # ==========================
    st.success("Artikel berhasil di-crawl dan dianalisis")

    st.subheader("Judul Artikel")
    st.write(data["crawled_title"])

    st.subheader("Tanggal Publikasi")
    st.write(data["crawled_date"])

    st.subheader("Tema ESG Dominan")
    st.info(dominant_esg)

    st.subheader("Skor Tema ESG")
    st.json(esg_scores)

    st.subheader("Isi Artikel")
    st.write(data["crawled_content"])
