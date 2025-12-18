import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ESG Content Crawling",
    layout="wide"
)

# =========================================================
# SESSION STATE INIT
# =========================================================
for key in ["crawled_data", "final_esg_category", "crawled_url"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================================================
# ESG KEYWORDS (RINGAN)
# =========================================================
ENV = ["iklim", "lingkungan", "emisi", "energi", "karbon", "hutan"]
SOC = ["sosial", "masyarakat", "pendidikan", "kesehatan", "karyawan"]
GOV = ["tata kelola", "transparansi", "audit", "korupsi", "etika"]

def keyword_classification(title):
    t = title.lower()
    if any(k in t for k in ENV):
        return "Environment"
    if any(k in t for k in SOC):
        return "Social"
    if any(k in t for k in GOV):
        return "Governance"
    return "Non-ESG"

# =========================================================
# SEMANTIC MODEL (MULTILINGUAL – STABLE)
# =========================================================
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

semantic_model = load_semantic_model()

THEMES = {
    "Environment": ["climate change", "renewable energy", "environmental impact"],
    "Social": ["social responsibility", "public welfare", "human rights"],
    "Governance": ["corporate governance", "transparency", "anti corruption"]
}

THEME_EMB = {
    k: semantic_model.encode(v, convert_to_tensor=True)
    for k, v in THEMES.items()
}

def semantic_classification(title):
    emb = semantic_model.encode(title, convert_to_tensor=True).unsqueeze(0)
    scores = {
        k: util.cos_sim(emb, v).max().item()
        for k, v in THEME_EMB.items()
    }
    best = max(scores, key=scores.get)
    return best, scores[best]

# =========================================================
# CRAWLERS PER WEBSITE
# =========================================================
def crawl_kompas(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.find("h1")
    date = soup.find("time")
    content = soup.find("div", class_=re.compile("read__content"))

    return {
        "crawled_title": title.get_text(strip=True) if title else "Tidak ditemukan",
        "crawled_date": date.get_text(strip=True) if date else "Tidak ditemukan",
        "crawled_content": " ".join(
            p.get_text(strip=True) for p in content.find_all("p")
        ) if content else ""
    }

def crawl_detik(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.find("h1")
    date = soup.find("div", class_="detail__date")
    content = soup.find("div", class_="detail__body-text")

    return {
        "crawled_title": title.get_text(strip=True) if title else "Tidak ditemukan",
        "crawled_date": date.get_text(strip=True) if date else "Tidak ditemukan",
        "crawled_content": " ".join(
            p.get_text(strip=True) for p in content.find_all("p")
        ) if content else ""
    }

def crawl_tribun(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.find("h1")
    date = soup.find("time")
    content = soup.find("div", class_="side-article txt-article")

    return {
        "crawled_title": title.get_text(strip=True) if title else "Tidak ditemukan",
        "crawled_date": date.get_text(strip=True) if date else "Tidak ditemukan",
        "crawled_content": " ".join(
            p.get_text(strip=True) for p in content.find_all("p")
        ) if content else ""
    }

# =========================================================
# UI
# =========================================================
st.title("Crawling Konten Artikel ESG")

website = st.selectbox(
    "Pilih Website Sumber",
    ["Kompas.com", "Detik.com", "Tribunnews.com"]
)

url = st.text_input("Masukkan URL Artikel")

if st.button("Crawl Artikel"):
    try:
        if website == "Kompas.com":
            data = crawl_kompas(url)
        elif website == "Detik.com":
            data = crawl_detik(url)
        else:
            data = crawl_tribun(url)

        # HARD LIMIT ISI (CLOUD SAFE)
        data["crawled_content"] = data["crawled_content"][:5000]

        st.session_state.crawled_data = data
        st.session_state.crawled_url = url

        # ESG Classification
        kw_cat = keyword_classification(data["crawled_title"])
        sem_cat, score = semantic_classification(data["crawled_title"])

        final_cat = kw_cat
        if kw_cat == "Non-ESG" and score >= 0.45:
            final_cat = sem_cat

        st.session_state.final_esg_category = final_cat

        # =================================================
        # DISPLAY
        # =================================================
        st.success("Artikel berhasil di-crawl")

        st.markdown("### Judul Artikel")
        st.write(data["crawled_title"])

        st.markdown("### Tanggal Publikasi")
        st.write(data["crawled_date"])

        st.markdown("### Isi Artikel")
        st.write(data["crawled_content"])

        st.markdown(f"### Kategori ESG: **{final_cat}**")

    except Exception as e:
        st.error("Gagal melakukan crawling. Pastikan URL dan website sesuai.")
