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
# ESG KEYWORDS (INDONESIA)
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
    "Environment": [
        "perubahan iklim", "energi terbarukan",
        "dampak lingkungan", "emisi karbon"
    ],
    "Social": [
        "tanggung jawab sosial", "kesejahteraan masyarakat",
        "hak asasi manusia", "kesehatan publik"
    ],
    "Governance": [
        "tata kelola perusahaan", "transparansi",
        "anti korupsi", "etika bisnis"
    ]
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
# CRAWLER: KOMPAS.COM 
# =========================================================
@st.cache_data(ttl=3600)
def crawl_kompas(url):
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/100.0.4896.127 Safari/537.36"
            )
        }

        req = requests.get(url, headers=headers, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title_tag = soup.find("h1", class_="read__title") or soup.find("h1")
        crawled_title = title_tag.get_text(strip=True) if title_tag else "Tidak ada judul"

        date_tag = soup.find("div", class_="read__time") or soup.find("time")
        crawled_date = date_tag.get_text(strip=True) if date_tag else "Tidak ada tanggal"

        full_text = ""
        main_div = soup.find("div", class_="read__content")

        if main_div:
            paragraphs = main_div.find_all("p")
            full_text = " ".join(p.get_text(strip=True) for p in paragraphs)
        else:
            fallback = soup.find("article") or soup.find("div", class_="clearfix")
            if fallback:
                paragraphs = fallback.find_all("p")
                full_text = " ".join(p.get_text(strip=True) for p in paragraphs)

        unwanted_patterns = [
            r"Baca juga :.*", r"Penulis :.*", r"Editor :.*",
            r"Sumber :.*", r"Ikuti kami.*", r"Simak berita.*"
        ]

        for p in unwanted_patterns:
            full_text = re.sub(p, "", full_text, flags=re.IGNORECASE | re.DOTALL)

        full_text = re.sub(r"\s+", " ", full_text).strip()

        return {
            "crawled_title": crawled_title,
            "crawled_date": crawled_date,
            "crawled_content": full_text or "Tidak ada konten"
        }

    except Exception as e:
        return {
            "crawled_title": "Gagal diakses",
            "crawled_date": "Gagal diakses",
            "crawled_content": f"Kesalahan: {e}"
        }

# =========================================================
# CRAWLER: TRIBUNNEWS.COM 
# =========================================================
@st.cache_data(ttl=3600)
def crawl_tribunnews(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title = soup.find("h1") or soup.find("title")
        date = soup.find("time")

        content_div = soup.find("div", class_="side-article txt-article multi-fontsize")

        text = content_div.get_text(" ", strip=True) if content_div else ""

        return {
            "crawled_title": title.get_text(strip=True) if title else "Tidak ada judul",
            "crawled_date": date.get_text(strip=True) if date else "Tidak ada tanggal",
            "crawled_content": text or "Tidak ada konten"
        }

    except Exception as e:
        return {
            "crawled_title": "Gagal diakses",
            "crawled_date": "Gagal diakses",
            "crawled_content": f"Kesalahan: {e}"
        }

# =========================================================
# CRAWLER: DETIK.COM 
# =========================================================
@st.cache_data(ttl=3600)
def crawl_detik(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title = soup.find("h1", class_="detail__title") or soup.find("title")
        date = soup.find("div", class_="detail__date")
        body = soup.find("div", class_="detail__body-text itp_bodycontent")

        paragraphs = body.find_all("p") if body else []
        text = " ".join(p.get_text(strip=True) for p in paragraphs)

        return {
            "crawled_title": title.get_text(strip=True) if title else "Tidak ada judul",
            "crawled_date": date.get_text(strip=True) if date else "Tidak ada tanggal",
            "crawled_content": text or "Tidak ada konten"
        }

    except Exception as e:
        return {
            "crawled_title": "Gagal diakses",
            "crawled_date": "Gagal diakses",
            "crawled_content": f"Kesalahan: {e}"
        }

# =========================================================
# UI
# =========================================================
st.title("Crawling Artikel ESG")

website = st.selectbox(
    "Pilih Website",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

url = st.text_input("Masukkan URL Artikel")

if st.button("Crawl"):
    if website == "Kompas.com":
        data = crawl_kompas(url)
    elif website == "Tribunnews.com":
        data = crawl_tribunnews(url)
    else:
        data = crawl_detik(url)

    # HARD LIMIT (CLOUD SAFE)
    data["crawled_content"] = data["crawled_content"][:5000]

    st.session_state.crawled_data = data
    st.session_state.crawled_url = url

    kw = keyword_classification(data["crawled_title"])
    sem, score = semantic_classification(data["crawled_title"])

    final = kw if kw != "Non-ESG" or score < 0.45 else sem
    st.session_state.final_esg_category = final

    st.markdown("### Judul")
    st.write(data["crawled_title"])

    st.markdown("### Tanggal Publikasi")
    st.write(data["crawled_date"])

    st.markdown("### Isi Artikel")
    st.write(data["crawled_content"])

    st.markdown(f"### Kategori ESG: **{final}**")
