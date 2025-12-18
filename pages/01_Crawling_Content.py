import streamlit as st
import requests
import re
from bs4 import BeautifulSoup

st.set_page_config(page_title="Crawling Artikel", layout="wide")
st.title("📰 Crawling Konten Artikel")

# =====================================================
# FUNGSI CRAWLING (KOMPAS, TRIBUN, DETIK)
# =====================================================

def crawl_kompas(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = requests.get(url, headers=headers, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title = soup.find("h1", class_="read__title") or soup.find("h1")
        date = soup.find("div", class_="read__time") or soup.find("time")
        content_div = soup.find("div", class_="read__content")

        paragraphs = content_div.find_all("p") if content_div else soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        content = re.sub(r"Baca juga :.*", "", content, flags=re.I)
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


def crawl_tribun(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title = soup.find("h1") or soup.find("title")
        date = soup.find("time")
        content_div = soup.find("div", class_="side-article")

        content = content_div.get_text(" ", strip=True) if content_div else ""

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


def crawl_detik(url):
    try:
        req = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, "lxml")

        title = soup.find("h1") or soup.find("title")
        date = soup.find("div", class_="detail__date")
        content_div = soup.find("div", class_="detail__body-text")

        paragraphs = content_div.find_all("p") if content_div else []
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

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

site = st.selectbox("Pilih Website", ["Kompas.com", "Tribunnews.com", "Detik.com"])
url = st.text_input("Masukkan URL Artikel")

if st.button("🚀 Crawl Artikel"):
    if not url:
        st.warning("Masukkan URL terlebih dahulu.")
    else:
        if "kompas" in site.lower():
            data = crawl_kompas(url)
        elif "tribun" in site.lower():
            data = crawl_tribun(url)
        else:
            data = crawl_detik(url)

        # ==========================
        # SIMPAN KE SESSION STATE
        # ==========================
        st.session_state.crawled_data = data
        st.session_state.crawled_url = url

        st.success("Artikel berhasil di-crawl")

        st.subheader("Judul")
        st.write(data["crawled_title"])

        st.subheader("Tanggal Publikasi")
        st.write(data["crawled_date"])

        st.subheader("Isi Artikel")
        st.write(data["crawled_content"])
