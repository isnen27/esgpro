# pages/01_crawling_content.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os

# --- Konfigurasi Halaman Streamlit ---
# Ini harus menjadi baris kode Streamlit pertama di halaman ini.
st.set_page_config(
    page_title="Crawling Content", # Nama yang akan muncul di sidebar
    page_icon="🔍", # Opsional: ikon untuk halaman ini
    layout="wide", # Opsional: tata letak halaman (bisa "centered" atau "wide")
)

# --- Fungsi Crawling untuk Kompas.com ---
@st.cache_data(ttl=3600) # Cache hasil crawling selama 1 jam
def crawl_kompas(url):
    """
    Mengambil judul, tanggal, dan isi artikel dari URL Kompas.com (termasuk subdomain).
    Struktur umum:
    - Judul: <h1 class="read__title">
    - Tanggal: <div class="read__time">
    - Isi artikel: <div class="read__content"> (kemudian mencari <p> di dalamnya)
    """
    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/100.0.4896.127 Safari/537.36'
            )
        }

        # Ambil halaman dengan timeout 10 detik
        req = requests.get(url, headers=headers, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, 'lxml')

        # ==========================
        # 1️⃣ Ambil Judul Artikel
        # ==========================
        title_tag = soup.find('h1', class_='read__title')
        if not title_tag:
            title_tag = soup.find('h1')  # fallback
        crawled_title = title_tag.get_text(strip=True) if title_tag else 'Tidak ada judul'

        # ==========================
        # 2️⃣ Ambil Tanggal Publikasi
        # ==========================
        date_tag = soup.find('div', class_='read__time')
        if not date_tag:
            date_tag = soup.find('time')
        crawled_date = date_tag.get_text(strip=True) if date_tag else 'Tidak ada tanggal'

        # ==========================
        # 3️⃣ Ambil Isi Artikel
        # ==========================
        crawled_content = 'Tidak ada konten'
        full_text = ''

        main_content_div = soup.find('div', class_='read__content')

        if main_content_div:
            paragraphs = main_content_div.find_all('p')
            if paragraphs:
                full_text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            else:
                full_text = main_content_div.get_text(separator=' ', strip=True)
        else:
            fallback_content_div = (
                soup.find('div', class_='clearfix') or
                soup.find('article')
            )
            if fallback_content_div:
                paragraphs = fallback_content_div.find_all('p')
                if paragraphs:
                    full_text = ' '.join(p.get_text(strip=True) for p in paragraphs)
                else:
                    full_text = fallback_content_div.get_text(separator=' ', strip=True)
            else:
                paragraphs = soup.find_all('p')
                full_text = ' '.join(p.get_text(strip=True) for p in paragraphs) if paragraphs else ''

        # ==========================
        # 4️⃣ Bersihkan teks
        # ==========================
        if full_text:
            unwanted_patterns = [
                r'Baca juga :.*', r'Baca Juga :.*', r'Penulis :.*', r'Editor :.*',
                r'Sumber :.*', r'Ikuti kami.*', r'Simak berita.*', r'Berita Terkait :',
                r'Lihat Artikel Asli', r'• .*', r'Otomatis Mode Gelap Mode Terang.*',
                r'Login Gabung KOMPAS.com.*', r'Berikan Masukanmu.*', r'Langganan Kompas.*',
            ]

            for pattern in unwanted_patterns:
                full_text = re.sub(pattern, '', full_text, flags=re.DOTALL | re.IGNORECASE).strip()

            full_text = re.sub(r'\s+', ' ', full_text).strip()

            if full_text:
                crawled_content = full_text

        return {
            'crawled_title': crawled_title,
            'crawled_date': crawled_date,
            'crawled_content': crawled_content
        }

    except requests.exceptions.Timeout:
        return {
            'crawled_title': 'Gagal diakses (Timeout)',
            'crawled_date': 'Gagal diakses (Timeout)',
            'crawled_content': f'Gagal diakses karena timeout setelah 10 detik: {url}'
        }
    except requests.exceptions.RequestException as e:
        return {
            'crawled_title': 'Gagal diakses',
            'crawled_date': 'Gagal diakses',
            'crawled_content': f'Gagal diakses karena kesalahan permintaan: {e}'
        }
    except Exception as e:
        return {
            'crawled_title': 'Kesalahan tak terduga',
            'crawled_date': 'Kesalahan tak terduga',
            'crawled_content': f'Kesalahan tak terduga: {e}'
        }

# --- Fungsi Crawling untuk Tribunnews.com ---
@st.cache_data(ttl=3600) # Cache hasil crawling selama 1 jam
def crawl_tribunnews(url):
    """
    Mengambil judul, tanggal, dan isi artikel dari URL tribunnews.com.
    Menambahkan penanganan error dan timeout.
    """
    try:
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, 'lxml')

        crawled_title = 'Tidak ada judul'
        try:
            title_tag = soup.find('h1', class_='f32 ln40 fbo txt-black display-block mt10')
            if title_tag:
                crawled_title = title_tag.get_text(strip=True)
            else:
                title_tag_fallback = soup.find('title')
                if title_tag_fallback:
                    crawled_title = title_tag_fallback.get_text(strip=True)
        except Exception:
            pass

        crawled_date = 'Tidak ada tanggal'
        try:
            date_tag = soup.find('time')
            if date_tag:
                crawled_date = date_tag.get_text(strip=True)
        except Exception:
            pass

        crawled_content = 'Tidak ada konten'
        try:
            content_div = soup.find('div', class_='side-article txt-article multi-fontsize')
            if content_div:
                full_text = content_div.get_text(separator=' ', strip=True)

                unwanted_patterns = [
                    r'Baca Juga :.*', r'Penulis :.*', r'Editor :.*', r'Sumber :.*',
                    r'Ikuti kami di Google News', r'Simak berita terbaru Tribunnews.com di Google News',
                    r'Berita Terkait :', r'Baca juga :', r'Lihat Artikel Asli', r'• .*',
                ]

                for pattern in unwanted_patterns:
                    full_text = re.sub(pattern, '', full_text, flags=re.DOTALL | re.IGNORECASE).strip()
                
                full_text = re.sub(r'\s+', ' ', full_text).strip()

                if full_text:
                    crawled_content = full_text

        except Exception:
            pass

        return {
            'crawled_title': crawled_title,
            'crawled_date': crawled_date,
            'crawled_content': crawled_content
        }

    except requests.exceptions.Timeout:
        return {
            'crawled_title': 'Gagal diakses (Timeout)',
            'crawled_date': 'Gagal diakses (Timeout)',
            'crawled_content': f'Gagal diakses karena timeout setelah 10 detik: {url}'
        }
    except requests.exceptions.RequestException as e:
        return {
            'crawled_title': 'Gagal diakses',
            'crawled_date': 'Gagal diakses',
            'crawled_content': f'Gagal diakses karena kesalahan permintaan: {e}'
        }
    except Exception as e:
        return {
            'crawled_title': 'Kesalahan tak terduga',
            'crawled_date': 'Kesalahan tak terduga',
            'crawled_content': f'Kesalahan tak terduga: {e}'
        }

# --- Fungsi Crawling untuk Detik.com ---
@st.cache_data(ttl=3600) # Cache hasil crawling selama 1 jam
def crawl_detik(url):
    """
    Mengambil judul, tanggal, dan isi artikel dari URL detik.com.
    Menambahkan penanganan error dan timeout.
    """
    try:
        req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}, timeout=10)
        req.raise_for_status()
        soup = BeautifulSoup(req.text, 'lxml')

        crawled_title = 'Tidak ada judul'
        try:
            title_tag = soup.find('h1', class_='detail__title')
            if title_tag:
                crawled_title = title_tag.get_text(strip=True)
            else:
                title_tag_fallback = soup.find('title')
                if title_tag_fallback:
                    crawled_title = title_tag_fallback.get_text(strip=True)
        except Exception:
            pass

        crawled_date = 'Tidak ada tanggal'
        try:
            date_tag = soup.find('div', class_='detail__date')
            if date_tag:
                crawled_date = date_tag.get_text(strip=True)
        except Exception:
            pass

        crawled_content = 'Tidak ada konten'
        try:
            content_div = soup.find('div', class_='detail__body-text itp_bodycontent')
            if content_div:
                paragraphs = content_div.find_all('p')
                crawled_content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                content_div_fallback = soup.find('div', class_='read__content')
                if content_div_fallback:
                    crawled_content = content_div_fallback.get_text(strip=True)
        except Exception:
            pass

        return {
            'crawled_title': crawled_title,
            'crawled_date': crawled_date,
            'crawled_content': crawled_content
        }

    except requests.exceptions.Timeout:
        return {
            'crawled_title': 'Gagal diakses (Timeout)',
            'crawled_date': 'Gagal diakses (Timeout)',
            'crawled_content': f'Gagal diakses karena timeout setelah 10 detik: {url}'
        }
    except requests.exceptions.RequestException as e:
        return {
            'crawled_title': 'Gagal diakses',
            'crawled_date': 'Gagal diakses',
            'crawled_content': f'Gagal diakses karena kesalahan permintaan: {e}'
        }
    except Exception as e:
        return {
            'crawled_title': 'Kesalahan tak terduga',
            'crawled_date': 'Kesalahan tak terduga',
            'crawled_content': f'Kesalahan tak terduga: {e}'
        }


# --- Konten Utama Aplikasi Streamlit ---
st.title("Crawling Content dari Berita Online")
st.write("Pilih website, masukkan URL artikel, dan dapatkan judul, tanggal, serta isi artikel.")

# --- 1. Pilih Website ---
st.subheader("1. Pilih Website Sumber")
website_choice = st.selectbox(
    "Pilih website yang akan di-crawl:",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

# --- 2. Input URL ---
st.subheader("2. Input URL Artikel")
default_url = ""
if website_choice == "Kompas.com":
    default_url = "https://www.kompas.com/global/read/2023/10/26/123456789/peran-indonesia-dalam-penanganan-perubahan-iklim"
elif website_choice == "Tribunnews.com":
    default_url = "https://www.tribunnews.com/bisnis/2024/01/15/perusahaan-ini-fokus-pada-energi-terbarukan-untuk-masa-depan"
elif website_choice == "Detik.com":
    default_url = "https://www.detik.com/finance/berita-ekonomi-bisnis/d-7000000/pemerintah-dorong-penerapan-tata-kelola-perusahaan-yang-baik"

url_input = st.text_input(f"Masukkan URL artikel dari {website_choice}:", default_url)

# Tombol untuk memulai proses crawling
if st.button("Mulai Crawling"):
    if not url_input:
        st.error("URL tidak boleh kosong. Silakan masukkan URL artikel.")
        st.stop()

    crawled_data = None
    with st.spinner(f"Sedang melakukan crawling dari {website_choice}..."):
        if website_choice == "Kompas.com":
            crawled_data = crawl_kompas(url_input)
        elif website_choice == "Tribunnews.com":
            crawled_data = crawl_tribunnews(url_input)
        elif website_choice == "Detik.com":
            crawled_data = crawl_detik(url_input)
        
    if crawled_data:
        st.subheader("3. Hasil Crawling")
        st.write(f"**URL Asal:** {url_input}")
        st.write(f"**Judul:** {crawled_data['crawled_title']}")
        st.write(f"**Tanggal Publikasi:** {crawled_data['crawled_date']}")
        
        st.markdown("---")
        st.subheader("Isi Artikel:")
        st.expander("Klik untuk melihat seluruh isi artikel", expanded=True).write(crawled_data['crawled_content'])
    else:
        st.error("Gagal mendapatkan data crawling. Silakan coba URL lain atau periksa koneksi Anda.")

st.markdown("---")
st.info("Catatan: Fungsi crawling sangat bergantung pada struktur HTML website. Perubahan pada website dapat mempengaruhi hasil crawling.")
