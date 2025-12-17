import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet as wn
import os
import requests
from bs4 import BeautifulSoup
import re

# --- Konfigurasi Streamlit ---
st.set_page_config(layout="wide", page_title="ESG Screening Tool")

# --- Inisialisasi Session State ---
if 'crawled_data' not in st.session_state:
    st.session_state.crawled_data = None
if 'final_esg_category' not in st.session_state:
    st.session_state.final_esg_category = None
if 'crawled_url' not in st.session_state:
    st.session_state.crawled_url = None

# --- 0️⃣ Download Data NLTK (WordNet) ---
@st.cache_resource
def download_nltk_data():
    """Mengunduh data NLTK yang diperlukan (WordNet dan OMW) jika belum ada."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

download_nltk_data()

# --- 1️⃣ Daftar dasar ESG ---
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

# --- 2️⃣ Fungsi untuk mengambil sinonim dari WordNet Bahasa ---
def get_synonyms_wordnet(word, lang='ind'):
    """Mengambil sinonim dari WordNet untuk kata tertentu dalam bahasa yang ditentukan."""
    try:
        syns = set()
        for syn in wn.synsets(word, lang=lang):
            for lemma in syn.lemmas(lang):
                syns.add(lemma.name().replace("_", " "))
        return list(syns)
    except Exception:
        return []

# --- 3️⃣ Fallback morfologi sederhana ---
def morphology_variants(word):
    """Menghasilkan variasi morfologi sederhana dari sebuah kata."""
    variants = set()
    if word.startswith("ber"):
        variants.add(word.replace("ber", "ke", 1))
    if word.startswith("ke"):
        variants.add(word.replace("ke", "ber", 1))
    if word.endswith("an"):
        variants.add(word[:-2])
    if not word.endswith("an"):
        variants.add(word + "an")
    variants.add(word)
    return list(variants)

# --- 4️⃣ Fallback semantik berbasis embedding similarity ---
def semantic_expand(base_list, model, top_k=3):
    """Memperluas daftar kata berdasarkan kemiripan semantik menggunakan model embedding."""
    corpus = list(set(base_list)) # Memastikan kata-kata unik
    if not corpus:
        return []
    
    corpus_emb = model.encode(corpus, convert_to_tensor=True)
    new_words = set(corpus)
    
    for i, word in enumerate(corpus):
        query_emb = corpus_emb[i]
        cos_scores = util.cos_sim(query_emb, corpus_emb)[0](citation_0)
        # Ambil top_k+1 hasil untuk menyertakan kata itu sendiri, lalu lewati
        top_results = torch.topk(cos_scores, k=min(top_k + 1, len(corpus)))
        for score, idx in zip(top_results[0](citation_0)[1:], top_results[1](citation_1)[1:]):  # lewati kata itu sendiri
            new_words.add(corpus[idx])
    return list(new_words)

# --- 5️⃣ Perluasan kata per kategori ---
def expand_category(base_words, model=None, target_size=1000):
    """Menggabungkan sinonim WordNet, variasi morfologi, dan perluasan semantik."""
    expanded = set(base_words)
    
    # Langkah 1: Tambah sinonim WordNet
    current_words_for_synonyms = list(expanded) # Iterasi di atas salinan
    for word in current_words_for_synonyms:
        for s in get_synonyms_wordnet(word):
            expanded.add(s)
        if len(expanded) >= target_size:
            break
            
    # Langkah 2: Tambah morfologi
    if len(expanded) < target_size:
        current_words_to_morph = list(expanded) # Iterasi di atas salinan
        for word in current_words_to_morph:
            for var in morphology_variants(word):
                expanded.add(var)
            if len(expanded) >= target_size:
                break
                
    # Langkah 3: Tambah semantik
    if len(expanded) < target_size and model is not None:
        expanded.update(semantic_expand(list(expanded), model))
        
    return list(expanded)[:target_size]

# --- 6️⃣ Load model untuk fallback semantik (SentenceTransformer for word expansion) ---
@st.cache_resource
def load_semantic_expansion_model():
    """Memuat model embedding untuk perluasan kata (cached)."""
    model_cache_dir = "./.model_cache"
    os.makedirs(model_cache_dir, exist_ok=True) # Pastikan direktori ada
    model = SentenceTransformer("asmud/nomic-embed-indonesian", trust_remote_code=True, cache_folder=model_cache_dir)
    return model

semantic_expansion_model = load_semantic_expansion_model()

# --- 7️⃣ Perluas semua kategori ---
env_keywords = expand_category(env_base, model=semantic_expansion_model, target_size=1000)
soc_keywords = expand_category(soc_base, model=semantic_expansion_model, target_size=1000)
gov_keywords = expand_category(gov_base, model=semantic_expansion_model, target_size=1000)

# --- Klasifikasi Cepat via Keyword ---
def classify_esg_fast(text_to_classify):
    """Mengklasifikasikan teks berdasarkan kata kunci yang diperluas."""
    j = str(text_to_classify).lower()
    if any(k in j for k in env_keywords):
        return "Environment"
    elif any(k in j for k in soc_keywords):
        return "Social"
    elif any(k in j for k in gov_keywords):
        return "Governance"
    else:
        return "Non-ESG"

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

# --- Tata Letak Aplikasi Streamlit ---
st.title("ESG Screening Tool")
st.markdown("""
Aplikasi ini melakukan *screening* tema ESG (Environment, Social, Governance) dari teks
menggunakan pendekatan berbasis kata kunci yang diperluas dan analisis semantik AI.
""")

st.subheader("Input URL Artikel untuk Screening ESG")

website_choice = st.radio(
    "Pilih website sumber artikel:",
    ("Kompas.com", "Tribunnews.com", "Detik.com")
)

user_url_input = st.text_input("Masukkan URL Artikel di sini:")

if st.button("Crawl dan Lakukan Screening ESG"):
    if not user_url_input:
        st.warning("Mohon masukkan URL artikel untuk memulai screening.")
    else:
        st.subheader("Hasil Crawling dan Screening ESG")

        crawled_data = None
        with st.spinner(f"Melakukan crawling dari {website_choice} ({user_url_input})..."):
            if website_choice == "Kompas.com":
                crawled_data = crawl_kompas(user_url_input)
            elif website_choice == "Tribunnews.com":
                crawled_data = crawl_tribunnews(user_url_input)
            elif website_choice == "Detik.com":
                crawled_data = crawl_detik(user_url_input)
            else:
                st.error("Pilihan website tidak valid.")
                crawled_data = {'crawled_title': 'Error', 'crawled_date': 'Error', 'crawled_content': 'Pilihan website tidak valid.'}

        # Simpan hasil crawling ke session state
        st.session_state.crawled_data = crawled_data
        st.session_state.crawled_url = user_url_input

        if crawled_data['crawled_title'].startswith('Gagal diakses') or crawled_data['crawled_title'].startswith('Kesalahan tak terduga'):
            st.error(f"Gagal melakukan crawling: {crawled_data['crawled_title']}")
            st.write(f"Detail: {crawled_data['crawled_content']}")
            st.session_state.final_esg_category = "
