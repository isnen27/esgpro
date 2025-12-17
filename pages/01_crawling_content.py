# pages/01_crawling_content.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util

# Import NLTK dan WordNet
import nltk
from nltk.corpus import wordnet as wn

# --- NLTK Data Download (Instruksi untuk pengguna) ---
# Pastikan Anda telah mengunduh data NLTK yang diperlukan:
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# Jika Anda mendeploy ke Streamlit Cloud, buat file `nltk.txt` di root proyek Anda
# dengan isi:
# wordnet
# omw-1.4

# --- Database Kata Dasar ESG ---
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

# --- Fungsi Perluasan Kata ---
@st.cache_data
def get_synonyms_wordnet(word, lang='ind'):
    """Mengambil sinonim dari WordNet."""
    syns = set()
    try:
        for syn in wn.synsets(word, lang=lang):
            for lemma in syn.lemmas(lang):
                syns.add(lemma.name().replace("_", " "))
    except Exception:
        pass # Handle case where word not found or lang not supported
    return list(syns)

@st.cache_data
def morphology_variants(word):
    """Menghasilkan varian morfologi sederhana."""
    variants = set()
    word_lower = word.lower()
    if word_lower.startswith("ber"):
        variants.add(word_lower.replace("ber", "ke", 1))
    if word_lower.startswith("ke"):
        variants.add(word_lower.replace("ke", "ber", 1))
    if word_lower.endswith("an"):
        variants.add(word_lower[:-2])
    if not word_lower.endswith("an"):
        variants.add(word_lower + "an")
    variants.add(word_lower)
    return list(variants)

@st.cache_data
def semantic_expand(base_list, model, top_k=3):
    """Memperluas daftar kata secara semantik menggunakan embedding similarity."""
    if not base_list:
        return []
    corpus = list(set(base_list)) # Hapus duplikat dan konversi ke list
    
    # Hindari encoding jika corpus terlalu besar atau kosong
    if not corpus:
        return []

    corpus_emb = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
    new_words = set(corpus)
    for i, word in enumerate(corpus):
        query_emb = corpus_emb[i]
        # Pastikan query_emb adalah 1D tensor untuk util.cos_sim
        if query_emb.dim() == 0: # Handle scalar case if it ever happens
            continue
        
        # Reshape query_emb to be 2D if it's 1D
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        cos_scores = util.cos_sim(query_emb, corpus_emb)[0]
        # top_k+1 karena hasil pertama adalah kata itu sendiri
        top_results = torch.topk(cos_scores, k=min(top_k + 1, len(corpus)))
        for score, idx in zip(top_results[0][1:], top_results[1][1:]):  # skip self
            new_words.add(corpus[idx])
    return list(new_words)

@st.cache_resource
def expand_category(base_words, model, target_size=100): # Mengurangi target_size default
    """
    Memperluas kategori kata menggunakan sinonim, morfologi, dan semantik.
    Fungsi ini di-cache karena bisa memakan waktu lama.
    """
    expanded = set(base_words)
    
    # Langkah 1: Tambah sinonim WordNet
    with st.spinner(f"Memperluas {len(base_words)} kata dasar dengan sinonim WordNet..."):
        current_words_to_expand = list(base_words)
        for word in current_words_to_expand:
            for s in get_synonyms_wordnet(word):
                expanded.add(s)
            if len(expanded) >= target_size:
                break
    
    # Langkah 2: Tambah morfologi
    if len(expanded) < target_size:
        with st.spinner(f"Memperluas {len(expanded)} kata dengan varian morfologi..."):
            current_words_to_expand = list(expanded) # Ambil yang sudah diperluas
            for word in current_words_to_expand:
                for var in morphology_variants(word):
                    expanded.add(var)
                if len(expanded) >= target_size:
                    break
    
    # Langkah 3: Tambah semantik
    if len(expanded) < target_size and model is not None:
        with st.spinner(f"Memperluas {len(expanded)} kata secara semantik (ini mungkin butuh waktu)..."):
            expanded.update(semantic_expand(list(expanded), model, top_k=5)) # top_k bisa disesuaikan
            
    return list(expanded)[:target_size]


# --- Model Caching ---
@st.cache_resource
def load_esg_model():
    """Loads the SentenceTransformer model for ESG semantic analysis."""
    with st.spinner("Memuat model AI untuk analisis semantik... Ini mungkin membutuhkan waktu beberapa saat."):
        cache_dir = "./model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        model = SentenceTransformer("indobenchmark/indobert-base-p2", cache_folder=cache_dir)
    return model

# --- Cached Expanded Themes Dictionary ---
@st.cache_resource
def get_expanded_themes_dict(model):
    """
    Mengembangkan daftar kata dasar ESG dan mengembalikannya sebagai kamus tema.
    Fungsi ini di-cache untuk menghindari pemrosesan ulang yang mahal.
    """
    st.info("Memulai proses perluasan database kata ESG (ini hanya akan berjalan sekali)...")
    # Mengurangi target_size untuk performa yang lebih baik di aplikasi web
    expanded_env = expand_category(env_base, model, target_size=100)
    expanded_soc = expand_category(soc_base, model, target_size=100)
    expanded_gov = expand_category(gov_base, model, target_size=100)
    st.success("Database kata ESG berhasil diperluas!")

    # Gabungkan semua kata yang diperluas ke dalam dictionary themes
    themes = {
        "Environment": expanded_env,
        "Social": expanded_soc,
        "Governance": expanded_gov
    }
    return themes


# --- Konten Utama Aplikasi Streamlit ---
st.title("Crawling Content & Prediksi Tema ESG")
st.write("Di sini Anda dapat memilih website, menginput URL, dan memprediksi tema ESG pada judul berita menggunakan AI Semantik dengan database kata yang diperkaya.")

# --- 1. Pilih Website ---
st.subheader("1. Pilih Website Sumber")
website_choice = st.selectbox(
    "Pilih website yang akan di-crawl:",
    ["Kompas.com", "Tribunnews.com", "Detik.com"]
)

# --- 2. Input URL ---
st.subheader("2. Input URL untuk Analisis")
# Contoh URL yang relevan dengan format yang Anda berikan
default_url = "https://www.kompas.com/global/read/2023/10/26/123456789/peran-indonesia-dalam-penanganan-perubahan-iklim"
if website_choice == "Tribunnews.com":
    default_url = "https://www.tribunnews.com/bisnis/2024/01/15/perusahaan-ini-fokus-pada-energi-terbarukan-untuk-masa-depan"
elif website_choice == "Detik.com":
    default_url = "https://www.detik.com/finance/berita-ekonomi-bisnis/d-7000000/pemerintah-dorong-penerapan-tata-kelola-perusahaan-yang-baik"

url_input = st.text_input(f"Masukkan URL dari {website_choice} untuk dianalisis:", default_url)

# --- Placeholder untuk crawling aktual ---
def simulate_crawling(url):
    """
    Fungsi dummy untuk mensimulasikan hasil crawling.
    Ganti dengan logika crawling Anda yang sebenarnya.
    Seharusnya mengembalikan DataFrame dengan kolom 'link'.
    """
    # Untuk demonstrasi, kita hanya membuat DataFrame dengan URL input
    return pd.DataFrame({"link": [url]})

# Tombol untuk memulai proses analisis
if st.button("Mulai Prediksi Tema ESG"):
    if not url_input:
        st.error("URL tidak boleh kosong. Silakan masukkan URL.")
        st.stop()

    # Simulasi crawling untuk mendapatkan DataFrame awal
    df = simulate_crawling(url_input)

    with st.spinner("Mengekstrak judul dari URL..."):
        df["url_lengkap"] = df["link"].apply(lambda x: "https:" + x if isinstance(x, str) and x.startswith("//") else x)
        df["judul"] = df["url_lengkap"].str.extract(r"/\d{4}/\d{2}/\d{2}/[^/]+/([^/]+)")[0]
        df["judul"] = df["judul"].fillna("").astype(str).str.replace('-', ' ')
    
    # --- Analisis Semantik (AI) ---
    st.subheader("3. Hasil Prediksi Tema ESG")
    
    model = load_esg_model() # Muat model yang di-cache
    themes = get_expanded_themes_dict(model) # Muat tema yang sudah diperluas dan di-cache

    with st.spinner("Menghitung embedding judul berita dan analisis kemiripan..."):
        valid_titles_mask = df["judul"].astype(str).str.len() > 0
        valid_titles = df.loc[valid_titles_mask, "judul"].astype(str).tolist()

        if not valid_titles:
            st.warning("Tidak ada judul berita yang valid untuk analisis semantik.")
            df["esg_similarity"] = 0.0
            df["Prediksi_AI"] = "Non-ESG"
        else:
            judul_embeddings = model.encode(valid_titles, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
            
            scores_env_full = np.zeros(len(df))
            scores_soc_full = np.zeros(len(df))
            scores_gov_full = np.zeros(len(df))

            # Encode tema yang diperluas
            theme_embeddings = {k: model.encode(v, convert_to_tensor=True, show_progress_bar=False) for k, v in themes.items()}

            scores_env_valid = util.cos_sim(judul_embeddings, theme_embeddings["Environment"]).max(dim=1).values.cpu().numpy()
            scores_soc_valid = util.cos_sim(judul_embeddings, theme_embeddings["Social"]).max(dim=1).values.cpu().numpy()
            scores_gov_valid = util.cos_sim(judul_embeddings, theme_embeddings["Governance"]).max(dim=1).values.cpu().numpy()

            scores_env_full[valid_titles_mask] = scores_env_valid
            scores_soc_full[valid_titles_mask] = scores_soc_valid
            scores_gov_full[valid_titles_mask] = scores_gov_valid

            df["esg_similarity"] = np.max([scores_env_full, scores_soc_full, scores_gov_full], axis=0)
            best_idx = np.argmax([scores_env_full, scores_soc_full, scores_gov_full], axis=0)
            df["Prediksi_AI"] = np.select(
                [best_idx == 0, best_idx == 1, best_idx == 2],
                ["Environment", "Social", "Governance"],
                default="Non-ESG"
            )
    
    # --- Terapkan Threshold Otomatis ---
    threshold = st.slider("Pilih nilai threshold kemiripan (skor di bawah ini akan dianggap 'Non-ESG'):", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    
    with st.spinner("Menerapkan threshold kemiripan..."):
        df["Tema_ESG_Final"] = df["Prediksi_AI"]
        df.loc[df["esg_similarity"] < threshold, "Tema_ESG_Final"] = "Non-ESG"

    st.write(f"**URL:** {url_input}")
    st.write(f"**Judul:** {df['judul'].iloc[0]}")
    st.write(f"**Prediksi Tema ESG:** **{df['Tema_ESG_Final'].iloc[0]}**")
    st.write(f"Skor Kemiripan AI Tertinggi: {df['esg_similarity'].iloc[0]:.4f}")


st.markdown("---")
st.info("Catatan: Fungsi crawling web aktual belum diimplementasikan. Data di atas adalah hasil simulasi dari URL input.")
st.warning("Proses perluasan database kata ESG hanya akan berjalan sekali saat aplikasi pertama kali dimuat atau setelah perubahan kode.")
