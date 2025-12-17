import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet as wn
import os

# --- Konfigurasi Streamlit ---
st.set_page_config(layout="wide", page_title="ESG Screening Tool")

# --- 0️⃣ Download Data NLTK (WordNet) ---
@st.cache_resource
def download_nltk_data():
    """Mengunduh data NLTK yang diperlukan (WordNet dan OMW) jika belum ada."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError: # Perbaikan di sini
        st.info("📦 Mengunduh data NLTK 'wordnet'...")
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError: # Perbaikan di sini
        st.info("📦 Mengunduh data NLTK 'omw-1.4' (Open Multilingual Wordnet)...")
        nltk.download('omw-1.4')
    st.success("✅ Data NLTK siap!")

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
        cos_scores = util.cos_sim(query_emb, corpus_emb)[0]
        # Ambil top_k+1 hasil untuk menyertakan kata itu sendiri, lalu lewati
        top_results = torch.topk(cos_scores, k=min(top_k + 1, len(corpus)))
        for score, idx in zip(top_results[0][1:], top_results[1][1:]):  # lewati kata itu sendiri
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
    st.info("🔍 Memuat model embedding untuk perluasan kata (asmud/nomic-embed-indonesian)...")
    # Gunakan folder cache kustom yang dapat ditulis di Streamlit Cloud
    model_cache_dir = "./.model_cache"
    os.makedirs(model_cache_dir, exist_ok=True) # Pastikan direktori ada
    model = SentenceTransformer("asmud/nomic-embed-indonesian", trust_remote_code=True, cache_folder=model_cache_dir)
    st.success("✅ Model perluasan kata berhasil dimuat!")
    return model

semantic_expansion_model = load_semantic_expansion_model()

# --- 7️⃣ Perluas semua kategori ---
st.header("Proses Perluasan Kata Kunci ESG")
with st.spinner("Memperluas kata kunci Lingkungan..."):
    env_keywords = expand_category(env_base, model=semantic_expansion_model, target_size=1000)
    st.write(f"🟢 Kata kunci Environment diperluas menjadi {len(env_keywords)} kata.")
with st.spinner("Memperluas kata kunci Sosial..."):
    soc_keywords = expand_category(soc_base, model=semantic_expansion_model, target_size=1000)
    st.write(f"🟣 Kata kunci Social diperluas menjadi {len(soc_keywords)} kata.")
with st.spinner("Memperluas kata kunci Tata Kelola..."):
    gov_keywords = expand_category(gov_base, model=semantic_expansion_model, target_size=1000)
    st.write(f"🔵 Kata kunci Governance diperluas menjadi {len(gov_keywords)} kata.")

# --- Klasifikasi Cepat via Keyword ---
def classify_esg_fast(judul):
    """Mengklasifikasikan judul berdasarkan kata kunci yang diperluas."""
    j = str(judul).lower()
    if any(k in j for k in env_keywords):
        return "Environment"
    elif any(k in j for k in soc_keywords):
        return "Social"
    elif any(k in j for k in gov_keywords):
        return "Governance"
    else:
        return "Non-ESG"

# --- Tata Letak Aplikasi Streamlit ---
st.title("ESG Screening Tool")
st.markdown("""
Aplikasi ini melakukan *screening* tema ESG (Environment, Social, Governance) dari teks
menggunakan pendekatan berbasis kata kunci yang diperluas dan analisis semantik AI.
""")

# --- Unggah File Pengguna ---
uploaded_file = st.file_uploader("Unggah file CSV atau Excel Anda (harus memiliki kolom 'judul')", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "judul" not in df.columns:
            st.error("File yang diunggah harus memiliki kolom bernama 'judul'.")
            df = None
        else:
            st.success("File berhasil diunggah! Menampilkan 5 baris pertama:")
            st.dataframe(df.head())
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")

if df is not None:
    st.subheader("1. Klasifikasi Awal Berbasis Kata Kunci")
    with st.spinner("Menerapkan klasifikasi kata kunci..."):
        df["Kategori_ESG"] = df["judul"].apply(classify_esg_fast)
    st.success("Klasifikasi kata kunci selesai!")
    st.dataframe(df[["judul", "Kategori_ESG"]].head())

    # --- 🧬 5️⃣ Analisis Semantik (AI) ---
    st.subheader("2. Analisis Semantik Lanjutan (Menggunakan AI)")

    @st.cache_resource
    def load_indobert_model():
        """Memuat model embedding IndoBERT (cached)."""
        st.info("🔍 Memuat model embedding IndoBERT (indobenchmark/indobert-base-p2)...")
        model_cache_dir = "./.model_cache"
        os.makedirs(model_cache_dir, exist_ok=True) # Pastikan direktori ada
        model = SentenceTransformer("indobenchmark/indobert-base-p2", cache_folder=model_cache_dir)
        st.success("✅ Model IndoBERT berhasil dimuat!")
        return model

    indobert_model = load_indobert_model()

    themes = {
        "Environment": ["environmental sustainability", "climate change", "renewable energy", "carbon neutral"],
        "Social": ["social responsibility", "community development", "human rights", "workplace equality"],
        "Governance": ["corporate governance", "anti corruption", "ethical management", "transparency"]
    }

    with st.spinner("Meng-encode tema ESG..."):
        theme_embeddings = {k: indobert_model.encode(v, convert_to_tensor=True) for k, v in themes.items()}
    st.success("Tema ESG berhasil di-encode!")

    with st.spinner("Meng-encode judul berita..."):
        # Pastikan kolom 'judul' adalah string sebelum encoding
        judul_embeddings = indobert_model.encode(df["judul"].astype(str).tolist(), batch_size=64, show_progress_bar=False, convert_to_tensor=True)
    st.success("Judul berita berhasil di-encode!")

    with st.spinner("Menghitung skor kemiripan semantik..."):
        scores_env = util.cos_sim(judul_embeddings, theme_embeddings["Environment"]).max(dim=1).values.cpu().numpy()
        scores_soc = util.cos_sim(judul_embeddings, theme_embeddings["Social"]).max(dim=1).values.cpu().numpy()
        scores_gov = util.cos_sim(judul_embeddings, theme_embeddings["Governance"]).max(dim=1).values.cpu().numpy()

        df["esg_similarity"] = np.max([scores_env, scores_soc, scores_gov], axis=0)
        best_idx = np.argmax([scores_env, scores_soc, scores_gov], axis=0)
        df["Prediksi_AI"] = np.select(
            [best_idx == 0, best_idx == 1, best_idx == 2],
            ["Environment", "Social", "Governance"],
            default="Non-ESG" # Default ini akan jarang tercapai karena argmax selalu memilih salah satu
        )
    st.success("Skor kemiripan semantik dihitung!")
    st.dataframe(df[["judul", "Kategori_ESG", "Prediksi_AI", "esg_similarity"]].head())

    # --- 🤖 6️⃣ Terapkan Threshold Otomatis ---
    st.subheader("3. Penerapan Threshold Otomatis untuk Finalisasi Klasifikasi")
    threshold = st.slider(
        "Pilih Threshold Kemiripan Semantik (untuk Non-ESG yang akan di-reklasifikasi oleh AI)",
        min_value=0.0, max_value=1.0, value=0.45, step=0.01
    )

    with st.spinner(f"Menerapkan threshold {threshold} untuk finalisasi klasifikasi..."):
        # Reklasifikasi item yang awalnya "Non-ESG" tetapi memiliki skor semantik tinggi
        mask_auto = (df["Kategori_ESG"] == "Non-ESG") & (df["esg_similarity"] >= threshold)
        df.loc[mask_auto, "Kategori_ESG"] = df.loc[mask_auto, "Prediksi_AI"]
    st.success("Finalisasi klasifikasi selesai!")

    st.subheader("Hasil Klasifikasi ESG Akhir")
    st.dataframe(df[["judul", "Kategori_ESG", "Prediksi_AI", "esg_similarity"]])
