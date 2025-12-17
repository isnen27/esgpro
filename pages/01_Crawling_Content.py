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

# --- Tata Letak Aplikasi Streamlit ---
st.title("ESG Screening Tool")
st.markdown("""
Aplikasi ini melakukan *screening* tema ESG (Environment, Social, Governance) dari teks
menggunakan pendekatan berbasis kata kunci yang diperluas dan analisis semantik AI.
""")

st.subheader("Input Artikel untuk Screening ESG")

st.info("""
**Catatan Penting:** Aplikasi ini tidak dapat melakukan *web crawling* secara langsung dari Streamlit Community Cloud
karena keterbatasan akses ke API eksternal dan pustaka *web scraping*.
Untuk melakukan *screening*, mohon **paste judul artikel (atau potongan teks relevan)** yang Anda dapatkan
dari hasil *crawling* Anda sendiri.
""")

website_choice = st.radio(
    "Pilih website (untuk referensi, tidak fungsional untuk crawling di aplikasi ini):",
    ("Kompas.com", "Tribunnews.com", "Detik.com")
)

user_url_input = st.text_input("URL Artikel (untuk referensi, tidak fungsional untuk crawling di aplikasi ini):")

crawled_title_input = st.text_area(
    "Paste Judul Artikel (atau potongan teks relevan) di sini:",
    height=100,
    placeholder="Contoh: Pertamina Patra Niaga Raih Penghargaan ESG Terbaik di Asia"
)

if st.button("Mulai Screening ESG"):
    if not crawled_title_input:
        st.warning("Mohon paste judul artikel atau teks untuk memulai screening.")
    else:
        st.subheader("Hasil Screening ESG")

        # 1. Klasifikasi Awal Berbasis Kata Kunci
        st.write("---")
        st.markdown("#### 1. Klasifikasi Awal Berbasis Kata Kunci")
        with st.spinner("Menerapkan klasifikasi kata kunci..."):
            esg_keyword_category = classify_esg_fast(crawled_title_input)
        st.success(f"**Klasifikasi Kata Kunci:** {esg_keyword_category}")

        # 2. Analisis Semantik Lanjutan (Menggunakan AI)
        st.write("---")
        st.markdown("#### 2. Analisis Semantik Lanjutan (Menggunakan AI)")

        @st.cache_resource
        def load_indobert_model():
            """Memuat model embedding IndoBERT (cached)."""
            model_cache_dir = "./.model_cache"
            os.makedirs(model_cache_dir, exist_ok=True) # Pastikan direktori ada
            model = SentenceTransformer("indobenchmark/indobert-base-p2", cache_folder=model_cache_dir)
            return model

        indobert_model = load_indobert_model()

        themes = {
            "Environment": ["environmental sustainability", "climate change", "renewable energy", "carbon neutral"],
            "Social": ["social responsibility", "community development", "human rights", "workplace equality"],
            "Governance": ["corporate governance", "anti corruption", "ethical management", "transparency"]
        }

        # Theme embeddings are loaded once due to @st.cache_resource
        theme_embeddings = {k: indobert_model.encode(v, convert_to_tensor=True) for k, v in themes.items()}

        with st.spinner("Meng-encode judul/teks untuk analisis semantik..."):
            title_embedding = indobert_model.encode(crawled_title_input, convert_to_tensor=True).unsqueeze(0) # Add batch dimension
        
        with st.spinner("Menghitung skor kemiripan semantik..."):
            scores_env = util.cos_sim(title_embedding, theme_embeddings["Environment"]).max(dim=1).values.cpu().numpy()
            scores_soc = util.cos_sim(title_embedding, theme_embeddings["Social"]).max(dim=1).values.cpu().numpy()
            scores_gov = util.cos_sim(title_embedding, theme_embeddings["Governance"]).max(dim=1).values.cpu().numpy()

            esg_similarity = np.max([scores_env, scores_soc, scores_gov], axis=0)[0]
            best_idx = np.argmax([scores_env, scores_soc, scores_gov], axis=0)[0]
            prediksi_ai = np.select(
                [best_idx == 0, best_idx == 1, best_idx == 2],
                ["Environment", "Social", "Governance"],
                default="Non-ESG"
            )
        st.success(f"**Prediksi AI (Semantik):** {prediksi_ai} (Kemiripan: {esg_similarity:.2f})")

        # 3. Penerapan Threshold Otomatis
        st.write("---")
        st.markdown("#### 3. Finalisasi Klasifikasi")
        threshold = st.slider(
            "Pilih Threshold Kemiripan Semantik (untuk Non-ESG yang akan di-reklasifikasi oleh AI)",
            min_value=0.0, max_value=1.0, value=0.45, step=0.01, key="final_threshold_single"
        )

        final_esg_category = esg_keyword_category
        if final_esg_category == "Non-ESG" and esg_similarity >= threshold:
            final_esg_category = prediksi_ai
            st.info(f"Judul awalnya 'Non-ESG' berdasarkan kata kunci, tetapi direklasifikasi menjadi **{final_esg_category}** oleh AI karena kemiripan semantik tinggi ({esg_similarity:.2f} >= {threshold}).")
        elif final_esg_category != "Non-ESG":
            st.info(f"Judul sudah diklasifikasikan sebagai **{final_esg_category}** berdasarkan kata kunci.")
        else:
            st.info(f"Judul tetap 'Non-ESG' karena tidak ada kata kunci yang cocok dan kemiripan semantik ({esg_similarity:.2f}) di bawah threshold ({threshold}).")

        st.markdown(f"## **Klasifikasi ESG Akhir: {final_esg_category}**")

        # Jika kategori akhir adalah Non-ESG, proses tidak dilanjutkan
        if final_esg_category == "Non-ESG":
            st.warning("Proses tidak dilanjutkan karena diklasifikasikan sebagai Non-ESG.")
        else:
            st.success("Proses dapat dilanjutkan karena diklasifikasikan sebagai ESG.")
