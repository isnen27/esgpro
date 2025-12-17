# pages/02_Analysis.py
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import heapq
import os
import pickle
from nltk.tokenize.punkt import PunktSentenceTokenizer # Impor kelas ini secara spesifik

st.set_page_config(layout="wide", page_title="ESG Analysis")
st.title("ESG Analysis Page")

st.markdown("---")

# --- NLTK Data Downloads (Cached) ---
@st.cache_resource
def download_nltk_data_for_analysis():
    """Mengunduh data NLTK yang diperlukan untuk analisis (cached)."""
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    os.makedirs(nltk_data_dir, exist_ok=True)

    # List of NLTK resources to download
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'words': 'corpora/words', # Diperlukan oleh maxent_ne_chunker
    }

    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path, paths=[nltk_data_dir])
        except LookupError:
            nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)

    # --- Explicitly load Indonesian PunktSentenceTokenizer ---
    # Inisialisasi di sini, akan diisi di blok try/except
    st.session_state._indonesian_punkt_tokenizer = None
    try:
        indonesian_punkt_file = nltk.data.find('tokenizers/punkt/indonesian.pickle', paths=[nltk_data_dir])
        with open(indonesian_punkt_file, 'rb') as f:
            st.session_state._indonesian_punkt_tokenizer = pickle.load(f)
        # st.success("Indonesian Punkt tokenizer loaded successfully.") # Hapus notifikasi ini
    except LookupError:
        st.error("Indonesian Punkt tokenizer pickle not found after 'punkt' download. Sentence tokenization for Indonesian will fail.")
        st.session_state._indonesian_punkt_tokenizer = None
    except Exception as e:
        st.error(f"Error loading Indonesian Punkt tokenizer: {e}. Sentence tokenization for Indonesian will fail.")
        st.session_state._indonesian_punkt_tokenizer = None

download_nltk_data_for_analysis()

# --- Fungsi untuk peringkasan TF-IDF ---
def summarize_text_tfidf(text, num_sentences=5):
    if not text or len(text.strip()) == 0:
        return "Tidak ada konten untuk diringkas."

    # Gunakan tokenizer yang sudah dimuat dari session_state
    if '_indonesian_punkt_tokenizer' in st.session_state and st.session_state._indonesian_punkt_tokenizer:
        sentences = st.session_state._indonesian_punkt_tokenizer.tokenize(text)
    else:
        # Jika tokenizer Indonesia gagal dimuat, kita tidak bisa melakukan sentence tokenization yang benar.
        # Kembalikan pesan kesalahan atau seluruh teks sebagai satu "kalimat".
        st.error("Peringkasan gagal: Indonesian Punkt tokenizer tidak tersedia.")
        return "Gagal meringkas: Indonesian Punkt tokenizer tidak tersedia."

    if len(sentences) <= num_sentences:
        return text # Jika teks pendek, kembalikan seluruh teks

    stop_words_id = set(stopwords.words('indonesian'))
    
    def preprocess(sentence):
        # Gunakan word_tokenize generik (default English). Ini adalah kompromi.
        # Untuk TF-IDF, ini mungkin cukup, tetapi tidak ideal untuk Bahasa Indonesia.
        words = word_tokenize(sentence.lower()) # Tanpa argumen 'language'
        # Filter kata-kata non-alfanumerik dan stopwords
        return [word for word in words if word.isalnum() and word not in stop_words_id]

    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=list(stop_words_id))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError: # Tangani kosakata kosong jika teks terlalu pendek atau semua stopwords
        return "Tidak dapat meringkas. Konten mungkin terlalu pendek atau tidak relevan."

    # Hitung skor untuk setiap kalimat
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        # Jumlahkan skor TF-IDF kata-kata dalam kalimat
        score = tfidf_matrix[i].sum() 
        sentence_scores[sentence] = score

    # Ambil N kalimat teratas
    # Gunakan heapq.nlargest untuk efisiensi
    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    return ' '.join(best_sentences)

# --- Fungsi untuk NER menggunakan NLTK ---
def perform_ner_nltk(text):
    if not text or len(text.strip()) == 0:
        return []

    # Tokenisasi kalimat terlebih dahulu menggunakan tokenizer yang sudah dimuat
    if '_indonesian_punkt_tokenizer' in st.session_state and st.session_state._indonesian_punkt_tokenizer:
        sentences = st.session_state._indonesian_punkt_tokenizer.tokenize(text)
    else:
        st.error("NER gagal: Indonesian Punkt tokenizer tidak tersedia.")
        return [] # Jika tokenizer Indonesia gagal dimuat, kembalikan daftar kosong untuk NER

    all_named_entities = []
    for sentence in sentences:
        # Tokenisasi kata dan POS tagging per kalimat
        # Penting: NLTK's ne_chunk dilatih dengan tag POS bahasa Inggris.
        # Menggunakan word_tokenize(..., language='english') dan pos_tag(..., lang='eng')
        # akan memberikan hasil yang lebih baik dengan ne_chunk meskipun teksnya Bahasa Indonesia.
        words = word_tokenize(sentence, language='english') 
        tagged_words = pos_tag(words, lang='eng') # Explicitly use English POS tagger

        # Named Entity Chunking
        tree = ne_chunk(tagged_words) 

        for subtree in tree.subtrees():
            if hasattr(subtree, 'label') and subtree.label() != 'S':
                entity_type = subtree.label()
                entity_name = " ".join([word for word, tag in subtree.leaves()])
                all_named_entities.append((entity_name, entity_type))
    
    return all_named_entities

# Cek apakah ada data di session_state dari halaman sebelumnya
if st.session_state.crawled_data and st.session_state.final_esg_category:
    st.success("Data artikel berhasil dimuat dari sesi screening!")
    
    crawled_url = st.session_state.crawled_url
    crawled_title = st.session_state.crawled_data['crawled_title']
    crawled_date = st.session_state.crawled_data['crawled_date']
    crawled_content = st.session_state.crawled_data['crawled_content']
    final_esg_category = st.session_state.final_esg_category

    st.markdown("### Artikel yang Sedang Dianalisis:")
    st.write(f"**URL Artikel:** {crawled_url}")
    st.write(f"**Judul Artikel:** {crawled_title}")
    st.write(f"**Tanggal Publikasi:** {crawled_date}")
    st.write(f"**Kategori ESG Akhir:** {final_esg_category}")
    
    with st.expander("Lihat Isi Artikel Lengkap"):
        st.write(crawled_content)

    st.markdown("---")
    st.markdown("#### Hasil Analisis Detail")

    # --- TF-IDF Summarization ---
    st.subheader("Peringkasan Artikel (TF-IDF)")
    num_sentences_summary = st.slider("Jumlah kalimat untuk ringkasan:", min_value=1, max_value=10, value=5)
    with st.spinner("Membuat ringkasan artikel..."):
        summary = summarize_text_tfidf(crawled_content, num_sentences=num_sentences_summary)
    st.write(summary)

    # --- NLTK NER ---
    st.subheader("Named Entity Recognition (NER) dengan NLTK")
    st.warning("Catatan: NLTK's `ne_chunk` primernya dilatih untuk Bahasa Inggris dan bekerja paling baik dengan POS tag Bahasa Inggris. Akurasi untuk Bahasa Indonesia mungkin terbatas.")
    with st.spinner("Mengekstrak entitas nama..."):
        entities = perform_ner_nltk(crawled_content)
    
    if entities:
        import pandas as pd
        entity_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
        st.dataframe(entity_df)
    else:
        st.info("Tidak ada entitas nama yang terdeteksi dalam artikel.")

    st.markdown("---")
    if st.button("Bersihkan Data & Kembali ke Screening"):
        st.session_state.crawled_data = None
        st.session_state.final_esg_category = None
        st.session_state.crawled_url = None
        st.warning("Data sesi telah dihapus. Silakan gunakan sidebar untuk kembali ke halaman utama untuk screening baru.")
        st.rerun()
else:
    st.warning("Tidak ada data artikel yang tersedia untuk analisis. Silakan kembali ke halaman utama untuk melakukan screening terlebih dahulu.")
    st.info("Gunakan sidebar di kiri untuk navigasi.")
