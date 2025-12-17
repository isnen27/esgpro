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
from nltk.tokenize.punkt import PunktSentenceTokenizer

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

    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'maxent_ne_chunker': 'chunkers/maxent_ne_chunker',
        'words': 'corpora/words',
    }

    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path, paths=[nltk_data_dir])
        except LookupError:
            nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)

    # --- Explicitly load Indonesian PunktSentenceTokenizer ---
    try:
        indonesian_punkt_path = os.path.join(nltk_data_dir, 'tokenizers', 'punkt', 'indonesian.pickle')
        if os.path.exists(indonesian_punkt_path):
            with open(indonesian_punkt_path, 'rb') as f:
                st.session_state._indonesian_punkt_tokenizer = pickle.load(f)
        else:
            st.error(f"Indonesian Punkt tokenizer pickle not found at {indonesian_punkt_path} after 'punkt' download. Sentence tokenization for Indonesian might fail.")
            st.session_state._indonesian_punkt_tokenizer = None
    except Exception as e:
        st.error(f"Error loading Indonesian Punkt tokenizer: {e}. Sentence tokenization for Indonesian might fail.")
        st.session_state._indonesian_punkt_tokenizer = None

download_nltk_data_for_analysis()

# --- Fungsi untuk peringkasan TF-IDF ---
def summarize_text_tfidf(text, num_sentences=5):
    if not text or len(text.strip()) == 0:
        return "Tidak ada konten untuk diringkas."

    if '_indonesian_punkt_tokenizer' in st.session_state and st.session_state._indonesian_punkt_tokenizer:
        sentences = st.session_state._indonesian_punkt_tokenizer.tokenize(text)
    else:
        st.warning("Menggunakan sentence tokenizer generik karena yang spesifik Indonesia tidak tersedia. Akurasi mungkin terpengaruh.")
        sentences = nltk.sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    stop_words_id = set(stopwords.words('indonesian'))
    
    def preprocess(sentence):
        words = word_tokenize(sentence.lower())
        return [word for word in words if word.isalnum() and word not in stop_words_id]

    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=list(stop_words_id))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return "Tidak dapat meringkas. Konten mungkin terlalu pendek atau tidak relevan."

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = tfidf_matrix[i].sum() 
        sentence_scores[sentence] = score

    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    return ' '.join(best_sentences)

# --- Fungsi untuk NER menggunakan NLTK ---
def perform_ner_nltk(text):
    if not text or len(text.strip()) == 0:
        return []

    if '_indonesian_punkt_tokenizer' in st.session_state and st.session_state._indonesian_punkt_tokenizer:
        sentences = st.session_state._indonesian_punkt_tokenizer.tokenize(text)
    else:
        st.warning("Menggunakan sentence tokenizer generik untuk NER karena yang spesifik Indonesia tidak tersedia. Akurasi mungkin terpengaruh.")
        sentences = nltk.sent_tokenize(text)

    all_named_entities = []
    for sentence in sentences:
        words = word_tokenize(sentence, language='english') 
        tagged_words = pos_tag(words, lang='eng')

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
    crawled_date = st.session_state.crawled_data['crawled_date'] # <--- BARIS INI YANG DIPERBAIKI
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
