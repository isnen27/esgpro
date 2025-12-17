# pages/02_Analysis.py
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import heapq # Untuk mengambil N elemen terbesar dari list

st.set_page_config(layout="wide", page_title="ESG Analysis")
st.title("ESG Analysis Page")

st.markdown("---")

# --- NLTK Data Downloads (Cached) ---
@st.cache_resource
def download_nltk_data_for_analysis():
    """Mengunduh data NLTK yang diperlukan untuk analisis (cached)."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
    try:
        nltk.data.find('corpora/words') # Diperlukan oleh maxent_ne_chunker
    except LookupError:
        nltk.download('words')

download_nltk_data_for_analysis()

# --- Fungsi untuk peringkasan TF-IDF ---
def summarize_text_tfidf(text, num_sentences=5):
    if not text or len(text.strip()) == 0:
        return "Tidak ada konten untuk diringkas."

    sentences = sent_tokenize(text, language='indonesian')
    if len(sentences) <= num_sentences:
        return text # Jika teks pendek, kembalikan seluruh teks

    # Preprocessing: lowercase, remove punctuation, remove stopwords
    stop_words_id = set(stopwords.words('indonesian'))
    
    def preprocess(sentence):
        words = word_tokenize(sentence.lower(), language='indonesian')
        return [word for word in words if word.isalnum() and word not in stop_words_id]

    # Buat TF-IDF Vectorizer
    # Gunakan fungsi preprocessing kustom untuk tokenizer
    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=list(stop_words_id))
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError: # Handle empty vocabulary if text is too short or all stopwords
        return "Tidak dapat meringkas. Konten mungkin terlalu pendek atau tidak relevan."

    # Hitung skor untuk setiap kalimat
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        score = tfidf_matrix[i].sum() # Jumlahkan skor TF-IDF kata-kata dalam kalimat
        sentence_scores[sentence] = score

    # Ambil N kalimat teratas
    # Gunakan heapq.nlargest untuk efisiensi
    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    return ' '.join(best_sentences)

# --- Fungsi untuk NER menggunakan NLTK ---
def perform_ner_nltk(text):
    if not text or len(text.strip()) == 0:
        return []

    # Tokenisasi kata dan POS tagging
    words = word_tokenize(text, language='indonesian')
    tagged_words = pos_tag(words)

    # Named Entity Chunking
    named_entities = []
    tree = ne_chunk(tagged_words)

    for subtree in tree.subtrees():
        if subtree.label() != 'S': # 'S' adalah label untuk kalimat
            entity_type = subtree.label()
            entity_name = " ".join([word for word, tag in subtree.leaves()])
            named_entities.append((entity_name, entity_type))
    
    return named_entities

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
    with st.spinner("Mengekstrak entitas nama..."):
        entities = perform_ner_nltk(crawled_content)
    
    if entities:
        entity_df = pd.DataFrame(entities, columns=['Entity', 'Type'])
        st.dataframe(entity_df)
    else:
        st.info("Tidak ada entitas nama yang terdeteksi dalam artikel.")

    st.markdown("---")
    if st.button("Bersihkan Data & Kembali ke Screening"):
        # Bersihkan session state agar halaman screening bisa memulai proses baru
        st.session_state.crawled_data = None
        st.session_state.final_esg_category = None
        st.session_state.crawled_url = None
        st.warning("Data sesi telah dihapus. Silakan gunakan sidebar untuk kembali ke halaman utama untuk screening baru.")
        st.rerun() # Muat ulang halaman analisis untuk merefleksikan penghapusan state
else:
    st.warning("Tidak ada data artikel yang tersedia untuk analisis. Silakan kembali ke halaman utama untuk melakukan screening terlebih dahulu.")
    st.info("Gunakan sidebar di kiri untuk navigasi.")
