# pages/02_Analysis.py

import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import networkx as nx
from pyvis.network import Network
import os
import re
import torch

# Untuk TF-IDF Summarization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Analysis",
    page_icon="📊",
    layout="wide",
)

# --- NLTK Data Download (Programmatic and Cached) ---
# Definisikan jalur kustom untuk data NLTK di dalam direktori aplikasi
NLTK_DATA_PATH = "./nltk_data"
# Tambahkan jalur ini ke jalur pencarian data NLTK
nltk.data.path.append(NLTK_DATA_PATH)

@st.cache_resource
def load_nltk_data():
    """
    Downloads NLTK data (punkt and stopwords) to a specified directory
    if not already present. This function is cached to run only once.
    """
    # Pastikan direktori data NLTK kustom ada
    os.makedirs(NLTK_DATA_PATH, exist_ok=True)

    with st.spinner(f"Memastikan data NLTK tersedia di {NLTK_DATA_PATH}..."):
        # Periksa dan unduh 'punkt'
        try:
            # Secara eksplisit periksa di jalur kustom kita
            nltk.data.find('tokenizers/punkt', path=[NLTK_DATA_PATH]) 
            st.success("NLTK 'punkt' data sudah ada.")
        except LookupError:
            st.warning("NLTK 'punkt' data tidak ditemukan. Mengunduh...")
            try:
                nltk.download('punkt', download_dir=NLTK_DATA_PATH)
                st.success("NLTK 'punkt' data berhasil diunduh.")
            except Exception as e:
                st.error(f"Gagal mengunduh NLTK 'punkt' data: {e}")
                st.stop()
        
        # Periksa dan unduh 'stopwords'
        try:
            # Secara eksplisit periksa di jalur kustom kita
            nltk.data.find('corpora/stopwords', path=[NLTK_DATA_PATH]) 
            st.success("NLTK 'stopwords' data sudah ada.")
        except LookupError:
            st.warning("NLTK 'stopwords' data tidak ditemukan. Mengunduh...")
            try:
                nltk.download('stopwords', download_dir=NLTK_DATA_PATH)
                st.success("NLTK 'stopwords' data berhasil diunduh.")
            except Exception as e:
                st.error(f"Gagal mengunduh NLTK 'stopwords' data: {e}")
                st.stop()
    st.success("Semua data NLTK siap!")

# Panggil fungsi pengunduhan NLTK di awal aplikasi
load_nltk_data()


st.title("Analisis Konten Artikel")
st.write("Halaman ini menampilkan ringkasan teks (TF-IDF), entitas NER (PER, ORG), dan knowledge graph dari konten yang di-crawl.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_ner_model():
    """Loads the NER model for Indonesian."""
    with st.spinner("Memuat model NER... (Ini mungkin membutuhkan waktu saat pertama kali)"):
        model_name = "flax-community/indonesian-ner"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
        model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir="./model_cache")
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    return ner_pipeline

# --- TF-IDF Summarization Function ---
@st.cache_data
def get_summary_tfidf(text, num_sentences=5):
    """Generates a summary of the given text using TF-IDF."""
    if not text or len(text.split()) < 50:
        return "Teks terlalu pendek untuk diringkas menggunakan TF-IDF."

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text # Jika jumlah kalimat kurang dari atau sama dengan yang diminta, kembalikan teks asli

    # Inisialisasi TF-IDF Vectorizer
    stop_words_id = set(stopwords.words('indonesian'))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words_id))

    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        feature_index = [vectorizer.vocabulary_.get(word) for word in word_tokenize(sentence.lower()) if word.isalpha() and word not in stop_words_id]
        feature_index = [idx for idx in feature_index if idx is not None]
        
        if feature_index:
            score = tfidf_matrix[i, feature_index].sum() / len(feature_index)
            sentence_scores[sentence] = score
        else:
            sentence_scores[sentence] = 0

    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    
    summary_sentences = []
    original_sentence_order = {sentence: i for i, sentence in enumerate(sentences)}
    
    top_sentences = [s[0] for s in ranked_sentences[:num_sentences]]
    
    summary_sentences = sorted(top_sentences, key=lambda s: original_sentence_order.get(s, len(sentences)))

    return " ".join(summary_sentences)


# --- NER Function ---
@st.cache_data
def get_ner_entities(text, ner_pipeline):
    """Extracts PER and ORG entities from text."""
    if not text:
        return {'PER': [], 'ORG': []}
    
    input_text = text[:2000] 
    
    entities = ner_pipeline(input_text)
    
    per_entities = []
    org_entities = []
    
    for entity in entities:
        if entity['entity_group'] == 'PER':
            per_entities.append(entity['word'])
        elif entity['entity_group'] == 'ORG':
            org_entities.append(entity['word'])
            
    return {'PER': sorted(list(set(per_entities))), 'ORG': sorted(list(set(org_entities)))}

# --- Knowledge Graph Generation Function ---
@st.cache_data
def generate_knowledge_graph(text, ner_results):
    """
    Generates a simple knowledge graph based on co-occurrence of PER and ORG entities within sentences.
    """
    if not text or (not ner_results['PER'] and not ner_results['ORG']):
        return "Tidak cukup entitas atau teks untuk membuat knowledge graph."

    G = nx.Graph()
    all_entities = ner_results['PER'] + ner_results['ORG']
    
    for entity in ner_results['PER']:
        G.add_node(entity, label=entity, group='PER', title=f"Orang: {entity}")
    for entity in ner_results['ORG']:
        G.add_node(entity, label=entity, group='ORG', title=f"Organisasi: {entity}")

    sentences = sent_tokenize(text)

    for sentence in sentences:
        found_entities_in_sentence = []
        for entity in all_entities:
            if re.search(r'\b' + re.escape(entity) + r'\b', sentence, re.IGNORECASE):
                found_entities_in_sentence.append(entity)
        
        for i in range(len(found_entities_in_sentence)):
            for j in range(i + 1, len(found_entities_in_sentence)):
                node1 = found_entities_in_sentence[i]
                node2 = found_entities_in_sentence[j]
                if G.has_edge(node1, node2):
                    G[node1][node2]['weight'] += 1
                    G[node1][node2]['title'] = f"Muncul bersama {G[node1][node2]['weight']} kali"
                else:
                    G.add_edge(node1, node2, weight=1, title="Muncul bersama 1 kali")

    if not G.nodes:
        return "Tidak ada hubungan yang teridentifikasi untuk membuat knowledge graph."

    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.toggle_physics(True)

    for node in G.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        net.add_node(node_id, 
                     label=node_data.get('label', node_id), 
                     group=node_data.get('group', 'default'),
                     title=node_data.get('title', node_id),
                     color='#FF5733' if node_data.get('group') == 'PER' else '#33FF57' if node_data.get('group') == 'ORG' else '#3366FF',
                     size=15 + G.degree(node_id) * 2
                    )

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], 
                     value=edge[2].get('weight', 1), 
                     title=edge[2].get('title', 'Muncul bersama'))

    path = 'html_files'
    if not os.path.exists(path):
        os.makedirs(path)
    net.save_graph(f'{path}/knowledge_graph.html')
    
    with open(f'{path}/knowledge_graph.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return html_content

# --- Main Application Logic ---
if 'crawled_content' not in st.session_state or not st.session_state.crawled_content:
    st.warning("Silakan kembali ke halaman 'Crawling Content' untuk mengambil artikel terlebih dahulu.")
    st.info("Anda bisa menggunakan URL contoh di halaman Crawling Content, lalu klik 'Mulai Crawling'.")
else:
    crawled_content = st.session_state.crawled_content
    crawled_title = st.session_state.crawled_title
    crawled_date = st.session_state.crawled_date
    crawled_url = st.session_state.crawled_url

    st.subheader(f"Analisis untuk Artikel: {crawled_title}")
    st.caption(f"URL: {crawled_url} | Tanggal: {crawled_date}")

    st.markdown("---")
    st.subheader("Teks Asli (Crawled Content)")
    with st.expander("Lihat Teks Asli", expanded=False):
        st.write(crawled_content)

    # --- Ringkasan Teks (TF-IDF) ---
    st.markdown("---")
    st.subheader("Ringkasan Teks (TF-IDF)")
    num_sentences_summary = st.slider("Jumlah kalimat dalam ringkasan:", min_value=1, max_value=10, value=5)
    summary = get_summary_tfidf(crawled_content, num_sentences=num_sentences_summary)
    st.info(summary)

    # --- Named Entity Recognition (NER) ---
    st.markdown("---")
    st.subheader("Entitas Teridentifikasi (NER)")
    ner_pipeline = load_ner_model()
    ner_entities = get_ner_entities(crawled_content, ner_pipeline)

    if ner_entities['PER'] or ner_entities['ORG']:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Orang (PER):**")
            if ner_entities['PER']:
                for per in ner_entities['PER']:
                    st.write(f"- {per}")
            else:
                st.write("Tidak ada entitas PER yang teridentifikasi.")
        with col2:
            st.write("**Organisasi (ORG):**")
            if ner_entities['ORG']:
                for org in ner_entities['ORG']:
                    st.write(f"- {org}")
            else:
                st.write("Tidak ada entitas ORG yang teridentifikasi.")
    else:
        st.write("Tidak ada entitas PER atau ORG yang teridentifikasi.")

    # --- Knowledge Graph ---
    st.markdown("---")
    st.subheader("Knowledge Graph")
    kg_html = generate_knowledge_graph(crawled_content, ner_entities)
    
    if isinstance(kg_html, str) and ("Tidak cukup entitas" in kg_html or "Tidak ada hubungan" in kg_html):
        st.warning(kg_html)
    elif kg_html:
        st.components.v1.html(kg_html, height=800)
    else:
        st.error("Gagal membuat knowledge graph.")
