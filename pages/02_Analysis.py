import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import networkx as nx
from pyvis.network import Network
import os
import re
import torch
from collections import Counter
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Analisis Konten Artikel")
st.write("Halaman ini menampilkan ringkasan teks (Frequency Based), entitas NER (PER, ORG), dan knowledge graph.")

# --- Helper Function: Simple Sentence Splitter (Tanpa NLTK) ---
def simple_sent_tokenize(text):
    """Memecah teks menjadi kalimat menggunakan RegEx sederhana."""
    # Memisahkan berdasarkan titik, tanda tanya, atau tanda seru yang diikuti spasi/akhir baris
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def simple_word_tokenize(text):
    """Membersihkan dan memecah kalimat menjadi kata."""
    return re.findall(r'\w+', text.lower())

# --- Model Loading (Cached) ---
@st.cache_resource
def load_ner_model():
    """
    Loads the NER model for Indonesian.
    Menggunakan cache_resource agar model hanya diload sekali di memori server.
    """
    # Cek apakah GPU tersedia
    device = 0 if torch.cuda.is_available() else -1
    
    with st.spinner("Memuat model NER... (Mohon tunggu, proses ini cukup berat)"):
        model_name = "flax-community/indonesian-ner"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
            return ner_pipeline
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

# --- Frequency-Based Summarization Function (Pengganti TF-IDF) ---
@st.cache_data
def get_summary_frequency(text, num_sentences=5):
    """
    Meringkas teks berdasarkan frekuensi kata.
    Kalimat yang mengandung kata-kata yang paling sering muncul akan diberi skor tinggi.
    """
    if not text:
        return "Tidak ada teks untuk diringkas."
    
    # 1. Tokenisasi Kalimat
    sentences = simple_sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    # 2. Hitung Frekuensi Kata (Stopwords manual sederhana untuk Bahasa Indonesia)
    stopwords_id = {
        'yang', 'di', 'dan', 'itu', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 
        'dalam', 'akan', 'pada', 'juga', 'saya', 'ke', 'karena', 'tersebut', 
        'bisa', 'ada', 'mereka', 'kata', 'adalah', 'atau', 'saat', 'oleh', 
        'sudah', 'telah', 'namun', 'tetapi', 'sebagai', 'dia', 'ia'
    }
    
    words = simple_word_tokenize(text)
    # Filter stopwords dan kata pendek
    clean_words = [w for w in words if w not in stopwords_id and len(w) > 2]
    word_freq = Counter(clean_words)
    
    if not word_freq:
        return text[:500] + "..." # Fallback jika tidak ada kata signifikan

    # Normalisasi frekuensi (max freq = 1)
    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    # 3. Beri Skor pada Kalimat
    sentence_scores = {}
    for sent in sentences:
        words_in_sent = simple_word_tokenize(sent)
        word_count_in_sent = len(words_in_sent)
        
        # Hindari kalimat terlalu pendek atau terlalu panjang
        if 10 < word_count_in_sent < 100:
            for word in words_in_sent:
                if word in word_freq:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_freq[word]
                    else:
                        sentence_scores[sent] += word_freq[word]

    # 4. Ambil N kalimat dengan skor tertinggi
    import heapq
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Urutkan kembali kalimat sesuai urutan kemunculan di teks asli agar alurnya pas
    summary_sentences_sorted = sorted(summary_sentences, key=lambda s: sentences.index(s))

    return " ".join(summary_sentences_sorted)

# --- NER Function ---
@st.cache_data
def get_ner_entities(text, _ner_pipeline):
    """Extracts PER and ORG entities from text."""
    if not text or _ner_pipeline is None:
        return {'PER': [], 'ORG': []}
    
    # Batasi input text agar tidak OOM (Out of Memory)
    input_text = text[:1500] 
    
    try:
        entities = _ner_pipeline(input_text)
    except Exception as e:
        return {'PER': [], 'ORG': []}
    
    per_entities = []
    org_entities = []
    
    for entity in entities:
        # Membersihkan spasi aneh dari tokenizer (misal ##word)
        word = entity['word'].replace('##', '')
        if entity['entity_group'] == 'PER':
            if len(word) > 2: per_entities.append(word)
        elif entity['entity_group'] == 'ORG':
            if len(word) > 2: org_entities.append(word)
            
    # Hapus duplikat dan urutkan
    return {'PER': sorted(list(set(per_entities))), 'ORG': sorted(list(set(org_entities)))}

# --- Knowledge Graph Generation Function ---
@st.cache_data
def generate_knowledge_graph(text, ner_results):
    """
    Generates a simple knowledge graph.
    """
    if not text or (not ner_results['PER'] and not ner_results['ORG']):
        return None

    G = nx.Graph()
    
    # Tambahkan Node
    for entity in ner_results['PER']:
        G.add_node(entity, label=entity, group='PER', title=f"Orang: {entity}")
    for entity in ner_results['ORG']:
        G.add_node(entity, label=entity, group='ORG', title=f"Organisasi: {entity}")

    sentences = simple_sent_tokenize(text)
    all_entities = ner_results['PER'] + ner_results['ORG']

    # Buat Edge berdasarkan kemunculan dalam satu kalimat
    for sentence in sentences:
        found_entities = []
        for entity in all_entities:
            # Cek eksistensi entity (case insensitive)
            if entity.lower() in sentence.lower():
                found_entities.append(entity)
        
        # Hubungkan entity yang muncul bersamaan
        for i in range(len(found_entities)):
            for j in range(i + 1, len(found_entities)):
                node1 = found_entities[i]
                node2 = found_entities[j]
                
                if node1 != node2:
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                        G[node1][node2]['title'] = f"Koneksi: {G[node1][node2]['weight']}"
                    else:
                        G.add_edge(node1, node2, weight=1, title="Koneksi: 1")

    if not G.nodes:
        return None

    # Visualisasi PyVis
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Konversi NetworkX ke PyVis
    for node in G.nodes(data=True):
        n_id = node[0]
        n_data = node[1]
        color = '#FF5733' if n_data.get('group') == 'PER' else '#33FF57'
        net.add_node(n_id, label=n_id, title=n_data.get('title'), color=color)
        
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]['weight'])

    # Simpan ke HTML string
    # Menggunakan opsi write_html untuk menghindari pembuatan file temporary yang kadang bermasalah di cloud
    try:
        # Trik untuk mendapatkan string HTML tanpa menyimpan file fisik jika memungkinkan
        # Atau simpan ke direktori /tmp yang aman di cloud
        path = '/tmp'
        if not os.path.exists(path):
            path = '.' # Fallback ke current dir
        
        output_path = f'{path}/kg.html'
        net.save_graph(output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except Exception as e:
        return f"<div>Error generating graph: {str(e)}</div>"

# --- Main Application Logic ---
if 'crawled_content' not in st.session_state or not st.session_state.crawled_content:
    st.warning("⚠️ Belum ada data artikel.")
    st.info("Silakan kembali ke halaman **Crawling** untuk mengambil artikel terlebih dahulu.")
else:
    crawled_content = st.session_state.crawled_content
    crawled_title = st.session_state.get('crawled_title', 'Tanpa Judul')
    crawled_url = st.session_state.get('crawled_url', '-')
    crawled_date = st.session_state.get('crawled_date', '-')

    st.subheader(f"Analisis: {crawled_title}")
    st.caption(f"Source: {crawled_url} | {crawled_date}")

    with st.expander("📝 Lihat Teks Asli"):
        st.write(crawled_content)

    # --- Ringkasan Teks (Frequency Based) ---
    st.markdown("### 📑 Ringkasan Teks")
    num_sentences = st.slider("Jumlah kalimat ringkasan:", 1, 10, 3)
    
    summary = get_summary_frequency(crawled_content, num_sentences=num_sentences)
    st.success(summary)

    # --- Named Entity Recognition (NER) ---
    st.markdown("### 🔍 Deteksi Entitas (NER)")
    
    ner_pipeline = load_ner_model()
    
    if ner_pipeline:
        ner_entities = get_ner_entities(crawled_content, ner_pipeline)

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Orang (Person):** {len(ner_entities['PER'])}")
            st.write(", ".join(ner_entities['PER']) if ner_entities['PER'] else "-")
        with col2:
            st.info(f"**Organisasi (Org):** {len(ner_entities['ORG'])}")
            st.write(", ".join(ner_entities['ORG']) if ner_entities['ORG'] else "-")

        # --- Knowledge Graph ---
        st.markdown("### 🕸️ Knowledge Graph")
        kg_html = generate_knowledge_graph(crawled_content, ner_entities)
        
        if kg_html and "<div" in kg_html: # Cek jika error string
             st.warning("Gagal membuat graph visual.")
        elif kg_html:
            components.html(kg_html, height=600, scrolling=True)
        else:
            st.write("Tidak cukup hubungan antar entitas untuk membuat graph.")
    else:
        st.error("Model NER gagal dimuat. Coba refresh halaman.")
