# pages/02_Analysis.py

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
    MENGGUNAKAN MODEL 'cahya/bert-base-indonesian-ner' YANG LEBIH STABIL.
    """
    # Cek apakah GPU tersedia
    device = 0 if torch.cuda.is_available() else -1
    
    with st.spinner("Memuat model NER... (Mohon tunggu, proses ini cukup berat)"):
        # MODEL DIGANTI KE YANG LEBIH STABIL
        model_name = "cahya/bert-base-indonesian-ner"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # aggregation_strategy="simple" sangat penting untuk menggabungkan B-PER dan I-PER menjadi satu entitas
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
            return ner_pipeline
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

# --- Frequency-Based Summarization Function ---
@st.cache_data
def get_summary_frequency(text, num_sentences=5):
    """
    Meringkas teks berdasarkan frekuensi kata (Metode Ringan tanpa Download NLTK).
    """
    if not text:
        return "Tidak ada teks untuk diringkas."
    
    # 1. Tokenisasi Kalimat
    sentences = simple_sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    # 2. Hitung Frekuensi Kata
    stopwords_id = {
        'yang', 'di', 'dan', 'itu', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 
        'dalam', 'akan', 'pada', 'juga', 'saya', 'ke', 'karena', 'tersebut', 
        'bisa', 'ada', 'mereka', 'kata', 'adalah', 'atau', 'saat', 'oleh', 
        'sudah', 'telah', 'namun', 'tetapi', 'sebagai', 'dia', 'ia', 'bahwa'
    }
    
    words = simple_word_tokenize(text)
    clean_words = [w for w in words if w not in stopwords_id and len(w) > 2]
    word_freq = Counter(clean_words)
    
    if not word_freq:
        return text[:500] + "..."

    max_freq = max(word_freq.values())
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    # 3. Beri Skor pada Kalimat
    sentence_scores = {}
    for sent in sentences:
        words_in_sent = simple_word_tokenize(sent)
        word_count_in_sent = len(words_in_sent)
        
        if 5 < word_count_in_sent < 100: # Filter kalimat terlalu pendek/panjang
            for word in words_in_sent:
                if word in word_freq:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_freq[word]
                    else:
                        sentence_scores[sent] += word_freq[word]

    # 4. Ambil N kalimat terbaik
    import heapq
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary_sentences_sorted = sorted(summary_sentences, key=lambda s: sentences.index(s))

    return " ".join(summary_sentences_sorted)

# --- NER Function ---
@st.cache_data
def get_ner_entities(text, _ner_pipeline):
    """Extracts PER and ORG entities from text."""
    if not text or _ner_pipeline is None:
        return {'PER': [], 'ORG': []}
    
    # Batasi input text agar memori aman
    input_text = text[:1500] 
    
    try:
        entities = _ner_pipeline(input_text)
    except Exception as e:
        st.warning(f"Error saat ekstraksi NER: {e}")
        return {'PER': [], 'ORG': []}
    
    per_entities = []
    org_entities = []
    
    for entity in entities:
        word = entity['word'].replace('##', '') # Bersihkan token BPE
        # Model Cahya menggunakan label 'PER' dan 'ORG' (kadang case sensitive tergantung mapping)
        # aggregation_strategy="simple" biasanya mengembalikan entity_group 'PER', 'ORG', 'LOC'
        
        group = entity.get('entity_group', entity.get('entity', ''))
        
        if group == 'PER' or group == 'B-PER' or group == 'I-PER':
            if len(word) > 2: per_entities.append(word)
        elif group == 'ORG' or group == 'B-ORG' or group == 'I-ORG':
            if len(word) > 2: org_entities.append(word)
            
    return {'PER': sorted(list(set(per_entities))), 'ORG': sorted(list(set(org_entities)))}

# --- Knowledge Graph Generation Function ---
@st.cache_data
def generate_knowledge_graph(text, ner_results):
    """Generates a simple knowledge graph."""
    if not text or (not ner_results['PER'] and not ner_results['ORG']):
        return None

    G = nx.Graph()
    
    for entity in ner_results['PER']:
        G.add_node(entity, label=entity, group='PER', title=f"Orang: {entity}")
    for entity in ner_results['ORG']:
        G.add_node(entity, label=entity, group='ORG', title=f"Organisasi: {entity}")

    sentences = simple_sent_tokenize(text)
    all_entities = ner_results['PER'] + ner_results['ORG']

    for sentence in sentences:
        found_entities = []
        for entity in all_entities:
            if entity in sentence: # Case sensitive check sederhana
                found_entities.append(entity)
        
        for i in range(len(found_entities)):
            for j in range(i + 1, len(found_entities)):
                node1 = found_entities[i]
                node2 = found_entities[j]
                if node1 != node2:
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                    else:
                        G.add_edge(node1, node2, weight=1)

    if not G.nodes:
        return None

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    for node in G.nodes(data=True):
        color = '#FF5733' if node[1].get('group') == 'PER' else '#33FF57'
        net.add_node(node[0], label=node[0], title=node[1].get('title'), color=color)
        
    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], value=edge[2]['weight'])

    # Simpan di /tmp agar aman di Cloud
    try:
        path = '/tmp'
        if not os.path.exists(path): path = '.' 
        output_path = f'{path}/kg.html'
        net.save_graph(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None

# --- Main Application Logic ---
if 'crawled_content' not in st.session_state or not st.session_state.crawled_content:
    st.warning("⚠️ Belum ada data artikel. Silakan Crawling dulu.")
else:
    crawled_content = st.session_state.crawled_content
    crawled_title = st.session_state.get('crawled_title', 'Tanpa Judul')

    st.subheader(f"Analisis: {crawled_title}")
    
    with st.expander("📝 Lihat Teks Asli"):
        st.write(crawled_content)

    st.markdown("### 📑 Ringkasan Teks")
    summary = get_summary_frequency(crawled_content)
    st.success(summary)

    st.markdown("### 🔍 Deteksi Entitas (NER)")
    ner_pipeline = load_ner_model()
    
    if ner_pipeline:
        ner_entities = get_ner_entities(crawled_content, ner_pipeline)
        
        c1, c2 = st.columns(2)
        c1.info(f"Orang: {len(ner_entities['PER'])}")
        c1.write(ner_entities['PER'])
        c2.info(f"Organisasi: {len(ner_entities['ORG'])}")
        c2.write(ner_entities['ORG'])

        st.markdown("### 🕸️ Knowledge Graph")
        kg_html = generate_knowledge_graph(crawled_content, ner_entities)
        if kg_html:
            components.html(kg_html, height=600, scrolling=True)
        else:
            st.write("Belum cukup data untuk graph.")
