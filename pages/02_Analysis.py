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

# --- Helper Function: Simple Sentence Splitter ---
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
    """Loads the NER model for Indonesian (Cahya)."""
    device = 0 if torch.cuda.is_available() else -1
    with st.spinner("Memuat model NER..."):
        model_name = "cahya/bert-base-indonesian-ner"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
            return ner_pipeline
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

# --- Frequency-Based Summarization Function ---
@st.cache_data
def get_summary_frequency(text, num_sentences=5):
    """Meringkas teks berdasarkan frekuensi kata."""
    if not text: return "Tidak ada teks."
    sentences = simple_sent_tokenize(text)
    if len(sentences) <= num_sentences: return text

    stopwords_id = {'yang', 'di', 'dan', 'itu', 'dengan', 'untuk', 'tidak', 'ini', 'dari', 'dalam', 'akan', 'pada', 'juga', 'saya', 'ke', 'karena', 'tersebut', 'bisa', 'ada', 'mereka', 'kata', 'adalah', 'atau', 'saat', 'oleh', 'sudah', 'telah', 'namun', 'tetapi', 'sebagai', 'dia', 'ia', 'bahwa'}
    words = simple_word_tokenize(text)
    clean_words = [w for w in words if w not in stopwords_id and len(w) > 2]
    word_freq = Counter(clean_words)
    if not word_freq: return text[:500] + "..."

    max_freq = max(word_freq.values())
    for word in word_freq: word_freq[word] = word_freq[word] / max_freq

    sentence_scores = {}
    for sent in sentences:
        words_in_sent = simple_word_tokenize(sent)
        if 5 < len(words_in_sent) < 100:
            for word in words_in_sent:
                if word in word_freq:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    import heapq
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return " ".join(sorted(summary_sentences, key=lambda s: sentences.index(s)))

# --- NER Function ---
@st.cache_data
def get_ner_entities(text, _ner_pipeline):
    """Extracts PER and ORG entities."""
    if not text or _ner_pipeline is None: return {'PER': [], 'ORG': []}
    input_text = text[:1500]
    try:
        entities = _ner_pipeline(input_text)
    except Exception: return {'PER': [], 'ORG': []}
    
    per_entities, org_entities = [], []
    for entity in entities:
        word = entity['word'].replace('##', '')
        group = entity.get('entity_group', entity.get('entity', ''))
        if 'PER' in group and len(word) > 2: per_entities.append(word)
        elif 'ORG' in group and len(word) > 2: org_entities.append(word)
            
    return {'PER': sorted(list(set(per_entities))), 'ORG': sorted(list(set(org_entities)))}

# --- Knowledge Graph Generation Function (UPDATED) ---
@st.cache_data
def generate_knowledge_graph(text, ner_results):
    """Generates a knowledge graph with white background and visible edge labels."""
    if not text or (not ner_results['PER'] and not ner_results['ORG']):
        return None

    G = nx.Graph()
    
    # Add Nodes
    for entity in ner_results['PER']:
        G.add_node(entity, label=entity, group='PER', title=f"Orang: {entity}")
    for entity in ner_results['ORG']:
        G.add_node(entity, label=entity, group='ORG', title=f"Organisasi: {entity}")

    sentences = simple_sent_tokenize(text)
    all_entities = ner_results['PER'] + ner_results['ORG']

    # Create Edges based on co-occurrence
    for sentence in sentences:
        found_entities = []
        sent_lower = sentence.lower() # Case insensitive check
        for entity in all_entities:
            if entity.lower() in sent_lower:
                found_entities.append(entity)
        
        for i in range(len(found_entities)):
            for j in range(i + 1, len(found_entities)):
                node1 = found_entities[i]
                node2 = found_entities[j]
                if node1 != node2:
                    # UPDATE 1: Menambahkan Label pada Edge NetworkX
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                        # Update label dengan jumlah baru
                        G[node1][node2]['label'] = f"Bersama ({G[node1][node2]['weight']})"
                    else:
                        # Label awal
                        G.add_edge(node1, node2, weight=1, label="Bersama (1)")

    if not G.nodes:
        return None

    # UPDATE 2: Background Putih & Font Hitam
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Opsi tambahan agar tampilan di background putih lebih rapi
    net.set_options("""
    var options = {
      "edges": {
        "color": {"inherit": true},
        "smooth": false,
        "font": {"size": 10, "align": "middle"} 
      },
      "physics": {
        "barnesHut": {"gravitationalConstant": -2000, "centralGravity": 0.3, "springLength": 95},
        "minVelocity": 0.75
      }
    }
    """)
    
    for node in G.nodes(data=True):
        # Menggunakan warna yang lebih gelap agar kontras di background putih
        color = '#D32F2F' if node[1].get('group') == 'PER' else '#388E3C' # Merah Tua & Hijau Tua
        net.add_node(node[0], label=node[0], title=node[1].get('title'), color=color)
        
    for edge in G.edges(data=True):
        # UPDATE 3: Meneruskan label ke PyVis
        net.add_edge(
            edge[0], 
            edge[1], 
            value=edge[2]['weight'], 
            label=edge[2]['label'] # Ini yang memunculkan teks di garis
        )

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
            st.write("Belum cukup data hubungan untuk graph.")
