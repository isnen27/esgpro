# pages/02_Analysis.py

import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import networkx as nx
from pyvis.network import Network
import os
import re
import torch

# --- Page Configuration ---
st.set_page_config(
    page_title="Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("Analisis Konten Artikel")
st.write("Halaman ini menampilkan ringkasan teks, entitas NER (PER, ORG), dan knowledge graph dari konten yang di-crawl.")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_summarizer_model():
    """Loads the summarization model for Indonesian."""
    with st.spinner("Memuat model summarization... (Ini mungkin membutuhkan waktu saat pertama kali)"):
        # Menggunakan model T5 yang di-fine-tune untuk ringkasan bahasa Indonesia
        model_name = "Wikidata/indonesian-t5-base-summarization"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="./model_cache")
        # Device 0 untuk GPU jika tersedia, -1 untuk CPU
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return summarizer

@st.cache_resource
def load_ner_model():
    """Loads the NER model for Indonesian."""
    with st.spinner("Memuat model NER... (Ini mungkin membutuhkan waktu saat pertama kali)"):
        # Menggunakan model NER bahasa Indonesia
        model_name = "flax-community/indonesian-ner"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model_cache")
        model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir="./model_cache")
        # aggregation_strategy="simple" menggabungkan token yang berdekatan dengan label yang sama
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    return ner_pipeline

# --- Summarization Function ---
@st.cache_data
def get_summary(text, summarizer_pipeline, max_length=150, min_length=30):
    """Generates a summary of the given text."""
    if not text or len(text.split()) < 50: # Hanya ringkas jika teks cukup panjang
        return "Teks terlalu pendek untuk diringkas."
    try:
        # Batasi panjang input untuk menghindari masalah memori/waktu proses yang terlalu lama
        # Model T5 biasanya memiliki batas input sekitar 512 token. Estimasi kasar: 1 token ~ 4 karakter.
        input_text = text[:4000] 
        summary = summarizer_pipeline(
            input_text, 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=False # Untuk output yang lebih deterministik
        )[0]['summary_text']
        return summary
    except Exception as e:
        st.error(f"Error saat membuat ringkasan: {e}")
        return "Gagal membuat ringkasan."

# --- NER Function ---
@st.cache_data
def get_ner_entities(text, ner_pipeline):
    """Extracts PER and ORG entities from text."""
    if not text:
        return {'PER': [], 'ORG': []}
    
    # Batasi panjang input untuk NER
    input_text = text[:2000] # Estimasi kasar untuk 512 token
    
    entities = ner_pipeline(input_text)
    
    per_entities = []
    org_entities = []
    
    for entity in entities:
        # Filter hanya untuk entitas PER (Person) dan ORG (Organization)
        if entity['entity_group'] == 'PER':
            per_entities.append(entity['word'])
        elif entity['entity_group'] == 'ORG':
            org_entities.append(entity['word'])
            
    # Kembalikan entitas unik dan terurut
    return {'PER': sorted(list(set(per_entities))), 'ORG': sorted(list(set(org_entities)))}

# --- Knowledge Graph Generation Function ---
@st.cache_data
def generate_knowledge_graph(text, ner_results):
    """
    Generates a simple knowledge graph based on co-occurrence of PER and ORG entities within sentences.
    """
    if not text or (not ner_results['PER'] and not ner_results['ORG']):
        return "Tidak cukup entitas atau teks untuk membuat knowledge graph."

    G = nx.Graph() # Buat objek graf NetworkX
    all_entities = ner_results['PER'] + ner_results['ORG']
    
    # Tambahkan node ke graf
    for entity in ner_results['PER']:
        G.add_node(entity, label=entity, group='PER', title=f"Orang: {entity}")
    for entity in ner_results['ORG']:
        G.add_node(entity, label=entity, group='ORG', title=f"Organisasi: {entity}")

    # Pisahkan teks menjadi kalimat (pendekatan sederhana)
    sentences = re.split(r'[.!?]\s*', text)

    # Tambahkan edge berdasarkan kemunculan bersama dalam kalimat
    for sentence in sentences:
        found_entities_in_sentence = []
        for entity in all_entities:
            # Gunakan regex untuk menemukan kecocokan kata utuh
            if re.search(r'\b' + re.escape(entity) + r'\b', sentence, re.IGNORECASE):
                found_entities_in_sentence.append(entity)
        
        # Buat edge antara semua pasangan entitas yang ditemukan dalam kalimat yang sama
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

    # Konversi ke Pyvis network untuk visualisasi interaktif
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.toggle_physics(True) # Aktifkan fisika untuk tata letak yang lebih baik

    for node in G.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        net.add_node(node_id, 
                     label=node_data.get('label', node_id), 
                     group=node_data.get('group', 'default'),
                     title=node_data.get('title', node_id),
                     color='#FF5733' if node_data.get('group') == 'PER' else '#33FF57' if node_data.get('group') == 'ORG' else '#3366FF',
                     size=15 + G.degree(node_id) * 2 # Ukuran node berdasarkan derajat (jumlah koneksi)
                    )

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1], 
                     value=edge[2].get('weight', 1), 
                     title=edge[2].get('title', 'Muncul bersama'))

    # Simpan network ke file HTML dan tampilkan
    path = 'html_files'
    if not os.path.exists(path):
        os.makedirs(path)
    net.save_graph(f'{path}/knowledge_graph.html')
    
    # Baca konten file HTML
    with open(f'{path}/knowledge_graph.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    return html_content

# --- Main Application Logic ---
# Periksa apakah konten sudah di-crawl dari halaman sebelumnya
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

    # --- Ringkasan Teks ---
    st.markdown("---")
    st.subheader("Ringkasan Teks")
    summarizer_pipeline = load_summarizer_model()
    summary = get_summary(crawled_content, summarizer_pipeline)
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
        # Tampilkan graf Pyvis menggunakan st.components.v1.html
        st.components.v1.html(kg_html, height=800)
    else:
        st.error("Gagal membuat knowledge graph.")
