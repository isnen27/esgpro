import streamlit as st
import re
import heapq
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import torch
import gc

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ESG Analysis",
    layout="wide"
)

torch.set_grad_enabled(False)

# =========================================================
# STOPWORDS INDONESIA (STATIC – CLOUD SAFE)
# =========================================================
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","pada","untuk","dengan","adalah",
    "sebagai","oleh","karena","atau","dalam","bahwa","akan","juga","dapat",
    "tidak","telah","lebih","saat","antara","hingga","agar","namun","sehingga",
    "tersebut","para","yakni","seperti","masih","harus","bagi","mereka","kami"
}

# =========================================================
# TF-IDF TEXT SUMMARIZATION (LOGIKA ASLI DIPERTAHANKAN)
# =========================================================
def summarize(text, n=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= n:
        return text

    def tokenizer(sentence):
        return [
            w for w in re.findall(r'\b\w+\b', sentence.lower())
            if w not in STOPWORDS_ID
        ]

    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = {
        sent: tfidf_matrix[idx].sum()
        for idx, sent in enumerate(sentences)
    }

    top_sentences = heapq.nlargest(n, sentence_scores, key=sentence_scores.get)
    return " ".join(top_sentences)

# =========================================================
# NER TRANSFORMER (LITE – STABLE)
# =========================================================
@st.cache_resource
def load_ner_model():
    return pipeline(
        "ner",
        model="cahya/indobert-lite-ner",
        aggregation_strategy="simple",
        device=-1
    )

def extract_entities(text, max_chars=2000):
    ner = load_ner_model()
    text = text[:max_chars]

    try:
        entities = ner(text)
        return [(e["word"], e["entity_group"]) for e in entities]
    except Exception:
        return []

# =========================================================
# KNOWLEDGE GRAPH (LIMITED – MEMORY SAFE)
# =========================================================
def render_knowledge_graph(entities, max_nodes=20):
    unique_entities = list(dict.fromkeys(entities))[:max_nodes]

    net = Network(
        height="600px",
        bgcolor="#ffffff",
        font_color="black"
    )

    for entity, label in unique_entities:
        net.add_node(entity, label=entity, title=label)

    for i in range(len(unique_entities) - 1):
        net.add_edge(
            unique_entities[i][0],
            unique_entities[i + 1][0]
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        components.html(
            open(tmp.name, "r", encoding="utf-8").read(),
            height=650,
            scrolling=True
        )

# =========================================================
# MAIN UI LOGIC
# =========================================================
st.title("ESG Content Analysis")

if "crawled_data" not in st.session_state or not st.session_state.crawled_data:
    st.info("Silakan lakukan crawling terlebih dahulu pada halaman sebelumnya.")
    st.stop()

content = st.session_state.crawled_data.get("crawled_content", "")

if not content.strip():
    st.warning("Konten artikel kosong atau tidak tersedia.")
    st.stop()

# -----------------------------
# SUMMARY
# -----------------------------
st.subheader("Ringkasan Artikel (TF-IDF)")
summary = summarize(content)
st.write(summary)

# -----------------------------
# NER
# -----------------------------
st.subheader("Named Entity Recognition (NER)")
entities = extract_entities(content)

if entities:
    df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
    st.dataframe(df_entities, use_container_width=True)
else:
    st.info("Tidak ditemukan entitas atau terjadi kegagalan ekstraksi.")

# -----------------------------
# KNOWLEDGE GRAPH
# -----------------------------
st.subheader("Knowledge Graph")
if entities:
    render_knowledge_graph(entities)
else:
    st.info("Knowledge Graph tidak dapat dibuat karena entitas kosong.")

# =========================================================
# CLEANUP (PREVENT MEMORY LEAK)
# =========================================================
gc.collect()
