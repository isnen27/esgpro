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
        task="ner",
        model="Davlan/bert-base-multilingual-cased-ner-hrl",
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

def extract_sentence_relations(text, entities):
    """
    Membuat relasi antar entitas jika muncul
    dalam kalimat yang sama.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relations = set()

    entity_names = [e[0] for e in entities]

    for sent in sentences:
        present = [e for e in entity_names if e in sent]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                relations.add((present[i], present[j]))

    return list(relations)

# =========================================================
# KNOWLEDGE GRAPH (LIMITED – MEMORY SAFE)
# =========================================================
def render_knowledge_graph(text, entities, max_nodes=20):
    entities = list(dict.fromkeys(entities))[:max_nodes]

    relations = extract_sentence_relations(text, entities)

    net = Network(
        height="600px",
        bgcolor="#ffffff",
        font_color="black"
    )

    # Tambahkan node
    for entity, label in entities:
        net.add_node(
            entity,
            label=entity,
            title=label
        )

    # Tambahkan edge berbasis kalimat
    for src, tgt in relations:
        net.add_edge(
            src,
            tgt,
            label="co-mentioned"
        )

    if not relations:
        # fallback minimal agar graph tidak kosong
        for i in range(len(entities) - 1):
            net.add_edge(entities[i][0], entities[i + 1][0], label="related")

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

if not entities:
    st.info("Tidak ditemukan entitas atau terjadi kegagalan ekstraksi.")
else:
    df = pd.DataFrame(entities, columns=["Entity", "Type"])
    df = df.drop_duplicates()

    ENTITY_LABELS = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Miscellaneous"
    }

    for ent_type, label in ENTITY_LABELS.items():
        subset = df[df["Type"] == ent_type]

        if not subset.empty:
            st.markdown(f"#### {label}")
            st.dataframe(
                subset[["Entity"]],
                use_container_width=True,
                hide_index=True
            )

# -----------------------------
# KNOWLEDGE GRAPH
# -----------------------------
st.subheader("Knowledge Graph")
if entities:
    render_knowledge_graph(content, entities)
else:
    st.info("Knowledge Graph tidak dapat dibuat karena entitas kosong.")

# =========================================================
# CLEANUP (PREVENT MEMORY LEAK)
# =========================================================
gc.collect()
