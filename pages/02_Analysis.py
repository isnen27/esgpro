import streamlit as st
import re
import heapq
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import nltk
import os

# =========================================================
# SETUP
# =========================================================
st.set_page_config(layout="wide", page_title="ESG Analysis")

@st.cache_resource
def setup_nltk():
    path = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(path, exist_ok=True)
    nltk.download("stopwords", download_dir=path, quiet=True)

setup_nltk()

# =========================================================
# TF-IDF SUMMARY (UNCHANGED)
# =========================================================
def summarize(text, n=5):
    sents = re.split(r'(?<=[.!?])\s+', text)
    if len(sents) <= n:
        return text

    sw = set(stopwords.words("indonesian"))

    def tok(s):
        return [w for w in re.findall(r'\b\w+\b', s.lower()) if w not in sw]

    vec = TfidfVectorizer(tokenizer=tok)
    mat = vec.fit_transform(sents)

    scores = {s: mat[i].sum() for i, s in enumerate(sents)}
    return " ".join(heapq.nlargest(n, scores, key=scores.get))

# =========================================================
# NER LITE
# =========================================================
@st.cache_resource
def load_ner():
    return pipeline(
        "ner",
        model="cahya/indobert-lite-ner",
        aggregation_strategy="simple",
        device=-1
    )

def ner_entities(text):
    ner = load_ner()
    text = text[:2000]  # HARD LIMIT
    return [(e["word"], e["entity_group"]) for e in ner(text)]

# =========================================================
# KNOWLEDGE GRAPH
# =========================================================
def show_graph(entities):
    entities = list(dict.fromkeys(entities))[:20]

    net = Network(height="600px", bgcolor="#fff")
    for e, t in entities:
        net.add_node(e, label=e, title=t)

    for i in range(len(entities)-1):
        net.add_edge(entities[i][0], entities[i+1][0])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        components.html(open(f.name).read(), height=650)

# =========================================================
# MAIN
# =========================================================
if st.session_state.crawled_data:
    text = st.session_state.crawled_data["crawled_content"]

    st.subheader("Ringkasan")
    st.write(summarize(text))

    st.subheader("NER")
    ents = ner_entities(text)
    st.dataframe(pd.DataFrame(ents, columns=["Entity", "Type"]))

    st.subheader("Knowledge Graph")
    if ents:
        show_graph(ents)
