import streamlit as st
import nltk
import os
import re
import heapq
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="ESG Analysis")
st.title("ESG Analysis Page")
st.markdown("---")

# =========================================================
# NLTK SETUP (STOPWORDS SAJA)
# =========================================================
@st.cache_resource
def download_nltk_stopwords():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.environ["NLTK_DATA"] = nltk_data_dir
    os.makedirs(nltk_data_dir, exist_ok=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", download_dir=nltk_data_dir, quiet=True)

download_nltk_stopwords()

# =========================================================
# SIMPLE SENTENCE TOKENIZER
# =========================================================
def simple_sentence_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]

# =========================================================
# TF-IDF SUMMARIZATION (DIPERTAHANKAN)
# =========================================================
def summarize_text_tfidf(text, num_sentences=5):
    if not text or not text.strip():
        return "Tidak ada konten untuk diringkas."

    sentences = simple_sentence_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    stop_words_id = set(stopwords.words("indonesian"))

    def preprocess(sentence):
        words = re.findall(r'\b\w+\b', sentence.lower())
        return [w for w in words if w.isalnum() and w not in stop_words_id]

    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words=list(stop_words_id))

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return "Tidak dapat meringkas konten."

    sentence_scores = {
        sentence: tfidf_matrix[i].sum()
        for i, sentence in enumerate(sentences)
    }

    best_sentences = heapq.nlargest(
        num_sentences, sentence_scores, key=sentence_scores.get
    )

    return " ".join(best_sentences)

# =========================================================
# TRANSFORMER INDONESIAN NER
# =========================================================
@st.cache_resource
def load_indo_ner_model():
    model_name = "cahya/bert-base-indonesian-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

def perform_ner_transformer(text):
    if not text or not text.strip():
        return []

    ner_pipeline = load_indo_ner_model()
    results = ner_pipeline(text)
    return [(r["word"], r["entity_group"]) for r in results]

# =========================================================
# KNOWLEDGE GRAPH BUILDER
# =========================================================
def build_knowledge_graph(entities):
    G = nx.Graph()

    unique_entities = {}
    for entity, ent_type in entities:
        unique_entities[entity] = ent_type

    for entity, ent_type in unique_entities.items():
        G.add_node(entity, label=entity, group=ent_type)

    entity_list = list(unique_entities.keys())
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            G.add_edge(entity_list[i], entity_list[j])

    return G

def visualize_knowledge_graph(G):
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black"
    )

    net.force_atlas_2based()

    for node, data in G.nodes(data=True):
        net.add_node(
            node,
            label=data.get("label"),
            group=data.get("group"),
            title=data.get("group")
        )

    for source, target in G.edges():
        net.add_edge(source, target)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html_path = tmp_file.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=650, scrolling=True)

# =========================================================
# MAIN PAGE
# =========================================================
if (
    "crawled_data" in st.session_state
    and st.session_state.crawled_data
    and "final_esg_category" in st.session_state
    and st.session_state.final_esg_category
):
    crawled_url = st.session_state.crawled_url
    crawled_title = st.session_state.crawled_data["crawled_title"]
    crawled_date = st.session_state.crawled_data["crawled_date"]
    crawled_content = st.session_state.crawled_data["crawled_content"]
    final_esg_category = st.session_state.final_esg_category

    st.markdown("### Artikel yang Dianalisis")
    st.write(f"**URL:** {crawled_url}")
    st.write(f"**Judul:** {crawled_title}")
    st.write(f"**Tanggal Publikasi:** {crawled_date}")
    st.write(f"**Kategori ESG:** {final_esg_category}")

    with st.expander("Lihat Isi Artikel Lengkap"):
        st.write(crawled_content)

    st.markdown("---")
    st.markdown("### Hasil Analisis")

    # -------------------------
    # SUMMARY
    # -------------------------
    st.subheader("Ringkasan Artikel (TF-IDF)")
    num_sentences_summary = st.slider(
        "Jumlah kalimat ringkasan",
        min_value=1,
        max_value=10,
        value=5
    )
    st.write(
        summarize_text_tfidf(
            crawled_content,
            num_sentences=num_sentences_summary
        )
    )

    # -------------------------
    # NER
    # -------------------------
    st.subheader("Named Entity Recognition (IndoNER)")
    entities = perform_ner_transformer(crawled_content)

    if entities:
        df_entities = pd.DataFrame(entities, columns=["Entity", "Type"])
        st.dataframe(df_entities, use_container_width=True)
    else:
        st.write("Tidak ada entitas terdeteksi.")

    # -------------------------
    # KNOWLEDGE GRAPH
    # -------------------------
    st.subheader("Knowledge Graph Entitas")
    if entities:
        G = build_knowledge_graph(entities)
        visualize_knowledge_graph(G)
    else:
        st.write("Knowledge Graph tidak dapat dibuat.")

    st.markdown("---")
    if st.button("Bersihkan Data & Kembali"):
        st.session_state.crawled_data = None
        st.session_state.final_esg_category = None
        st.session_state.crawled_url = None
        st.rerun()

else:
    st.write("Tidak ada data artikel untuk dianalisis.")
