import streamlit as st
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ESG Sentiment Recommendation",
    layout="wide"
)

st.title("Rekomendasi Strategis Berbasis Sentimen ESG")
st.write(
    "Halaman ini memberikan rekomendasi tindakan berdasarkan "
    "hasil prediksi sentimen Environment, Social, dan Governance."
)

# =====================================================
# SENTIMENT & RECOMMENDATION MAPPING
# =====================================================
SENTIMENT_MAP = {
    1: "Neutral",
    2: "Positive",
    3: "Negative"
}

RECOMMENDATION_MAP = {
    "Positive": (
        "Sentimen positif terdeteksi. "
        "Disarankan untuk **mempertahankan dan memperkuat praktik yang ada**, "
        "serta melanjutkan komunikasi publik secara konsisten."
    ),
    "Neutral": (
        "Sentimen netral terdeteksi. "
        "Disarankan untuk **melakukan mitigasi risiko**, "
        "meningkatkan transparansi, dan memantau isu secara berkala."
    ),
    "Negative": (
        "Sentimen negatif terdeteksi. "
        "Disarankan untuk **melakukan counter issue**, "
        "mengambil tindakan korektif, dan menyiapkan strategi komunikasi krisis."
    )
}

# =====================================================
# LOAD STOPWORDS
# =====================================================
@st.cache_data
def load_stopwords():
    path = "stopword.txt"
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())

STOPWORDS = load_stopwords()

# =====================================================
# TEXT PREPROCESSING
# =====================================================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    return (
        joblib.load("model_Y1_sentiment_environment_RandomForest.joblib"),
        joblib.load("model_Y2_sentiment_social_RandomForest.joblib"),
        joblib.load("model_Y3_sentiment_governance_RandomForest.joblib"),
    )

model_env, model_soc, model_gov = load_models()

# =====================================================
# TF-IDF VECTORIZER
# =====================================================
@st.cache_resource
def load_vectorizer():
    return TfidfVectorizer()

vectorizer = load_vectorizer()

# =====================================================
# VALIDATE SESSION STATE
# =====================================================
if (
    "crawled_data" not in st.session_state
    or not st.session_state.crawled_data
    or "crawled_content" not in st.session_state.crawled_data
):
    st.warning("⚠️ Artikel belum tersedia. Silakan lakukan crawling terlebih dahulu.")
    st.stop()

raw_text = st.session_state.crawled_data["crawled_content"]

# =====================================================
# DISPLAY ARTICLE
# =====================================================
st.subheader("Artikel yang Dianalisis")
with st.expander("Tampilkan isi artikel"):
    st.write(raw_text)

# =====================================================
# PREPROCESSING
# =====================================================
processed_text = preprocess_text(raw_text)

st.subheader("Hasil Preprocessing (Corpus)")
st.code(processed_text[:1200] + ("..." if len(processed_text) > 1200 else ""))

# =====================================================
# VECTORIZE
# =====================================================
X = vectorizer.fit_transform([processed_text])

# =====================================================
# PREDICTION
# =====================================================
y_env = int(model_env.predict(X)[0])
y_soc = int(model_soc.predict(X)[0])
y_gov = int(model_gov.predict(X)[0])

label_env = SENTIMENT_MAP.get(y_env, "Unknown")
label_soc = SENTIMENT_MAP.get(y_soc, "Unknown")
label_gov = SENTIMENT_MAP.get(y_gov, "Unknown")

# =====================================================
# DISPLAY SENTIMENT
# =====================================================
st.subheader("Hasil Prediksi Sentimen ESG")

col1, col2, col3 = st.columns(3)
col1.metric("Environment", label_env)
col2.metric("Social", label_soc)
col3.metric("Governance", label_gov)

# =====================================================
# STRATEGIC RECOMMENDATION
# =====================================================
st.subheader("Rekomendasi Strategis")

def show_recommendation(aspect, sentiment):
    st.markdown(f"**{aspect}**")
    st.write(RECOMMENDATION_MAP.get(sentiment, "Tidak ada rekomendasi."))

show_recommendation("Environment", label_env)
show_recommendation("Social", label_soc)
show_recommendation("Governance", label_gov)

# =====================================================
# FINAL MESSAGE
# =====================================================
st.success("Analisis sentimen dan rekomendasi strategis ESG berhasil ditampilkan.")
