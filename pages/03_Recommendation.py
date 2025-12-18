import streamlit as st
import re
import joblib
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ESG Sentiment Recommendation",
    layout="wide"
)

st.title("Rekomendasi Strategis Berbasis Sentimen ESG")

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
        "Pertahankan dan perkuat praktik ESG serta komunikasi publik."
    ),
    "Neutral": (
        "Sentimen netral terdeteksi. "
        "Lakukan mitigasi risiko dan pemantauan isu secara berkala."
    ),
    "Negative": (
        "Sentimen negatif terdeteksi. "
        "Lakukan counter issue, tindakan korektif, dan strategi komunikasi krisis."
    )
}

# =====================================================
# LOAD STOPWORDS
# =====================================================
@st.cache_data
def load_stopwords():
    with open("stopword.txt", encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())

STOPWORDS = load_stopwords()

# =====================================================
# PREPROCESSING (HARUS SAMA DENGAN TRAINING)
# =====================================================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# =====================================================
# LOAD MODELS & TF-IDF (INI KUNCI UTAMA)
# =====================================================
@st.cache_resource
def load_artifacts():
    return (
        joblib.load("tfidf_vectorizer.joblib"),
        joblib.load("model_Y1_sentiment_environment_RandomForest.joblib"),
        joblib.load("model_Y2_sentiment_social_RandomForest.joblib"),
        joblib.load("model_Y3_sentiment_governance_RandomForest.joblib"),
    )

vectorizer, model_env, model_soc, model_gov = load_artifacts()

# =====================================================
# VALIDATE SESSION STATE
# =====================================================
if (
    "crawled_data" not in st.session_state
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
# PREPROCESS + VECTORIZE (TRANSFORM, BUKAN FIT)
# =====================================================
processed_text = preprocess_text(raw_text)
X = vectorizer.transform([processed_text])

# =====================================================
# PREDICTION
# =====================================================
y_env = int(model_env.predict(X)[0])
y_soc = int(model_soc.predict(X)[0])
y_gov = int(model_gov.predict(X)[0])

label_env = SENTIMENT_MAP[y_env]
label_soc = SENTIMENT_MAP[y_soc]
label_gov = SENTIMENT_MAP[y_gov]

# =====================================================
# DISPLAY RESULT
# =====================================================
st.subheader("Hasil Prediksi Sentimen ESG")

col1, col2, col3 = st.columns(3)
col1.metric("Environment", label_env)
col2.metric("Social", label_soc)
col3.metric("Governance", label_gov)

# =====================================================
# RECOMMENDATION
# =====================================================
st.subheader("Rekomendasi Strategis")

st.markdown("**Environment**")
st.write(RECOMMENDATION_MAP[label_env])

st.markdown("**Social**")
st.write(RECOMMENDATION_MAP[label_soc])

st.markdown("**Governance**")
st.write(RECOMMENDATION_MAP[label_gov])

st.success("Analisis sentimen dan rekomendasi berhasil dibuat.")
