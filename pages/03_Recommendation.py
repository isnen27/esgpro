import streamlit as st
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

st.set_page_config(page_title="ESG Recommendation", layout="wide")

# =====================================================
# CONSTANTS & MAPPING
# =====================================================

SENTIMENT_LABEL = {
    1: "Neutral",
    2: "Positive",
    3: "Negative"
}

SENTIMENT_EMOJI = {
    "Positive": "🟢",
    "Neutral": "🟡",
    "Negative": "🔴"
}

RECOMMENDATION_MAP = {
    "Positive": "Pertahankan strategi dan komunikasi ESG yang sudah berjalan dengan baik.",
    "Neutral": "Lakukan mitigasi risiko dan perkuat kebijakan ESG untuk mencegah isu berkembang.",
    "Negative": "Segera lakukan counter issue, perbaikan kebijakan, dan komunikasi krisis ESG."
}

# =====================================================
# LOAD RESOURCE
# =====================================================

@st.cache_resource
def load_models():
    model_env = joblib.load("model_Y1_sentiment_environment_RandomForest.joblib")
    model_soc = joblib.load("model_Y2_sentiment_social_RandomForest.joblib")
    model_gov = joblib.load("model_Y3_sentiment_governance_RandomForest.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model_env, model_soc, model_gov, vectorizer

def load_stopwords():
    with open("stopword.txt", encoding="utf-8") as f:
        return set(f.read().splitlines())

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_text(text, stopwords):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return " ".join(tokens)

# =====================================================
# PDF GENERATOR (SAFE PATH)
# =====================================================

def generate_pdf(results, overall_risk):
    file_path = "esg_recommendation.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>ESG SENTIMENT & STRATEGIC RECOMMENDATION REPORT</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Overall ESG Risk Level:</b> {overall_risk}", styles["Normal"]))
    story.append(Spacer(1, 12))

    for aspect, data in results.items():
        story.append(Paragraph(
            f"<b>{aspect}</b> : {data['emoji']} {data['label']}",
            styles["Heading3"]
        ))
        story.append(Paragraph(data["recommendation"], styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    return file_path

# =====================================================
# MAIN UI
# =====================================================

st.title("📊 Rekomendasi Strategis Berbasis Sentimen ESG")

if "crawled_content" not in st.session_state or not st.session_state["crawled_content"]:
    st.warning("⚠️ Artikel belum tersedia. Silakan lakukan crawling terlebih dahulu.")
    st.stop()

# Load data
article_text = st.session_state["crawled_content"]

stopwords = load_stopwords()
model_env, model_soc, model_gov, vectorizer = load_models()

processed_text = preprocess_text(article_text, stopwords)
X = vectorizer.transform([processed_text])

# =====================================================
# PREDICTION
# =====================================================

y_env = int(model_env.predict(X)[0])
y_soc = int(model_soc.predict(X)[0])
y_gov = int(model_gov.predict(X)[0])

results = {
    "Environment": {
        "score": y_env,
        "label": SENTIMENT_LABEL[y_env],
        "emoji": SENTIMENT_EMOJI[SENTIMENT_LABEL[y_env]],
        "recommendation": RECOMMENDATION_MAP[SENTIMENT_LABEL[y_env]]
    },
    "Social": {
        "score": y_soc,
        "label": SENTIMENT_LABEL[y_soc],
        "emoji": SENTIMENT_EMOJI[SENTIMENT_LABEL[y_soc]],
        "recommendation": RECOMMENDATION_MAP[SENTIMENT_LABEL[y_soc]]
    },
    "Governance": {
        "score": y_gov,
        "label": SENTIMENT_LABEL[y_gov],
        "emoji": SENTIMENT_EMOJI[SENTIMENT_LABEL[y_gov]],
        "recommendation": RECOMMENDATION_MAP[SENTIMENT_LABEL[y_gov]]
    }
}

# =====================================================
# OVERALL RISK & HIGHLIGHT
# =====================================================

avg_score = (y_env + y_soc + y_gov) / 3

if avg_score >= 2.5:
    overall_risk = "🔴 HIGH RISK"
elif avg_score >= 1.8:
    overall_risk = "🟡 MEDIUM RISK"
else:
    overall_risk = "🟢 LOW RISK"

most_problematic = max(results.items(), key=lambda x: x[1]["score"])

# =====================================================
# DISPLAY RESULT
# =====================================================

st.subheader("📌 Hasil Prediksi Sentimen")

cols = st.columns(3)
for col, (aspect, data) in zip(cols, results.items()):
    col.metric(
        label=f"{aspect}",
        value=f"{data['emoji']} {data['label']}"
    )

st.divider()

st.subheader("⚠️ Highlight Utama")
st.error(
    f"Aspek paling bermasalah: **{most_problematic[0]}** "
    f"({most_problematic[1]['label']})"
)

st.subheader("📈 Overall ESG Risk Level")
st.info(overall_risk)

st.subheader("🧭 Rekomendasi Strategis")
for aspect, data in results.items():
    st.markdown(
        f"**{aspect} {data['emoji']} ({data['label']})**  \n"
        f"- {data['recommendation']}"
    )

# =====================================================
# EXPORT PDF
# =====================================================

st.divider()

if st.button("⬇️ Download Rekomendasi dalam PDF"):
    pdf_path = generate_pdf(results, overall_risk)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Klik untuk download PDF",
            data=f,
            file_name="esg_recommendation.pdf",
            mime="application/pdf"
        )
