import streamlit as st
import re
import joblib
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ESG Sentiment Recommendation",
    layout="wide"
)

st.title("📊 ESG Sentiment & Strategic Recommendation")

# =====================================================
# MAPPING
# =====================================================
SENTIMENT_MAP = {
    1: "Neutral",
    2: "Positive",
    3: "Negative"
}

SENTIMENT_BADGE = {
    "Positive": "🟢 Positive",
    "Neutral": "🟡 Neutral",
    "Negative": "🔴 Negative"
}

RECOMMENDATION_MAP = {
    "Positive": "Pertahankan dan perkuat praktik ESG serta komunikasi publik.",
    "Neutral": "Lakukan mitigasi risiko, peningkatan transparansi, dan monitoring isu.",
    "Negative": "Lakukan counter issue, tindakan korektif, dan strategi komunikasi krisis."
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
# PREPROCESSING
# =====================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# =====================================================
# LOAD MODEL + TFIDF
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
# PREPROCESS + VECTORIZE
# =====================================================
processed_text = preprocess_text(raw_text)
X = vectorizer.transform([processed_text])

# =====================================================
# PREDICTION
# =====================================================
results = {
    "Environment": SENTIMENT_MAP[int(model_env.predict(X)[0])],
    "Social": SENTIMENT_MAP[int(model_soc.predict(X)[0])],
    "Governance": SENTIMENT_MAP[int(model_gov.predict(X)[0])]
}

# =====================================================
# DISPLAY SENTIMENT
# =====================================================
st.subheader("🔍 Hasil Prediksi Sentimen")

cols = st.columns(3)
for col, (aspect, sentiment) in zip(cols, results.items()):
    col.metric(aspect, SENTIMENT_BADGE[sentiment])

# =====================================================
# OVERALL ESG RISK
# =====================================================
st.subheader("⚠️ Overall ESG Risk Level")

if "Negative" in results.values():
    overall_risk = "🔴 HIGH RISK"
elif "Neutral" in results.values():
    overall_risk = "🟠 MEDIUM RISK"
else:
    overall_risk = "🟢 LOW RISK"

st.markdown(f"### {overall_risk}")

# =====================================================
# HIGHLIGHT MOST PROBLEMATIC ASPECT
# =====================================================
st.subheader("🚨 Aspek Paling Bermasalah")

problematic = [
    a for a, s in results.items()
    if s in ("Negative", "Neutral")
]

if problematic:
    for a in problematic:
        st.write(f"- **{a}** → {SENTIMENT_BADGE[results[a]]}")
else:
    st.success("Semua aspek ESG dalam kondisi positif.")

# =====================================================
# RECOMMENDATION
# =====================================================
st.subheader("🧭 Rekomendasi Strategis")

for aspect, sentiment in results.items():
    st.markdown(f"**{aspect}**")
    st.write(RECOMMENDATION_MAP[sentiment])

# =====================================================
# EXPORT TO PDF
# =====================================================
st.subheader("📄 Export Rekomendasi")

def generate_pdf(results, overall_risk):
    file_path = "/mnt/data/esg_recommendation.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ESG Sentiment & Recommendation Report", styles["Title"]))
    story.append(Paragraph(f"Overall ESG Risk Level: {overall_risk}", styles["Normal"]))

    for aspect, sentiment in results.items():
        story.append(Paragraph(f"<b>{aspect}</b>: {sentiment}", styles["Normal"]))
        story.append(Paragraph(RECOMMENDATION_MAP[sentiment], styles["Normal"]))

    doc.build(story)
    return file_path

if st.button("⬇️ Download PDF"):
    pdf_path = generate_pdf(results, overall_risk)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Klik untuk download",
            f,
            file_name="ESG_Recommendation.pdf",
            mime="application/pdf"
        )

st.success("Analisis ESG selesai dan siap digunakan.")
