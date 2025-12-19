import streamlit as st
import re
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="ESG Recommendation", layout="wide")
st.title("📊 Rekomendasi Strategis ESG")

# =====================================================
# VALIDASI SESSION STATE
# =====================================================

if "crawled_data" not in st.session_state:
    st.warning("⚠️ Artikel belum tersedia. Silakan lakukan crawling terlebih dahulu.")
    st.stop()

article_text = st.session_state.crawled_data.get("crawled_content", "")

# =====================================================
# LOAD RESOURCE
# =====================================================

@st.cache_resource
def load_resources():
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model_env = joblib.load("model_Y1_sentiment_environment_KNeighbors.joblib")
    model_soc = joblib.load("model_Y2_sentiment_social_KNeighbors.joblib")
    model_gov = joblib.load("model_Y3_sentiment_governance_KNeighbors.joblib")
    return vectorizer, model_env, model_soc, model_gov

def load_stopwords():
    with open("stopword.txt", encoding="utf-8") as f:
        return set(f.read().splitlines())

vectorizer, model_env, model_soc, model_gov = load_resources()
stopwords = load_stopwords()

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 2]
    return " ".join(tokens)

processed = preprocess(article_text)
X = vectorizer.transform([processed])

# =====================================================
# PREDIKSI
# =====================================================

LABEL = {1: "Neutral", 2: "Positive", 3: "Negative"}
EMOJI = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}
RECO = {
    "Positive": "Pertahankan strategi dan komunikasi ESG yang sudah berjalan dengan baik.",
    "Neutral": "Lakukan mitigasi risiko dan perkuat kebijakan ESG untuk mencegah isu berkembang.",
    "Negative": "Segera lakukan counter issue, perbaikan kebijakan, dan komunikasi krisis ESG."
}

scores = {
    "Environment": LABEL[int(model_env.predict(X)[0])],
    "Social": LABEL[int(model_soc.predict(X)[0])],
    "Governance": LABEL[int(model_gov.predict(X)[0])]
}

# =====================================================
# TAMPILAN
# =====================================================

cols = st.columns(3)
for col, (k, v) in zip(cols, scores.items()):
    col.metric(k, f"{EMOJI[v]} {v}")

if "Negative" in scores.values():
    risk = "🔴 HIGH RISK"
elif "Neutral" in scores.values():
    risk = "🟡 MEDIUM RISK"
else:
    risk = "🟢 LOW RISK"

st.subheader("📈 Overall ESG Risk Level")
st.info(risk)

st.subheader("🧭 Rekomendasi Strategis")
for k, v in scores.items():
    st.markdown(f"**{k}** → {RECO[v]}")

# =====================================================
# EXPORT PDF
# =====================================================

def generate_pdf():
    file_path = "esg_recommendation.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Ambil data dari session state
    article_data = st.session_state.crawled_data
    title = article_data.get("crawled_title", "N/A")
    date = article_data.get("crawled_date", "N/A")
    content = article_data.get("crawled_content", "")

    # Header Laporan
    story.append(Paragraph("ESG Strategic Recommendation Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # --- BAGIAN 1: INFORMASI ARTIKEL ---
    story.append(Paragraph("Informasi Artikel", styles["Heading2"]))
    story.append(Paragraph(f"<b>Judul:</b> {title}", styles["Normal"]))
    story.append(Paragraph(f"<b>Tanggal:</b> {date}", styles["Normal"]))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Isi Artikel:", styles["Heading3"]))
    # Menggunakan gaya 'BodyText' agar teks panjang teratur dengan baik
    story.append(Paragraph(content[:1500] + "..." if len(content) > 1500 else content, styles["Normal"])) 
    story.append(Spacer(1, 20))
    story.append(Paragraph("<hr/>", styles["Normal"])) # Garis pemisah sederhana

    # --- BAGIAN 2: HASIL ANALISIS ---
    story.append(Paragraph("Hasil Analisis & Rekomendasi", styles["Heading2"]))
    story.append(Paragraph(f"<b>Overall Risk Level:</b> {risk}", styles["Normal"]))
    story.append(Spacer(1, 10))

    for k, v in scores.items():
        story.append(Paragraph(f"{k}: {v}", styles["Heading3"]))
        story.append(Paragraph(f"Rekomendasi: {RECO[v]}", styles["Normal"]))
        story.append(Spacer(1, 5))

    doc.build(story)
    return file_path

if st.button("⬇️ Download PDF"):
    path = generate_pdf()
    with open(path, "rb") as f:
        st.download_button("📥 Download", f, file_name="ESG_Recommendation.pdf")
