# evaluation.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Untuk memuat model yang sudah dilatih

def app():
    st.title("Evaluasi Model")
    st.write("Di sini Anda dapat mengevaluasi performa model Anda.")

    st.subheader("Unggah Data Uji dan Model")

    uploaded_data_file = st.file_uploader("Pilih file CSV data uji", type=["csv"])
    uploaded_model_file = st.file_uploader("Pilih file model (misal: .pkl)", type=["pkl"])

    if uploaded_data_file is not None and uploaded_model_file is not None:
        test_data = pd.read_csv(uploaded_data_file)
        st.write("Data uji yang diunggah:")
        st.dataframe(test_data.head())

        # Memuat model
        try:
            model = joblib.load(uploaded_model_file)
            st.success("Model berhasil dimuat!")
            st.write(f"Jenis model: {type(model).__name__}")
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return

        st.subheader("Pengaturan Evaluasi")

        # Asumsi: kolom target adalah 'target' atau bisa dipilih
        target_column = st.selectbox("Pilih kolom target:", test_data.columns.tolist())
        if target_column not in test_data.columns:
            st.error("Kolom target tidak ditemukan dalam data.")
            return

        feature_columns = [col for col in test_data.columns if col != target_column]
        if not feature_columns:
            st.error("Tidak ada kolom fitur yang tersedia untuk evaluasi.")
            return

        st.write(f"Fitur yang akan digunakan: {', '.join(feature_columns)}")

        # Memisahkan fitur (X) dan target (y)
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]

        # Pastikan X_test memiliki format yang sesuai dengan model
        # Ini adalah bagian krusial: model yang dilatih harus memiliki fitur yang sama
        # dan dalam urutan yang sama dengan data uji.
        # Untuk contoh ini, kita akan asumsikan model dilatih dengan fitur numerik.
        # Anda mungkin perlu melakukan preprocessing yang sama seperti saat melatih model.
        try:
            # Jika model adalah klasifikasi dan memiliki metode predict_proba
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            else: # Jika model hanya memiliki metode predict
                y_pred = model.predict(X_test)
                y_pred_proba = None # Tidak bisa menghitung ROC AUC tanpa probabilitas

            st.subheader("Hasil Evaluasi")

            st.write(f"**Akurasi:** {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"**Presisi:** {precision_score(y_test, y_pred, average='weighted'):.4f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.4f}")
            st.write(f"**F1-Score:** {f1_score(y_test, y_pred, average='weighted'):.4f}")

            if y_pred_proba is not None:
                try:
                    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_pred_proba):.4f}")
                except ValueError:
                    st.warning("ROC AUC Score tidak dapat dihitung karena hanya ada satu kelas dalam y_true atau y_score.")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Prediksi')
            ax.set_ylabel('Aktual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi atau evaluasi: {e}")
            st.warning("Pastikan data uji memiliki format (kolom dan tipe data) yang sama dengan data yang digunakan untuk melatih model.")

    else:
        st.info("Silakan unggah data uji CSV dan file model (.pkl) untuk memulai evaluasi.")

    st.markdown("---")
    st.write("Bagian ini akan terus dikembangkan untuk metrik evaluasi yang lebih mendalam.")
