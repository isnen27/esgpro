# preprocessing.py
import streamlit as st
import pandas as pd

def app():
    st.title("Preprocessing Data")
    st.write("Di sini Anda dapat melakukan langkah-langkah preprocessing pada data Anda.")

    st.subheader("Unggah Data")
    uploaded_file = st.file_uploader("Pilih file CSV untuk preprocessing", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data asli:")
        st.dataframe(data.head())

        st.subheader("Langkah Preprocessing")
        # Contoh opsi preprocessing
        remove_duplicates = st.checkbox("Hapus Duplikat", value=True)
        handle_missing = st.checkbox("Tangani Nilai Hilang (isi dengan median/mean)", value=False)
        text_cleaning = st.checkbox("Bersihkan Teks (lowercase, hapus tanda baca)", value=True)

        if st.button("Terapkan Preprocessing"):
            processed_data = data.copy()
            if remove_duplicates:
                processed_data.drop_duplicates(inplace=True)
                st.success("Duplikat dihapus.")
            if handle_missing:
                # Contoh sederhana, Anda perlu menyesuaikannya dengan kolom numerik/kategorikal
                for col in processed_data.select_dtypes(include=['number']).columns:
                    processed_data[col].fillna(processed_data[col].median(), inplace=True)
                st.success("Nilai hilang ditangani.")
            if text_cleaning:
                # Placeholder untuk fungsi pembersihan teks
                # Misalnya: processed_data['text_column'] = processed_data['text_column'].apply(your_text_cleaning_function)
                st.info("Pembersihan teks diterapkan (fungsi belum diimplementasikan).")

            st.write("Data setelah Preprocessing:")
            st.dataframe(processed_data.head())

            # Opsi untuk mengunduh data yang sudah diproses
            csv = processed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Data yang Diproses",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
            )
    else:
        st.info("Silakan unggah file CSV untuk memulai preprocessing.")

    st.markdown("---")
    st.write("Bagian ini akan terus dikembangkan untuk fitur preprocessing yang lebih spesifik.")
