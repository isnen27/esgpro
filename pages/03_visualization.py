# pages/visualization.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Visualisasi Data")
st.write("Di sini Anda dapat membuat berbagai visualisasi dari data Anda.")

st.subheader("Unggah Data")
uploaded_file = st.file_uploader("Pilih file CSV untuk visualisasi", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.dataframe(data.head())

    st.subheader("Pilih Jenis Visualisasi")
    chart_type = st.selectbox(
        "Pilih jenis chart:",
        ["Histogram", "Scatter Plot", "Bar Chart", "Box Plot"]
    )

    if chart_type == "Histogram":
        st.write("### Histogram")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
            if selected_col:
                fig, ax = plt.subplots()
                sns.histplot(data[selected_col], kde=True, ax=ax)
                ax.set_title(f"Distribusi {selected_col}")
                st.pyplot(fig)
        else:
            st.warning("Tidak ada kolom numerik untuk histogram.")

    elif chart_type == "Scatter Plot":
        st.write("### Scatter Plot")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Pilih kolom X:", numeric_cols)
            y_col = st.selectbox("Pilih kolom Y:", [col for col in numeric_cols if col != x_col])
            if x_col and y_col:
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)
                ax.set_title(f"Scatter Plot {x_col} vs {y_col}")
                st.pyplot(fig)
        else:
            st.warning("Dibutuhkan setidaknya dua kolom numerik untuk scatter plot.")

    elif chart_type == "Bar Chart":
        st.write("### Bar Chart")
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            selected_col = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
            if selected_col:
                fig, ax = plt.subplots()
                sns.countplot(y=data[selected_col], ax=ax, order=data[selected_col].value_counts().index)
                ax.set_title(f"Frekuensi {selected_col}")
                st.pyplot(fig)
        else:
            st.warning("Tidak ada kolom kategorikal untuk bar chart.")

    elif chart_type == "Box Plot":
        st.write("### Box Plot")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols and categorical_cols:
            numeric_col = st.selectbox("Pilih kolom numerik:", numeric_cols)
            category_col = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
            if numeric_col and category_col:
                fig, ax = plt.subplots()
                sns.boxplot(x=data[category_col], y=data[numeric_col], ax=ax)
                ax.set_title(f"Box Plot {numeric_col} berdasarkan {category_col}")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
        else:
            st.warning("Dibutuhkan kolom numerik dan kategorikal untuk box plot.")

else:
    st.info("Silakan unggah file CSV untuk memulai visualisasi.")

st.markdown("---")
st.write("Bagian ini akan terus dikembangkan untuk visualisasi yang lebih interaktif dan spesifik.")
