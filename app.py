import streamlit as st
import pandas as pd

from utils.data_loader import load_csv_info
from utils.preprocess import prepare_features
from utils.recommenders import build_models_and_recommend
from utils.chatbot import find_title_in_query

st.set_page_config(page_title="SPK Recommender (General CSV)", layout="wide")
st.title("SPK Rekomendasi Multi-Metode — Bring here ur CSV")

# ============================
# UPLOAD CSV
# ============================
uploaded = st.file_uploader("Upload CSV (any dataset)", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
else:
    df = pd.read_csv(uploaded)

    # Sidebar informasi dataset
    st.sidebar.header("Dataset preview & pilihan")
    st.sidebar.write(f"Jumlah baris: {len(df)} — Jumlah kolom: {len(df.columns)}")

    # ============================
    # PREVIEW
    # ============================
    st.subheader("Preview dataset")
    st.dataframe(df.head(10))

    # ============================
    # PILIH IDENTIFIER
    # ============================
    st.sidebar.markdown("Pilih kolom identifier")
    id_col = st.sidebar.selectbox("Identifier column", options=df.columns.tolist())

    # Kolom teks
    st.sidebar.markdown("Pilih kolom teks yang akan dipakai untuk TF-IDF")
    text_cols = st.sidebar.multiselect("Text columns (TF-IDF)", options=df.columns.tolist())

    # Kolom numerik
    st.sidebar.markdown("Pilih kolom numerik untuk CBR / scaling")
    num_cols = st.sidebar.multiselect(
        "Numeric columns",
        options=df.select_dtypes(include=["number"]).columns.tolist()
    )

    # ============================
    # BOBOT METODE
    # ============================
    st.sidebar.markdown("Bobot metode (total tidak harus 1, akan dinormalisasi)")
    w_text = st.sidebar.slider("Bobot Content-Based (text)", 0.0, 1.0, 0.5)
    w_num = st.sidebar.slider("Bobot Numeric (CBR)", 0.0, 1.0, 0.3)
    w_cluster = st.sidebar.slider("Bobot Cluster", 0.0, 1.0, 0.1)

    # ============================
    # BUILD MODEL
    # ============================
    if st.button("Proses: Bangun model & Index"):
        with st.spinner("Membangun fitur & model..."):
            meta = load_csv_info(df)
            features = prepare_features(df, text_cols=text_cols, num_cols=num_cols)

            # Build hybrid engine
            st.session_state.engine = build_models_and_recommend(df, id_col=id_col, features=features)

        st.success("Selesai membangun model. Sekarang coba rekomendasi atau chat!")

    # ============================
    # REKOMENDASI MANUAL (SEED)
    # ============================
    st.subheader("Pilih item sebagai seed (atau gunakan chat di bawah)")
    seed = st.selectbox("Pilih item (seed)", options=df[id_col].astype(str).tolist())
    topn = st.number_input("Jumlah rekomendasi (top N)", min_value=1, max_value=50, value=5)

    if st.button("Dapatkan Rekomendasi dari seed"):
        if "engine" not in st.session_state:
            st.error("Model belum dibangun. Silakan klik 'Proses: Bangun model & Index' terlebih dahulu.")
        else:
            recs_df = st.session_state.engine.recommend_by_title(
                seed,
                topn=topn,
                weights={"text": w_text, "num": w_num, "cluster": w_cluster}
            )

            st.write(recs_df)

            csv = recs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download rekomendasi (CSV)",
                data=csv,
                file_name="recommendations.csv",
                mime='text/csv'
            )

    # ============================
    # CHATBOT
    # ============================
    st.markdown("---")
    st.subheader("Chatbot Q&A")

    q = st.text_input("Tanyakan ke sistem")

    if st.button("Kirim pertanyaan"):
        if "engine" not in st.session_state:
            st.error("Model belum dibangun. Silakan klik 'Proses: Bangun model & Index' terlebih dahulu.")
        else:
            title_in_query = find_title_in_query(q, df, id_col)

            if title_in_query:
                st.write(f"Terdeteksi judul: {title_in_query} — memberikan rekomendasi...")

                recs_df = st.session_state.engine.recommend_by_title(
                    title_in_query,
                    topn=topn,
                    weights={"text": w_text, "num": w_num, "cluster": w_cluster}
                )
                st.write(recs_df)

            else:
                st.warning(
                    "Tidak menemukan judul yang cocok di dataset. "
                    "Coba masukkan judul secara tepat atau pilih dari dropdown seed."
                )
