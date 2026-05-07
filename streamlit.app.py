import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. KONFIGURASI FITUR (Disesuaikan dengan Gambar Workflow Anda)
# -----------------------------------------------------------------------------
# Nama key (LB, LT, dsb) harus sama persis dengan header di file Excel/Orange Anda.
FEATURE_CONFIG = {
    "LB": {
        "label": "Luas Bangunan (m²)",
        "min": 0, "max": 2000, "default": 100
    },
    "LT": {
        "label": "Luas Tanah (m²)",
        "min": 0, "max": 5000, "default": 120
    },
    "KT": {
        "label": "Jumlah Kamar Tidur",
        "min": 0, "max": 15, "default": 3
    },
    "KM": {
        "label": "Jumlah Kamar Mandi",
        "min": 0, "max": 10, "default": 2
    },
    "GRS": {
        "label": "Kapasitas Garasi (Mobil)",
        "min": 0, "max": 10, "default": 1
    }
}

# -----------------------------------------------------------------------------
# 2. FUNGSI LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Menggunakan nama file asli yang Anda upload
    MODEL_PATH = Path(__file__).parent / "adaboost rumah.pkcls"
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File {MODEL_PATH.name} tidak ditemukan. Pastikan file sudah di-upload ke GitHub.")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# -----------------------------------------------------------------------------
# 3. FUNGSI PREDIKSI
# -----------------------------------------------------------------------------
def predict_price(model, input_df):
    try:
        # Coba cara standar scikit-learn
        return model.predict(input_df)
    except:
        # Fallback untuk format internal Orange
        import Orange
        # Membuat domain sesuai input
        attrs = [Orange.data.ContinuousVariable(c) for c in input_df.columns]
        domain = Orange.data.Domain(attrs)
        # Konversi dataframe ke Orange Table
        data = Orange.data.Table.from_numpy(domain, input_df.to_numpy())
        return model(data)

# -----------------------------------------------------------------------------
# 4. INTERFACE APLIKASI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠")
    
    st.title("🏠 Estimasi Harga Rumah (AdaBoost)")
    st.write("Aplikasi ini memprediksi harga berdasarkan fitur dari model Orange Anda.")
    
    # Sidebar Info
    st.sidebar.header("Detail Model")
    st.sidebar.info("Model: AdaBoost\nFile: adaboost rumah.pkcls")

    # Load Model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Form Input
    input_values = {}
    with st.form("input_form"):
        st.subheader("Input Spesifikasi Rumah")
        
        # Grid layout untuk input
        col1, col2 = st.columns(2)
        
        for i, (key, conf) in enumerate(FEATURE_CONFIG.items()):
            target_col = col1 if i % 2 == 0 else col2
            input_values[key] = target_col.number_input(
                conf["label"], 
                min_value=conf["min"], 
                max_value=conf["max"], 
                value=conf["default"]
            )
            
        submitted = st.form_submit_button("Hitung Estimasi Harga")

    if submitted:
        # Buat DataFrame dengan urutan kolom yang benar
        df_input = pd.DataFrame([input_values])
        
        with st.spinner("Menganalisis data..."):
            try:
                prediction = predict_price(model, df_input)
                hasil = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
                
                st.divider()
                st.subheader("Hasil Prediksi:")
                # Tampilkan dalam format mata uang (asumsi Rupiah)
                st.success(f"### Estimasi Harga: Rp {hasil:,.0f}")
                
            except Exception as e:
                st.error("Gagal melakukan prediksi. Pastikan urutan fitur di model sama.")
                st.exception(e)

if __name__ == "__main__":
    main()
