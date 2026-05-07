import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. KONFIGURASI FITUR (Sesuai Gambar Select Columns Orange)
# -----------------------------------------------------------------------------
# Pastikan nama key (LB, LT, KT, KM, GRS) sama persis dengan kolom di file Excel/Orange
FEATURE_CONFIG = {
    "LB": {
        "label": "Luas Bangunan (m²)",
        "min": 10, "max": 2000, "default": 100
    },
    "LT": {
        "label": "Luas Tanah (m²)",
        "min": 10, "max": 5000, "default": 120
    },
    "KT": {
        "label": "Jumlah Kamar Tidur",
        "min": 1, "max": 20, "default": 3
    },
    "KM": {
        "label": "Jumlah Kamar Mandi",
        "min": 1, "max": 15, "default": 2
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
    # Menggunakan nama file asli: 'adaboost rumah.pkcls'
    MODEL_PATH = Path(__file__).parent / "adaboost rumah.pkcls"
    
    if not MODEL_PATH.exists():
        st.error(f"File model '{MODEL_PATH.name}' tidak ditemukan di repository GitHub!")
        st.stop()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# -----------------------------------------------------------------------------
# 3. FUNGSI PREDIKSI (Fallback Orange)
# -----------------------------------------------------------------------------
def predict_price(model, input_df):
    try:
        # Mencoba prediksi standard (scikit-learn style)
        return model.predict(input_df)
    except:
        # Fallback jika model memerlukan format Orange Table
        import Orange
        # Membuat domain sesuai input data
        attributes = [Orange.data.ContinuousVariable(name) for name in input_df.columns]
        domain = Orange.data.Domain(attributes)
        # Konversi ke format Orange
        orange_table = Orange.data.Table.from_numpy(domain, input_df.to_numpy())
        return model(orange_table)

# -----------------------------------------------------------------------------
# 4. INTERFACE PENGGUNA (UI)
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Estimasi Harga Rumah", page_icon="🏠")

    # Sidebar
    st.sidebar.title("Informasi Model")
    st.sidebar.info("Model: AdaBoost Regressor\nSumber: Orange Data Mining")
    st.sidebar.write("Pastikan file `adaboost rumah.pkcls` berada di root folder GitHub.")

    # Main Area
    st.title("🏠 Prediksi Harga Rumah")
    st.write("Gunakan slider atau input angka di bawah ini untuk menghitung estimasi harga rumah.")
    
    model = load_model()

    # Form Input
    with st.form("house_form"):
        st.subheader("Parameter Properti")
        
        # Mengatur tampilan dalam 2 kolom
        col1, col2 = st.columns(2)
        input_data = {}

        for i, (key, cfg) in enumerate(FEATURE_CONFIG.items()):
            target_col = col1 if i % 2 == 0 else col2
            input_data[key] = target_col.number_input(
                cfg["label"], 
                min_value=cfg["min"], 
                max_value=cfg["max"], 
                value=cfg["default"]
            )
        
        submitted = st.form_submit_button("Hitung Prediksi Harga")

    if submitted:
        # Menyiapkan DataFrame
        df_input = pd.DataFrame([input_data])
        
        with st.spinner("Menghitung..."):
            try:
                prediction = predict_price(model, df_input)
                # Mengambil nilai tunggal dari hasil prediksi
                final_val = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
                
                st.success("### Hasil Estimasi")
                # Format mata uang (Contoh: Rp 1.500.000.000)
                st.metric(label="Harga Prediksi", value=f"Rp {final_val:,.0f}")
                
            except Exception as e:
                st.error("Terjadi kesalahan pada proses prediksi.")
                st.exception(e)

if __name__ == "__main__":
    main()
