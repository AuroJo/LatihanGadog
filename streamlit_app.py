import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ---------------------------------------------------------
# 1. KONFIGURASI FITUR (Sesuai Select Columns Orange)
# ---------------------------------------------------------
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
        "min": 1, "max": 15, "default": 3
    },
    "KM": {
        "label": "Jumlah Kamar Mandi",
        "min": 1, "max": 15, "default": 2
    },
    "GRS": {
        "label": "Garasi (Mobil)",
        "min": 0, "max": 10, "default": 1
    }
}

# ---------------------------------------------------------
# 2. FUNGSI LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    # Pastikan file ini ada di root repository GitHub Anda
    MODEL_NAME = "adaboost rumah.pkcls"
    MODEL_PATH = Path(__file__).parent / MODEL_NAME
    
    if not MODEL_PATH.exists():
        st.error(f"File model '{MODEL_NAME}' tidak ditemukan di repository!")
        st.stop()
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------------------------------------
# 3. FUNGSI PREDIKSI
# ---------------------------------------------------------
def predict_price(model, input_df):
    try:
        # Coba prediksi standard scikit-learn
        return model.predict(input_df)
    except:
        # Fallback untuk format khusus Orange Data Mining
        import Orange
        attributes = [Orange.data.ContinuousVariable(name) for name in input_df.columns]
        domain = Orange.data.Domain(attributes)
        orange_table = Orange.data.Table.from_numpy(domain, input_df.to_numpy())
        return model(orange_table)

# ---------------------------------------------------------
# 4. TAMPILAN APLIKASI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠")

    st.title("🏠 Prediksi Harga Rumah (AdaBoost)")
    st.write("Aplikasi ini memprediksi harga berdasarkan spesifikasi bangunan.")
    
    # Load Model
    model = load_model()

    # Form Input
    with st.form("input_form"):
        st.subheader("Masukkan Detail Properti")
        
        col1, col2 = st.columns(2)
        user_inputs = {}

        for i, (key, cfg) in enumerate(FEATURE_CONFIG.items()):
            # Bagi input ke dalam 2 kolom agar rapi
            target_col = col1 if i % 2 == 0 else col2
            user_inputs[key] = target_col.number_input(
                cfg["label"], 
                min_value=cfg["min"], 
                max_value=cfg["max"], 
                value=cfg["default"]
            )
            
        predict_button = st.form_submit_button("Hitung Estimasi Harga")

    if predict_button:
        # Susun input menjadi DataFrame
        df_input = pd.DataFrame([user_inputs])
        
        with st.spinner("Sedang menghitung..."):
            try:
                prediction = predict_price(model, df_input)
                # Ambil nilai prediksi
                final_price = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
                
                st.divider()
                st.success(f"### Estimasi Harga: Rp {final_price:,.0f}")
                
            except Exception as e:
                st.error("Terjadi error saat menjalankan prediksi.")
                st.exception(e)

if __name__ == "__main__":
    main()