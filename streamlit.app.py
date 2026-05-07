import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. KONFIGURASI FITUR (PENTING: Sesuaikan dengan dataset asli Anda di Orange)
# -----------------------------------------------------------------------------
# GANTI nama keys di bawah ini agar SAMA PERSIS dengan nama kolom (variabel)
# pada dataset yang Anda gunakan saat melatih model di Orange Data Mining!
FEATURE_CONFIG = {
    "luas_tanah": {
        "type": "numeric",
        "input": "number",
        "min": 0,
        "max": 10000,
        "default": 100
    },
    "luas_bangunan": {
        "type": "numeric",
        "input": "number",
        "min": 0,
        "max": 10000,
        "default": 80
    },
    "kamar_tidur": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 20,
        "default": 3
    },
    "kamar_mandi": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 10,
        "default": 2
    },
    "lokasi": {
        "type": "categorical",
        "options": ["Pusat Kota", "Pinggiran Kota", "Pedesaan"] # Ganti sesuai data asli
    }
}

# -----------------------------------------------------------------------------
# 2. FUNGSI LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Memuat file model dari repository lokal GitHub.
    Menggunakan nama file 'adaboost rumah.pkcls' sesuai file yang diunggah.
    """
    # Mencari file model di folder yang sama dengan streamlit_app.py
    MODEL_PATH = Path(__file__).parent / "adaboost rumah.pkcls"
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_PATH}. Pastikan file 'adaboost rumah.pkcls' sudah di-upload ke GitHub.")
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# -----------------------------------------------------------------------------
# 3. FUNGSI PREDIKSI & FALLBACK ORANGE
# -----------------------------------------------------------------------------
def predict_with_orange_fallback(model, input_df):
    try:
        import Orange
    except ImportError:
        raise ImportError("Library 'orange3' tidak tersedia. Pastikan ada di requirements.txt")

    attributes = []
    for col_name in input_df.columns:
        if pd.api.types.is_numeric_dtype(input_df[col_name]):
            attributes.append(Orange.data.ContinuousVariable(col_name))
        else:
            if col_name in FEATURE_CONFIG and FEATURE_CONFIG[col_name]["type"] == "categorical":
                options = FEATURE_CONFIG[col_name]["options"]
            else:
                options = [str(x) for x in input_df[col_name].unique()]
            attributes.append(Orange.data.DiscreteVariable(col_name, values=options))
    
    domain = Orange.data.Domain(attributes)
    X = np.zeros((1, len(attributes)))
    
    for i, col in enumerate(input_df.columns):
        val = input_df[col].iloc[0]
        if isinstance(attributes[i], Orange.data.ContinuousVariable):
            X[0, i] = float(val)
        else:
            X[0, i] = attributes[i].to_val(str(val))
            
    orange_table = Orange.data.Table.from_numpy(domain, X)
    hasil = model(orange_table)
    return hasil

def predict_with_model(model, input_df):
    try:
        # Coba cara standar scikit-learn
        return model.predict(input_df)
    except Exception as e_sklearn:
        try:
            # Jika error, gunakan cara spesifik Orange Data Mining
            return predict_with_orange_fallback(model, input_df)
        except Exception as e_orange:
            raise RuntimeError(f"Gagal prediksi.\nSklearn: {e_sklearn}\nOrange: {e_orange}")

# -----------------------------------------------------------------------------
# 4. UI & MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Prediksi Harga Rumah (AdaBoost)", page_icon="🏠", layout="centered")
    
    with st.sidebar:
        st.header("Informasi")
        st.write("Aplikasi ini memprediksi nilai rumah menggunakan model **AdaBoost** dari Orange Data Mining.")
        st.write("File model: `adaboost rumah.pkcls`")
        st.info("Isi parameter rumah di form, lalu klik Prediksi.")

    st.title("🏠 Aplikasi Prediksi Harga Rumah")
    st.write("Masukkan spesifikasi rumah untuk melihat estimasi hasil prediksi.")
    st.divider()

    # Load Model
    try:
        model = load_model()
    except Exception as e:
        st.error("Gagal memuat model. Periksa pesan error berikut:")
        st.code(str(e))
        st.stop()

    # Buat Form Input
    input_data = {}
    with st.form("prediction_form"):
        st.subheader("Spesifikasi Rumah")
        
        for feature_name, config in FEATURE_CONFIG.items():
            label = feature_name.replace("_", " ").title()
            
            if config["type"] == "numeric":
                if config["input"] == "slider":
                    input_data[feature_name] = st.slider(label, config["min"], config["max"], config["default"])
                else:
                    input_data[feature_name] = st.number_input(label, config["min"], config["max"], config["default"])
            
            elif config["type"] == "categorical":
                input_data[feature_name] = st.selectbox(label, config["options"])
                
        submitted = st.form_submit_button("Prediksi Sekarang")

    # Eksekusi Prediksi
    if submitted:
        input_df = pd.DataFrame([input_data], columns=FEATURE_CONFIG.keys())
        st.write("### Data Input Anda:")
        st.dataframe(input_df, use_container_width=True)
        
        with st.spinner("Menghitung prediksi..."):
            try:
                hasil = predict_with_model(model, input_df)
                st.success("Prediksi Berhasil!")
                
                # Format dan tampilkan hasil prediksi
                nilai_prediksi = hasil[0] if isinstance(hasil, (list, np.ndarray)) else hasil
                st.metric(label="Estimasi Hasil (Prediksi)", value=f"{nilai_prediksi:,.2f}")
                    
            except Exception as e: