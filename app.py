
import streamlit as st
import pandas as pd
import joblib
import os

# Judul aplikasi
st.title("Analisis Dampak Lingkungan Urban Heat Island")

# Cek apakah file model tersedia
model_path = "random_forest_classifier.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' tidak ditemukan. Pastikan file ini berada di folder yang sama.")
    st.stop()

# Load model
model = joblib.load(model_path)

# Input dari pengguna
st.sidebar.header("Input Data Lingkungan")
def user_input_features():
    suhu = st.sidebar.slider("Suhu (°C)", 20.0, 45.0, 30.0)
    kelembaban = st.sidebar.slider("Kelembaban (%)", 10.0, 100.0, 50.0)
    kecepatan_angin = st.sidebar.slider("Kecepatan Angin (m/s)", 0.0, 15.0, 5.0)
    polusi = st.sidebar.slider("Polusi (PM2.5)", 0.0, 300.0, 50.0)
    data = {
        "suhu": suhu,
        "kelembaban": kelembaban,
        "kecepatan_angin": kecepatan_angin,
        "polusi": polusi
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediksi
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output prediksi
st.subheader("Hasil Prediksi")
kelas = ["Rendah", "Sedang", "Tinggi"]  # Sesuaikan label klasifikasi
st.write(f"Kategori Dampak: **{kelas[prediction[0]]}**")

st.subheader("Probabilitas")
proba_df = pd.DataFrame(prediction_proba, columns=kelas)
st.write(proba_df)
