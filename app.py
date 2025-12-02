import streamlit as st
import pandas as pd
import pickle
import os

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Walmart Stockout Predictor",
    page_icon="📦",
    layout="centered"
)

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    file_path = 'walmart_stockout_model.pkl'
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# --- BAGIAN UTAMA APLIKASI ---

st.title("📦 Walmart Stockout Prediction")
st.markdown("---")
st.write("Prediksi risiko kehabisan stok (Stockout) berdasarkan kondisi gudang saat ini.")

# Coba load model
model_data = load_model()

if model_data is None:
    st.error("⚠️ File model tidak ditemukan!")
    st.warning("Silakan jalankan file `model.py` terlebih dahulu untuk melatih dan menyimpan model.")
    st.code("python model.py", language="bash")
    st.stop() # Hentikan aplikasi jika model tidak ada

model = model_data['model']
scaler = model_data['scaler']

# --- SIDEBAR INPUT ---
st.sidebar.header("📝 Input Data Gudang")

def user_input_features():
    inventory_level = st.sidebar.number_input("Level Inventaris (Unit)", min_value=0, value=100)
    reorder_point = st.sidebar.number_input("Titik Reorder (Unit)", min_value=0, value=50)
    lead_time = st.sidebar.slider("Waktu Tunggu Supplier (Hari)", 1, 30, 7)
    forecasted_demand = st.sidebar.number_input("Forecast Permintaan (Unit)", min_value=0, value=120)
    
    data = {
        'inventory_level': inventory_level,
        'reorder_point': reorder_point,
        'supplier_lead_time': lead_time,
        'forecasted_demand': forecasted_demand
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Tampilkan input user di halaman utama
st.subheader("Parameter Input:")
st.dataframe(input_df, hide_index=True)

# --- TOMBOL PREDIKSI ---
if st.button("🔍 Analisis Risiko", type="primary"):
    
    # 1. Preprocessing Input (Scaling)
    # Kita harus menggunakan scaler yang sama persis dengan saat training
    input_scaled = scaler.transform(input_df)
    
    # 2. Prediksi
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] # Ambil probabilitas kelas 1 (Stockout)
    
    # 3. Tampilkan Hasil
    st.subheader("Hasil Analisis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("🚨 POTENSI STOCKOUT")
            st.markdown(f"**Risiko: Tinggi ({probability*100:.1f}%)**")
        else:
            st.success("✅ STOK AMAN")
            st.markdown(f"**Risiko: Rendah ({probability*100:.1f}%)**")
            
    with col2:
        st.write("Progress Risiko:")
        if probability > 0.5:
            st.progress(probability, text="DANGER")
        else:
            st.progress(probability, text="SAFE")

    # 4. Rekomendasi Bisnis
    st.markdown("---")
    st.subheader("💡 Rekomendasi Tindakan")
    
    if prediction == 1:
        st.warning("""
        **Terdeteksi risiko kehabisan stok!**
        1. Segera lakukan **Emergency Order** ke supplier.
        2. Cek apakah ada stok di gudang cabang lain.
        3. Tunda promosi untuk produk ini sementara waktu.
        """)
    else:
        st.info("""
        **Kondisi stok stabil.**
        1. Lanjutkan jadwal pemesanan reguler.
        2. Monitor level inventaris secara berkala.
        """)
