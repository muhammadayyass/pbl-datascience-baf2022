import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="e-KYC Anti-Fraud Gatekeeper", page_icon="🛡️", layout="wide")

st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); padding:20px; border-radius:10px; text-align:center;">
<h1 style="color:white; margin:0;">🛡️ Digital Onboarding: Anti-Fraud Gatekeeper</h1>
<p style="color:#e0f2fe; margin:0;">Audit Kepatuhan BAF NeurIPS 2022 (HistGradientBoosting Engine)</p>
</div>
<br>
""", unsafe_allow_html=True)

# 2. Fungsi untuk memuat artifak AI
@st.cache_resource
def load_artifacts():
    model = joblib.load('baf_histgb_model.pkl')
    scaler = joblib.load('baf_scaler.pkl')
    threshold = joblib.load('baf_threshold.pkl')
    cols_fs = joblib.load('baf_cols_fs.pkl')
    cols_scale = joblib.load('baf_cols_scale.pkl')
    template_mentah = joblib.load('baf_template.pkl')
    return model, scaler, threshold, cols_fs, cols_scale, template_mentah

try:
    model, scaler, threshold, cols_fs, cols_scale, template_mentah = load_artifacts()
except Exception as e:
    st.error("⚠️ SISTEM TERHENTI: Artifak model (.pkl) tidak ditemukan di direktori GitHub.")
    st.stop()

# 3. Layout Antarmuka
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📝 Formulir Pendaftaran Digital")
    umur = st.slider("Kelompok Usia Pendaftar", 10, 90, 30, help="Evaluasi: Model berisiko bias terhadap pendaftar usia lanjut.")
    pendapatan = st.slider("Estimasi Pendapatan (Decile)", 0.1, 0.9, 0.5, step=0.1)
    kredit = st.slider("Skor Risiko Kredit Biro", 0, 500, 150, step=10)
    velocity = st.slider("Velocity (Kecepatan Pengisian Form 6 Jam)", 0, 20000, 5000, step=500)
    
    col_a, col_b = st.columns(2)
    with col_a:
        perumahan = st.selectbox("Status Hunian", ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG'])
        os_hp = st.selectbox("OS Perangkat", ['windows', 'linux', 'macintosh', 'x11', 'other'])
    with col_b:
        pekerjaan = st.selectbox("Status Pekerjaan", ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'])
        sumber = st.selectbox("Sumber Pendaftaran", ['INTERNET', 'TELEAPP'])

with col2:
    st.markdown("### ⚙️ Eksekutif Control (Regulasi)")
    st.info(f"Threshold Bawaan Model (5% FPR Kepatuhan) berada pada probabilitas: **{threshold*100:.2f}%**")
    sensitivitas = st.slider("Toleransi Manajemen Bank (%)", 1, 99, 50, help="Geser ke kiri untuk membuat AI lebih galak.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    tombol_evaluasi = st.button("🚀 Audit Pendaftar Sekarang", use_container_width=True, type="primary")

# 4. Mesin Pemroses Utama
if tombol_evaluasi:
    with st.spinner('Menjalankan Audit AI HistGradientBoosting...'):
        # Membangun matriks nol dinamis
        df_live = pd.DataFrame(0, index=[0], columns=cols_fs)

        for col in cols_fs:
            if col in template_mentah:
                df_live.at[0, col] = template_mentah[col]
        
        df_live.at[0, 'customer_age'] = umur
        df_live.at[0, 'income'] = pendapatan
        df_live.at[0, 'credit_risk_score'] = kredit
        df_live.at[0, 'velocity_6h'] = velocity
        
        if 'velocity_6h_log' in cols_fs:
            df_live.at[0, 'velocity_6h_log'] = np.log1p(velocity)
            
        dummy_mappings = {
            f"housing_status_{perumahan}": 1,
            f"employment_status_{pekerjaan}": 1,
            f"device_os_{os_hp}": 1,
            f"source_{sumber}": 1
        }
        for feature_name, value in dummy_mappings.items():
            if feature_name in cols_fs:
                df_live.at[0, feature_name] = value
            
        kolom_tersedia_scale = [c for c in cols_scale if c in df_live.columns]
        df_live[kolom_tersedia_scale] = scaler.transform(df_live[kolom_tersedia_scale])

        prob_fraud = model.predict_proba(df_live)[0][1]
        batas_blokir = threshold * (sensitivitas / 50.0)

        st.markdown("---")
        st.markdown("### 🖥️ Putusan Core Banking System")
        
        if prob_fraud >= batas_blokir:
            st.error(f"""
            **❌ STATUS REGISTRASI: DITOLAK (DIBLOKIR)**
            
            **🤖 Audit e-KYC (Machine Learning):**
            * Klasifikasi: **TERINDIKASI APPLICATION FRAUD SANGAT KUAT 🚨**
            * Skor Kecurigaan Profil AI: **{prob_fraud*100:.2f}%**
            * Ambang Toleransi Bank Anda: **{batas_blokir*100:.2f}%**
            """)
        else:
            st.success(f"""
            **✅ STATUS REGISTRASI: DITERIMA (SAH)**
            
            **🤖 Audit e-KYC (Machine Learning):**
            * Klasifikasi: **PROFIL WAJAR DAN AMAN 🛡️**
            * Skor Kecurigaan Profil AI: **{prob_fraud*100:.2f}%**
            * Ambang Toleransi Bank Anda: **{batas_blokir*100:.2f}%**
            """)

        # Peringatan Bias Usia Lanjut
        if umur >= 56:
            st.warning("""
            ⚠️ **FAIRNESS AUDIT ALERT (PREDICTIVE EQUALITY VIOLATION)** ⚠️
            
            Sistem mendeteksi bahwa pendaftar berada pada kelompok usia rentan **(> 55 Tahun)**. Model ini secara sistemik terbukti memiliki tingkat kesalahan tolak (*False Positive Rate*) sebesar **13.77%** pada kelompok usia ini, yaitu **8.4x lipat lebih tinggi** dibandingkan pemohon muda. 
            *Rekomendasi Kebijakan:* Lakukan verifikasi manual tambahan (Human-in-the-loop) untuk mencegah diskriminasi digital.
            """)
