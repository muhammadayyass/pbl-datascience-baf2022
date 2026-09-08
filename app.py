import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="e-KYC Anti-Fraud Gatekeeper", page_icon="🛡️", layout="wide")

# Header Profesional
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); padding:20px; border-radius:10px; text-align:center;">
<h1 style="color:white; margin:0;">🛡️ Digital Onboarding: Anti-Fraud Gatekeeper</h1>
<p style="color:#e0f2fe; margin:0;">Audit Kepatuhan BAF NeurIPS 2022 (HistGradientBoosting Engine)</p>
</div>
<br>
""", unsafe_allow_html=True)

# 2. Memuat Artifak (.pkl) dari GitHub
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
    st.error(f"⚠️ SISTEM TERHENTI: Artifak model (.pkl) bermasalah. Detail Error: {e}")
    st.stop()

# 3. Layout Antarmuka
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📝 Formulir Pendaftaran Digital")
    umur = st.slider("Kelompok Usia Pendaftar (Binned)", 10, 90, 30, help="Perhatian: Evaluasi menunjukkan model rentan bias terhadap pendaftar usia lanjut.")
    pendapatan = st.slider("Estimasi Pendapatan (Decile)", 0.1, 0.9, 0.5, step=0.1)
    kredit = st.slider("Skor Risiko Kredit Biro", 0, 500, 150, step=10)
    velocity = st.slider("Velocity (Kecepatan Pengisian Form 6 Jam)", 0, 20000, 5000, step=100, help="Di dataset BAF, pengisian form yang terlalu lambat justru merupakan indikasi kuat penipuan.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        perumahan = st.selectbox(
            "Status Hunian", 
            ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG'], 
            help="Kode terenkripsi bawaan dataset untuk menjaga privasi nasabah (Privacy Preserving)."
        )
        os_hp = st.selectbox("OS Perangkat Pendaftar", ['windows', 'linux', 'macintosh', 'x11', 'other'])
    with col_b:
        pekerjaan = st.selectbox(
            "Status Pekerjaan", 
            ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'], 
            help="Kode terenkripsi bawaan dataset untuk menjaga privasi nasabah (Privacy Preserving)."
        )
        sumber = st.selectbox("Sumber Pendaftaran", ['INTERNET', 'TELEAPP'])

with col2:
    st.markdown("### ⚙️ Eksekutif Control (Regulasi)")
    st.info(f"Threshold Bawaan Model (5% FPR Kepatuhan) berada pada probabilitas: **{threshold*100:.2f}%**")
    sensitivitas = st.slider("Toleransi Manajemen Bank (%)", 1, 99, 50, help="Geser ke kiri untuk membuat AI lebih ketat (galak), ke kanan untuk lebih longgar.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    tombol_evaluasi = st.button("🚀 Audit Pendaftar Sekarang", use_container_width=True, type="primary")

# 4. Mesin Pemroses Utama Terlindungi
if tombol_evaluasi:
    with st.spinner('Menjalankan Audit AI HistGradientBoosting...'):
        
        # Keamanan Data Type: Membangun matriks dengan float (0.0) agar Pandas tidak crash
        df_live = pd.DataFrame(0.0, index=[0], columns=cols_fs)

        # Mengisi dengan nilai Median default (Dikonversi ke float secara eksplisit)
        for col in cols_fs:
            if col in template_mentah:
                df_live.at[0, col] = float(template_mentah[col])
        
        # Menimpa Data Nyata dari User Interface
        df_live.at[0, 'customer_age'] = float(umur)
        df_live.at[0, 'income'] = float(pendapatan)
        df_live.at[0, 'credit_risk_score'] = float(kredit)
        df_live.at[0, 'velocity_6h'] = float(velocity)
        
        # Log transformasi (Wajib sesuai pipeline asli)
        if 'velocity_6h_log' in cols_fs:
            df_live.at[0, 'velocity_6h_log'] = np.log1p(velocity)
            
        # Logika One-Hot Encoding Dinamis Anti-Error
        dummy_mappings = {
            f"housing_status_{perumahan}": 1.0,
            f"employment_status_{pekerjaan}": 1.0,
            f"device_os_{os_hp}": 1.0,
            f"source_{sumber}": 1.0
        }
        for feature_name, value in dummy_mappings.items():
            if feature_name in cols_fs:
                df_live.at[0, feature_name] = value
            
        # Standardisasi Skala
        kolom_tersedia_scale = [c for c in cols_scale if c in df_live.columns]
        if kolom_tersedia_scale:
            df_live[kolom_tersedia_scale] = scaler.transform(df_live[kolom_tersedia_scale])

        # Prediksi HistGradientBoosting
        prob_fraud = model.predict_proba(df_live)[0][1]
        batas_blokir = threshold * (sensitivitas / 50.0)

        # 5. Layar Keputusan dan Notifikasi
        st.markdown("---")
        st.markdown("### 🖥️ Putusan Core Banking System")
        
        if prob_fraud >= batas_blokir:
            st.error(f"""
            **❌ STATUS REGISTRASI: DITOLAK (DIBLOKIR)**
            
            **🤖 Klasifikasi Sistem Keamanan:**
            * Indikasi: **APPLICATION FRAUD SANGAT KUAT 🚨**
            * Skor Kecurigaan Profil AI: **{prob_fraud*100:.2f}%**
            * Ambang Toleransi Bank Anda: **{batas_blokir*100:.2f}%**
            
            **Tindakan Rekomendasi:** Registrasi dibekukan. Pendaftar wajib diarahkan ke verifikasi Biometrik (Video Call) lanjutan.
            """)
        else:
            st.success(f"""
            **✅ STATUS REGISTRASI: DITERIMA (SAH)**
            
            **🤖 Klasifikasi Sistem Keamanan:**
            * Indikasi: **PROFIL WAJAR DAN AMAN 🛡️**
            * Skor Kecurigaan Profil AI: **{prob_fraud*100:.2f}%**
            * Ambang Toleransi Bank Anda: **{batas_blokir*100:.2f}%**
            
            **Tindakan Rekomendasi:** Lolos screening e-KYC. Rekening digital langsung diterbitkan.
            """)

        # Fitur Peringatan Bias Usia (Fairness Audit)
        if umur >= 56:
            st.warning("""
            ⚠️ **FAIRNESS AUDIT ALERT (PREDICTIVE EQUALITY VIOLATION)** ⚠️
            
            Sistem mendeteksi bahwa pendaftar berada pada kelompok usia rentan **(> 55 Tahun)**. Berdasarkan audit metrik keadilan (*Fairness*) dari dataset NeurIPS, model ini secara sistemik memiliki tingkat kesalahan tolak (*False Positive Rate*) sebesar **13.77%** pada kelompok usia ini, yaitu **8.4x lipat lebih tinggi** dibandingkan pemohon usia muda (18-25 tahun). 
            
            *Rekomendasi Kebijakan:* Jika pendaftar ini ditolak oleh mesin, sangat disarankan untuk melakukan evaluasi manual oleh agen (Human-in-the-Loop) guna mencegah diskriminasi inklusi keuangan.
            """)
