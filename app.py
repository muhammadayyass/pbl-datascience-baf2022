import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Konfigurasi Halaman (Clean & Corporate Layout)
st.set_page_config(page_title="Digital Onboarding Risk Assessment", layout="wide")

# Custom CSS untuk tampilan minimalis dan enterprise
st.markdown("""
    <style>
    .main-header {font-size: 28px; font-weight: 600; color: #1e293b; margin-bottom: 0px; padding-bottom: 0px;}
    .sub-header {font-size: 16px; color: #64748b; margin-top: 0px; margin-bottom: 25px;}
    .stButton>button {background-color: #0f172a; color: white; border-radius: 4px; font-weight: 500;}
    .stButton>button:hover {background-color: #334155; border: 1px solid #334155;}
    .metric-card {background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Digital Onboarding Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Application Fraud Detection System | NeurIPS 2022 Benchmark</p>', unsafe_allow_html=True)
st.markdown("---")

# 2. Memuat Artifak Model
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
    st.error(f"Sistem tidak dapat memuat artifak model. Detail teknis: {e}")
    st.stop()

# 3. Layout Antarmuka Terstruktur
col_input, col_result = st.columns([1.3, 1])

with col_input:
    st.subheader("Data Pendaftar")
    
    with st.container(border=True):
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            umur = st.number_input("Usia Pendaftar", min_value=18, max_value=90, value=30)
            pendapatan = st.slider("Desil Pendapatan", 0.1, 0.9, 0.5, step=0.1)
        with row1_col2:
            kredit = st.number_input("Skor Risiko Biro Kredit", min_value=0, max_value=500, value=150)
            velocity = st.number_input("Kecepatan Pengisian (Velocity 6h)", min_value=0, max_value=20000, value=5000)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            perumahan = st.selectbox("Kode Status Hunian", ['BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG'])
            os_hp = st.selectbox("Sistem Operasi Perangkat", ['windows', 'linux', 'macintosh', 'x11', 'other'])
        with row2_col2:
            pekerjaan = st.selectbox("Kode Status Pekerjaan", ['CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG'])
            sumber = st.selectbox("Kanal Pendaftaran", ['INTERNET', 'TELEAPP'])

with col_result:
    st.subheader("Pengaturan & Hasil Evaluasi")
    
    with st.expander("⚙️ Parameter Regulasi Risiko", expanded=True):
        st.caption(f"Batas probabilitas dasar model disetel pada {threshold*100:.2f}% untuk menjaga rasio False Positive Rate maksimal 5%.")
        sensitivitas = st.slider("Penyesuaian Toleransi Risiko (%)", 1, 99, 50, help="< 50: Lebih konservatif/ketat. > 50: Lebih moderat/longgar.")
    
    tombol_evaluasi = st.button("Jalankan Asesmen Risiko", use_container_width=True)

# 4. Mesin Pemroses Utama
if tombol_evaluasi:
    with st.spinner('Memproses data melalui Decision Engine...'):
        
        # Keamanan Data Type Float 
        df_live = pd.DataFrame(0.0, index=[0], columns=cols_fs)

        # Pengisian Imputasi Median
        for col in cols_fs:
            if col in template_mentah:
                df_live.at[0, col] = float(template_mentah[col])
        
        # Penimpaan Fitur Utama
        df_live.at[0, 'customer_age'] = float(umur)
        df_live.at[0, 'income'] = float(pendapatan)
        df_live.at[0, 'credit_risk_score'] = float(kredit)
        df_live.at[0, 'velocity_6h'] = float(velocity)
        
        if 'velocity_6h_log' in cols_fs:
            df_live.at[0, 'velocity_6h_log'] = np.log1p(velocity)
            
        # One-Hot Encoding Dinamis
        dummy_mappings = {
            f"housing_status_{perumahan}": 1.0,
            f"employment_status_{pekerjaan}": 1.0,
            f"device_os_{os_hp}": 1.0,
            f"source_{sumber}": 1.0
        }
        for feature_name, value in dummy_mappings.items():
            if feature_name in cols_fs:
                df_live.at[0, feature_name] = value
            
        # Standardisasi Data
        kolom_tersedia_scale = [c for c in cols_scale if c in df_live.columns]
        if kolom_tersedia_scale:
            df_live[kolom_tersedia_scale] = scaler.transform(df_live[kolom_tersedia_scale])

        # Eksekusi Prediksi
        prob_fraud = model.predict_proba(df_live)[0][1]
        batas_blokir = threshold * (sensitivitas / 50.0)

        # 5. Tampilan Hasil Keputusan (Corporate Style)
        st.markdown("---")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Probabilitas Fraud", value=f"{prob_fraud*100:.2f}%", delta=f"Batas: {batas_blokir*100:.2f}%", delta_color="inverse")
            
        with res_col2:
            if prob_fraud >= batas_blokir:
                st.error("**KEPUTUSAN SISTEM: PENOLAKAN OTOMATIS (REJECT)**")
                st.markdown("""
                **Deskripsi Risiko:**
                Pendaftar menunjukkan indikasi profil *Application Fraud* atau Pencurian Identitas tingkat tinggi yang melampaui parameter toleransi risiko bank.
                
                **Rekomendasi Tindakan:**
                Tangguhkan proses *onboarding*. Alihkan proses ke investigasi unit anti-fraud atau jadwalkan prosedur verifikasi biometrik tingkat lanjut.
                """)
            else:
                st.success("**KEPUTUSAN SISTEM: DISETUJUI (APPROVE)**")
                st.markdown("""
                **Deskripsi Risiko:**
                Profil pendaftar berada dalam batas kewajaran dan tidak menunjukkan anomali risiko tingkat tinggi.
                
                **Rekomendasi Tindakan:**
                Lanjutkan proses *digital onboarding* ke tahap pembuatan rekening sistem inti (*Core Banking*).
                """)

        # Peringatan Bias (Compliance Notice)
        if umur >= 56:
            st.warning("""
            **Pemberitahuan Kepatuhan (Compliance Notice): Potensi Disparitas Demografis**
            
            Sistem mengidentifikasi pemohon berada pada kelompok rentang usia senior (> 55 Tahun). Metrik *Predictive Equality* historis menunjukkan bahwa model memiliki *False Positive Rate* (Tingkat Penolakan Keliru) sebesar 13.77% pada segmen usia ini. 
            
            *Pedoman Tata Kelola:* Keputusan penolakan otomatis pada profil ini wajib melewati proses eskalasi peninjauan manual (Human-in-the-Loop) untuk memitigasi risiko pelanggaran regulasi inklusi keuangan.
            """)
