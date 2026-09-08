# Bank Account Fraud (BAF) Detection & Fairness Audit
**End-to-End Machine Learning Pipeline for Digital Onboarding Risk Assessment**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pbl-datascience-baf2022-8lmr9vkarc7mfckfaea8hy.streamlit.app/)

## Executive Summary
Proyek ini mengimplementasikan model *Machine Learning* tingkat lanjut untuk mendeteksi *Application Fraud* pada proses pembukaan rekening bank digital (Digital Onboarding). Dibangun menggunakan dataset **Bank Account Fraud (BAF) NeurIPS 2022**, sistem ini tidak hanya berfokus pada akurasi prediktif, tetapi juga pada kepatuhan regulasi melalui audit keadilan algoritma (*Fairness Audit*).

## Business Problem & Methodology
Dalam ekosistem sistem pembayaran digital, penipuan identitas memiliki dampak finansial yang signifikan. Namun, penolakan keliru (*False Positive*) terhadap nasabah sah dapat merusak reputasi institusi dan melanggar prinsip inklusi keuangan.

Proyek ini menggunakan protokol evaluasi ketat yang diadopsi dari standar riset industri:
- **Temporal Split Validation:** Model dilatih pada data historis (Bulan 0-5) dan diuji pada data masa depan (Bulan 6-7) untuk mengukur ketahanan terhadap evolusi pola kejahatan (*Concept Drift*).
- **Business-Centric Metric:** Evaluasi tidak menggunakan akurasi biasa, melainkan **Recall @ 5% FPR** guna memastikan batas toleransi salah tolak nasabah sah maksimal berada di angka 5%.
- **Predictive Equality (Fairness):** Mengukur disparitas tingkat penolakan antar kelompok demografi untuk mencegah diskriminasi sistemik.

## Technical Stack & Performance
- **Algoritma:** `HistGradientBoostingClassifier` (Dipilih karena efisiensi komputasi pada jutaan baris data dan performa superior dibanding Random Forest).
- **Skor AUC:** ~0.89
- **Recall @ 5% FPR:** ~53.54%
- **Fairness Insight:** Analisis mendalam menemukan disparitas *False Positive Rate* sebesar 8.4x lipat pada kelompok pemohon lanjut usia (56+), memicu implementasi fitur peringatan *Human-in-the-Loop* pada antarmuka produksi.

## Repository Structure
- `app.py`: *Source code* untuk antarmuka web Streamlit.
- `Base.csv`: (Tidak disertakan dalam repositori karena ukuran file 1 Juta baris, silakan merujuk ke sumber asli Kaggle).
- `*.pkl`: Artifak model, *scaler*, dan *threshold* hasil pelatihan.
- `requirements.txt`: Dependensi sistem.

## Author
**Kanaya Salsabila Setiawan**  
*Data Analyst Project Portfolio*
