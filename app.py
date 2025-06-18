import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Penipuan Kartu Kredit",
    page_icon="💳",
    layout="wide"
)

# Inisialisasi session state di awal untuk menyimpan status dan data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Judul Aplikasi ---
st.title("Aplikasi Deteksi Penipuan Kartu Kredit 💳")
st.write("Aplikasi ini menggunakan model Machine Learning (LightGBM) untuk mendeteksi transaksi kartu kredit yang berpotensi penipuan.")

# --- Fungsi Cache untuk Pra-pemrosesan ---
@st.cache_data
def preprocess_data(df):
    try:
        cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num', 'trans_date_trans_time']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df.drop(columns=existing_cols_to_drop, inplace=True)
        
        categorical_features = ["merchant", "category", "gender", "job"]
        for feature in categorical_features:
            if feature in df.columns:
                encoder = LabelEncoder()
                df[feature] = encoder.fit_transform(df[feature])
        
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
        return None

# --- SIDEBAR & LOGIKA PEMUATAN DATA ---
st.sidebar.header("Pengaturan Data")
file_path = 'fraudTrain.csv'

if os.path.exists(file_path):
    st.sidebar.success(f"✅ File '{file_path}' ditemukan.")
    if st.sidebar.button("Muat & Proses Data Lokal", key="load_local"):
        with st.spinner("Memuat dan memproses data..."):
            df_raw = pd.read_csv(file_path)
            st.session_state.df = preprocess_data(df_raw)
            st.session_state.data_loaded = True
            st.rerun()

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    f"Jika '{file_path}' tidak ada, unggah data Anda (.csv)",
    type=["csv"]
)

if uploaded_file is not None and not st.session_state.get('data_loaded', False):
    with st.spinner("Memuat dan memproses data yang diunggah..."):
        df_raw = pd.read_csv(uploaded_file)
        st.session_state.df = preprocess_data(df_raw)
        st.session_state.data_loaded = True
        st.rerun()

if st.session_state.data_loaded and st.session_state.df is not None:
    
    df = st.session_state.df
    
    features_anova = ['amt', 'category', 'gender', 'unix_time', 'city_pop', 'lat', 'merch_lat', 'merch_long', 'long', 'merchant', 'job']
    target_variable = 'is_fraud'
    
    existing_features = [f for f in features_anova if f in df.columns]
    missing_features = [f for f in features_anova if f not in df.columns]

    if target_variable not in df.columns or not existing_features:
         st.error(f"Dataset tidak valid. Kolom target '{target_variable}' atau kolom fitur tidak ditemukan.")
    else:
        if missing_features:
            st.warning(f"Fitur berikut tidak ditemukan dan akan diabaikan: {missing_features}")

        X = df[existing_features]
        y = df[target_variable]

        tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Data", "⚙️ Pelatihan & Evaluasi Model", "🔍 Hasil Deteksi"])

        with tab1:
            st.header("Ringkasan Dataset")
            st.dataframe(df.head())
            
            st.subheader("Distribusi Kelas Target (is_fraud)")
            fraud_dist = y.value_counts()
            st.bar_chart(fraud_dist)
            st.write(f"Jumlah Transaksi Sah: **{fraud_dist.get(0, 0)}**")
            st.write(f"Jumlah Transaksi Penipuan: **{fraud_dist.get(1, 0)}**")

        with tab2:
            st.header("Pelatihan Model Deteksi Penipuan")
            st.info("Klik tombol di bawah untuk memulai proses pelatihan. Proses ini mungkin memakan waktu beberapa menit.")

            if st.button("🚀 Mulai Pelatihan dan Evaluasi"):
                with st.spinner("Mohon tunggu, proses pelatihan sedang berjalan..."):
                    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, stratify=y)
                    
                    over = SMOTE(sampling_strategy=0.5, random_state=42)
                    under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
                    pipeline = Pipeline(steps=[('u', under), ('o', over)])
                    x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)
                    
                    lgbm_clf = lgb.LGBMClassifier(random_state=42)
                    lgbm_clf.fit(x_train_res, y_train_res)
                    
                    predictions = lgbm_clf.predict(x_test)
                    
                    st.session_state['model'] = lgbm_clf
                    st.session_state['x_test'] = x_test
                    st.session_state['y_test'] = y_test
                    st.session_state['predictions'] = predictions
                    st.session_state['model_trained'] = True

                st.success("✅ Model berhasil dilatih dan dievaluasi!")
                
                st.subheader("Laporan Klasifikasi")
                report = classification_report(y_test, predictions, target_names=['Bukan Penipuan', 'Penipuan'], output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Bukan Penipuan', 'Penipuan'], 
                            yticklabels=['Bukan Penipuan', 'Penipuan'],
                            ax=ax)
                plt.title('Confusion Matrix Hasil Deteksi')
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot(fig)

        with tab3:
            st.header("Hasil Deteksi pada Data Uji")
            if 'model_trained' not in st.session_state or not st.session_state.get('model_trained', False):
                st.warning("⚠️ Harap latih model terlebih dahulu di tab 'Pelatihan & Evaluasi Model'.")
            else:
                x_test = st.session_state['x_test']
                y_test = st.session_state['y_test']
                predictions = st.session_state['predictions']
                
                detection_results = pd.DataFrame({
                    'Data Asli (Indeks)': x_test.index,
                    'Jumlah Transaksi ($)': x_test['amt'],
                    'Status Aktual': y_test,
                    'Status Prediksi': predictions
                }).reset_index(drop=True)

                detection_results['Hasil Deteksi'] = np.where(detection_results['Status Aktual'] == detection_results['Status Prediksi'], 'Benar ✅', 'Salah ❌')
                detection_results.replace({'Status Aktual': {0: 'Bukan Fraud', 1: 'Fraud'}, 'Status Prediksi': {0: 'Bukan Fraud', 1: 'Fraud'}}, inplace=True)
                
                filter_option = st.selectbox(
                    "Tampilkan contoh untuk:",
                    ('Semua Hasil', 'Deteksi Fraud (Aktual)', 'Deteksi Bukan Fraud (Aktual)', 'Prediksi Salah')
                )
                filtered_df = pd.DataFrame()
                
                # 2. Filter data berdasarkan pilihan, tapi JANGAN tampilkan dulu
                if filter_option == 'Deteksi Fraud (Aktual)':
                    st.subheader("Contoh Transaksi yang Sebenarnya adalah Penipuan")
                    filtered_df = detection_results[detection_results['Status Aktual'] == 'Fraud']
                elif filter_option == 'Deteksi Bukan Fraud (Aktual)':
                    st.subheader("Contoh Transaksi yang Sebenarnya Bukan Penipuan")
                    filtered_df = detection_results[detection_results['Status Aktual'] == 'Bukan Fraud']
                elif filter_option == 'Prediksi Salah':
                    st.subheader("Contoh di Mana Prediksi Model Salah")
                    filtered_df = detection_results[detection_results['Hasil Deteksi'] == 'Salah ❌']
                else: # 'Semua Hasil'
                    st.subheader("Seluruh Hasil Deteksi pada Data Uji")
                    filtered_df = detection_results

                # 3. Cek ukuran DataFrame yang sudah difilter, batasi jika perlu
                if len(filtered_df) > 1000:
                    st.info(f"Menampilkan 1.000 baris pertama dari total {len(filtered_df)} baris untuk menjaga performa aplikasi.")
                    display_df = filtered_df.head(1000)
                else:
                    display_df = filtered_df
                
                # 4. Tampilkan DataFrame yang ukurannya sudah aman dengan format
                st.dataframe(display_df.style.format({'Jumlah Transaksi ($)': '${:,.2f}'}))

else:
    st.info("Silakan muat dan proses data melalui panel di sebelah kiri untuk memulai.")