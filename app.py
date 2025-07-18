import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Credit Card Fraud Detection (Random Forest)",
    page_icon="💳",
    layout="wide"
)

# --- Kamus Terjemahan ---
lang_dict = {
    "id": {
        "page_title": "Deteksi Penipuan Kartu Kredit (Random Forest)",
        "app_title": "Sistem Deteksi Penipuan Kartu Kredit 💳",
        "app_subtitle": "Aplikasi ini menggunakan model **Random Forest** untuk mendeteksi transaksi kartu kredit yang berpotensi penipuan, sesuai dengan alur analisis pada notebook referensi.",
        "sidebar_lang_select": "Pilih Bahasa",
        "sidebar_header": "1. Pengaturan Data",
        "local_files_found": "✅ Ditemukan 'fraudTrain.csv' dan 'fraudTest.csv'.",
        "select_file_prompt": "Pilih file data untuk dianalisis:",
        "load_selected_button": "Muat Data Pilihan",
        "loading_file": "Memuat {file}...",
        "single_file_found": "✅ File data lokal '{file}' ditemukan.",
        "load_single_button": "Muat {file}",
        "no_local_files": "Tidak ada file data lokal ('fraudTrain.csv' atau 'fraudTest.csv') yang ditemukan.",
        "upload_prompt": "Atau unggah file data Anda (.csv)",
        "loading_uploaded": "Memuat data yang diunggah...",
        "info_load_data": "Silakan muat data melalui panel di sebelah kiri untuk memulai analisis.",
        "tab_eda": "📊 Analisis Data Eksplorasi (EDA)",
        "tab_training": "⚙️ Pelatihan & Evaluasi Model",
        "tab_analysis": "🔍 Analisis Hasil Deteksi",
        "eda_header": "Ringkasan dan Analisis Data Awal",
        "eda_desc": "Tahap ini menampilkan ringkasan data mentah sebelum diproses lebih lanjut.",
        "df_head": "Lima Baris Pertama Data",
        "df_info": "Informasi Dataset",
        "df_desc": "Statistik Deskriptif",
        "target_dist_header": "Distribusi Kelas Target (is_fraud)",
        "target_dist_desc": "Dataset ini sangat tidak seimbang, di mana jumlah transaksi penipuan jauh lebih sedikit.",
        "legit_transactions": "Jumlah Transaksi Sah (0):",
        "fraud_transactions": "Jumlah Transaksi Penipuan (1):",
        "no_target_column": "Kolom 'is_fraud' tidak ditemukan dalam data ini untuk analisis distribusi.",
        "training_header": "Pelatihan Model Random Forest (Tanpa SMOTE)",
        "training_desc": "Model terbaik berdasarkan F1-Score dari notebook adalah **Random Forest** yang dilatih pada data asli (tidak seimbang). Klik tombol di bawah untuk memulai proses pelatihan.",
        "training_button": "🚀 Latih Model Random Forest",
        "training_error_no_target": "Tidak dapat melatih model karena kolom target 'is_fraud' tidak ada dalam dataset.",
        "training_spinner": "Mohon tunggu, proses pra-pemrosesan dan pelatihan sedang berjalan...",
        "training_success": "✅ Model Random Forest berhasil dilatih dan dievaluasi!",
        "eval_results_header": "Hasil Evaluasi Model",
        "accuracy_label": "Akurasi Model:",
        "classification_report": "Laporan Klasifikasi",
        "confusion_matrix": "Confusion Matrix",
        "cm_title": "Confusion Matrix Hasil Deteksi",
        "cm_xlabel": "Prediksi",
        "cm_ylabel": "Aktual",
        "class_names": ["Bukan Penipuan", "Penipuan"],
        "analysis_header": "Analisis Mendalam pada Hasil Deteksi",
        "analysis_warning": "⚠️ Harap latih model terlebih dahulu di tab 'Pelatihan & Evaluasi Model'.",
        "feature_importance_header": "Tingkat Kepentingan Fitur (Feature Importance)",
        "feature_importance_desc": "Fitur yang paling berpengaruh dalam prediksi model Random Forest.",
        "feature_importance_title": "Feature Importance - Random Forest",
        "predicted_fraud_header": "Analisis Transaksi yang Diprediksi sebagai Penipuan",
        "predicted_fraud_count": "Model memprediksi **{count}** transaksi sebagai penipuan pada data uji.",
        "predicted_fraud_stats": "Statistik deskriptif untuk jumlah transaksi (`amt`) yang diprediksi sebagai penipuan:",
        "actual_col": "is_fraud_aktual",
        "predicted_col": "is_fraud_prediksi",
        "status_legit": "Bukan Fraud",
        "status_fraud": "Fraud"
    },
    "en": {
        "page_title": "Credit Card Fraud Detection (Random Forest)",
        "app_title": "Credit Card Fraud Detection System 💳",
        "app_subtitle": "This application uses a **Random Forest** model to detect potentially fraudulent credit card transactions, following the analysis flow from the reference notebook.",
        "sidebar_lang_select": "Select Language",
        "sidebar_header": "1. Data Settings",
        "local_files_found": "✅ Found 'fraudTrain.csv' and 'fraudTest.csv'.",
        "select_file_prompt": "Select a data file to analyze:",
        "load_selected_button": "Load Selected Data",
        "loading_file": "Loading {file}...",
        "single_file_found": "✅ Local data file '{file}' found.",
        "load_single_button": "Load {file}",
        "no_local_files": "No local data files ('fraudTrain.csv' or 'fraudTest.csv') were found.",
        "upload_prompt": "Or upload your data file (.csv)",
        "loading_uploaded": "Loading uploaded data...",
        "info_load_data": "Please load data via the panel on the left to start the analysis.",
        "tab_eda": "📊 Exploratory Data Analysis (EDA)",
        "tab_training": "⚙️ Model Training & Evaluation",
        "tab_analysis": "🔍 Detection Result Analysis",
        "eda_header": "Initial Data Summary and Analysis",
        "eda_desc": "This stage shows a summary of the raw data before further processing.",
        "df_head": "First Five Rows of Data",
        "df_info": "Dataset Information",
        "df_desc": "Descriptive Statistics",
        "target_dist_header": "Target Class Distribution (is_fraud)",
        "target_dist_desc": "This dataset is highly imbalanced, where the number of fraudulent transactions is much smaller.",
        "legit_transactions": "Number of Legitimate Transactions (0):",
        "fraud_transactions": "Number of Fraudulent Transactions (1):",
        "no_target_column": "The 'is_fraud' column was not found in this data for distribution analysis.",
        "training_header": "Random Forest Model Training (Without SMOTE)",
        "training_desc": "The best model based on F1-Score from the notebook is **Random Forest** trained on the original (imbalanced) data. Click the button below to start the training process.",
        "training_button": "🚀 Train Random Forest Model",
        "training_error_no_target": "Cannot train the model because the target column 'is_fraud' is not in the dataset.",
        "training_spinner": "Please wait, preprocessing and training are in progress...",
        "training_success": "✅ Random Forest model successfully trained and evaluated!",
        "eval_results_header": "Model Evaluation Results",
        "accuracy_label": "Model Accuracy:",
        "classification_report": "Classification Report",
        "confusion_matrix": "Confusion Matrix",
        "cm_title": "Detection Result Confusion Matrix",
        "cm_xlabel": "Predicted",
        "cm_ylabel": "Actual",
        "class_names": ["Not Fraud", "Fraud"],
        "analysis_header": "In-depth Analysis of Detection Results",
        "analysis_warning": "⚠️ Please train the model first in the 'Model Training & Evaluation' tab.",
        "feature_importance_header": "Feature Importance",
        "feature_importance_desc": "The most influential features in the Random Forest model's predictions.",
        "feature_importance_title": "Feature Importance - Random Forest",
        "predicted_fraud_header": "Analysis of Transactions Predicted as Fraud",
        "predicted_fraud_count": "The model predicted **{count}** transactions as fraudulent in the test data.",
        "predicted_fraud_stats": "Descriptive statistics for the transaction amount (`amt`) of predicted frauds:",
        "actual_col": "is_fraud_actual",
        "predicted_col": "is_fraud_predicted",
        "status_legit": "Not Fraud",
        "status_fraud": "Fraud"
    }
}

# --- Inisialisasi Session State ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'lang' not in st.session_state:
    st.session_state.lang = "id"

# --- Sidebar untuk Pilihan Bahasa ---
lang_option = st.sidebar.selectbox(
    label="Pilih Bahasa / Select Language",
    options=["Indonesia", "English"],
    index=0 if st.session_state.lang == "id" else 1
)
st.session_state.lang = "id" if lang_option == "Indonesia" else "en"
texts = lang_dict[st.session_state.lang]

# --- Judul Aplikasi ---
st.title(texts["app_title"])
st.write(texts["app_subtitle"])

# --- Fungsi-Fungsi ---
@st.cache_data
def load_data(file_path=None, uploaded_file=None):
    try:
        if uploaded_file:
            return pd.read_csv(uploaded_file)
        elif file_path and os.path.exists(file_path):
            return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
    return None

def engineer_features(df):
    df_copy = df.copy()
    cols_to_drop = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 'unix_time']
    df_copy = df_copy.drop(columns=cols_to_drop, errors='ignore')
    df_copy['trans_date_trans_time'] = pd.to_datetime(df_copy['trans_date_trans_time'])
    df_copy['hour'] = df_copy['trans_date_trans_time'].dt.hour
    df_copy['day'] = df_copy['trans_date_trans_time'].dt.day
    df_copy['month'] = df_copy['trans_date_trans_time'].dt.month
    df_copy['day_of_week'] = df_copy['trans_date_trans_time'].dt.dayofweek
    df_copy = df_copy.drop(columns=['trans_date_trans_time'])
    return df_copy

# --- Sidebar untuk Pemuatan Data ---
st.sidebar.header(texts["sidebar_header"])
file_options = ['fraudTrain.csv', 'fraudTest.csv']
available_files = [f for f in file_options if os.path.exists(f)]

if len(available_files) == 2:
    st.sidebar.success(texts["local_files_found"])
    chosen_file_option = st.sidebar.selectbox(
        texts["select_file_prompt"],
        available_files
    )
    if st.sidebar.button(texts["load_selected_button"], key="load_selected"):
        with st.spinner(texts["loading_file"].format(file=chosen_file_option)):
            st.session_state.df_raw = load_data(file_path=chosen_file_option)
            if st.session_state.df_raw is not None:
                st.session_state.data_loaded = True
                st.rerun()

elif len(available_files) == 1:
    st.sidebar.success(texts["single_file_found"].format(file=available_files[0]))
    if st.sidebar.button(texts["load_single_button"].format(file=available_files[0]), key="load_single"):
        with st.spinner(texts["loading_file"].format(file=available_files[0])):
            st.session_state.df_raw = load_data(file_path=available_files[0])
            if st.session_state.df_raw is not None:
                st.session_state.data_loaded = True
                st.rerun()
else:
    st.sidebar.warning(texts["no_local_files"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    texts["upload_prompt"],
    type=["csv"]
)

if uploaded_file is not None:
    if not st.session_state.get('upload_processed', False):
        with st.spinner(texts["loading_uploaded"]):
            st.session_state.df_raw = load_data(uploaded_file=uploaded_file)
            if st.session_state.df_raw is not None:
                st.session_state.data_loaded = True
                st.session_state.upload_processed = True
                st.rerun()
else:
    st.session_state.upload_processed = False

# --- Tampilan Utama Aplikasi ---
if not st.session_state.data_loaded:
    st.info(texts["info_load_data"])
else:
    df = st.session_state.df_raw
    X = df.drop('is_fraud', axis=1, errors='ignore')
    y = df['is_fraud'] if 'is_fraud' in df.columns else None

    tab1, tab2, tab3 = st.tabs([
        texts["tab_eda"],
        texts["tab_training"],
        texts["tab_analysis"]
    ])

    with tab1:
        st.header(texts["eda_header"])
        st.markdown(texts["eda_desc"])
        st.subheader(texts["df_head"])
        st.dataframe(df.head())
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(texts["df_info"])
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        with col2:
            st.subheader(texts["df_desc"])
            st.dataframe(df.describe())
        if y is not None:
            st.subheader(texts["target_dist_header"])
            st.write(texts["target_dist_desc"])
            fraud_dist = y.value_counts()
            st.bar_chart(fraud_dist)
            st.write(f"{texts['legit_transactions']} **{fraud_dist.get(0, 0)}**")
            st.write(f"{texts['fraud_transactions']} **{fraud_dist.get(1, 0)}**")
        else:
            st.warning(texts["no_target_column"])

    with tab2:
        st.header(texts["training_header"])
        st.info(texts["training_desc"])
        if y is None:
            st.error(texts["training_error_no_target"])
        elif st.button(texts["training_button"]):
            with st.spinner(texts["training_spinner"]):
                df_processed = engineer_features(df)
                X_processed = df_processed.drop('is_fraud', axis=1)
                y_processed = df_processed['is_fraud']
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
                )
                X_test_original = X_test.copy()
                categorical_cols = X_train.select_dtypes(include=['object']).columns
                numerical_cols = X_train.select_dtypes(include=np.number).columns
                encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    all_categories = pd.concat([X_train[col], X_test[col]]).dropna().unique()
                    le.fit(all_categories)
                    X_train[col] = X_train[col].apply(lambda x: x if x not in le.classes_ else le.transform([x])[0])
                    X_test[col] = X_test[col].apply(lambda x: -1 if x not in le.classes_ else le.transform([x])[0])
                    encoders[col] = le
                scaler = StandardScaler()
                X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
                X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
                model_rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
                model_rf.fit(X_train, y_train)
                y_pred_rf = model_rf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_rf)
                report = classification_report(y_test, y_pred_rf, target_names=texts["class_names"], output_dict=True)
                cm = confusion_matrix(y_test, y_pred_rf)
                st.session_state.model_results = {
                    'model': model_rf, 'report': report, 'cm': cm, 'accuracy': accuracy,
                    'X_test_original': X_test_original, 'y_test': y_test, 'y_pred': y_pred_rf,
                    'feature_names': X_processed.columns
                }
                st.session_state.model_trained = True
            st.success(texts["training_success"])

        if st.session_state.model_trained:
            results = st.session_state.model_results
            st.subheader(texts["eval_results_header"])
            st.write(f"**{texts['accuracy_label']}** `{results['accuracy'] * 100:.2f}%`")
            st.subheader(texts["classification_report"])
            st.dataframe(pd.DataFrame(results['report']).transpose())
            st.subheader(texts["confusion_matrix"])
            fig, ax = plt.subplots()
            sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Oranges',
                        xticklabels=texts["class_names"], yticklabels=texts["class_names"], ax=ax)
            plt.title(texts["cm_title"])
            plt.xlabel(texts["cm_xlabel"])
            plt.ylabel(texts["cm_ylabel"])
            st.pyplot(fig)

    with tab3:
        st.header(texts["analysis_header"])
        if not st.session_state.model_trained:
            st.warning(texts["analysis_warning"])
        else:
            results = st.session_state.model_results
            model = results['model']
            feature_names = results['feature_names']
            st.subheader(texts["feature_importance_header"])
            st.write(texts["feature_importance_desc"])
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
            plt.title(texts["feature_importance_title"])
            st.pyplot(fig)
            st.subheader(texts["predicted_fraud_header"])
            X_test_original = results['X_test_original'].copy()
            X_test_original[texts["actual_col"]] = results['y_test']
            X_test_original[texts["predicted_col"]] = results['y_pred']
            predicted_frauds = X_test_original[X_test_original[texts["predicted_col"]] == 1].copy()
            predicted_frauds.replace({
                texts["actual_col"]: {0: texts["status_legit"], 1: texts["status_fraud"]},
                texts["predicted_col"]: {0: texts["status_legit"], 1: texts["status_fraud"]}
            }, inplace=True)
            st.write(texts["predicted_fraud_count"].format(count=len(predicted_frauds)))
            st.dataframe(predicted_frauds.head(20))
            st.write(texts["predicted_fraud_stats"])
            st.dataframe(predicted_frauds['amt'].describe())
