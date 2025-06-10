# app.py (Resized Pie Chart)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# --- Pengaturan Halaman Utama ---
st.set_page_config(page_title="Analisis Fraud Deteksi Otomatis", layout="wide")

# --- Fungsi-Fungsi Bantuan ---
@st.cache_data
def run_evaluation(data):
    st.info("Memulai proses... Ini mungkin memakan waktu beberapa saat tergantung pada ukuran dataset.")
    
    # === Analisis Awal Dataset yang Diunggah ===
    st.subheader("Analisis Awal Dataset")
    fraud_count = data['Class'].value_counts()
    
    # === UKURAN GAMBAR DIUBAH DI SINI ===
    fig_pie, ax_pie = plt.subplots(figsize=(4, 3)) # Diubah dari (5, 5) menjadi (4, 3)
    
    ax_pie.pie(fraud_count, labels=['Normal', 'Fraud'], autopct='%1.2f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    ax_pie.set_title('Distribusi Kelas pada Dataset yang Diunggah')
    st.pyplot(fig_pie)
    
    # === TAHAP 1: PEMILIHAN FITUR (ANOVA) ===
    progress_bar = st.progress(0, text="Tahap 1: Pemilihan Fitur...")
    
    X_full = data.loc[:, :'Amount']
    y_full = data.loc[:, 'Class']

    with st.spinner('Memilih fitur terbaik...'):
        featurescore = pd.DataFrame(
            data=SelectKBest(score_func=f_classif, k='all').fit(X_full, y_full).scores_,
            index=X_full.columns,
            columns=["ANOVA Score"]
        ).sort_values(by="ANOVA Score", ascending=False)
        
        model_data = data.drop(columns=list(featurescore.index[20:]))
        X = model_data.drop('Class', axis=1)
        y = model_data['Class']
    
    st.write("✅ 20 Fitur teratas telah dipilih berdasarkan skor ANOVA.")
    progress_bar.progress(10, text="Tahap 1 Selesai.")

    # === TAHAP 2: PEMISAHAN DATA (MENCEGAH DATA LEAKAGE) ===
    st.write("Memisahkan dataset menjadi data latih (80%) dan data uji (20%).")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    progress_bar.progress(20, text="Tahap 2 Selesai.")

    # === TAHAP 3: BALANCING DATA LATIH (HANYA PADA DATA LATIH) ===
    st.write("Melakukan balancing pada data latih menggunakan SMOTE dan Undersampling.")
    with st.spinner('Menyeimbangkan data latih...'):
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
        pipeline_balancer = Pipeline(steps=[('u', under), ('o', over)])
        x_train_res, y_train_res = pipeline_balancer.fit_resample(x_train, y_train)
    progress_bar.progress(40, text="Tahap 3 Selesai.")

    # === TAHAP 4: PELATIHAN & EVALUASI MODEL ===
    st.write("Melatih dan mengevaluasi 5 model klasifikasi...")
    
    models = [
        ("Logistic Regression", LogisticRegression(random_state=0, C=10, penalty='l2', max_iter=1000)),
        ("K-Nearest Neighbors", KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1)),
        ("Support Vector Classifier", SVC(kernel='linear', C=0.1, probability=True, random_state=42)),
        ("Decision Tree Classifier", DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1)),
        ("Random Forest Classifier", RandomForestClassifier(max_depth=4, random_state=0))
    ]

    results = []
    
    for i, (name, model) in enumerate(models):
        with st.expander(f"Hasil untuk Model: **{name}**"):
            st.write(f"Memproses {name}...")
            
            with st.spinner(f'Melatih dan mengevaluasi {name}...'):
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                cv_score = cross_val_score(model, x_train_res, y_train_res, cv=cv, scoring='roc_auc').mean()
                
                model.fit(x_train_res, y_train_res)
                
                y_pred = model.predict(x_test)
                y_pred_proba = model.predict_proba(x_test)[:, 1]
                
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                
                results.append({
                    "ML Algorithm": name,
                    "Cross Validation ROC AUC": cv_score,
                    "Test ROC AUC": roc_auc,
                    "F1 Score (Fraud)": f1
                })

                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.metric(label="Cross-Validation Score (Mean)", value=f"{cv_score:.2%}")
                    st.metric(label="ROC AUC Score (Test)", value=f"{roc_auc:.2%}")
                    st.metric(label="F1 Score (Test)", value=f"{f1:.2%}")
                
                with col2:
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
                    ax_cm.set_title('Confusion Matrix')
                    ax_cm.set_xlabel('Prediksi')
                    ax_cm.set_ylabel('Aktual')
                    st.pyplot(fig_cm)
                
                with col3:
                    fig_roc, ax_roc = plt.subplots()
                    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax_roc)
                    ax_roc.set_title('Kurva ROC (Test Data)')
                    st.pyplot(fig_roc)
        
        progress_bar.progress(40 + ((i + 1) * 12), text=f"Model {name} selesai dievaluasi.")

    return pd.DataFrame(results)

# --- Tampilan Utama Aplikasi ---
st.title("Aplikasi Analisis & Deteksi Penipuan Kartu Kredit")
st.markdown("---")

st.header("1. Unggah Dataset Anda")
st.write("Unggah file CSV dengan format yang sama seperti dataset 'creditcard.csv'. Aplikasi akan secara otomatis melakukan seluruh proses analisis.")

uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset berhasil diunggah!")
        st.write("Beberapa baris pertama dari dataset Anda:")
        st.dataframe(data.head())
        
        if st.button("Mulai Proses dan Evaluasi Model", use_container_width=True, type="primary"):
            st.markdown("---")
            st.header("2. Hasil Evaluasi Model")
            
            results_df = run_evaluation(data)
            
            st.markdown("---")
            st.header("3. Rangkuman Performa Seluruh Model")
            
            results_df_display = results_df.copy()
            results_df_display['Cross Validation ROC AUC'] = results_df_display['Cross Validation ROC AUC'].map('{:.2%}'.format)
            results_df_display['Test ROC AUC'] = results_df_display['Test ROC AUC'].map('{:.2%}'.format)
            results_df_display['F1 Score (Fraud)'] = results_df_display['F1 Score (Fraud)'].map('{:.2%}'.format)
            
            st.dataframe(results_df_display.set_index('ML Algorithm'))
            
            st.markdown("---")
            st.header("4. Kesimpulan")
            
            best_model = results_df.loc[results_df['F1 Score (Fraud)'].idxmax()]
            best_model_name = best_model['ML Algorithm']
            best_f1_score = best_model['F1 Score (Fraud)']
            
            success_message = (
                f"**Model Terbaik Berdasarkan F1-Score adalah: {best_model_name}** "
                f"dengan **F1-Score sebesar {best_f1_score:.2%}**."
            )
            st.success(success_message)

    except Exception as e:
        st.error(f"Terjadi error saat memproses file: {e}")

else:
    st.info("Menunggu file CSV diunggah untuk memulai analisis.")