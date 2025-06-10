import streamlit as st
import pandas as pd
import numpy as np
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
import io

# --- Pengaturan Halaman Utama ---
st.set_page_config(page_title="System Deteksi Penipuan Kartu Kredit", layout="wide")
st.title("System Deteksi Penipuan Kartu Kredit (Credit Card Fraud)")
st.markdown("---")


# Fungsi untuk mendapatkan metrik ringkasan
def get_summary_metrics(models, x_train_res, y_train_res, x_test, y_test, method_name):
    records = []
    
    st.subheader(f"Hasil Evaluasi untuk Fitur dari {method_name}")
    
    for idx, (name, clf) in enumerate(models, start=1):
        with st.expander(f"Hasil untuk Model: **{name}**"):
            
            with st.spinner(f"Melatih dan mengevaluasi {name} pada fitur {method_name}..."):
                clf.fit(x_train_res, y_train_res)
                cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                cv_score = cross_val_score(clf, x_train_res, y_train_res, cv=cv, scoring='roc_auc').mean()
                
                y_pred = clf.predict(x_test)
                y_pred_proba = clf.predict_proba(x_test)[:,1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                
                # Menampilkan metrik
                st.write(f"**Cross Validation ROC AUC:** `{cv_score:.2%}`")
                st.write(f"**Test ROC AUC:** `{roc_auc:.2%}`")
                st.write(f"**F1 Score (Fraud):** `{f1:.2%}`")

                # Menampilkan plot
                fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Ukuran gambar bisa disesuaikan
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
                axes[0].set_title('Confusion Matrix', fontsize=20)
                
                # ROC Curve
                RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=axes[1])
                axes[1].set_title('Kurva ROC', fontsize=20)

                st.pyplot(fig)

                records.append({
                    "No.": idx,
                    "ML Algorithm": name,
                    "Cross Validation ROC AUC": f"{cv_score:.2%}",
                    "Test ROC AUC": f"{roc_auc:.2%}",
                    "F1 Score (Fraud)": f"{f1:.2%}"
                })

    return pd.DataFrame(records)

# --- Tampilan Utama ---
st.header("Unggah Dataset")
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diunggah!")
    st.write("Beberapa baris pertama dari dataset:")
    st.dataframe(data.head())
            
    if st.button("Mulai Proses Analisis Lengkap", use_container_width=True, type="primary"):
        # === PERSIAPAN DATA ===
        st.header("1. Persiapan Data")
        st.write("**Isi Dataset:**")
        st.dataframe(data.describe())
        
        st.write("**Informasi Dataset:**")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # === VISUALISASI DATA ===
        st.header("2. Visualisasi Distribusi Kelas")
        fraud_count = data['Class'].value_counts()
        
        fig_dist, axes_dist = plt.subplots(1, 2, figsize=(10, 10))
        
        # Pie Chart
        axes_dist[0].pie(fraud_count, labels=['Normal', 'Fraud'], autopct='%1.2f%%', startangle=90)
        axes_dist[0].set_title('Persentase: Fraud vs. No Fraud', fontsize=20)
        
        # Count Plot
        sns.countplot(x='Class', data=data, ax=axes_dist[1], palette="pastel")
        axes_dist[1].set_title('Jumlah Kasus Fraud (Penipuan)', fontsize=20)
        axes_dist[1].set_xticklabels(['Normal (0)', 'Fraud (1)'])
        
        st.pyplot(fig_dist)
        st.markdown("---")

        # === OUTLINE BAGIAN 3: PEMILIHAN FITUR (Tidak ditampilkan, hanya proses) ===
        
        # === OUTLINE BAGIAN 4 & 5: PEMODELAN & EVALUASI ===
        st.header("3. Evaluasi Model")
        
        # Model 1: Fitur dari Analisis Plot Korelasi
        model1_data = data[['V3','V4','V7','V10','V11','V12','V14','V16','V17','Class']].copy()
        X1 = model1_data.drop('Class', axis=1)
        y1 = model1_data['Class']
        x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.20, random_state=2, stratify=y1)
        
        # Model 2: Seleksi Fitur dari Skor ANOVA
        X_full = data.loc[:, :'Amount']
        y_full = data.loc[:, 'Class']
        featurescore = pd.DataFrame(data=SelectKBest(score_func=f_classif, k='all').fit(X_full, y_full).scores_, index=X_full.columns, columns=["ANOVA Score"]).sort_values(by="ANOVA Score", ascending=False)
        model2_data = data.drop(columns=list(featurescore.index[20:]))
        X2 = model2_data.drop('Class', axis=1)
        y2 = model2_data['Class']
        x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.20, random_state=2, stratify=y2)
        
        # Pipeline Balancer
        pipeline_balancer = Pipeline(steps=[('u', RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
                                            ('o', SMOTE(sampling_strategy=0.5, random_state=42))])
        
        x_train1_res, y_train1_res = pipeline_balancer.fit_resample(x_train1, y_train1)
        x_train2_res, y_train2_res = pipeline_balancer.fit_resample(x_train2, y_train2)
        
        models = [
            ("Logistic Regression", LogisticRegression(random_state=0, C=10, penalty='l2', max_iter=1000)),
            ("K-Nearest Neighbors", KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1)),
            ("Support Vector Classifier", SVC(kernel='linear', C=0.1, probability=True, random_state=42)),
            ("Decision Tree Classifier", DecisionTreeClassifier(random_state=1000, max_depth=4, min_samples_leaf=1)),
            ("Random Forest Classifier", RandomForestClassifier(max_depth=4, random_state=0))
        ]

        # Menjalankan evaluasi untuk kedua metode dan menampilkan hasilnya
        results_corr = get_summary_metrics(models, x_train1_res, y_train1_res, x_test1, y_test1, "Plot Korelasi")
        results_anova = get_summary_metrics(models, x_train2_res, y_train2_res, x_test2, y_test2, "Skor ANOVA")
        
        st.markdown("---")

        # === OUTLINE BAGIAN 6: RANGKUMAN DAN KESIMPULAN ===
        st.header("4. Rangkuman Hasil & Kesimpulan")
        st.subheader("Tabel Rangkuman Hasil Akhir")
        st.write("**Hasil Metode Fitur Korelasi**")
        st.dataframe(results_corr)
        st.write("**Hasil Metode Fitur ANOVA**")
        st.dataframe(results_anova)
        
        # Gabungkan untuk kesimpulan
        results_corr['Metode Fitur'] = 'Plot Korelasi'
        results_anova['Metode Fitur'] = 'Skor ANOVA'
        all_results = pd.concat([results_corr, results_anova], ignore_index=True)
        all_results['F1_Score_Num'] = all_results['F1 Score (Fraud)'].str.replace('%','').astype(float)
        
        best_overall = all_results.loc[all_results['F1_Score_Num'].idxmax()]
        
        st.subheader("Kesimpulan")
        st.code(
            f'{"="*25} KESIMPULAN {"="*25}\n\n'
            f"Berdasarkan evaluasi komparatif dari metrik F1-Score, model dengan kinerja terbaik adalah '{best_overall['ML Algorithm']}'.\n"
            f"Performa optimal ini dicapai saat model dilatih menggunakan set fitur dari '{best_overall['Metode Fitur']}'.\n"
            f"Model tersebut berhasil mencapai F1-Score sebesar {best_overall['F1 Score (Fraud)']} untuk mendeteksi kasus penipuan (fraud).\n\n"
            f"Analisis:\n"
            f"F1-Score menjadi metrik acuan utama karena mampu memberikan penilaian yang seimbang antara Precision dan Recall.\n"
            f"Dalam konteks deteksi penipuan, hal ini sangat krusial. Nilai F1-Score yang tinggi pada model '{best_overall['ML Algorithm']}'\n"
            f"mengindikasikan bahwa model tersebut tidak hanya efektif dalam memaksimalkan deteksi kasus penipuan (recall tinggi),\n"
            f"tetapi juga mampu meminimalkan jumlah transaksi sah yang keliru diklasifikasikan sebagai penipuan (precision tinggi).\n"
            f'{"="*62}',
            language=None
        )