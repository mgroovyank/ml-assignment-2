import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
)

st.set_page_config(page_title="Mobile Price Evaluator", layout="wide")
st.title("📱 Mobile Price Classification Model Evaluation Dashboard")
st.write("Upload your validation data to evaluate and compare pre-trained Machine Learning models instantly without retraining.")

# Sidebar Controls
st.sidebar.header("📁 Step 1: Upload Evaluation Data")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Step 2: Choose Classifier")
model_option = st.sidebar.selectbox(
    "Select Model Architecture",
    ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Gaussian Naive Bayes", "Support Vector Machine")
)

model_mapping = {
    "Logistic Regression": "model/Logistic_Regression_model.joblib",
    "Decision Tree": "model/Decision_Tree_model.joblib",
    "K-Nearest Neighbors": "model/KNN_model.joblib",
    "Gaussian Naive Bayes": "model/Naive_Bayes_model.joblib",
    "Support Vector Machine": "model/SVM_model.joblib"
}

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        
        if 'price_range' not in test_df.columns:
            st.error("❌ Structure Error: The target column 'price_range' must be explicitly present in the uploaded test file.")
        else:
            st.success(f"✅ Successfully loaded test split for analysis matrix calculations.")
            
            X_test_raw = test_df.drop(columns=['price_range'])
            y_test = test_df['price_range']
            
            # Apply identical training feature engineering
            X_test_processed = X_test_raw.copy()
            X_test_processed['ram_battery_interaction'] = X_test_processed['ram'] * X_test_processed['battery_power']
            X_test_processed['total_megapixels'] = X_test_processed['pc'] + X_test_processed['fc']
            
            cols_to_drop = ['fc', 'pc', 'clock_speed', 'mobile_wt', 'wifi']
            X_test_processed = X_test_processed.drop(columns=[col for col in cols_to_drop if col in X_test_processed.columns])
            
            # Standardize vectors
            scaler = joblib.load('model/scaler.joblib')
            X_inference = scaler.transform(X_test_processed)
            
            # Execute inference
            chosen_model_path = model_mapping[model_option]
            model = joblib.load(chosen_model_path)
            
            y_pred = model.predict(X_inference)
            y_prob = model.predict_proba(X_inference)
            
            # Visual Layout Grid
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            
            st.subheader(f"📊 {model_option} Operational Metrics Breakdown")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{acc * 100:.2f}%")
            m2.metric("AUC Score", f"{auc:.4f}")
            m3.metric("Precision", f"{prec:.4f}")
            m4.metric("Recall", f"{rec:.4f}")
            m5.metric("F1 Score", f"{f1:.4f}")
            m6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)
                
            with col2:
                st.subheader("🎯 Visual Confusion Matrix Heatmap")
                cm_array = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    cm_array, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low', 'Med', 'High', 'V.High'],
                    yticklabels=['Low', 'Med', 'High', 'V.High'], ax=ax
                )
                plt.ylabel('Actual Class Label')
                plt.xlabel('Predicted Class Label')
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Failed to process deployment metrics. Error: {e}")
else:
    st.info("ℹ️ Welcome! Please drop your exported `test_data.csv` file into the sidebar uploader area on the left.")