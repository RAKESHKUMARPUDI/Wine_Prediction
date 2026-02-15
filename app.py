import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Wine Quality AI", layout="wide")

# Load Scaler
scaler = pickle.load(open("model/scaler.pkl", "rb"))

# --- SIDEBAR ---
st.sidebar.header("1. Model Selection")
model_options = ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
selected_model_name = st.sidebar.selectbox("Select Model", model_options)
model = pickle.load(open(f"model/{selected_model_name}.pkl", "rb"))

st.sidebar.header("2. Data Source")
data_source = st.sidebar.radio("Choose Test Data:", ("Use Default Test File", "Upload New CSV"))

test_df = None

if data_source == "Use Default Test File":
    try:
        test_df = pd.read_csv("test_data.csv")
        st.sidebar.success("Loaded default test_data.csv")
    except FileNotFoundError:
        st.sidebar.error("test_data.csv not found in repository!")

else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

# --- MAIN PAGE LOGIC ---
if test_df is not None:
    if 'target' in test_df.columns:
        X_test_new = test_df.drop('target', axis=1)
        y_test_new = test_df['target']
        
        # Preprocess & Predict
        X_scaled = scaler.transform(X_test_new)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Display Metrics
        st.header(f"Results for {selected_model_name.replace('_', ' ')}")
        cols = st.columns(6)
        metrics = [
            ("Accuracy", accuracy_score(y_test_new, y_pred)),
            ("AUC Score", roc_auc_score(y_test_new, y_proba)),
            ("Precision", precision_score(y_test_new, y_pred)),
            ("Recall", recall_score(y_test_new, y_pred)),
            ("F1 Score", f1_score(y_test_new, y_pred)),
            ("MCC", matthews_corrcoef(y_test_new, y_pred))
        ]
        for i, (label, val) in enumerate(metrics):
            cols[i].metric(label, f"{val:.3f}")

        # Confusion Matrix & Report
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test_new, y_pred), annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)
            
        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test_new, y_pred, output_dict=True)
            st.table(pd.DataFrame(report).transpose())
    else:
        st.error("CSV must contain a 'target' column.")
else:
    st.info("Select a data source in the sidebar to begin.")