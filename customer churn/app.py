# Databricks notebook source
import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Churn Intelligence Portal", layout="wide")

# 2. Load Data and Model
@st.cache_resource
def load_assets():
    df = pd.read_csv('Final_Customer_Churn_Priority_List.csv')
    with open('churn_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    # Re-creating the explainer for SHAP
    explainer = shap.TreeExplainer(model)
    return df, model, explainer

df, model, explainer = load_assets()

# 3. Sidebar Header
st.sidebar.title("🛠️ Control Panel")
st.sidebar.markdown("Filter customers to investigate risk levels and triggers.")

# 4. Main Dashboard Header
st.title("🛡️ Customer Churn & Retention Dashboard")
st.markdown("---")

# 5. Top Level Metrics (The "Business View")
col1, col2, col3 = st.columns(3)
high_risk_count = len(df[df['Risk_Level'] == 'High Risk'])
avg_prob = df['Churn_Probability'].mean()

col1.metric("Total High Risk", high_risk_count, delta_color="inverse")
col2.metric("Avg. Churn Probability", f"{avg_prob:.2%}")
col3.metric("Retention Strategy", "Active", delta="10% Target")

# 6. Customer Selection & SHAP (The "Technical View")
st.subheader("🔍 Individual Customer Deep-Dive")
selected_id = st.selectbox("Search Customer by ID:", df['cust_id'].unique())

# Get data for the selected customer
cust_data = df[df['cust_id'] == selected_id].iloc[0]
# We need to drop the extra columns we added (probability, risk, action) 
# to match the model's expected input features
features = df.drop(['cust_id', 'churn', 'Churn_Probability', 'Risk_Level', 'Recommended_Action', 'Marketing_Strategy'], axis=1, errors='ignore')
cust_features = features[df['cust_id'] == selected_id]

# Generate SHAP Values for this customer
shap_values = explainer.shap_values(cust_features)

# Layout for specific customer details
left_col, right_col = st.columns([1, 2])

with left_col:
    st.write(f"**Risk Level:** {cust_data['Risk_Level']}")
    st.write(f"**Probability:** {cust_data['Churn_Probability']:.2%}")
    st.info(f"**Recommendation:** \n{cust_data['Recommended_Action']}")

with right_col:
    st.write("**Why is this customer at risk? (SHAP Explanation)**")
    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], cust_features.iloc[0], show=False)
    st.pyplot(fig)

# 7. Data Table
st.markdown("---")
st.subheader("📋 Full Priority List")
st.dataframe(filtered_df := df[['cust_id', 'Risk_Level', 'Churn_Probability', 'Recommended_Action']])