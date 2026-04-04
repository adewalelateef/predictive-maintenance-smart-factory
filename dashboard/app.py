'''Building a dashboard with Dash and Streamlit'''

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="Smart Factory Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Smart Factory Predictive Maintenance System")
st.markdown("**IIoT Sensor Data • XGBoost • SHAP Explainability**")
st.caption("Aerospace CNC Machine Failure Prediction")

# ------------------- Sidebar -------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Make Prediction", "📊 SHAP Explainability", "🔬 What-if Simulator", "💰 Business Impact"]
)

st.sidebar.markdown("---")
st.sidebar.info("Model: Tuned XGBoost with Optuna")

# ------------------- Load Model (We'll improve this later) -------------------
@st.cache_resource
def load_model():
    # For now, we'll use a placeholder. Later we'll load the real model
    return None  # Placeholder

model = load_model()

# ------------------- Home Page -------------------
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Overview")
        st.write("""
        This dashboard predicts machine failures in aerospace CNC machines using IIoT sensor data.
        It combines advanced feature engineering, a tuned XGBoost model, and SHAP explainability.
        """)
        
        st.success("✅ Model is ready")
        st.info("📌 Current best model: Tuned XGBoost (F1 ≈ 0.72 for failure class)")
    
    with col2:
        st.subheader("Key Capabilities")
        st.markdown("""
        - Real-time failure prediction
        - SHAP explainability (why the model decided)
        - Interactive What-if simulator
        - Business cost savings calculator
        """)

    st.markdown("---")
    st.subheader("How to use")
    st.write("1. Go to **Make Prediction** to test the model")
    st.write("2. Use **SHAP Explainability** to understand decisions")
    st.write("3. Try **What-if Simulator** to test scenarios")

# ------------------- Make Prediction Page -------------------
elif page == "🔮 Make Prediction":
    st.header("🔮 Make Failure Prediction")
    
    tab1, tab2 = st.tabs(["📤 Upload Data", "📋 Use Sample Data"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload your IIoT sensor CSV file", type=["csv"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            st.info("Prediction functionality will be added in the next step")
    
    with tab2:
        if st.button("Use Sample Data"):
            st.success("Sample data loaded!")
            st.info("Prediction will appear here after model integration")
            
    st.markdown("---")
    st.subheader("Prediction Result")
    st.info("The prediction gauge and probability will appear here once the model is connected.")

# ------------------- Placeholder Pages -------------------
elif page == "📊 SHAP Explainability":
    st.header("📊 SHAP Explainability")
    st.info("SHAP plots will be displayed here in the next update.")

elif page == "🔬 What-if Simulator":
    st.header("🔬 What-if Simulator")
    st.info("Interactive sliders for testing scenarios will be added soon.")

elif page == "💰 Business Impact":
    st.header("💰 Business Impact Calculator")
    st.info("Estimated cost savings calculator coming in the next step.")

# Footer
st.markdown("---")
st.caption("Built as part of Personal Industry 4.0 Project | MSc Future Industries @ IT:U Austria")