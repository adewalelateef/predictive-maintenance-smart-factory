"""
Smart Factory Predictive Maintenance Dashboard
End-to-end IIoT sensor data analysis with ML explainability
"""

import sys
import os
import streamlit as st

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration
from utils.constants import PAGE_CONFIG, PAGES
from utils.model import load_model

# Import pages
from _pages import home, prediction, whatif, shap, business_impact

# Configure Streamlit
st.set_page_config(**PAGE_CONFIG)

# Load model into session state
if 'model' not in st.session_state:
    st.session_state.model = load_model()

# UI Setup
st.title("🏭 Smart Factory Predictive Maintenance System")
st.markdown("**IIoT Sensor Data • Tuned XGBoost • SHAP Explainability**")
st.caption("Industrial CNC Milling Machine Failure Prediction")

# Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.info("Model: Tuned XGBoost (Optuna)")

# ====================== PAGE ROUTING ======================
if page == "🏠 Home":
    home.render()

elif page == "🔮 Make Prediction":
    prediction.render()

elif page == "🔬 What-if Simulator":
    whatif.render()

elif page == "📊 SHAP Explainability":
    shap.render()

elif page == "💰 Business Impact":
    business_impact.render()

else:
    st.info(f"{page} page is under construction.")

st.markdown("---")
st.markdown("An Industry 4.0 Project by [Adewale Lateef](https://github.com/adewalelateef)")