"""Home page of the dashboard."""

import streamlit as st
from utils.model import is_model_ready


def render():
    """Render the home page."""
    st.subheader("Project Overview")
    st.write(
        "End-to-end predictive maintenance dashboard for industrial CNC milling machines. "
        "Powered by machine learning explainability (SHAP) and XGBoost prediction."
    )
    
    if is_model_ready(st.session_state.get('model')):
        st.success("✅ Model is ready")
    else:
        st.warning("Model not loaded yet")
    
    # Dataset Information
    st.markdown("---")
    st.subheader("📊 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Source**")
        st.markdown(
            "[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)  \n"
            "UCI Machine Learning Repository"
        )
        st.markdown("\n**Data Characteristics**")
        st.markdown(
            "- **Samples**: 10,000 machine operating points\n"
            "- **Features**: 5 sensor readings + 1 failure indicator\n"
            "- **Machine Type**: Industrial CNC milling machines\n"
            "- **Data Type**: 🔬 Synthetic data for model development and testing"
        )
    
    with col2:
        st.markdown("**Sensor Features**")
        st.markdown(
            "- 🌡️ Air Temperature [K]\n"
            "- 🌡️ Process Temperature [K]\n"
            "- ⚙️ Rotational Speed [rpm]\n"
            "- 🔧 Torque [Nm]\n"
            "- ⏱️ Tool Wear [min]\n"
            "- ⚠️ Machine Failure (target)"
        )
    
    # Model Information
    st.markdown("---")
    st.subheader("🤖 Model Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algorithm", "XGBoost", "Tree-based")
    with col2:
        st.metric("Hyperparameter Tuning", "Optuna", "Bayesian Optimization")
    with col3:
        st.metric("Explainability", "SHAP", "Feature Importance")
    
    st.info(
        "💡 **How to Use**: Navigate through the sidebar to explore predictions, "
        "simulate what-if scenarios, and understand model decisions through SHAP explanations."
    )
