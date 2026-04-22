"""Model loading and management utilities."""

import streamlit as st
import joblib
from .constants import MODEL_PATH


@st.cache_resource
def load_model():
    """Load and cache the trained XGBoost model."""
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model not found: {str(e)}")
        return None


def is_model_ready(model) -> bool:
    """Check if model is available."""
    return model is not None
