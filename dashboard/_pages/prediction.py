"""Make Prediction page."""

import pandas as pd
import streamlit as st
from utils.model import is_model_ready
from utils.prediction import get_prediction
from utils.constants import DATASET_PATH


def render():
    """Render the make prediction page."""
    st.header("🔮 Make Failure Prediction")
    model = st.session_state.get('model')
    
    if not is_model_ready(model):
        st.error("Model not loaded")
        return
    
    if st.button("🚀 Run Prediction on Sample Data (raw ai4i2020.csv)", type="primary"):
        try:
            df = pd.read_csv(DATASET_PATH)
            st.success("Sample raw data loaded")
            
            with st.spinner("Applying feature engineering..."):
                result = get_prediction(model, df.iloc[[0]])
            
            if not result['success']:
                st.error(f"Prediction failed: {result['error']}")
                return
            
            st.success("✅ Feature engineering completed!")
            
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                if result['prediction'] == 1:
                    st.error("⚠️ FAILURE PREDICTED")
                else:
                    st.success("✅ No Failure Predicted")
            
            with col2:
                st.metric("Failure Probability", f"{result['probability']:.1%}")
            
            st.dataframe(result['X_pred'])
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.info("Tip: The first few rows usually show 0% probability because they are normal operation.")
