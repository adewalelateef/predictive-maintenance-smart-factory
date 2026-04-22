"""SHAP Explainability page."""

import pandas as pd
import streamlit as st
import numpy as np
from utils.model import is_model_ready
from utils.shap_utils import (
    get_shap_availability,
    get_shap_explainer,
    compute_shap_explanation,
    build_local_explanation_dataframe,
    load_shap_top_features,
)
from utils.visualization import plot_shap_local_explanation, plot_global_feature_importance
from utils.constants import SHAP_EDUCATION, DATASET_PATH, SHAP_FEATURES_PATH
from src.pipeline import apply_feature_engineering


def render():
    """Render the SHAP explainability page."""
    st.header("📊 SHAP Explainability")
    st.markdown("**Understanding why the model makes its predictions**")
    
    # Education section
    with st.expander(SHAP_EDUCATION['title'], expanded=False):
        st.markdown(SHAP_EDUCATION['content'])

    model = st.session_state.get('model')
    if not is_model_ready(model):
        st.error("Model not loaded")
        return

    if not get_shap_availability():
        st.error("SHAP package not available. Please install it: pip install shap")
        return

    # Data source selection
    st.subheader("Step 1: Select Data Source")
    
    has_whatif_inputs = hasattr(st.session_state, 'whatif_inputs') and st.session_state.whatif_inputs
    
    if has_whatif_inputs:
        data_options = ["📁 ai4i2020.csv (select row)", "⚙️ Current What-if Simulator Input"]
    else:
        data_options = ["📁 ai4i2020.csv (select row)"]
    
    data_source = st.radio("Choose data source:", data_options, index=0)

    # Load data and prepare input
    use_whatif = data_source == "⚙️ Current What-if Simulator Input"
    row_index = None
    input_row = None

    if use_whatif:
        whatif = st.session_state.whatif_inputs
        input_row = pd.DataFrame({
            'Air temperature [K]': [whatif['air_temp']],
            'Process temperature [K]': [whatif['process_temp']],
            'Rotational speed [rpm]': [whatif['rpm']],
            'Torque [Nm]': [whatif['torque']],
            'Tool wear [min]': [whatif['tool_wear']]
        })
        st.success("✅ Using current What-if Simulator inputs")
        with st.expander("📋 What-if Input Values"):
            st.dataframe(input_row, use_container_width=True)
    else:
        try:
            sample_df = pd.read_csv(DATASET_PATH)
            st.success("✅ Loaded ai4i2020.csv")
            
            st.subheader("Step 2: Select Sample to Explain")
            row_index = st.slider(
                "Select row to explain",
                0,
                len(sample_df) - 1,
                min(50, len(sample_df) - 1)
            )
            
            with st.expander("📋 Raw Sensor Values (this row)"):
                st.dataframe(sample_df.iloc[[row_index]], use_container_width=True)
        except Exception as e:
            st.error(f"Could not load ai4i2020.csv: {str(e)}")
            return

    if not use_whatif:
        st.subheader("Step 2: Using File Input")

    # Compute SHAP explanation
    if st.button("🚀 Explain" + (" What-if Input" if use_whatif else " This Sample"), type="primary"):
        with st.spinner("Processing data and computing SHAP values..."):
            try:
                if not use_whatif:
                    input_row = sample_df.iloc[[int(row_index)]]
                
                processed = apply_feature_engineering(input_row)
                model_features = model.feature_names_in_
                available_features = [f for f in model_features if f in processed.columns]
                X_row = processed[available_features]
                
                explainer = get_shap_explainer(model)
                shap_result = compute_shap_explanation(model, X_row, explainer)
                
                if not shap_result['success']:
                    st.error(f"SHAP computation failed: {shap_result['error']}")
                    return
                
                # Store in session state
                st.session_state.shap_explanation = {
                    'prediction': shap_result['prediction'],
                    'probability': shap_result['probability'],
                    'X_row': X_row,
                    'shap_values': shap_result['shap_values'],
                    'base_value': shap_result['base_value'],
                }
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

    # Display results if available
    if hasattr(st.session_state, 'shap_explanation') and st.session_state.shap_explanation:
        exp = st.session_state.shap_explanation
        
        # Prediction results
        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if exp['prediction'] == 1:
                st.error("⚠️ FAILURE PREDICTED")
            else:
                st.success("✅ No Failure Predicted")

        with col2:
            st.metric("Failure Probability", f"{exp['probability']:.1%}")

        with col3:
            approx_prob = 1 / (1 + np.exp(-(exp['base_value'] + exp['shap_values'].sum())))
            st.metric("SHAP-Verified Probability", f"{approx_prob:.1%}")

        # Local explanation
        st.markdown("---")
        st.subheader("Local Explanation – Why This Prediction?")
        st.write(f"Base model output: {exp['base_value']:.3f}  →  After applying feature effects:")

        local_df = build_local_explanation_dataframe(exp['X_row'], exp['shap_values'])

        st.write("**Top contributing features (sorted by magnitude):**")
        col1, col2 = st.columns([2, 1])

        with col1:
            plot_shap_local_explanation(local_df, n_features=12)

        with col2:
            st.markdown("**Color Legend:**")
            st.markdown("🔴 **Red** = Pushes toward failure")
            st.markdown("🔵 **Blue** = Pushes toward no failure")
            st.markdown("")
            st.markdown("**Size** = Strength of effect")

        st.write("**Detailed Breakdown:**")
        display_df = local_df[["Feature", "Value", "SHAP Impact", "Impact Magnitude"]].head(15).copy()
        display_df["Value"] = display_df["Value"].apply(lambda x: f"{x:.2f}")
        display_df["SHAP Impact"] = display_df["SHAP Impact"].apply(lambda x: f"{x:.4f}")
        display_df["Impact Magnitude"] = display_df["Impact Magnitude"].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True)

        st.caption(
            f"💡 **Insight**: Sum of all SHAP values ({exp['shap_values'].sum():.4f}) + "
            f"base value ({exp['base_value']:.4f}) ≈ final log-odds producing {exp['probability']:.1%} failure probability."
        )

        # Global feature importance
        st.markdown("---")
        st.subheader("Global Feature Importance (Across Dataset)")

        top_features_df = load_shap_top_features(SHAP_FEATURES_PATH)
        if not top_features_df.empty:
            top_n = st.slider(
                "Number of top features to display",
                5, 20, 10,
                key="global_features_slider"
            )
            plot_global_feature_importance(top_features_df, n_features=top_n)
            st.write(
                "These features have the strongest average influence on model predictions "
                "across the dataset."
            )
        else:
            st.warning("Precomputed global SHAP data not available.")
