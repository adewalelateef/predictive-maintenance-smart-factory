"""What-if Simulator page."""

import pandas as pd
import streamlit as st
from utils.model import is_model_ready
from utils.prediction import prepare_input_data, get_prediction, get_risk_level
from utils.constants import FEATURE_RANGES, FEATURE_NAMES, RISK_THRESHOLDS, RISK_LEVELS


def render():
    """Render the what-if simulator page."""
    st.header("🔬 What-if Simulator")
    st.markdown("**Adjust sensor values** and instantly see how failure risk changes.")
    
    model = st.session_state.get('model')
    if not is_model_ready(model):
        st.error("Model not loaded")
        return

    # Collect sensor inputs
    st.subheader("Current Sensor Readings")
    col1, col2 = st.columns(2)

    inputs = {}
    with col1:
        inputs['air_temp'] = st.slider(
            "Air Temperature [K]",
            *FEATURE_RANGES['air_temp'],
            step=0.1
        )
        inputs['process_temp'] = st.slider(
            "Process Temperature [K]",
            *FEATURE_RANGES['process_temp'],
            step=0.1
        )
        inputs['rpm'] = st.slider(
            "Rotational Speed [rpm]",
            *FEATURE_RANGES['rpm'],
            step=10
        )
    
    with col2:
        inputs['torque'] = st.slider(
            "Torque [Nm]",
            *FEATURE_RANGES['torque'],
            step=0.1
        )
        inputs['tool_wear'] = st.slider(
            "Tool Wear [min]",
            *FEATURE_RANGES['tool_wear'],
            step=1
        )

    # Store inputs in session state for SHAP page
    st.session_state.whatif_inputs = inputs

    # Get prediction
    input_data = prepare_input_data(**inputs)
    result = get_prediction(model, input_data)

    if not result['success']:
        st.error(f"Prediction failed: {result['error']}")
        return

    probability = result['probability']
    risk_level = get_risk_level(probability, RISK_THRESHOLDS)
    risk_text, risk_type = RISK_LEVELS[risk_level]

    # Display result
    st.subheader("Prediction Result")

    col1, col2 = st.columns([3, 1])
    with col1:
        if risk_type == 'success':
            st.success(risk_text + " — No immediate action needed")
        elif risk_type == 'warning':
            st.warning(risk_text + " — Monitor closely, risk is increasing")
        else:
            st.error(risk_text + " — Failure is likely, take preventive action")
    
    with col2:
        st.metric("Failure Probability", f"{probability:.1%}")

    # Key Risk Drivers
    st.subheader("Key Risk Drivers")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        stress = result['processed_data']['torque_x_toolwear'].iloc[0]
        st.metric("⚙️ Mechanical Stress (Torque × Tool Wear)", f"{stress:.1f}")
        st.progress(min(stress / 8000, 1.0))
        st.caption("Strongest predictor of failure")

    with plot_col2:
        power = result['processed_data']['power_proxy'].iloc[0]
        st.metric("⚡ Power Proxy (Torque × RPM)", f"{power:,.0f}")
        st.progress(min(power / 100000, 1.0))
        st.caption("Overall mechanical load")

    st.caption(
        "**Insight**: High mechanical stress combined with worn tools "
        "is the strongest driver of failure in this model."
    )
