import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import apply_feature_engineering

st.set_page_config(page_title="Smart Factory Predictive Maintenance", page_icon="🏭", layout="wide")

st.title("🏭 Smart Factory Predictive Maintenance System")
st.markdown("**IIoT Sensor Data • Tuned XGBoost • SHAP Explainability**")
st.caption("Aerospace CNC Machine Failure Prediction")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Make Prediction", "🔬 What-if Simulator", "📊 SHAP Explainability", "💰 Business Impact"])

st.sidebar.markdown("---")
st.sidebar.info("Model: Tuned XGBoost (Optuna)")

# Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('src/models/final_xgb_model.pkl')
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except:
        st.sidebar.error("❌ Model not found")
        return None

model = load_model()

# ====================== PAGES ======================
if page == "🏠 Home":
    st.subheader("Project Overview")
    st.write("End-to-end predictive maintenance dashboard for aerospace CNC machines.")
    if model is not None:
        st.success("✅ Model is ready")
    else:
        st.warning("Model not loaded yet")

elif page == "🔮 Make Prediction":
    st.header("🔮 Make Failure Prediction")
    
    if st.button("🚀 Run Prediction on Sample Data (raw ai4i2020.csv)", type="primary"):
        try:
            df = pd.read_csv("data/ai4i2020.csv")
            st.success("Sample raw data loaded")
            
            with st.spinner("Applying feature engineering..."):
                processed_data = apply_feature_engineering(df)
            
            st.success("✅ Feature engineering completed!")
            
            if model is not None:
                model_features = model.feature_names_in_
                available_features = [f for f in model_features if f in processed_data.columns]
                X_pred = processed_data[available_features].iloc[[0]]
                
                prediction = model.predict(X_pred, validate_features=False)[0]
                probability = model.predict_proba(X_pred, validate_features=False)[0][1]
                
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error(f"⚠️ FAILURE PREDICTED")
                    else:
                        st.success(f"✅ No Failure Predicted")
                
                with col2:
                    st.metric("Failure Probability", f"{probability:.1%}")
                
                st.dataframe(X_pred)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.info("Tip: The first few rows usually show 0% probability because they are normal operation.")

elif page == "🔬 What-if Simulator":
    st.header("🔬 What-if Simulator")
    st.markdown("**Adjust sensor values** and instantly see how failure risk changes.")

    if model is None:
        st.error("Model not loaded")
        st.stop()

    # Sensor Inputs
    st.subheader("Current Sensor Readings")
    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.slider("Air Temperature [K]", 290.0, 310.0, 298.1, step=0.1)
        process_temp = st.slider("Process Temperature [K]", 300.0, 320.0, 308.6, step=0.1)
        rpm = st.slider("Rotational Speed [rpm]", 1000, 3000, 1551, step=10)
    with col2:
        torque = st.slider("Torque [Nm]", 10.0, 80.0, 42.8, step=0.1)
        tool_wear = st.slider("Tool Wear [min]", 0, 300, 0, step=1)

    # Create input row
    input_data = pd.DataFrame({
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rpm],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })

    with st.spinner("Running feature engineering & prediction..."):
        processed = apply_feature_engineering(input_data)
        
        model_features = model.feature_names_in_
        available = [f for f in model_features if f in processed.columns]
        X_pred = processed[available]
        
        prediction = model.predict(X_pred, validate_features=False)[0]
        probability = model.predict_proba(X_pred, validate_features=False)[0][1]

    # ====================== COLOR-CODED RESULT ======================
    st.subheader("Prediction Result")

    if probability < 0.20:
        st.success("🟢 SAFE — No immediate action needed")
    elif probability < 0.60:
        st.warning("🟡 WARNING — Monitor closely, risk is increasing")
    else:
        st.error("🔴 HIGH RISK — Failure is likely, take preventive action")

    col1, col2 = st.columns([3, 1])
    with col1:
        if prediction == 1:
            st.error("⚠️ FAILURE PREDICTED")
        else:
            st.success("✅ No Failure Predicted")
    with col2:
        st.metric("Failure Probability", f"{probability:.1%}")

    # ====================== KEY RISK DRIVERS ======================
    st.subheader("Key Risk Drivers")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        stress = processed['torque_x_toolwear'].iloc[0]
        st.metric("⚙️ Mechanical Stress (Torque × Tool Wear)", f"{stress:.1f}")
        st.progress(min(stress / 8000, 1.0))
        st.caption("Strongest predictor of failure")

    with plot_col2:
        power = processed['power_proxy'].iloc[0]
        st.metric("⚡ Power Proxy (Torque × RPM)", f"{power:,.0f}")
        st.progress(min(power / 100000, 1.0))
        st.caption("Overall mechanical load")

    st.caption("**Insight**: High mechanical stress combined with worn tools is the strongest driver of failure in this model.")


else:
    st.info(f"{page} page is under construction.")

st.markdown("---")
st.caption("Built as part of Personal Industry 4.0 Project")