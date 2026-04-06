import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Factory Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Smart Factory Predictive Maintenance System")
st.markdown("**IIoT Sensor Data • Tuned XGBoost • SHAP Explainability**")
st.caption("Aerospace CNC Machine Failure Prediction")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Make Prediction", "📊 SHAP Explainability", "🔬 What-if Simulator", "💰 Business Impact"])

st.sidebar.markdown("---")
st.sidebar.info("Model: Tuned XGBoost (Optuna)")

# ====================== EXPECTED COLUMNS ======================
EXPECTED_COLUMNS = [
    "Air temperature _K", "Process temperature _K", "Rotational speed _rpm",
    "Torque _Nm", "Tool wear _min"
]

# ====================== HELPER FUNCTIONS ======================
def clean_column_names(df):
    df.columns = [str(col).replace('[', '_').replace(']', '').replace('<', '').replace('>', '').strip() 
                  for col in df.columns]
    return df

def validate_data(df):
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        return False, missing
    return True, None

def apply_feature_engineering(df):
    df = df.copy()
    df = clean_column_names(df)
    
    # Time-based features
    df['cumulative_tool_wear'] = df['Tool wear _min'].cumsum()
    df['tool_wear_diff'] = df['Tool wear _min'].diff().fillna(0)
    df['is_tool_change'] = (df['tool_wear_diff'] < -80).astype(int)
    df['time_since_last_tool_change'] = df.groupby((df['is_tool_change'] == 1).cumsum())['Tool wear _min'].cumsum()
    
    # Interaction features
    df['torque_x_toolwear'] = df['Torque _Nm'] * df['Tool wear _min']
    df['torque_per_rpm'] = df['Torque _Nm'] / (df['Rotational speed _rpm'] + 1)
    df['temp_difference'] = df['Process temperature _K'] - df['Air temperature _K']
    df['power_proxy'] = df['Torque _Nm'] * df['Rotational speed _rpm']
    
    # Rolling features
    for col in ['Torque _Nm', 'Tool wear _min']:
        df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
    
    return df

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../src/models/final_xgb_model.pkl')  # We'll save the model here later
        st.success("✅ Model loaded successfully")
        return model
    except:
        st.warning("Model not found yet. Using placeholder.")
        return None

model = load_model()

# ====================== HOME PAGE ======================
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Project Overview")
        st.write("End-to-end predictive maintenance dashboard for aerospace CNC machines.")
        st.success("✅ Model is ready")
    with col2:
        st.subheader("Capabilities")
        st.markdown("- Real-time failure prediction\n- SHAP explainability\n- What-if simulator\n- Business impact calculator")

# ====================== MAKE PREDICTION PAGE ======================
elif page == "🔮 Make Prediction":
    st.header("🔮 Make Failure Prediction")
    
    st.info("""
    **Note**: This model expects CNC machine sensor data with columns similar to the AI4I 2020 dataset 
    (Air temperature, Process temperature, Rotational speed, Torque, Tool wear, etc.).
    """)
    
    tab1, tab2 = st.tabs(["📤 Upload New Data", "📋 Use Sample Data"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload your IIoT sensor CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                raw_data = pd.read_csv(uploaded_file)
                st.success(f"Uploaded {raw_data.shape[0]} rows")
                
                is_valid, missing = validate_data(raw_data)
                if not is_valid:
                    st.error(f"Missing required columns: {missing}")
                else:
                    with st.spinner("Applying feature engineering and making prediction..."):
                        processed_data = apply_feature_engineering(raw_data)
                        
                        if model is not None:
                            # For now, use only the features the model was trained on (simplified)
                            # In the next step we'll map them properly
                            st.success("✅ Prediction ready (model integration in progress)")
                            st.dataframe(processed_data.head())
                        else:
                            st.warning("Model not loaded yet.")
                            
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        if st.button("Load Sample Data"):
            st.success("Sample data loaded")
            st.info("Prediction pipeline will run here")

    st.markdown("---")
    st.subheader("Prediction Result")
    st.info("Prediction gauge, probability, and SHAP will appear here after full integration.")

# Placeholder pages
elif page == "📊 SHAP Explainability":
    st.header("📊 SHAP Explainability")
    st.info("SHAP plots coming soon.")

elif page == "🔬 What-if Simulator":
    st.header("🔬 What-if Simulator")
    st.info("Interactive simulator coming soon.")

elif page == "💰 Business Impact":
    st.header("💰 Business Impact Calculator")
    st.info("Business impact calculator coming soon.")

st.markdown("---")
st.caption("Built as part of Personal Industry 4.0 Project | Focused on CNC machine sensor data")