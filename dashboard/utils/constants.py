"""Configuration and constants for the predictive maintenance dashboard."""

# Data Paths
DATA_DIR = "data"
DATASET_PATH = f"{DATA_DIR}/ai4i2020.csv"
SHAP_FEATURES_PATH = f"{DATA_DIR}/shap_top_features.csv"
MODEL_PATH = "src/models/final_xgb_model.pkl"

# Machine Configuration
MACHINE_TYPE = "Industrial CNC Milling"
FEATURE_NAMES = {
    'air_temp': 'Air temperature [K]',
    'process_temp': 'Process temperature [K]',
    'rpm': 'Rotational speed [rpm]',
    'torque': 'Torque [Nm]',
    'tool_wear': 'Tool wear [min]',
}

FEATURE_RANGES = {
    'air_temp': (290.0, 310.0, 298.1),      # (min, max, default)
    'process_temp': (300.0, 320.0, 308.6),
    'rpm': (1000, 3000, 1551),
    'torque': (10.0, 80.0, 42.8),
    'tool_wear': (0, 300, 0),
}

# Business Impact Constants
COST_PER_FAILURE = 18500  # USD
DOWNTIME_PER_FAILURE = 14  # hours
HOURLY_PRODUCTION_VALUE = 1250  # USD
FAILURE_PREVENTION_RATE = 0.65  # 65% reduction (based on model recall)

# Risk Thresholds
RISK_THRESHOLDS = {
    'safe': 0.20,           # Green zone
    'warning': 0.60,        # Yellow zone
    'critical': 1.0,        # Red zone
}

RISK_LEVELS = {
    'safe': ('🟢 SAFE', 'success'),
    'warning': ('🟡 WARNING', 'warning'),
    'critical': ('🔴 HIGH RISK', 'error'),
}

# UI Configuration
PAGE_CONFIG = {
    'page_title': "Smart Factory Predictive Maintenance",
    'page_icon': "🏭",
    'layout': "wide"
}

PAGES = [
    "🏠 Home",
    "🔮 Make Prediction",
    "🔬 What-if Simulator",
    "📊 SHAP Explainability",
    "💰 Business Impact"
]

# SHAP Configuration
SHAP_EDUCATION = {
    'title': '❓ What is SHAP and how do I read it?',
    'content': """
**SHAP (SHapley Additive exPlanations)** is a method to explain what each feature contributes to a prediction.

**How to interpret the visualizations:**

1. **Local Explanation (Bar Chart)**
   - Shows the top features influencing **this specific prediction**
   - **Red bars** = pushed the model toward predicting **failure**
   - **Blue bars** = pushed the model toward predicting **no failure**
   - **Longer bars** = stronger impact on the prediction

2. **SHAP Values Explained**
   - Each feature gets a **SHAP value** (positive or negative number)
   - Positive SHAP = increases failure probability
   - Negative SHAP = decreases failure probability
   - Sum of all SHAP values = how much the model moved from baseline to final prediction

3. **The Math (Simple Version)**
   - Base Model Output = model's average prediction (≈0.5 for balanced data)
   - + All SHAP Values = total feature contributions
   - = Final Prediction (converted to failure probability %)

4. **Global Feature Importance**
   - Shows which features **consistently matter** across all predictions
   - Green bar = how often this feature influences predictions on average

**Example:**
- If Mechanical Stress has a red bar of +0.3, it means this high stress pushed the failure probability up by ~30%
- If Tool Wear has a blue bar of -0.1, it means low tool wear pushed it down by ~10%
"""
}
