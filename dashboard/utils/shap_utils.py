"""SHAP explainability utilities."""

import pandas as pd
import numpy as np
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_shap_availability() -> bool:
    """Check if SHAP is available."""
    return SHAP_AVAILABLE


@st.cache_resource
def get_shap_explainer(_model):
    """Cache the SHAP TreeExplainer instance.
    
    Args:
        _model: Trained tree-based model (underscore prevents Streamlit hashing)
    
    Returns:
        SHAP TreeExplainer or None
    """
    if _model is None:
        return None
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(_model)


def get_positive_class_shap_values(raw_shap_values: np.ndarray) -> np.ndarray:
    """Normalize SHAP outputs into a 1D contribution vector for positive class.
    
    Handles multiple SHAP output formats from different XGBoost versions.
    
    Args:
        raw_shap_values: Raw SHAP output from explainer
    
    Returns:
        1D array of SHAP values for positive class
    
    Raises:
        ValueError: If output format is unexpected
    """
    if isinstance(raw_shap_values, list):
        if len(raw_shap_values) > 1:
            return np.asarray(raw_shap_values[1])[0]
        return np.asarray(raw_shap_values[0])[0]

    shap_array = np.asarray(raw_shap_values)

    if shap_array.ndim == 2:
        return shap_array[0]

    if shap_array.ndim == 3:
        # Format: (samples, features, classes)
        if shap_array.shape[-1] == 2:
            return shap_array[0, :, 1]
        # Format: (classes, samples, features)
        if shap_array.shape[0] == 2:
            return shap_array[1, 0, :]

    raise ValueError(f"Unexpected SHAP output format: {shap_array.shape}")


def get_positive_class_base_value(expected_value) -> float:
    """Return scalar expected value for positive class in binary classification.
    
    Args:
        expected_value: Base value from SHAP explainer
    
    Returns:
        Float value representing baseline log-odds
    """
    if isinstance(expected_value, list):
        return float(expected_value[1] if len(expected_value) > 1 else expected_value[0])

    expected_arr = np.asarray(expected_value)
    if expected_arr.ndim == 0:
        return float(expected_arr)
    if expected_arr.size >= 2:
        return float(expected_arr.flatten()[1])
    return float(expected_arr.flatten()[0])


@st.cache_data
def load_shap_top_features(path: str) -> pd.DataFrame:
    """Load precomputed global SHAP feature importance.
    
    Args:
        path: Path to shap_top_features.csv
    
    Returns:
        DataFrame with feature importance or empty DataFrame if file not found
    """
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def compute_shap_explanation(model, X_row: pd.DataFrame, explainer=None) -> dict:
    """Compute SHAP values for a single sample.
    
    Args:
        model: Trained model
        X_row: Single row DataFrame (processed features)
        explainer: SHAP explainer instance (will create if None)
    
    Returns:
        Dictionary with prediction, probability, and SHAP values
    """
    try:
        if explainer is None:
            explainer = get_shap_explainer(model)
        
        if explainer is None:
            return {'success': False, 'error': 'SHAP explainer not available'}
        
        prediction = model.predict(X_row, validate_features=False)[0]
        probability = model.predict_proba(X_row, validate_features=False)[0][1]
        
        raw_shap_values = explainer.shap_values(X_row, check_additivity=False)
        shap_values_row = get_positive_class_shap_values(raw_shap_values)
        base_value = get_positive_class_base_value(explainer.expected_value)
        
        return {
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'shap_values': shap_values_row,
            'base_value': float(base_value),
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"SHAP computation failed: {str(e)}"
        }


def build_local_explanation_dataframe(X_row: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """Build DataFrame for local SHAP explanation visualization.
    
    Args:
        X_row: Single row of features
        shap_values: SHAP values for features
    
    Returns:
        DataFrame sorted by impact magnitude
    """
    df = pd.DataFrame({
        "Feature": X_row.columns,
        "Value": X_row.iloc[0].values,
        "SHAP Impact": shap_values,
    })
    df["Impact Magnitude"] = df["SHAP Impact"].abs()
    return df.sort_values("Impact Magnitude", ascending=False)
