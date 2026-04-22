"""Prediction utilities for model inference."""

import pandas as pd
import numpy as np
from src.pipeline import apply_feature_engineering


def prepare_input_data(**kwargs) -> pd.DataFrame:
    """Create a DataFrame from sensor input parameters.
    
    Args:
        **kwargs: Feature names and values (e.g., air_temp=298.1, process_temp=308.6)
    
    Returns:
        DataFrame with properly named columns
    """
    feature_mapping = {
        'air_temp': 'Air temperature [K]',
        'process_temp': 'Process temperature [K]',
        'rpm': 'Rotational speed [rpm]',
        'torque': 'Torque [Nm]',
        'tool_wear': 'Tool wear [min]',
    }
    
    data = {feature_mapping[k]: [v] for k, v in kwargs.items() if k in feature_mapping}
    return pd.DataFrame(data)


def get_prediction(model, input_df: pd.DataFrame) -> dict:
    """Run model prediction on input data.
    
    Args:
        model: Trained XGBoost model
        input_df: Input data (raw sensor values)
    
    Returns:
        Dictionary with prediction, probability, and processed features
    """
    try:
        processed = apply_feature_engineering(input_df)
        model_features = model.feature_names_in_
        available_features = [f for f in model_features if f in processed.columns]
        X_pred = processed[available_features]
        
        prediction = model.predict(X_pred, validate_features=False)[0]
        probability = model.predict_proba(X_pred, validate_features=False)[0][1]
        
        return {
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'processed_data': processed,
            'X_pred': X_pred,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'prediction': None,
            'probability': None,
            'processed_data': None,
            'X_pred': None,
            'error': str(e)
        }


def get_risk_level(probability: float, thresholds: dict) -> str:
    """Determine risk level based on failure probability.
    
    Args:
        probability: Predicted failure probability
        thresholds: Risk threshold configuration
    
    Returns:
        Risk level string: 'safe', 'warning', or 'critical'
    """
    if probability < thresholds['safe']:
        return 'safe'
    elif probability < thresholds['warning']:
        return 'warning'
    return 'critical'
