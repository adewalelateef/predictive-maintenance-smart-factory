
import pandas as pd

def clean_column_names(df):
    """Clean column names to match training format"""
    df = df.copy()
    df.columns = [str(col).replace('[', '_').replace(']', '').replace('<', '').replace('>', '')
                  for col in df.columns]
    return df


def apply_feature_engineering(df):
    """Full feature engineering - exact match to your notebook"""
    df = clean_column_names(df)

    # ------------------------------------------------------------------
    # 1. Time-based features
    # ------------------------------------------------------------------
    df['cumulative_tool_wear'] = df['Tool wear _min'].cumsum()
    df['tool_wear_diff'] = df['Tool wear _min'].diff().fillna(0)
    df['is_tool_change'] = (df['tool_wear_diff'] < -80).astype(int)
    df['time_since_last_tool_change'] = df.groupby((df['is_tool_change'] == 1).cumsum())['Tool wear _min'].cumsum()
    df['running_tool_wear'] = df['Tool wear _min'].copy()

    # ------------------------------------------------------------------
    # 2. Rolling Window Features
    # ------------------------------------------------------------------
    sensor_cols = ['Torque _Nm', 'Rotational speed _rpm', 
                   'Process temperature _K', 'Air temperature _K', 
                   'Tool wear _min']
    windows = [5, 10]

    for col in sensor_cols:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()

    # ------------------------------------------------------------------
    # 3. Interaction & Domain Features
    # ------------------------------------------------------------------
    df['torque_x_toolwear'] = df['Torque _Nm'] * df['Tool wear _min']
    df['torque_per_rpm'] = df['Torque _Nm'] / (df['Rotational speed _rpm'] + 1)
    df['temp_difference'] = df['Process temperature _K'] - df['Air temperature _K']
    df['power_proxy'] = df['Torque _Nm'] * df['Rotational speed _rpm']
    df['tool_wear_rate'] = df['Tool wear _min'].diff().fillna(0)

    # Binary flags
    df['high_stress'] = ((df['Torque _Nm'] > df['Torque _Nm'].quantile(0.75)) & 
                         (df['Tool wear _min'] > 150)).astype(int)

    df['poor_heat_dissipation'] = ((df['temp_difference'] < 5) & 
                                   (df['Rotational speed _rpm'] < 1400)).astype(int)

    # ------------------------------------------------------------------
    # Final cleanup (remove temporary helper columns)
    # ------------------------------------------------------------------
    helper_cols = ['tool_wear_diff', 'is_tool_change']
    for col in helper_cols:
        if col in df.columns:
            df = df.drop(columns=col)

    return df