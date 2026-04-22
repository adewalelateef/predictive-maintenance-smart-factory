"""Visualization utilities for plots and charts."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import streamlit as st


def close_plot(fig):
    """Properly close matplotlib figure to free memory."""
    plt.close(fig)


def plot_shap_local_explanation(local_df: pd.DataFrame, n_features: int = 12) -> None:
    """Plot local SHAP explanation as horizontal bar chart.
    
    Args:
        local_df: DataFrame from build_local_explanation_dataframe
        n_features: Number of top features to display
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = local_df.head(n_features).sort_values("SHAP Impact", ascending=True)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in plot_df["SHAP Impact"]]
    
    ax.barh(plot_df["Feature"], plot_df["SHAP Impact"], color=colors)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_title("Feature Contributions to Failure Risk (SHAP Values)", 
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig)
    close_plot(fig)


def plot_global_feature_importance(features_df: pd.DataFrame, n_features: int = 10) -> None:
    """Plot global SHAP feature importance.
    
    Args:
        features_df: DataFrame with 'feature_name' and 'mean_abs_shap' columns
        n_features: Number of top features to display
    """
    if features_df.empty:
        st.warning("No feature importance data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = features_df.head(n_features).copy()
    
    ax.barh(range(len(plot_df)), plot_df["mean_abs_shap"], color="#2ecc71")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature_name"])
    ax.set_xlabel("Mean |SHAP| Value")
    ax.set_title("Global Feature Importance (Mean Absolute SHAP)", 
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig)
    close_plot(fig)


def plot_cost_comparison(reactive_cost: float, predictive_cost: float) -> None:
    """Plot cost comparison between reactive and predictive maintenance.
    
    Args:
        reactive_cost: Total annual cost with reactive maintenance
        predictive_cost: Total annual cost with predictive maintenance
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    categories = ["Reactive (No Model)", "Predictive (This System)"]
    costs = [reactive_cost, predictive_cost]
    colors = ["#d62728", "#2ecc71"]
    
    bars = ax.bar(categories, costs, color=colors, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Annual Cost ($)", fontsize=11)
    ax.set_title("Cost Comparison: Reactive vs Predictive Maintenance", 
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x/1000:.0f}K"))
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height/1000:.1f}K',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    close_plot(fig)


def plot_roi_projection(years: list, cumulative_savings: list) -> None:
    """Plot ROI projection over years.
    
    Args:
        years: List of years [1, 2, 3, ...]
        cumulative_savings: List of cumulative savings for each year
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(years, cumulative_savings, marker='o', linewidth=2.5, 
            markersize=10, color="#2ecc71", label="Cumulative Savings")
    ax.fill_between(years, cumulative_savings, alpha=0.3, color="#2ecc71")
    
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cumulative Savings ($)", fontsize=11)
    ax.set_title("ROI Projection", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x/1000:.0f}K"))
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(years)
    
    for year, savings in zip(years, cumulative_savings):
        ax.text(year, savings, f"${savings/1000:.0f}K",
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    close_plot(fig)
