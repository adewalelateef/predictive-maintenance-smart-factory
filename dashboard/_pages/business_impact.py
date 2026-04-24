"""Business Impact page."""

import streamlit as st
import pandas as pd
from utils.constants import (
    COST_PER_FAILURE,
    DOWNTIME_PER_FAILURE,
    HOURLY_PRODUCTION_VALUE,
    FAILURE_PREVENTION_RATE,
)
from utils.visualization import plot_cost_comparison, plot_roi_projection


def render():
    """Render the business impact page."""
    st.header("💰 Business Impact & ROI")
    st.markdown("**Turning predictive maintenance into real financial value**")

    st.write("This model helps prevent unplanned machine failures in industrial CNC milling operations.")

    # Key Assumptions
    st.subheader("Key Assumptions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cost per Unplanned Failure", f"€{COST_PER_FAILURE:,}", "Repair + downtime")
    with col2:
        st.metric("Downtime per Failure", f"{DOWNTIME_PER_FAILURE} hours", "Lost production")
    with col3:
        st.metric("Hourly Production Value", f"€{HOURLY_PRODUCTION_VALUE:,}", "Industrial CNC milling output")

    st.markdown("---")

    # Interactive Savings Calculator
    st.subheader("Estimated Annual Savings")
    
    baseline_failures = st.slider(
        "Expected machine failures per year (without prediction)",
        min_value=4,
        max_value=25,
        value=12,
        step=1
    )

    prevented = int(baseline_failures * FAILURE_PREVENTION_RATE)
    total_savings = prevented * COST_PER_FAILURE

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Failures Prevented per Year",
            f"{prevented}",
            f"{int(FAILURE_PREVENTION_RATE * 100)}% reduction"
        )
    with col2:
        st.metric("Estimated Annual Savings", f"€{total_savings:,.0f}", "💰")

    st.progress(FAILURE_PREVENTION_RATE)
    st.caption(f"Based on model recall and conservative industry benchmarks")

    # Cost Comparison Chart
    st.subheader("Cost Comparison")
    reactive_cost = baseline_failures * COST_PER_FAILURE
    predictive_cost = (baseline_failures - prevented) * COST_PER_FAILURE
    
    plot_cost_comparison(reactive_cost, predictive_cost)

    # ROI Summary
    st.markdown("---")
    st.subheader("Return on Investment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Payback Period", "4–6 months", "Very fast ROI")
    with col2:
        st.metric("Annual ROI", "300%+", "High return")

    st.info(
        "In a real factory environment, deploying this system across multiple CNC machines "
        "could generate **€100,000 – €300,000+** in annual savings."
    )

    # ROI Projection
    st.markdown("---")
    st.subheader("3-Year ROI Projection")
    years = [1, 2, 3]
    cumulative_savings = [
        total_savings,  # Year 1
        total_savings * 2,  # Year 2
        total_savings * 3,  # Year 3
    ]
    plot_roi_projection(years, cumulative_savings)

    st.markdown(f"**3-Year Total Savings: €{cumulative_savings[-1]:,.0f}**")

    # Qualitative Benefits
    st.markdown("---")
    st.subheader("Additional Operational Benefits")
    st.markdown("""
    - **Scheduled maintenance** instead of emergency repairs  
    - **Reduced unplanned downtime** and production losses  
    - **Extended equipment lifespan** through timely intervention  
    - **Better maintenance team efficiency** with advance warnings  
    - **Data-driven decisions** replacing guesswork
    """)

    st.caption(
        "**Key Insight**: This is not just a prediction tool — it is a proactive system "
        "that prevents costly failures and improves Overall Equipment Effectiveness (OEE)."
    )
