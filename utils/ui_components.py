import pandas as pd
import streamlit as st

def money(x):
    if pd.isna(x):
        return ""
    return f"${x:,.0f}"

def american(x):
    if pd.isna(x):
        return ""
    x = int(round(x))
    return f"+{x}" if x > 0 else str(x)

def render_metric(label, value, subtext="", accent="neutral"):
    color = {
        "green": "#22C55E",
        "red": "#EF4444",
        "blue": "#3B82F6",
        "amber": "#F59E0B",
        "purple": "#A855F7",
        "neutral": "#F9FAFB",
    }.get(accent, "#F9FAFB")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
def pct(x):
    if pd.isna(x):
        return ""
    return f"{x * 100:.1f}%"
