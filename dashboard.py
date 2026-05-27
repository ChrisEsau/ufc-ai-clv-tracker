# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import streamlit.components.v1 as components
import plotly.graph_objects as go

from utils.theme import apply_theme
from tabs.betting_board import render_betting_board
from tabs.line_movement import render_line_movement
#from tabs.model_lab import render_model_lab
#from tabs.data_maintenance import render_data_maintenance

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UFC Betting Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def pct(x):
    if pd.isna(x):
        return ""
    return f"{x * 100:.1f}%"


def pct_already(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}%"


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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("UFC Intelligence")
st.sidebar.caption("Betting Board Control Plane")

if st.sidebar.button("Clear cache / reload"):
    st.cache_data.clear()
    st.rerun()

# ------------------------------------------------------------
# Optional workflow controls
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Controls")

GITHUB_OWNER = "ChrisEsau"
GITHUB_REPO = "ufc-ai-clv-tracker"


def trigger_workflow(workflow_file):
    try:
        github_token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        return None, "Missing Streamlit secret: GITHUB_TOKEN"

    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/"
        f"{workflow_file}/dispatches"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
    }

    response = requests.post(
        url,
        headers=headers,
        json={"ref": "main"},
        timeout=20,
    )

    return response.status_code, response.text


if st.sidebar.button("Run Model Predictions"):
    status, msg = trigger_workflow("run-model-predictions.yml")
    if status == 204:
        st.sidebar.success("Model prediction workflow started.")
    else:
        st.sidebar.error(f"Workflow failed: {status} {msg}")

if st.sidebar.button("Run Market Update"):
    status, msg = trigger_workflow("run-market-update.yml")
    if status == 204:
        st.sidebar.success("Market update workflow started.")
    else:
        st.sidebar.error(f"Workflow failed: {status} {msg}")

if st.sidebar.button("Run Betting Decision"):
    status, msg = trigger_workflow("run-betting-decision.yml")
    if status == 204:
        st.sidebar.success("Betting decision workflow started.")
    else:
        st.sidebar.error(f"Workflow failed: {status} {msg}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Betting Board",
        "📈 Line Movement / CLV",
        "🧠 Model Lab",
        "🛠️ Data Maintenance",
    ]
)

with tab1:
    render_betting_board()

with tab2:
    render_line_movement()

#with tab3:
#    render_model_lab()

#with tab4:
#    render_data_maintenance()
