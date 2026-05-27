# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import streamlit.components.v1 as components
import plotly.graph_objects as go

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

# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0B1220 0%, #111827 45%, #0F172A 100%);
        color: #F9FAFB;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #334155;
    }

    h1 {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #F9FAFB !important;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1600px;
    }

    .metric-card {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.32);
        min-height: 118px;
    }

    .metric-label {
        color: #93A4BA;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-subtext {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 8px;
    }

    .section-header {
        font-size: 24px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 12px;
        color: #F8FAFC;
        letter-spacing: -0.03em;
    }

    .panel {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    }

    .small-muted {
        color: #94A3B8;
        font-size: 13px;
    }

    div[data-testid="stDataFrame"] {
        background: #1E293B;
        border-radius: 14px;
        border: 1px solid #334155;
        overflow: hidden;
    }

    .stSelectbox label,
    .stMultiSelect label,
    .stRadio label {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }

    .stButton button {
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #475569;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton button:hover {
        background: #334155;
        color: #FFFFFF;
        border-color: #60A5FA;
    }

    .positive {
        color: #22C55E;
        font-weight: 800;
    }

    .negative {
        color: #EF4444;
        font-weight: 800;
    }

    .neutral {
        color: #CBD5E1;
        font-weight: 700;
    }

    .status-pill {
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        display: inline-block;
    }

    .pill-official {
        background: rgba(16,185,129,0.20);
        color: #34D399;
        border: 1px solid rgba(16,185,129,0.45);
    }

    .pill-watchlist {
        background: rgba(245,158,11,0.18);
        color: #FBBF24;
        border: 1px solid rgba(245,158,11,0.45);
    }

    .pill-danger {
        background: rgba(239,68,68,0.18);
        color: #F87171;
        border: 1px solid rgba(239,68,68,0.45);
    }

    .pill-info {
        background: rgba(59,130,246,0.18);
        color: #60A5FA;
        border: 1px solid rgba(59,130,246,0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    /* Selectbox container */
    div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        min-height: 42px !important;
    }

    /* Selected text */
    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }

    /* Dropdown arrow */
    div[data-baseweb="select"] svg {
        color: #CBD5E1 !important;
        fill: #CBD5E1 !important;
    }

    /* Input text area */
    div[data-baseweb="select"] input {
        color: #F8FAFC !important;
    }

    /* Selectbox label */
    label[data-testid="stWidgetLabel"] {
        color: #CBD5E1 !important;
        font-weight: 700 !important;
        font-size: 13px !important;
    }

    /* Dropdown menu */
    ul[role="listbox"] {
        background-color: #111827 !important;
        border: 1px solid #334155 !important;
    }

    ul[role="listbox"] li {
        background-color: #111827 !important;
        color: #F8FAFC !important;
    }

    ul[role="listbox"] li:hover {
        background-color: #1E293B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
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
