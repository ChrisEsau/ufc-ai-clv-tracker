# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================


import streamlit as st

from utils.theme import apply_theme
from tabs.betting_board_v2 import render_betting_board
from tabs.line_movement import render_line_movement
from tabs.bankroll import render_bankroll
from tabs.model_lab import render_model_lab
from tabs.data_maintenance import render_data_maintenance
from utils.sidebar import render_sidebar
import utils.model_lab_workflows as model_lab_workflows
from utils.model_lab_feature_selection import render_feature_checklist

# ============================================================
# PAGE CONFIG
# ============================================================

apply_theme()

# Model Lab toggle polish. Keep this narrow: only Streamlit toggle widgets.
st.markdown(
    """
    <style>
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] label *,
    div[data-testid="stToggle"] p,
    div[data-testid="stToggle"] span {
        color: #f8fafc !important;
        background: transparent !important;
        background-color: transparent !important;
        opacity: 1 !important;
    }
    div[data-testid="stToggle"] [role="switch"][aria-checked="true"] {
        background-color: #2563eb !important;
        border-color: #60a5fa !important;
    }
    div[data-testid="stToggle"] [role="switch"][aria-checked="true"] * {
        background-color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Keep existing Model Lab save behavior, but replace the old text override
# feature selector with the explicit checkbox bundle selector.
model_lab_workflows._render_feature_bundle_editor = render_feature_checklist

page = render_sidebar()

if page == "Betting Board":
    render_betting_board()
elif page == "Line Movement / CLV":
    render_line_movement()
elif page in {"Bankroll", "Bet Ledger / Bankroll"}:
    render_bankroll()
elif page == "Model Lab":
    render_model_lab()
elif page == "Data Maintenance":
    render_data_maintenance()
