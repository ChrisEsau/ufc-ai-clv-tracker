# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================


import streamlit as st

from utils.theme import apply_theme
from tabs.betting_board_v2 import render_betting_board
from tabs.line_movement import render_line_movement
from tabs.bankroll import render_bankroll
from tabs.model_lab_refactored import render_model_lab
from tabs.data_maintenance import render_data_maintenance
from utils.sidebar import render_sidebar

# ============================================================
# PAGE CONFIG
# ============================================================

apply_theme()

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
