# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================


import streamlit as st

from utils.theme import apply_theme
from tabs.betting_board_v2 import render_betting_board
from tabs.market_intelligence import render_market_intelligence
from tabs.bankroll_market_risk import render_bankroll
from tabs.model_lab_refactored import render_model_lab
from tabs.data_maintenance import render_data_maintenance
from tabs.operations_center import render_operations_center
from utils.sidebar_refactored import render_sidebar

# ============================================================
# PAGE CONFIG
# ============================================================

apply_theme()

page = render_sidebar()

if page == "Betting Board":
    render_betting_board()
elif page == "Market Intelligence":
    render_market_intelligence()
elif page in {"Bankroll", "Bet Ledger / Bankroll"}:
    render_bankroll()
elif page == "Model Lab":
    render_model_lab()
elif page == "Data Maintenance":
    render_data_maintenance()
elif page == "Operations Center":
    render_operations_center()
