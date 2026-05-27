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
from tabs.model_lab import render_model_lab
from tabs.data_maintenance import render_data_maintenance
from utils.sidebar import render_sidebar

# ============================================================
# PAGE CONFIG
# ============================================================

apply_theme()

st.title("UFC Betting Intelligence Platform")
st.caption(
    "Model probability, market movement, EV, CLV, and betting decision intelligence."
)

page = render_sidebar()

if page == "Betting Board":
    render_betting_board()
elif page == "Line Movement / CLV":
    render_line_movement()
elif page == "Model Lab":
    render_model_lab()
elif page == "Data Maintenance":
    render_data_maintenance()

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

with tab3:
    render_model_lab()

with tab4:
    render_data_maintenance()
