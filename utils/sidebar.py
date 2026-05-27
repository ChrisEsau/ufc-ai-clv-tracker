# utils/sidebar.py

import streamlit as st


def render_sidebar():
    
    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=240,
    )

    st.sidebar.markdown(
        '<div class="sidebar-section">WORKFLOW</div>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "",
        [
            "Betting Board",
            "Line Movement / CLV",
            "Model Lab",
            "Data Maintenance",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        '<div class="sidebar-section">QUICK ACTIONS</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("▶ Run Full Pipeline"):
        st.sidebar.info("Pipeline trigger placeholder")

    if st.sidebar.button("🔄 Clear Cache / Refresh"):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("⬇ Export Board"):
        st.sidebar.info("Export placeholder")

    st.sidebar.markdown(
        '<div class="sidebar-section">FILTER PRESETS</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.selectbox(
        "Select Preset",
        [
            "Default",
            "Official Bets Only",
            "Watchlist",
            "High EV",
            "Data Issues",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-version">
            v1.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page
