import streamlit as st



def render_sidebar():

    # =========================================================
    # LOGO
    # =========================================================

    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=190,
    )

    st.sidebar.markdown("---")

    # =========================================================
    # WORKFLOW NAVIGATION
    # =========================================================

    st.sidebar.markdown(
        '<div class="sidebar-section">WORKFLOW</div>',
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Betting Board"

    if st.sidebar.button("🎯  Betting Board", use_container_width=True):
        st.session_state.page = "Betting Board"
        st.rerun()

    if st.sidebar.button("📈  Line Movement / CLV", use_container_width=True):
        st.session_state.page = "Line Movement / CLV"
        st.rerun()

    if st.sidebar.button("💰  Bet Ledger / Bankroll", use_container_width=True):
        st.session_state.page = "Bet Ledger / Bankroll"
        st.rerun()

    if st.sidebar.button("🧪  Model Lab", use_container_width=True):
        st.session_state.page = "Model Lab"
        st.rerun()

    if st.sidebar.button("🛠️  Data Maintenance", use_container_width=True):
        st.session_state.page = "Data Maintenance"
        st.rerun()

    page = st.session_state.page

    st.sidebar.markdown("---")

    # =========================================================
    # QUICK ACTIONS
    # =========================================================

    st.sidebar.markdown(
        '<div class="sidebar-section">QUICK ACTIONS</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("▶  Run Pipeline", use_container_width=True):
        st.sidebar.info("Pipeline trigger placeholder")

    if st.sidebar.button("⬇  Export Board", use_container_width=True):
        st.sidebar.info("Export placeholder")

    st.sidebar.markdown("---")

    # =========================================================
    # FILTER PRESETS
    # =========================================================

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

