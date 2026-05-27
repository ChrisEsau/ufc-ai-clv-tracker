# utils/sidebar.py

import streamlit as st


def render_sidebar():
    
    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=360,
    )

    st.sidebar.markdown(
        '<div class="sidebar-section">WORKFLOW</div>',
        unsafe_allow_html=True,
    )
    
    nav_items = {
        "Betting Board": "🎯",
        "Line Movement / CLV": "📈",
        "Model Lab": "🧪",
        "Data Maintenance": "🗄️",
    }
    
    if "page" not in st.session_state:
        st.session_state.page = "Betting Board"
    
    for label, icon in nav_items.items():
        active = st.session_state.page == label
        css_class = "nav-card active-nav" if active else "nav-card"
    
        if st.sidebar.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
        ):
            st.session_state.page = label
            st.rerun()
    
    page = st.session_state.page

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
