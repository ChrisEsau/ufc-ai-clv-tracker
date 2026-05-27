import streamlit as st


def render_sidebar():
    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=190,
    )

    st.sidebar.markdown(
        '<div class="sidebar-section">WORKFLOW</div>',
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Betting Board"

    nav_items = [
        ("Betting Board", "🎯"),
        ("Line Movement / CLV", "📈"),
        ("Model Lab", "🧪"),
        ("Data Maintenance", "🗄️"),
    ]

    for label, icon in nav_items:
        button_label = f"{icon}  {label}"

        if st.sidebar.button(
            button_label,
            key=f"nav_{label}",
            use_container_width=True,
        ):
            st.session_state.page = label
            st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        '<div class="sidebar-section">QUICK ACTIONS</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("▶  Run Pipeline", use_container_width=True):
        st.sidebar.info("Pipeline trigger placeholder")

    if st.sidebar.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("⬇  Export Board", use_container_width=True):
        st.sidebar.info("Export placeholder")

    if st.sidebar.button("🔔  Notification Log", use_container_width=True):
        st.sidebar.info("Notification log placeholder")

    st.sidebar.markdown("---")

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

    return st.session_state.page
