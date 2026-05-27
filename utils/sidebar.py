import streamlit as st


def render_sidebar():

    # =========================================================
    # LOGO
    # =========================================================

    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=320,
    )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    # =========================================================
    # WORKFLOW
    # =========================================================

    st.sidebar.markdown(
        '<div class="sidebar-section">WORKFLOW</div>',
        unsafe_allow_html=True,
    )

    workflow_items = [
        ("Betting Board", "assets/icons/betting_board.png"),
        ("Line Movement / CLV", "assets/icons/line_movement.png"),
        ("Model Lab", "assets/icons/model_lab.png"),
        ("Data Maintenance", "assets/icons/data_maintenance.png"),
    ]

    if "page" not in st.session_state:
        st.session_state.page = "Betting Board"

    for label, icon_path in workflow_items:

        active = st.session_state.page == label

        row_class = (
            "workflow-row-active"
            if active
            else "workflow-row"
        )

        st.sidebar.markdown(
            f'<div class="{row_class}">',
            unsafe_allow_html=True,
        )

        cols = st.sidebar.columns([1, 5])

        with cols[0]:
            st.image(
                icon_path,
                width=20,
            )

        with cols[1]:
            if st.button(
                label,
                key=f"workflow_{label}",
                use_container_width=True,
            ):
                st.session_state.page = label
                st.rerun()

        st.sidebar.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    page = st.session_state.page

    st.sidebar.markdown("---")

    # =========================================================
    # QUICK ACTIONS
    # =========================================================

    st.sidebar.markdown(
        '<div class="sidebar-section">QUICK ACTIONS</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "▶  Run Pipeline",
        use_container_width=True,
    ):
        st.sidebar.info("Pipeline trigger placeholder")

    if st.sidebar.button(
        "🔄  Refresh Data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button(
        "⬇  Export Board",
        use_container_width=True,
    ):
        st.sidebar.info("Export placeholder")

    if st.sidebar.button(
        "🔔  Notification Log",
        use_container_width=True,
    ):
        st.sidebar.info("Notification log placeholder")

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
