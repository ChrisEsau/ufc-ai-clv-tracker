import streamlit as st
import base64

from st_click_detector import click_detector


# =========================================================
# HELPERS
# =========================================================

def image_to_base64(path):

    with open(path, "rb") as f:

        return base64.b64encode(
            f.read()
        ).decode()


# =========================================================
# SIDEBAR
# =========================================================

def render_sidebar():

    # -----------------------------------------------------
    # LOGO
    # -----------------------------------------------------

    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=190,
    )

    st.sidebar.markdown(
        "<div style='height:18px;'></div>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # WORKFLOW HEADER
    # -----------------------------------------------------

    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            WORKFLOW
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # NAVIGATION ITEMS
    # -----------------------------------------------------

    workflow_items = [
        (
            "Betting Board",
            "assets/icons/betting_board.png",
        ),

        (
            "Line Movement / CLV",
            "assets/icons/line_movement.png",
        ),

        (
            "Model Lab",
            "assets/icons/model_lab.png",
        ),

        (
            "Data Maintenance",
            "assets/icons/data_maintenance.png",
        ),
    ]

    if "page" not in st.session_state:

        st.session_state.page = "Betting Board"

    # -----------------------------------------------------
    # CLICKABLE PNG NAVIGATION
    # -----------------------------------------------------

    for label, image_path in workflow_items:

        encoded = image_to_base64(
            image_path
        )

        clicked = click_detector(
            f'''
            <a id="{label}">
                <img
                    src="data:image/png;base64,{encoded}"
                    width="100%"
                    style="
                        margin-bottom:10px;
                        border-radius:14px;
                        cursor:pointer;
                        transition:all 0.2s ease;
                    "
                >
            </a>
            '''
        )

        if clicked == label:

            st.session_state.page = label

            st.rerun()

    # -----------------------------------------------------
    # CURRENT PAGE
    # -----------------------------------------------------

    page = st.session_state.page

    st.sidebar.markdown("---")

    # -----------------------------------------------------
    # QUICK ACTIONS
    # -----------------------------------------------------

    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            QUICK ACTIONS
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "▶ Run Pipeline",
        use_container_width=True,
    ):

        st.sidebar.info(
            "Pipeline trigger placeholder"
        )

    if st.sidebar.button(
        "🔄 Refresh Data",
        use_container_width=True,
    ):

        st.cache_data.clear()

        st.rerun()

    if st.sidebar.button(
        "⬇ Export Board",
        use_container_width=True,
    ):

        st.sidebar.info(
            "Export placeholder"
        )

    if st.sidebar.button(
        "🔔 Notification Log",
        use_container_width=True,
    ):

        st.sidebar.info(
            "Notification placeholder"
        )

    st.sidebar.markdown("---")

    # -----------------------------------------------------
    # FILTER PRESETS
    # -----------------------------------------------------

    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            FILTER PRESETS
        </div>
        """,
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

    # -----------------------------------------------------
    # VERSION
    # -----------------------------------------------------

    st.sidebar.markdown(
        """
        <div class="sidebar-version">
            v1.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page

