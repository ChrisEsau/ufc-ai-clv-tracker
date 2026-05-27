import streamlit as st


def render_sidebar():

    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=190,
    )

    st.sidebar.markdown("---")

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

    for label, image_path in workflow_items:
        st.sidebar.image(image_path, use_container_width=True)

        if st.sidebar.button(
            f"Open {label}",
            key=f"workflow_{label}",
            use_container_width=True,
        ):
            st.session_state.page = label
            st.rerun()

    page = st.session_state.page

    st.sidebar.markdown("---")

    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return page

