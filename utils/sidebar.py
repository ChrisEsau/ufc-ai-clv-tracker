# utils/sidebar.py

import streamlit as st


def set_page_from_query():
    page = st.query_params.get("page", "Betting Board")
    return page


def nav_item(label, icon_svg, key):
    current_page = set_page_from_query()
    active_class = "active-workflow" if current_page == label else ""

    st.sidebar.markdown(
        f"""
        <a class="workflow-link {active_class}" href="?page={key}" target="_self">
            <span class="workflow-svg">{icon_svg}</span>
            <span>{label}</span>
        </a>
        """,
        unsafe_allow_html=True,
    )


target_svg = """<svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2a10 10 0 1 0 10 10h-2a8 8 0 1 1-8-8V2zm0 4a6 6 0 1 0 6 6h-2a4 4 0 1 1-4-4V6zm0 4a2 2 0 1 0 0 4a2 2 0 0 0 0-4z"/></svg>"""
chart_svg = """<svg viewBox="0 0 24 24"><path fill="currentColor" d="M3 19h18v2H3V3h2v14h16v2H3zm4-3l4-4l3 3l6-7l1.5 1.3l-7.4 8.6l-3.1-3.1L8.4 17L7 16z"/></svg>"""
flask_svg = """<svg viewBox="0 0 24 24"><path fill="currentColor" d="M9 2h6v2h-1v5.1l5.7 9.9A2 2 0 0 1 18 22H6a2 2 0 0 1-1.7-3L10 9.1V4H9V2zm3 8l-5.8 10h11.6L12 10z"/></svg>"""
database_svg = """<svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 3C7 3 3 4.8 3 7v10c0 2.2 4 4 9 4s9-1.8 9-4V7c0-2.2-4-4-9-4zm0 2c4.2 0 7 1.2 7 2s-2.8 2-7 2s-7-1.2-7-2s2.8-2 7-2zm0 14c-4.2 0-7-1.2-7-2v-2.1c1.6 1.3 4.3 2.1 7 2.1s5.4-.8 7-2.1V17c0 .8-2.8 2-7 2zm0-4c-4.2 0-7-1.2-7-2v-2.1c1.6 1.3 4.3 2.1 7 2.1s5.4-.8 7-2.1V13c0 .8-2.8 2-7 2z"/></svg>"""


def render_sidebar():
    st.sidebar.image(
        "assets/ufc_betting_logo.png",
        width=190,
    )

    st.sidebar.markdown(
        '<div class="sidebar-section">WORKFLOW</div>',
        unsafe_allow_html=True,
    )

    nav_item("Betting Board", target_svg, "Betting%20Board")
    nav_item("Line Movement / CLV", chart_svg, "Line%20Movement%20/%20CLV")
    nav_item("Model Lab", flask_svg, "Model%20Lab")
    nav_item("Data Maintenance", database_svg, "Data%20Maintenance")

    return set_page_from_query()
