import streamlit as st


def render_section_header(title):
    st.markdown(
        f'''
        <div class="section-header">
            {title}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_panel_open():
    st.markdown(
        """
        <div class="panel">
        """,
        unsafe_allow_html=True,
    )


def render_panel_close():
    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(status):

    status_classes = {
        "OFFICIAL BET": "pill-official",
        "WATCHLIST": "pill-watchlist",
        "INVALID MODEL DATA": "pill-danger",
        "LOW ODDS MATCH": "pill-danger",
        "SPARSE FEATURES": "pill-danger",
    }

    css_class = status_classes.get(
        status,
        "pill-info",
    )

    return f"""
    <span class="status-pill {css_class}">
        {status}
    </span>
    """
