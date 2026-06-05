from __future__ import annotations

import streamlit as st

NAV_ITEMS = [
    ("Betting Board", "▣", "Live EV board"),
    ("Line Movement / CLV", "↗", "Market tracking"),
    ("Model Lab", "⌘", "Model diagnostics"),
    ("Data Maintenance", "▤", "Ingestion control"),
    ("Bankroll", "▥", "Ledger and risk"),
]


def _sidebar_section(label: str) -> None:
    st.sidebar.markdown(f'<div class="sidebar-section">{label}</div>', unsafe_allow_html=True)


def render_sidebar():
    """Render persistent left navigation without changing workspace backends."""

    st.sidebar.image("assets/ufc_betting_logo.png", width=190)
    st.sidebar.markdown(
        '<div class="sidebar-note">INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True,
    )

    if "page" not in st.session_state:
        st.session_state.page = "Betting Board"
    if st.session_state.page == "Bet Ledger / Bankroll":
        st.session_state.page = "Bankroll"

    _sidebar_section("Workspaces")
    for page, icon, caption in NAV_ITEMS:
        active = page == st.session_state.page
        label = f"{icon}  {page}"
        if st.sidebar.button(label, use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.page = page
            st.rerun()
        if active:
            st.sidebar.markdown(
                f'<div class="sidebar-note" style="margin:-.35rem 0 .35rem 1rem;color:#9aa8bd;">{caption}</div>',
                unsafe_allow_html=True,
            )

    st.sidebar.markdown("---")
    page = st.session_state.page

    if page == "Betting Board":
        _sidebar_section("Filters")
        st.sidebar.caption("Use in-page controls for event, confidence, odds, and EV thresholds.")
        st.sidebar.toggle("Show only positive EV", value=True, key="sidebar_positive_ev_hint")
        st.sidebar.toggle("Hide fights without odds", value=True, key="sidebar_odds_hint")
        _sidebar_section("Legend")
        st.sidebar.markdown("🟢 Strong Bet  \n🔵 Lean Bet  \n🟡 Watchlist  \n⚪ Pass")
    elif page == "Line Movement / CLV":
        _sidebar_section("CLV Summary")
        st.sidebar.caption("Market snapshots, closing lines, and CLV results are loaded from canonical artifacts.")
        st.sidebar.selectbox("Market Type", ["Moneyline"], key="sidebar_clv_market")
    elif page == "Model Lab":
        _sidebar_section("Model Lab Navigation")
        st.sidebar.markdown("▣ Model Performance  \n▤ Feature Importance  \n↗ Live Prediction Audit")
        st.sidebar.caption("Read-only diagnostics; no retraining controls are added.")
    elif page == "Data Maintenance":
        _sidebar_section("Data Maintenance")
        st.sidebar.markdown("▣ Dataset Health  \n↗ Event Discovery  \n▤ Final Staged Review  \n⌁ Audit History")
        st.sidebar.caption("Following the consolidated Final Staged Review architecture.")
    elif page == "Bankroll":
        _sidebar_section("Bankroll Navigation")
        st.sidebar.markdown("▣ Overview  \n▤ Bet Ledger  \n↗ Performance  \n⚙ Risk Settings")

    st.sidebar.markdown("---")
    _sidebar_section("Quick Actions")
    if st.sidebar.button("↻  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("Workflow-specific actions remain inside the workspaces that consume their artifacts.")

    st.sidebar.markdown(
        """
        <div class="sidebar-version">
            UFC AI Betting Intelligence<br/>v1.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page
