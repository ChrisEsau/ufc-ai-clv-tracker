from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.betting_board_artifacts import load_upcoming_events
from utils.data_loader import load_parquet
from pipeline.common.paths import BETTING_BOARD_PATH

NAV_ITEMS = [
    ("Betting Board", "▣", "Live EV board"),
    ("Line Movement / CLV", "↗", "Market tracking"),
    ("Model Lab", "⌘", "Model diagnostics"),
    ("Data Maintenance", "▤", "Ingestion control"),
    ("Bankroll", "▥", "Ledger and risk"),
]


def _sidebar_section(label: str, compact: bool = False) -> None:
    compact_class = " compact" if compact else ""
    st.sidebar.markdown(
        f'<div class="sidebar-section{compact_class}">{label}</div>', unsafe_allow_html=True
    )


def _betting_board_event_names() -> list[str]:
    """Return event filter choices sorted by upcoming date, nearest first."""

    event_dates: dict[str, pd.Timestamp] = {}
    events, _ = load_upcoming_events()
    if events is not None and not events.empty:
        name_column = next(
            (column for column in ["ufcstats_event_name", "event_name"] if column in events.columns),
            None,
        )
        date_column = next(
            (column for column in ["ufcstats_event_date", "event_date"] if column in events.columns),
            None,
        )
        if name_column:
            for _, row in events.iterrows():
                name = str(row.get(name_column) or "").strip()
                if not name:
                    continue
                parsed_date = pd.to_datetime(row.get(date_column), errors="coerce") if date_column else pd.NaT
                current_date = event_dates.get(name, pd.NaT)
                if pd.isna(current_date) or (not pd.isna(parsed_date) and parsed_date < current_date):
                    event_dates[name] = parsed_date

    board = load_parquet(BETTING_BOARD_PATH)
    if board is not None and not board.empty and "event_name" in board.columns:
        for name in board["event_name"].dropna().astype(str):
            clean_name = name.strip()
            if clean_name and clean_name not in event_dates:
                event_dates[clean_name] = pd.NaT

    sorted_events = sorted(
        event_dates.items(),
        key=lambda item: (pd.isna(item[1]), item[1] if not pd.isna(item[1]) else pd.Timestamp.max, item[0]),
    )
    return ["All Events", *[name for name, _ in sorted_events]]


def _betting_board_date_bounds() -> tuple:
    events, _ = load_upcoming_events()
    dates = []
    if events is not None and not events.empty:
        for column in ["ufcstats_event_date", "event_date"]:
            if column in events.columns:
                parsed = pd.to_datetime(events[column], errors="coerce").dropna()
                dates.extend(parsed.dt.date.tolist())
                break
    if not dates:
        return ()
    return (min(dates), max(dates))


def _render_betting_board_filters() -> None:
    _sidebar_section("Filters", compact=True)
    event_names = _betting_board_event_names()
    if st.session_state.get("bb_filter_event") not in event_names:
        st.session_state["bb_filter_event"] = "All Events"
    st.sidebar.selectbox(
        "Event",
        event_names,
        key="bb_filter_event",
    )

    date_bounds = _betting_board_date_bounds()
    if date_bounds:
        st.sidebar.date_input("Date Range", value=date_bounds, key="bb_filter_date_range")
    else:
        st.sidebar.caption("Date Range: no upcoming dates available")

    st.sidebar.slider(
        "Odds Range",
        min_value=-500,
        max_value=500,
        value=st.session_state.get("bb_filter_odds_range", (-250, 400)),
        step=10,
        key="bb_filter_odds_range",
    )
    st.sidebar.selectbox(
        "EV Threshold ($)",
        [0, 25, 50, 75, 100, 150, 200],
        index=2,
        key="bb_filter_ev_threshold",
    )
    st.sidebar.selectbox(
        "Min Model Confidence (%)",
        [0, 50, 60, 70, 75, 80, 85, 90],
        index=3,
        key="bb_filter_min_confidence",
    )
    st.sidebar.radio(
        "Kelly Stake Sizing",
        ["1/2 Kelly", "1/4 Kelly"],
        horizontal=True,
        key="bb_kelly_mode",
    )
    st.sidebar.toggle("Show Only Positive EV", value=True, key="bb_filter_positive_ev")
    st.sidebar.toggle("Hide Fights Without Odds", value=True, key="bb_filter_hide_missing_odds")


def render_sidebar():
    """Render persistent left navigation without changing workspace backends."""

    st.sidebar.image("assets/ufc_betting_logo.png", width=230)
    if "page" not in st.session_state:
        st.session_state.page = "Betting Board"
    if st.session_state.page == "Bet Ledger / Bankroll":
        st.session_state.page = "Bankroll"

    _sidebar_section("Workspaces")
    for page, icon, caption in NAV_ITEMS:
        active = page == st.session_state.page
        label = f"{icon}  {page}"
        if st.sidebar.button(
            label, use_container_width=True, type="primary" if active else "secondary"
        ):
            st.session_state.page = page
            st.rerun()

    st.sidebar.markdown('<div class="sidebar-divider compact"></div>', unsafe_allow_html=True)
    page = st.session_state.page

    if page == "Betting Board":
        _render_betting_board_filters()
    elif page == "Line Movement / CLV":
        _sidebar_section("Filters", compact=True)
        st.sidebar.selectbox("Market Type", ["Moneyline"], key="sidebar_clv_market")
        st.sidebar.selectbox("View", ["Movement", "CLV Results"], key="sidebar_clv_view")
        st.sidebar.caption("Market snapshots, closing lines, and CLV results are loaded from canonical artifacts.")
    elif page == "Model Lab":
        _sidebar_section("Filters", compact=True)
        st.sidebar.selectbox("Model View", ["Model Performance", "Feature Importance", "Live Prediction Audit"], key="sidebar_model_lab_view")
        st.sidebar.caption("Read-only diagnostics; no retraining controls are added.")
    elif page == "Data Maintenance":
        _sidebar_section("Filters", compact=True)
        st.sidebar.selectbox("Workflow Area", ["Dataset Health", "Event Discovery", "Final Staged Review", "Audit History"], key="sidebar_dm_area")
        st.sidebar.caption("Following the consolidated Final Staged Review architecture.")
    elif page == "Bankroll":
        _sidebar_section("Filters", compact=True)
        st.sidebar.selectbox("Ledger View", ["Overview", "Bet Ledger", "Performance", "Risk Settings"], key="sidebar_bankroll_view")

    st.sidebar.markdown("---")
    _sidebar_section("Quick Actions")
    if st.sidebar.button("↻  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(
        "Workflow-specific actions remain inside the workspaces that consume their artifacts."
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-version">
            UFC AI Betting Intelligence<br/>v1.0.0
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page
