from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.betting_board_artifacts import load_upcoming_events
from utils.bankroll_artifacts import load_bet_ledger
from utils.data_loader import load_parquet
from pipeline.common.paths import BETTING_BOARD_PATH, MARKET_SNAPSHOTS_PATH

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



def _clv_snapshots() -> pd.DataFrame:
    snapshots = load_parquet(MARKET_SNAPSHOTS_PATH)
    if snapshots is None or snapshots.empty:
        return pd.DataFrame()
    snapshots = snapshots.copy()
    snapshots["snapshot_timestamp"] = pd.to_datetime(
        snapshots.get("snapshot_timestamp"), utc=True, errors="coerce"
    )
    snapshots["commence_time"] = pd.to_datetime(
        snapshots.get("commence_time"), utc=True, errors="coerce"
    )
    return snapshots


def _clv_event_names(snapshots: pd.DataFrame) -> list[str]:
    if snapshots.empty or "event_name" not in snapshots.columns:
        return ["All Events"]
    event_dates = (
        snapshots.dropna(subset=["event_name"])
        .groupby("event_name")["commence_time"]
        .min()
        .sort_values(na_position="last")
    )
    return ["All Events", *[str(name) for name in event_dates.index if str(name).strip()]]


def _clv_date_bounds(snapshots: pd.DataFrame) -> tuple:
    if snapshots.empty:
        return ()
    dates = snapshots["commence_time"].dropna()
    if dates.empty:
        dates = snapshots["snapshot_timestamp"].dropna()
    if dates.empty:
        return ()
    return (dates.min().date(), dates.max().date())


def _clv_books(snapshots: pd.DataFrame) -> list[str]:
    if snapshots.empty or "bookmaker" not in snapshots.columns:
        return ["All Books"]
    books = sorted({str(book).strip() for book in snapshots["bookmaker"].dropna() if str(book).strip()})
    return ["All Books", *books]


def _clv_sidebar_summary(snapshots: pd.DataFrame) -> dict[str, str]:
    if snapshots.empty:
        return {
            "Bets Tracked": "0",
            "Beat Closing Line %": "0.0%",
            "Average CLV": "+0.0%",
            "Median CLV": "+0.0%",
            "Total CLV": "+0.0%",
        }
    grouped = snapshots.sort_values("snapshot_timestamp").groupby(["fight_id", "bookmaker"], dropna=False)
    values = []
    for _, group in grouped:
        first = group.iloc[0]
        last = group.iloc[-1]
        pick = str(first.get("model_pick", "")).strip()
        use_blue = pick and pick == str(first.get("blue_fighter", "")).strip()
        taken = first.get("blue_american_odds" if use_blue else "red_american_odds")
        closing = last.get("blue_american_odds" if use_blue else "red_american_odds")
        try:
            taken = float(taken)
            closing = float(closing)
        except (TypeError, ValueError):
            continue
        taken_decimal = 1 + taken / 100 if taken > 0 else 1 + 100 / abs(taken) if taken else None
        closing_decimal = 1 + closing / 100 if closing > 0 else 1 + 100 / abs(closing) if closing else None
        if taken_decimal and closing_decimal:
            values.append((taken_decimal / closing_decimal - 1) * 100)
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        beat = avg = median = total = 0.0
    else:
        beat = (series >= 0).mean() * 100
        avg = series.mean()
        median = series.median()
        total = series.sum()
    return {
        "Bets Tracked": f"{len(series):,}",
        "Beat Closing Line %": f"{beat:.1f}%",
        "Average CLV": f"{avg:+.1f}%",
        "Median CLV": f"{median:+.1f}%",
        "Total CLV": f"{total:+.1f}%",
    }


def _render_clv_filters() -> None:
    _sidebar_section("Filters", compact=True)
    snapshots = _clv_snapshots()
    bounds = _clv_date_bounds(snapshots)
    if bounds:
        st.sidebar.date_input("Date Range", value=bounds, key="sidebar_clv_date_range")
    else:
        st.sidebar.caption("Date Range: no market dates available")

    events = _clv_event_names(snapshots)
    if st.session_state.get("sidebar_clv_event") not in events:
        st.session_state["sidebar_clv_event"] = "All Events"
    st.sidebar.selectbox("Event", events, key="sidebar_clv_event")

    books = _clv_books(snapshots)
    if st.session_state.get("sidebar_clv_book") not in books:
        st.session_state["sidebar_clv_book"] = "All Books"
    st.sidebar.selectbox("Sportsbook", books, key="sidebar_clv_book")
    st.sidebar.selectbox("Market Type", ["Moneyline"], key="sidebar_clv_market")
    st.sidebar.toggle("Show My Bets Only", value=False, key="sidebar_clv_my_bets_only")

    st.sidebar.markdown('<div class="sidebar-divider compact"></div>', unsafe_allow_html=True)
    _sidebar_section("CLV Summary (All-Time)", compact=True)
    for label, value in _clv_sidebar_summary(snapshots).items():
        st.sidebar.markdown(
            f'<div class="sidebar-stat-row"><span>{label}</span><strong>{value}</strong></div>',
            unsafe_allow_html=True,
        )
    st.sidebar.caption("All times shown in America/Chicago")

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
        _render_clv_filters()
    elif page == "Model Lab":
        _sidebar_section("Filters", compact=True)
        st.sidebar.selectbox("Model View", ["Model Performance", "Feature Importance", "Live Prediction Audit"], key="sidebar_model_lab_view")
        st.sidebar.caption("Read-only diagnostics; no retraining controls are added.")
    elif page == "Data Maintenance":
        _sidebar_section("Filters", compact=True)
        st.sidebar.selectbox("Workflow Area", ["Dataset Health", "Event Discovery", "Final Staged Review", "Audit History"], key="sidebar_dm_area")
        st.sidebar.caption("Following the consolidated Final Staged Review architecture.")
    elif page == "Bankroll":
        _sidebar_section("Bankroll Status", compact=True)
        ledger = load_bet_ledger()
        open_count = 0 if ledger.empty else int(ledger["result"].astype(str).str.lower().isin(["open", "pending", ""]).sum())
        st.sidebar.caption(f"{len(ledger):,} ledger bets · {open_count:,} open")

    st.sidebar.markdown("---")
    _sidebar_section("Quick Actions")
    if page == "Bankroll":
        if st.sidebar.button("＋  Add New Bet", use_container_width=True):
            st.session_state["bankroll_dialog"] = "add"
            st.rerun()
        if st.sidebar.button("◎  Settle Bet", use_container_width=True):
            st.session_state["bankroll_dialog"] = "settle"
            st.rerun()
        ledger = load_bet_ledger()
        st.sidebar.download_button(
            "⇩  Export Ledger",
            data="" if ledger.empty else ledger.to_csv(index=False),
            file_name="ufc_bet_ledger.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=ledger.empty,
        )
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
