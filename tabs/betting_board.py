import numpy as np
import pandas as pd
import streamlit as st

from pipeline.common.paths import BETTING_BOARD_PATH
from utils.betting_board_artifacts import (
    event_label,
    get_betting_artifact_status,
    get_upcoming_artifact_status,
    load_upcoming_events,
    load_upcoming_fights,
)
from utils.data_loader import load_parquet
from utils.dm_workflow_status import remember_launched_workflow, render_workflow_status
from utils.github_actions import trigger_workflow
from utils.panels import render_section_header
from utils.ui_components import american, money, pct, render_metric


REFRESH_UPCOMING_WORKFLOW = "run-refresh-upcoming-events.yml"
SELECTED_EVENT_WORKFLOW = "run-betting-board-selected-event.yml"


def _artifact_health_table():
    status = pd.concat(
        [
            get_upcoming_artifact_status().assign(group="Card selection"),
            get_betting_artifact_status().assign(group="Betting outputs"),
        ],
        ignore_index=True,
    )
    return status[["group", "artifact", "exists", "size", "modified_utc", "path"]]


def _selected_event_id(event_row):
    return event_row.get("ufcstats_event_id") or event_row.get("event_id")


def render_upcoming_event_selection():
    render_section_header("Upcoming Event Selection")

    with st.expander("Select an upcoming UFCStats event for betting predictions", expanded=True):
        st.caption(
            "Refresh the UFCStats upcoming-events artifact, choose a card, then launch the selected-event "
            "betting workflow. The workflow builds the live card, model predictions, market odds, and "
            "betting board outputs from canonical data/ paths."
        )

        st.dataframe(_artifact_health_table(), use_container_width=True, hide_index=True)

        control_cols = st.columns([1, 1])

        with control_cols[0]:
            if st.button("Refresh Upcoming Events", use_container_width=True):
                ok, msg = trigger_workflow(REFRESH_UPCOMING_WORKFLOW)
                if ok:
                    remember_launched_workflow(
                        "betting_refresh_upcoming_events",
                        "Refresh Upcoming Events",
                        REFRESH_UPCOMING_WORKFLOW,
                    )
                    st.success(msg)
                else:
                    st.error(msg)

        with control_cols[1]:
            st.caption("Refresh before selecting a new card if UFCStats has changed.")

        render_workflow_status("betting_refresh_upcoming_events")

        events, events_error = load_upcoming_events()
        fights, fights_error = load_upcoming_fights()

        if events_error:
            st.warning(events_error)
            return None

        if events.empty:
            st.warning("No upcoming events are available yet. Refresh upcoming events first.")
            return None

        events = events.sort_values("ufcstats_event_date", na_position="last").reset_index(drop=True)
        event_options = events.to_dict("records")
        selected_event = st.selectbox(
            "Upcoming event",
            options=event_options,
            format_func=event_label,
            key="betting_selected_upcoming_event",
        )

        selected_event_id = _selected_event_id(selected_event)

        event_cols = [
            column
            for column in [
                "ufcstats_event_id",
                "ufcstats_event_date",
                "ufcstats_event_name",
                "ufcstats_event_location",
                "ufcstats_event_url",
            ]
            if column in events.columns
        ]
        st.dataframe(pd.DataFrame([selected_event])[event_cols], use_container_width=True, hide_index=True)

        if fights_error:
            st.warning(fights_error)
        elif not fights.empty and "event_id" in fights.columns:
            selected_fights = fights[fights["event_id"].astype(str) == str(selected_event_id)]
            fight_cols = [
                column
                for column in [
                    "fight_order",
                    "red_fighter",
                    "blue_fighter",
                    "weight_class",
                    "fight_id",
                ]
                if column in selected_fights.columns
            ]
            st.markdown(f"**Selected card fights:** {len(selected_fights)}")
            st.dataframe(selected_fights[fight_cols], use_container_width=True, hide_index=True)

        if st.button("Run Betting Predictions for Selected Event", type="primary", use_container_width=True):
            ok, msg = trigger_workflow(
                SELECTED_EVENT_WORKFLOW,
                inputs={"event_id": str(selected_event_id)},
            )
            if ok:
                remember_launched_workflow(
                    "betting_selected_event",
                    "Run Betting Predictions for Selected Event",
                    SELECTED_EVENT_WORKFLOW,
                    inputs={"event_id": str(selected_event_id)},
                )
                st.success(msg)
            else:
                st.error(msg)

        render_workflow_status("betting_selected_event")

        return selected_event


def render_board_filters(board):
    with st.expander("Betting Board Filters", expanded=True):
        events = sorted(board["event_name"].dropna().unique().tolist()) if "event_name" in board.columns else []
        selected_event = st.selectbox("Event", ["All Events"] + events)

        status_order = [
            "OFFICIAL BET",
            "WATCHLIST",
            "LOW ODDS MATCH",
            "SPARSE FEATURES",
            "INVALID MODEL DATA",
            "NO BET",
        ]
        available_statuses = [s for s in status_order if "bet_status" in board.columns and s in board["bet_status"].dropna().unique()]
        selected_statuses = st.multiselect("Bet status", available_statuses, default=available_statuses)
        show_only_actionable = st.checkbox("Show only actionable statuses", value=False)
        min_ev = st.slider("Minimum EV", min_value=-100.0, max_value=100.0, value=-100.0, step=1.0)
        min_confidence = st.slider("Minimum confidence", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

    filtered = board.copy()

    if selected_event != "All Events" and "event_name" in filtered.columns:
        filtered = filtered[filtered["event_name"] == selected_event]

    if selected_statuses and "bet_status" in filtered.columns:
        filtered = filtered[filtered["bet_status"].isin(selected_statuses)]

    if show_only_actionable and "bet_status" in filtered.columns:
        filtered = filtered[filtered["bet_status"].isin(["OFFICIAL BET", "WATCHLIST"])]

    ev_col = "best_ev_pct" if "best_ev_pct" in filtered.columns else "best_ev"
    if ev_col in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered[ev_col], errors="coerce").fillna(-999) >= min_ev]

    if "best_confidence" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["best_confidence"], errors="coerce").fillna(0) >= min_confidence]

    return filtered


def render_summary_cards(filtered):
    total_fights = len(filtered)
    official_bets = int((filtered["bet_status"] == "OFFICIAL BET").sum()) if "bet_status" in filtered.columns else 0
    watchlist = int((filtered["bet_status"] == "WATCHLIST").sum()) if "bet_status" in filtered.columns else 0
    best_ev_col = "best_ev_pct" if "best_ev_pct" in filtered.columns else "best_ev"
    best_ev = filtered[best_ev_col].max() if best_ev_col in filtered.columns and not filtered.empty else np.nan
    recommended_stake = filtered["recommended_stake"].sum() if "recommended_stake" in filtered.columns and not filtered.empty else 0
    latest_market_time = str(filtered["snapshot_timestamp"].max()) if "snapshot_timestamp" in filtered.columns and not filtered.empty else "N/A"

    cols = st.columns(6)
    with cols[0]:
        render_metric("Fights", total_fights)
    with cols[1]:
        render_metric("Official Bets", official_bets)
    with cols[2]:
        render_metric("Watchlist", watchlist)
    with cols[3]:
        render_metric("Best EV", f"{best_ev:.1f}%" if pd.notna(best_ev) else "N/A")
    with cols[4]:
        render_metric("Total Stake", money(recommended_stake))
    with cols[5]:
        render_metric("Latest Market", latest_market_time[:16])


def build_display_frame(filtered):
    display = filtered.copy()

    if "red_fighter" in display.columns and "blue_fighter" in display.columns:
        display["fight"] = display["red_fighter"].fillna("") + " vs " + display["blue_fighter"].fillna("")

    if "best_american_odds" in display.columns:
        display["odds_display"] = display["best_american_odds"].apply(american)

    for column in ["best_prob", "best_implied_prob", "best_edge"]:
        if column in display.columns:
            display[f"{column}_display"] = display[column].apply(pct)

    ev_col = "best_ev_pct" if "best_ev_pct" in display.columns else "best_ev"
    if ev_col in display.columns:
        display["best_ev_display"] = display[ev_col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    if "best_confidence" in display.columns:
        display["best_confidence_display"] = display["best_confidence"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    if "recommended_stake" in display.columns:
        display["stake_display"] = display["recommended_stake"].apply(money)

    status_rank = {
        "OFFICIAL BET": 0,
        "WATCHLIST": 1,
        "LOW ODDS MATCH": 2,
        "SPARSE FEATURES": 3,
        "INVALID MODEL DATA": 4,
        "NO BET": 5,
    }
    display["_status_rank"] = display["bet_status"].map(status_rank).fillna(99) if "bet_status" in display.columns else 99

    sort_cols = ["_status_rank"]
    ascending = [True]
    ev_sort_col = "best_ev_pct" if "best_ev_pct" in display.columns else "best_ev"
    if ev_sort_col in display.columns:
        sort_cols.append(ev_sort_col)
        ascending.append(False)

    return display.sort_values(sort_cols, ascending=ascending, na_position="last")


def render_action_board(filtered):
    render_section_header("Primary Action Board")

    display = build_display_frame(filtered)
    main_cols = [
        "bet_status",
        "fight",
        "best_side",
        "odds_display",
        "best_prob_display",
        "best_implied_prob_display",
        "best_edge_display",
        "best_ev_display",
        "best_confidence_display",
        "stake_display",
        "bet_reason",
    ]
    main_cols = [column for column in main_cols if column in display.columns]
    st.dataframe(display[main_cols], use_container_width=True, hide_index=True)


def render_status_and_diagnostics(filtered):
    render_section_header("Status Breakdown")
    if "bet_status" in filtered.columns:
        status_counts = filtered["bet_status"].value_counts().rename_axis("status").reset_index(name="count")
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No bet_status column found.")

    render_section_header("Artifact Diagnostics")
    st.dataframe(get_betting_artifact_status(), use_container_width=True, hide_index=True)


def render_selected_fight_detail(filtered):
    render_section_header("Selected Fight Detail")
    if filtered.empty:
        st.info("No fights match the current filters.")
        return

    display = build_display_frame(filtered)
    if "fight" not in display.columns:
        st.info("Fight names are not available in this artifact.")
        return

    selected_fight = st.selectbox("Inspect fight", display["fight"].dropna().unique().tolist())
    selected_rows = display[display["fight"] == selected_fight]
    detail_cols = [
        "event_name",
        "fight",
        "best_side",
        "bet_status",
        "best_american_odds",
        "best_prob",
        "best_implied_prob",
        "best_edge",
        "best_ev",
        "best_confidence",
        "recommended_stake",
        "bet_reason",
        "odds_match_score",
        "odds_match_type",
    ]
    detail_cols = [column for column in detail_cols if column in selected_rows.columns]
    st.dataframe(selected_rows[detail_cols], use_container_width=True, hide_index=True)


def render_betting_board():
    st.title("UFC Betting Intelligence Platform")
    st.caption("Betting Board — event selection, model probability, market odds, EV, quality gates, and recommended action.")

    render_upcoming_event_selection()

    board = load_parquet(BETTING_BOARD_PATH)

    if board.empty:
        st.warning("No betting board data found. Select an upcoming event and run the betting workflow.")
        return

    filtered = render_board_filters(board)
    render_summary_cards(filtered)
    render_action_board(filtered)
    render_status_and_diagnostics(filtered)
    render_selected_fight_detail(filtered)

    with st.expander("Raw Betting Board Data", expanded=False):
        st.dataframe(board, use_container_width=True, hide_index=True)
