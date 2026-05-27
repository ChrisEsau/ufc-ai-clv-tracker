import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_parquet
from utils.ui_components import render_metric, money, american, pct

from utils.panels import (
    render_section_header,
    render_panel_open,
    render_panel_close,
    render_status_pill,
)

def render_betting_board():
    board = load_parquet("ufc_betting_board.parquet")

    if board.empty:
        st.warning("No betting board data found. Run the betting decision workflow.")
        return

    # paste the rest of your Betting Board code here
    # filters
    # summary cards
    # action board
    # diagnostics
    # selected fight detail
    # selected fight line movement
    # raw debug view
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    board = load_parquet("ufc_betting_board.parquet")
    
    # ============================================================
    # HEADER
    # ============================================================
    
    st.title("UFC Betting Intelligence Platform")
    st.caption("Betting Board — model probability, market odds, EV, quality gates, and recommended action.")
    
    if board.empty:
        st.warning("No betting board data found. Run the betting decision workflow.")
        st.stop()
    
    # ============================================================
    # FILTERS
    # ============================================================
    
    events = (
        sorted(board["event_name"].dropna().unique().tolist())
        if "event_name" in board.columns
        else []
    )
       
    status_order = [
        "OFFICIAL BET",
        "WATCHLIST",
        "LOW ODDS MATCH",
        "SPARSE FEATURES",
        "INVALID MODEL DATA",
        "NO BET",
    ]
    
    available_statuses = (
        [s for s in status_order if s in board["bet_status"].dropna().unique()]
        if "bet_status" in board.columns
        else []
    )
       
    filtered = board.copy()
    
    if selected_event != "All Events" and "event_name" in filtered.columns:
        filtered = filtered[filtered["event_name"] == selected_event]
    
    if selected_statuses and "bet_status" in filtered.columns:
        filtered = filtered[filtered["bet_status"].isin(selected_statuses)]
    
    if show_only_actionable and "bet_status" in filtered.columns:
        filtered = filtered[
            filtered["bet_status"].isin(["OFFICIAL BET", "WATCHLIST"])
        ]
    
    # ============================================================
    # SUMMARY CARDS
    # ============================================================
    
    total_fights = len(filtered)
    
    official_bets = (
        int((filtered["bet_status"] == "OFFICIAL BET").sum())
        if "bet_status" in filtered.columns
        else 0
    )
    
    watchlist = (
        int((filtered["bet_status"] == "WATCHLIST").sum())
        if "bet_status" in filtered.columns
        else 0
    )
    
    best_ev = (
        filtered["best_ev"].max()
        if "best_ev" in filtered.columns and not filtered.empty
        else np.nan
    )
    
    recommended_stake = (
        filtered["recommended_stake"].sum()
        if "recommended_stake" in filtered.columns and not filtered.empty
        else 0
    )
    
    latest_market_time = (
        str(filtered["snapshot_timestamp"].max())
        if "snapshot_timestamp" in filtered.columns and not filtered.empty
        else "N/A"
    )
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        render_metric("Fights", total_fights)
    
    with c2:
        render_metric("Official Bets", official_bets)
    
    with c3:
        render_metric("Watchlist", watchlist)
    
    with c4:
        render_metric("Best EV", f"{best_ev:.1f}%" if pd.notna(best_ev) else "N/A")
    
    with c5:
        render_metric("Total Stake", money(recommended_stake))
    
    with c6:
        render_metric("Latest Market", latest_market_time[:16])
    
    # ============================================================
    # MAIN ACTION BOARD
    # ============================================================
    
    render_section_header("Primary Action Board")
    
    display = filtered.copy()
    
    # Add display-friendly columns
    if "red_fighter" in display.columns and "blue_fighter" in display.columns:
        display["fight"] = display["red_fighter"] + " vs " + display["blue_fighter"]
    
    if "best_american_odds" in display.columns:
        display["odds_display"] = display["best_american_odds"].apply(american)
    
    for col in [
        "best_prob",
        "best_implied_prob",
        "best_edge",
    ]:
        if col in display.columns:
            display[f"{col}_display"] = display[col].apply(pct)
    
    ev_col = "best_ev_pct" if "best_ev_pct" in display.columns else "best_ev"
    
    display["best_ev_display"] = display[ev_col].apply(
        lambda x: f"{x:.1f}%"
        if pd.notna(x)
        else ""
    )
    
    if "best_confidence" in display.columns:
        display["best_confidence_display"] = display["best_confidence"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    
    if "recommended_stake" in display.columns:
        display["stake_display"] = display["recommended_stake"].apply(money)
    
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
    
    main_cols = [c for c in main_cols if c in display.columns]
    
    status_rank = {
        "OFFICIAL BET": 0,
        "WATCHLIST": 1,
        "LOW ODDS MATCH": 2,
        "SPARSE FEATURES": 3,
        "INVALID MODEL DATA": 4,
        "NO BET": 5,
    }
    
    if "bet_status" in display.columns:
        display["_status_rank"] = display["bet_status"].map(status_rank).fillna(99)
    else:
        display["_status_rank"] = 99
    
    display = display.sort_values(
        ["_status_rank", "best_ev"],
        ascending=[True, False],
        na_position="last",
    )
    
    st.dataframe(
        display[main_cols],
        use_container_width=True,
        hide_index=True,
    )
    
    # ============================================================
    # STATUS BREAKDOWN
    # ============================================================
    
    render_section_header("Status Breakdown")
    
    if "bet_status" in filtered.columns:
    
        status_counts = (
            filtered["bet_status"]
            .value_counts()
            .rename_axis("status")
            .reset_index(name="count")
        )
    
        st.dataframe(
            status_counts,
            use_container_width=True,
            hide_index=True,
        )
    
    else:
        st.info("No bet_status column found.")
    
    # ============================================================
    # FILTER DIAGNOSTICS
    # ============================================================
    
    render_section_header("Filter Diagnostics")
    
    diagnostic_cols = [
        "red_fighter",
        "blue_fighter",
        "bet_status",
        "failed_filters",
        "passes_model_quality_filter",
        "passes_feature_validation_filter",
        "passes_odds_match_filter",
        "passes_edge_filter",
        "passes_confidence_filter",
        "passes_odds_range_filter",
        "passes_positive_ev_filter",
        "passes_all_bet_filters",
        "odds_match_score",
        "odds_match_type",
        "nonzero_feature_count",
        "zero_feature_pct",
    ]
    
    diagnostic_cols = [c for c in diagnostic_cols if c in filtered.columns]
    
    st.dataframe(
        filtered[diagnostic_cols],
        use_container_width=True,
        hide_index=True,
    )
    
    # ============================================================
    # SELECTED FIGHT DETAIL
    # ============================================================
    
    render_section_header("Selected Fight Detail")
    
    fight_options = (
        display["fight"].dropna().tolist()
        if "fight" in display.columns
        else []
    )
    
    if fight_options:
        selected_fight = st.selectbox(
            "Select fight",
            fight_options,
        )
    
        fight_detail = display[display["fight"] == selected_fight].copy()
    
        st.dataframe(
            fight_detail[
                [
                    c for c in [
                        "event_name",
                        "fight",
                        "bet_status",
                        "best_side",
                        "red_model_prob",
                        "blue_model_prob",
                        "red_american_odds",
                        "blue_american_odds",
                        "red_implied_prob",
                        "blue_implied_prob",
                        "red_ev",
                        "blue_ev",
                        "best_ev",
                        "recommended_stake",
                        "failed_filters",
                        "bet_reason",
                        "red_feature_match",
                        "blue_feature_match",
                        "odds_match_score",
                        "odds_match_type",
                    ]
                    if c in fight_detail.columns
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    # ============================================================
    # SELECTED FIGHT LINE MOVEMENT
    # ============================================================

    render_section_header("Selected Fight Line Movement")
    
    snapshots = load_parquet("ufc_market_snapshots.parquet")
    
    if snapshots.empty:
        st.info("No market snapshot history found yet.")
    else:
        if fight_options:
            selected_row = display[display["fight"] == selected_fight].iloc[0]
            selected_fight_id = selected_row["fight_id"]
    
            fight_snapshots = snapshots[
                snapshots["fight_id"] == selected_fight_id
            ].copy()
    
            if fight_snapshots.empty:
                st.info("No snapshot history found for this fight yet.")
            else:
                fight_snapshots["snapshot_timestamp"] = pd.to_datetime(
                    fight_snapshots["snapshot_timestamp"],
                    utc=True,
                    errors="coerce",
                )
    
                fight_snapshots = fight_snapshots.sort_values(
                    "snapshot_timestamp"
                )
    
                chart_metric = st.selectbox(
                    "Line movement metric",
                    [
                        "American Odds",
                        "Implied Probability",
                    ],
                )
    
                if chart_metric == "American Odds":
                    chart_df = fight_snapshots[
                        [
                            "snapshot_timestamp",
                            "red_american_odds",
                            "blue_american_odds",
                        ]
                    ].rename(
                        columns={
                            "red_american_odds": selected_row["red_fighter"],
                            "blue_american_odds": selected_row["blue_fighter"],
                        }
                    )
                else:
                    chart_df = fight_snapshots[
                        [
                            "snapshot_timestamp",
                            "red_implied_prob",
                            "blue_implied_prob",
                        ]
                    ].rename(
                        columns={
                            "red_implied_prob": selected_row["red_fighter"],
                            "blue_implied_prob": selected_row["blue_fighter"],
                        }
                    )
    
                    chart_df[selected_row["red_fighter"]] = (
                        chart_df[selected_row["red_fighter"]] * 100
                    )
    
                    chart_df[selected_row["blue_fighter"]] = (
                        chart_df[selected_row["blue_fighter"]] * 100
                    )
    
                chart_df = chart_df.set_index("snapshot_timestamp")
    
                st.line_chart(
                    chart_df,
                    use_container_width=True,
                )
    
                st.subheader("Snapshot History")
    
                snapshot_display_cols = [
                    "snapshot_timestamp",
                    "bookmaker",
                    "red_fighter",
                    "blue_fighter",
                    "red_american_odds",
                    "blue_american_odds",
                    "red_implied_prob",
                    "blue_implied_prob",
                    "odds_match_score",
                    "odds_match_type",
                ]
    
                snapshot_display_cols = [
                    c for c in snapshot_display_cols
                    if c in fight_snapshots.columns
                ]
    
                st.dataframe(
                    fight_snapshots[snapshot_display_cols],
                    use_container_width=True,
                    hide_index=True,
                )
    # ============================================================
    # RAW DEBUG VIEW
    # ============================================================
    
    with st.expander("Raw Betting Board Data"):
        st.dataframe(
            board,
            use_container_width=True,
        )
