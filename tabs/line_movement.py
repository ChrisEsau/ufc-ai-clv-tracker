import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

from pipeline.common.paths import MARKET_SNAPSHOTS_PATH
from utils.clv_artifacts import (
    artifact_readiness_summary,
    get_clv_artifact_status,
    market_snapshot_coverage_summary,
)
from utils.data_loader import load_parquet
from utils.ui_components import render_metric
from utils.ui.sections import page_header

from utils.panels import render_section_header


def render_clv_artifact_health():
    render_section_header("CLV Artifact Health")

    status = get_clv_artifact_status()
    summary = artifact_readiness_summary(status)

    cols = st.columns(5)
    cols[0].metric("Required ready", f"{summary['required_ready']} / {summary['required_total']}")
    cols[1].metric("Missing required", summary["missing_required"])
    cols[2].metric("Empty required", summary["empty_required"])
    cols[3].metric("Optional missing", summary["optional_missing"])
    cols[4].metric("Stale ready", summary["stale_ready_artifacts"])

    if summary["ready_to_review"]:
        st.success("Required CLV artifacts are present and non-empty.")
    else:
        st.warning("One or more required CLV artifacts are missing or empty. Run the market update / CLV workflows before trusting CLV output.")

    display_cols = [
        "artifact",
        "required_for",
        "health",
        "freshness",
        "optional",
        "rows",
        "size",
        "age_hours",
        "modified_utc",
        "path",
    ]
    st.dataframe(status[display_cols], use_container_width=True, hide_index=True)


def render_market_snapshot_coverage(snapshots):
    render_section_header("Market Snapshot Coverage")

    summary = market_snapshot_coverage_summary(snapshots)

    cols = st.columns(4)
    cols[0].metric("Snapshot rows", summary["snapshot_rows"])
    cols[1].metric("Tracked fights", summary["unique_fights"])
    cols[2].metric("Tracked events", summary["unique_events"])
    cols[3].metric("Sportsbooks", summary["unique_sportsbooks"])

    cols = st.columns(4)
    cols[0].metric("Freshness", summary["freshness"])
    snapshot_age = (
        "n/a"
        if summary["snapshot_age_hours"] is None
        else f"{summary['snapshot_age_hours']:.2f}h"
    )
    cols[1].metric("Snapshot age", snapshot_age)
    cols[2].metric("Matched rows", summary["matched_rows"])
    cols[3].metric("Unmatched rows", summary["unmatched_rows"])

    if summary["freshness"] == "fresh":
        st.success("Market snapshots are fresh.")
    elif summary["freshness"] == "aging":
        st.warning("Market snapshots are aging. Consider refreshing market odds before making timing-sensitive decisions.")
    elif summary["freshness"] == "stale":
        st.error("Market snapshots are stale. Refresh market odds before trusting line movement or CLV output.")
    else:
        st.warning("Market snapshot freshness could not be determined.")

    detail = pd.DataFrame([summary])
    display_cols = [
        "latest_snapshot",
        "snapshot_age_hours",
        "freshness",
        "snapshot_rows",
        "unique_events",
        "unique_fights",
        "unique_sportsbooks",
        "matched_rows",
        "unmatched_rows",
        "missing_red_odds",
        "missing_blue_odds",
    ]
    st.dataframe(detail[display_cols], use_container_width=True, hide_index=True)


def render_line_movement():
    # ============================================================
    # LINE MOVEMENT / CLV SECTION
    # ============================================================
    page_header(
        "Line Movement / CLV",
        "Track market movement, sportsbook snapshots, and closing line value performance.",
        kicker="Market Intelligence",
    )
    render_clv_artifact_health()
    
    snapshots = load_parquet(MARKET_SNAPSHOTS_PATH)
    render_market_snapshot_coverage(snapshots)
    
    if snapshots.empty:
        st.info("No market snapshots found yet.")
    else:
        snapshots["snapshot_timestamp"] = pd.to_datetime(
            snapshots["snapshot_timestamp"],
            utc=True,
            errors="coerce",
        )
    
        snapshots = snapshots.dropna(
            subset=[
                "fight_id",
                "snapshot_timestamp",
                "red_american_odds",
                "blue_american_odds",
            ]
        ).copy()
    
        tracked_fights = snapshots["fight_id"].nunique()
        total_snapshots = len(snapshots)
        latest_snapshot = snapshots["snapshot_timestamp"].max()
    
        latest_rows = (
            snapshots.sort_values("snapshot_timestamp")
            .groupby("fight_id")
            .tail(1)
        )
    
        first_rows = (
            snapshots.sort_values("snapshot_timestamp")
            .groupby("fight_id")
            .head(1)
        )
    
        movement_df = first_rows[
            [
                "fight_id",
                "red_fighter",
                "blue_fighter",
                "red_american_odds",
                "blue_american_odds",
                "red_implied_prob",
                "blue_implied_prob",
            ]
        ].rename(
            columns={
                "red_american_odds": "opening_red_odds",
                "blue_american_odds": "opening_blue_odds",
                "red_implied_prob": "opening_red_implied",
                "blue_implied_prob": "opening_blue_implied",
            }
        ).merge(
            latest_rows[
                [
                    "fight_id",
                    "red_american_odds",
                    "blue_american_odds",
                    "red_implied_prob",
                    "blue_implied_prob",
                    "snapshot_timestamp",
                ]
            ].rename(
                columns={
                    "red_american_odds": "current_red_odds",
                    "blue_american_odds": "current_blue_odds",
                    "red_implied_prob": "current_red_implied",
                    "blue_implied_prob": "current_blue_implied",
                    "snapshot_timestamp": "latest_snapshot",
                }
            ),
            on="fight_id",
            how="left",
        )
    
        movement_df["red_implied_move"] = (
            movement_df["current_red_implied"]
            - movement_df["opening_red_implied"]
        )
    
        movement_df["blue_implied_move"] = (
            movement_df["current_blue_implied"]
            - movement_df["opening_blue_implied"]
        )
    
        movement_df["largest_abs_move"] = movement_df[
            [
                "red_implied_move",
                "blue_implied_move",
            ]
        ].abs().max(axis=1)
    
        snapshot_counts = (
            snapshots.groupby("fight_id")
            .size()
            .reset_index(name="snapshot_count")
        )
    
        movement_df = movement_df.merge(
            snapshot_counts,
            on="fight_id",
            how="left",
        )
    
        largest_move = (
            movement_df["largest_abs_move"].max()
            if not movement_df.empty
            else 0
        )
    
        m1, m2, m3, m4 = st.columns(4)
    
        with m1:
            render_metric("Tracked Fights", tracked_fights, accent="blue")
    
        with m2:
            render_metric("Total Snapshots", total_snapshots, accent="purple")
    
        with m3:
            render_metric("Largest Move", f"{largest_move * 100:.1f}%", accent="green")
    
        with m4:
            render_metric("Latest Snapshot", str(latest_snapshot)[:16], accent="neutral")
    
        # --------------------------------------------------------
        # CARD-STYLE MARKET MOVERS
        # --------------------------------------------------------
    
        render_section_header("Largest Market Movers")
    
        mover_display = movement_df.copy()
    
        mover_display["fight"] = (
            mover_display["red_fighter"]
            + " vs "
            + mover_display["blue_fighter"]
        )
    
        styled_df = mover_display.sort_values(
            "largest_abs_move",
            ascending=False,
        ).copy()
    
        def move_color(move):
            if move > 0:
                return "#22C55E"
            if move < 0:
                return "#EF4444"
            return "#CBD5E1"
    
        cards_html = """
        <div style="
            display:grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap:16px;
            margin-bottom:30px;
        ">
        """
    
        for _, row in styled_df.iterrows():
    
            red_move = row["red_implied_move"] * 100
            blue_move = row["blue_implied_move"] * 100
    
            red_color = move_color(red_move)
            blue_color = move_color(blue_move)
    
            larger_side = "Red" if abs(red_move) >= abs(blue_move) else "Blue"
            larger_move = max(abs(red_move), abs(blue_move))
    
            badge_color = "#EF4444" if larger_side == "Red" else "#3B82F6"
    
            cards_html += f"""
            <div style="
                background: linear-gradient(180deg, #1E293B 0%, #141C2E 100%);
                border: 1px solid #334155;
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 14px 32px rgba(0,0,0,0.32);
            ">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:flex-start;
                    margin-bottom:14px;
                ">
                    <div>
                        <div style="
                            font-size:15px;
                            font-weight:800;
                            color:#F8FAFC;
                            line-height:1.25;
                        ">
                            {row['fight']}
                        </div>
                        <div style="
                            font-size:12px;
                            color:#94A3B8;
                            margin-top:5px;
                        ">
                            {row['snapshot_count']} snapshots · latest {str(row['latest_snapshot'])[:16]}
                        </div>
                    </div>
    
                    <div style="
                        background:{badge_color}33;
                        color:{badge_color};
                        border:1px solid {badge_color}66;
                        border-radius:999px;
                        padding:5px 9px;
                        font-size:12px;
                        font-weight:800;
                    ">
                        {larger_side} {larger_move:.1f}%
                    </div>
                </div>
    
                <div style="
                    display:grid;
                    grid-template-columns: 1fr 1fr;
                    gap:12px;
                ">
                    <div style="
                        background:#0F172A;
                        border:1px solid #334155;
                        border-radius:14px;
                        padding:14px;
                    ">
                        <div style="
                            color:#94A3B8;
                            font-size:11px;
                            text-transform:uppercase;
                            font-weight:700;
                            letter-spacing:0.06em;
                            margin-bottom:8px;
                        ">
                            Red Side
                        </div>
    
                        <div style="
                            color:#F8FAFC;
                            font-weight:800;
                            font-size:14px;
                            margin-bottom:10px;
                        ">
                            {row['red_fighter']}
                        </div>
    
                        <div style="display:flex;justify-content:space-between;color:#CBD5E1;font-size:13px;margin-bottom:6px;">
                            <span>Open</span>
                            <strong>{row['opening_red_odds']}</strong>
                        </div>
    
                        <div style="display:flex;justify-content:space-between;color:#CBD5E1;font-size:13px;margin-bottom:6px;">
                            <span>Current</span>
                            <strong>{row['current_red_odds']}</strong>
                        </div>
    
                        <div style="display:flex;justify-content:space-between;color:{red_color};font-size:14px;font-weight:900;">
                            <span>Move</span>
                            <span>{red_move:+.1f}%</span>
                        </div>
                    </div>
    
                    <div style="
                        background:#0F172A;
                        border:1px solid #334155;
                        border-radius:14px;
                        padding:14px;
                    ">
                        <div style="
                            color:#94A3B8;
                            font-size:11px;
                            text-transform:uppercase;
                            font-weight:700;
                            letter-spacing:0.06em;
                            margin-bottom:8px;
                        ">
                            Blue Side
                        </div>
    
                        <div style="
                            color:#F8FAFC;
                            font-weight:800;
                            font-size:14px;
                            margin-bottom:10px;
                        ">
                            {row['blue_fighter']}
                        </div>
    
                        <div style="display:flex;justify-content:space-between;color:#CBD5E1;font-size:13px;margin-bottom:6px;">
                            <span>Open</span>
                            <strong>{row['opening_blue_odds']}</strong>
                        </div>
    
                        <div style="display:flex;justify-content:space-between;color:#CBD5E1;font-size:13px;margin-bottom:6px;">
                            <span>Current</span>
                            <strong>{row['current_blue_odds']}</strong>
                        </div>
    
                        <div style="display:flex;justify-content:space-between;color:{blue_color};font-size:14px;font-weight:900;">
                            <span>Move</span>
                            <span>{blue_move:+.1f}%</span>
                        </div>
                    </div>
                </div>
            </div>
            """
    
        cards_html += """
        </div>
        """
    
        components.html(
            cards_html,
            height=720,
            scrolling=True,
        )
    
        # --------------------------------------------------------
        # SELECTED FIGHT LINE CHART
        # --------------------------------------------------------
    
        render_section_header("Selected Fight Line Chart")
    
        fight_options_movement = (
            mover_display["fight"].dropna().tolist()
        )
    
        control1, control2, control3 = st.columns([2.5, 1.3, 1])
        
        with control1:
            selected_movement_fight = st.selectbox(
                "Select Fight",
                fight_options_movement,
                key="movement_fight_selector",
            )
        
        with control2:
            chart_metric = st.selectbox(
                "Chart Metric",
                [
                    "American Odds",
                    "Implied Probability",
                ],
                key="movement_chart_metric",
            )
        
        with control3:
            time_window = st.selectbox(
                "Time Window",
                [
                    "All",
                    "1H",
                    "6H",
                    "24H",
                ],
                index=0,
                key="movement_time_window",
            )
        
        selected_movement_row = mover_display[
            mover_display["fight"] == selected_movement_fight
        ].iloc[0]
        
        selected_movement_fight_id = selected_movement_row["fight_id"]
        
        fight_history = snapshots[
            snapshots["fight_id"] == selected_movement_fight_id
        ].copy()
        
        fight_history = fight_history.sort_values(
            "snapshot_timestamp"
        )
        
        if time_window != "All":
            hours = {
                "1H": 1,
                "6H": 6,
                "24H": 24,
            }[time_window]
        
            cutoff_time = fight_history["snapshot_timestamp"].max() - pd.Timedelta(hours=hours)
        
            fight_history = fight_history[
                fight_history["snapshot_timestamp"] >= cutoff_time
            ].copy()
        
        if chart_metric == "American Odds":
            chart_df = fight_history[
                [
                    "snapshot_timestamp",
                    "red_american_odds",
                    "blue_american_odds",
                ]
            ].rename(
                columns={
                    "red_american_odds": selected_movement_row["red_fighter"],
                    "blue_american_odds": selected_movement_row["blue_fighter"],
                }
            )
        else:
            chart_df = fight_history[
                [
                    "snapshot_timestamp",
                    "red_implied_prob",
                    "blue_implied_prob",
                ]
            ].rename(
                columns={
                    "red_implied_prob": selected_movement_row["red_fighter"],
                    "blue_implied_prob": selected_movement_row["blue_fighter"],
                }
            )
    
            chart_df[selected_movement_row["red_fighter"]] = (
                chart_df[selected_movement_row["red_fighter"]] * 100
            )
    
            chart_df[selected_movement_row["blue_fighter"]] = (
                chart_df[selected_movement_row["blue_fighter"]] * 100
            )
    
        chart_df = chart_df.set_index("snapshot_timestamp")
    
        # Replace st.line_chart(...) with Plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df[selected_movement_row["red_fighter"]],
            mode="lines+markers",
            name=f'{selected_movement_row["red_fighter"]} (Red)',
            line=dict(
                color="#EF4444",
                width=3,
            ),
            marker=dict(
                size=7,
                color="#EF4444",
                line=dict(
                    color="#FFFFFF",
                    width=2,
                ),
            ),
        ))
        
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df[selected_movement_row["blue_fighter"]],
            mode="lines+markers",
            name=f'{selected_movement_row["blue_fighter"]} (Blue)',
            line=dict(
                color="#3B82F6",
                width=3,
            ),
            marker=dict(
                size=7,
                color="#3B82F6",
                line=dict(
                    color="#FFFFFF",
                    width=2,
                ),
            ),
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1E293B",
            plot_bgcolor="#172033",
            font=dict(color="#F8FAFC"),
            margin=dict(l=40, r=20, t=30, b=40),
            height=420,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                color="#F8FAFC",
                size=13,
                ),
            )
        )
        
        fig.update_xaxes(
            gridcolor="rgba(148,163,184,0.15)",
            showline=True,
            linecolor="#475569",
            tickfont=dict(
                color="#E2E8F0",
                size=12,
            ),
            title_font=dict(
                color="#F8FAFC",
                size=14,
            ),
        )
        
        fig.update_yaxes(
            gridcolor="rgba(148,163,184,0.15)",
            showline=True,
            linecolor="#475569",
            tickfont=dict(
                color="#E2E8F0",
                size=12,
            ),
            title_font=dict(
                color="#F8FAFC",
                size=14,
            ),
            title="Implied Probability (%)",
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
        # --------------------------------------------------------
        # SNAPSHOT HISTORY TABLE
        # --------------------------------------------------------
    
        render_section_header("Snapshot History")
    
        snapshot_cols = [
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
    
        snapshot_cols = [
            c for c in snapshot_cols
            if c in fight_history.columns
        ]
    
        st.dataframe(
            fight_history[snapshot_cols],
            use_container_width=True,
            hide_index=True,
        )
