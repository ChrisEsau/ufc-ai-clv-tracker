import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="UFC CLV Dashboard",
    layout="wide",
)

st.title("UFC CLV Dashboard")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

clv = pd.read_parquet("ufc_clv_results.csv")
closing = pd.read_parquet("ufc_closing_lines.csv")
latest = pd.read_parquet("ufc_latest_market_snapshot.csv")

# ------------------------------------------------------------
# Clean CLV columns
# ------------------------------------------------------------

clv = clv.rename(columns={
    "event_name_x": "event_name",
    "snapshot_run_id_x": "snapshot_run_id",
})

# ------------------------------------------------------------
# Top Metrics
# ------------------------------------------------------------

st.header("CLV Summary")

total_bets = len(clv)
bets_with_close = clv["closing_odds"].notna().sum()

beat_close_rate = (
    clv["beat_closing_line"].mean()
    if "beat_closing_line" in clv.columns and total_bets > 0
    else 0
)

avg_clv = (
    clv["clv_diff"].mean()
    if "clv_diff" in clv.columns and total_bets > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Tracked Bets", total_bets)
col2.metric("Bets With Close", bets_with_close)
col3.metric("Beat Close %", f"{beat_close_rate:.1%}")
col4.metric("Average CLV", f"{avg_clv:.2%}")

# ------------------------------------------------------------
# CLV Results
# ------------------------------------------------------------

st.header("Official Bets / CLV")

clv_display_cols = [
    "bet_id",
    "event_name",
    "best_side",
    "bookmaker",
    "bet_odds",
    "closing_odds",
    "bet_implied_prob",
    "closing_implied_prob",
    "clv_diff",
    "beat_closing_line",
    "best_prob",
    "best_ev",
    "best_confidence",
    "recommended_stake",
]

clv_display_cols = [
    col for col in clv_display_cols
    if col in clv.columns
]

st.dataframe(
    clv[clv_display_cols],
    use_container_width=True,
)

# ------------------------------------------------------------
# Closing Lines
# ------------------------------------------------------------

st.header("Closing / Latest Pre-Fight Lines")

closing_display_cols = [
    "event_name",
    "fighter_name",
    "opponent_name",
    "sportsbook",
    "market_type",
    "closing_odds",
    "closing_implied_prob",
    "closing_line_status",
    "minutes_before_fight",
    "commence_time",
]

closing_display_cols = [
    col for col in closing_display_cols
    if col in closing.columns
]

st.dataframe(
    closing[closing_display_cols],
    use_container_width=True,
)

# ------------------------------------------------------------
# Latest Market Snapshot
# ------------------------------------------------------------

st.header("Latest Market Snapshot")

latest_display_cols = [
    "event_name",
    "fighter_name",
    "opponent_name",
    "sportsbook",
    "market_type",
    "american_odds",
    "implied_prob",
    "snapshot_timestamp",
    "commence_time",
]

latest_display_cols = [
    col for col in latest_display_cols
    if col in latest.columns
]

st.dataframe(
    latest[latest_display_cols],
    use_container_width=True,
)
