import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="UFC CLV Dashboard",
    layout="wide",
)

st.title("UFC CLV Dashboard")

# ------------------------------------------------------------
# Load parquet files safely
# ------------------------------------------------------------

@st.cache_data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
        return pd.DataFrame()

clv = load_parquet("ufc_clv_results.parquet")
closing = load_parquet("ufc_closing_lines.parquet")
snapshots = load_parquet("ufc_market_snapshots.parquet")
movement = load_parquet("ufc_line_movement.parquet")

# ------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------

st.header("CLV Summary")

col1, col2, col3, col4 = st.columns(4)

total_bets = len(clv)

bets_with_close = (
    clv["closing_odds"].notna().sum()
    if "closing_odds" in clv.columns
    else 0
)

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

col1.metric("Tracked Bets", total_bets)
col2.metric("Bets With Close", bets_with_close)
col3.metric("Beat Close %", f"{beat_close_rate:.1%}")
col4.metric("Average CLV", f"{avg_clv:.2%}")

# ------------------------------------------------------------
# CLV Results
# ------------------------------------------------------------

st.header("Official Bets / CLV")

if clv.empty:
    st.warning("No CLV results found.")
else:
    clv = clv.rename(columns={
        "event_name_x": "event_name",
        "snapshot_run_id_x": "snapshot_run_id",
    })

    clv_cols = [
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

    clv_cols = [c for c in clv_cols if c in clv.columns]

    st.dataframe(
        clv[clv_cols],
        use_container_width=True,
    )

# ------------------------------------------------------------
# Closing Lines
# ------------------------------------------------------------

st.header("Closing / Latest Pre-Fight Lines")

if closing.empty:
    st.warning("No closing lines found.")
else:
    closing_cols = [
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

    closing_cols = [c for c in closing_cols if c in closing.columns]

    st.dataframe(
        closing[closing_cols],
        use_container_width=True,
    )

# ------------------------------------------------------------
# Latest Snapshot
# ------------------------------------------------------------

st.header("Latest Market Snapshot")

if snapshots.empty:
    st.warning("No market snapshots found.")
else:
    snapshots["snapshot_timestamp"] = pd.to_datetime(
        snapshots["snapshot_timestamp"],
        utc=True,
        errors="coerce",
    )

    latest_time = snapshots["snapshot_timestamp"].max()

    latest = snapshots[
        snapshots["snapshot_timestamp"] == latest_time
    ].copy()

    st.caption(f"Latest snapshot: {latest_time}")

    latest_cols = [
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

    latest_cols = [c for c in latest_cols if c in latest.columns]

    st.dataframe(
        latest[latest_cols],
        use_container_width=True,
    )
    st.header("Line Movement / Steam")

if movement.empty:
    st.warning("No line movement data found.")
else:
    movement_cols = [
        "event_name",
        "fighter_name",
        "opponent_name",
        "sportsbook",
        "market_type",
        "opening_odds",
        "latest_odds",
        "opening_implied_prob",
        "latest_implied_prob",
        "implied_prob_movement",
        "is_steam_move",
        "steam_direction",
    ]

    movement_cols = [c for c in movement_cols if c in movement.columns]

    st.dataframe(
        movement[movement_cols],
        use_container_width=True,
    )
    
# ------------------------------------------------------------
# Line History Chart
# ------------------------------------------------------------

st.header("Line History Chart")

if snapshots.empty:
    st.warning("No snapshot history found.")
else:
    chart_df = snapshots.copy()

    chart_df["snapshot_timestamp"] = pd.to_datetime(
        chart_df["snapshot_timestamp"],
        utc=True,
        errors="coerce",
    )

    fighters = sorted(chart_df["fighter_name"].dropna().unique())

    selected_fighters = st.multiselect(
        "Select fighters to chart",
        fighters,
        default=fighters[:2],
    )

    chart_metric = st.selectbox(
        "Chart metric",
        [
            "american_odds",
            "implied_prob",
        ],
    )

    filtered_chart_df = chart_df[
        chart_df["fighter_name"].isin(selected_fighters)
    ].copy()

    if filtered_chart_df.empty:
        st.info("Select at least one fighter.")
    else:
        st.line_chart(
            filtered_chart_df,
            x="snapshot_timestamp",
            y=chart_metric,
            color="fighter_name",
            use_container_width=True,
        )
