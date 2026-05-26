import streamlit as st
import pandas as pd

st.set_page_config(page_title="UFC CLV Tracker", layout="wide")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: #0b111a;
    color: #e6edf3;
}
section[data-testid="stSidebar"] {
    background: #111827;
}
.card {
    background: #151d2a;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #243244;
    box-shadow: 0 8px 24px rgba(0,0,0,.25);
}
.metric-title {
    color: #9ca3af;
    font-size: 14px;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    color: #22c55e;
}
.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 20px;
}
[data-testid="stDataFrame"] {
    background: #151d2a;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

clv = load_parquet("ufc_clv_results.parquet")
closing = load_parquet("ufc_closing_lines.parquet")
snapshots = load_parquet("ufc_market_snapshots.parquet")
movement = load_parquet("ufc_line_movement.parquet")

st.sidebar.title("UFC CLV Tracker")
st.sidebar.caption("Track Closing Line Value in Real Time")

event_filter = st.sidebar.selectbox(
    "Event",
    ["All Events"] + sorted(snapshots["event_name"].dropna().unique().tolist())
    if not snapshots.empty and "event_name" in snapshots.columns else ["All Events"]
)

steam_only = st.sidebar.toggle("Steam Moves Only", value=False)

st.title("Dashboard Overview")
st.caption("Data updates automatically from GitHub Actions.")

# -----------------------------
# Metrics
# -----------------------------
total_bets = len(clv)
beat_close_rate = clv["beat_closing_line"].mean() if not clv.empty and "beat_closing_line" in clv else 0
avg_clv = clv["clv_diff"].mean() if not clv.empty and "clv_diff" in clv else 0
steam_moves = movement["is_steam_move"].sum() if not movement.empty and "is_steam_move" in movement else 0

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f'<div class="card"><div class="metric-title">Tracked Bets</div><div class="metric-value">{total_bets}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="card"><div class="metric-title">Beat Close %</div><div class="metric-value">{beat_close_rate:.1%}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="card"><div class="metric-title">Average CLV</div><div class="metric-value">{avg_clv:.2%}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="card"><div class="metric-title">Steam Moves</div><div class="metric-value">{int(steam_moves)}</div></div>', unsafe_allow_html=True)

# -----------------------------
# Tables
# -----------------------------
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">CLV Summary</div>', unsafe_allow_html=True)
    if clv.empty:
        st.warning("No CLV data.")
    else:
        clv = clv.rename(columns={"event_name_x": "event_name"})
        cols = [
            "event_name", "best_side", "bookmaker",
            "bet_odds", "closing_odds", "clv_diff",
            "beat_closing_line", "best_ev", "recommended_stake"
        ]
        st.dataframe(clv[[c for c in cols if c in clv.columns]], use_container_width=True)

with right:
    st.markdown('<div class="section-title">Latest Closing Lines</div>', unsafe_allow_html=True)
    if closing.empty:
        st.warning("No closing line data.")
    else:
        cols = [
            "fighter_name", "opponent_name", "sportsbook",
            "closing_odds", "closing_implied_prob",
            "closing_line_status"
        ]
        st.dataframe(closing[[c for c in cols if c in closing.columns]], use_container_width=True)

left2, right2 = st.columns(2)

with left2:
    st.markdown('<div class="section-title">Latest Market Snapshot</div>', unsafe_allow_html=True)
    if snapshots.empty:
        st.warning("No snapshot data.")
    else:
        snapshots["snapshot_timestamp"] = pd.to_datetime(snapshots["snapshot_timestamp"], utc=True, errors="coerce")
        latest_time = snapshots["snapshot_timestamp"].max()
        latest = snapshots[snapshots["snapshot_timestamp"] == latest_time].copy()
        cols = ["fighter_name", "opponent_name", "sportsbook", "american_odds", "implied_prob", "snapshot_timestamp"]
        st.dataframe(latest[[c for c in cols if c in latest.columns]], use_container_width=True)

with right2:
    st.markdown('<div class="section-title">Line Movement / Steam</div>', unsafe_allow_html=True)
    if movement.empty:
        st.warning("No movement data.")
    else:
        movement_view = movement.copy()
        if steam_only and "is_steam_move" in movement_view.columns:
            movement_view = movement_view[movement_view["is_steam_move"] == True]
        cols = [
            "fighter_name", "opponent_name", "opening_odds", "latest_odds",
            "implied_prob_movement", "is_steam_move", "steam_direction"
        ]
        st.dataframe(movement_view[[c for c in cols if c in movement_view.columns]], use_container_width=True)

# -----------------------------
# Chart
# -----------------------------
st.markdown('<div class="section-title">Line History Chart</div>', unsafe_allow_html=True)

if snapshots.empty:
    st.warning("No historical snapshot data.")
else:
    chart_df = snapshots.copy()
    chart_df["snapshot_timestamp"] = pd.to_datetime(chart_df["snapshot_timestamp"], utc=True, errors="coerce")

    fighters = sorted(chart_df["fighter_name"].dropna().unique())
    selected = st.multiselect("Select fighters", fighters, default=fighters[:2])

    metric = st.selectbox("Metric", ["implied_prob", "american_odds"])

    chart_df = chart_df[chart_df["fighter_name"].isin(selected)]

    if not chart_df.empty:
        st.line_chart(
            chart_df,
            x="snapshot_timestamp",
            y=metric,
            color="fighter_name",
            use_container_width=True,
        )
