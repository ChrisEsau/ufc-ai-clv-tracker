# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import streamlit.components.v1 as components
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="UFC Betting Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME / CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0B1220 0%, #111827 45%, #0F172A 100%);
        color: #F9FAFB;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #334155;
    }

    h1 {
        font-size: 34px !important;
        font-weight: 800 !important;
        color: #F9FAFB !important;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #F9FAFB !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1600px;
    }

    .metric-card {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.32);
        min-height: 118px;
    }

    .metric-label {
        color: #93A4BA;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-subtext {
        color: #94A3B8;
        font-size: 13px;
        margin-top: 8px;
    }

    .section-header {
        font-size: 24px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 12px;
        color: #F8FAFC;
        letter-spacing: -0.03em;
    }

    .panel {
        background: linear-gradient(180deg, #1E293B 0%, #172033 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    }

    .small-muted {
        color: #94A3B8;
        font-size: 13px;
    }

    div[data-testid="stDataFrame"] {
        background: #1E293B;
        border-radius: 14px;
        border: 1px solid #334155;
        overflow: hidden;
    }

    .stSelectbox label,
    .stMultiSelect label,
    .stRadio label {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }

    .stButton button {
        background: #1E293B;
        color: #F8FAFC;
        border: 1px solid #475569;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton button:hover {
        background: #334155;
        color: #FFFFFF;
        border-color: #60A5FA;
    }

    .positive {
        color: #22C55E;
        font-weight: 800;
    }

    .negative {
        color: #EF4444;
        font-weight: 800;
    }

    .neutral {
        color: #CBD5E1;
        font-weight: 700;
    }

    .status-pill {
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        display: inline-block;
    }

    .pill-official {
        background: rgba(16,185,129,0.20);
        color: #34D399;
        border: 1px solid rgba(16,185,129,0.45);
    }

    .pill-watchlist {
        background: rgba(245,158,11,0.18);
        color: #FBBF24;
        border: 1px solid rgba(245,158,11,0.45);
    }

    .pill-danger {
        background: rgba(239,68,68,0.18);
        color: #F87171;
        border: 1px solid rgba(239,68,68,0.45);
    }

    .pill-info {
        background: rgba(59,130,246,0.18);
        color: #60A5FA;
        border: 1px solid rgba(59,130,246,0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def pct(x):
    if pd.isna(x):
        return ""
    return f"{x * 100:.1f}%"


def pct_already(x):
    if pd.isna(x):
        return ""
    return f"{x:.1f}%"


def money(x):
    if pd.isna(x):
        return ""
    return f"${x:,.0f}"


def american(x):
    if pd.isna(x):
        return ""
    x = int(round(x))
    return f"+{x}" if x > 0 else str(x)


def render_metric(label, value, subtext="", accent="neutral"):
    color = {
        "green": "#22C55E",
        "red": "#EF4444",
        "blue": "#3B82F6",
        "amber": "#F59E0B",
        "purple": "#A855F7",
        "neutral": "#F9FAFB",
    }.get(accent, "#F9FAFB")

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("UFC Intelligence")
st.sidebar.caption("Betting Board Control Plane")

if st.sidebar.button("Clear cache / reload"):
    st.cache_data.clear()
    st.rerun()

# ------------------------------------------------------------
# Optional workflow controls
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Controls")

GITHUB_OWNER = "ChrisEsau"
GITHUB_REPO = "ufc-ai-clv-tracker"


def trigger_workflow(workflow_file):
    try:
        github_token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        return None, "Missing Streamlit secret: GITHUB_TOKEN"

    url = (
        f"https://api.github.com/repos/"
        f"{GITHUB_OWNER}/{GITHUB_REPO}/actions/workflows/"
        f"{workflow_file}/dispatches"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
    }

    response = requests.post(
        url,
        headers=headers,
        json={"ref": "main"},
        timeout=20,
    )

    return response.status_code, response.text


if st.sidebar.button("Run Model Predictions"):
    status, msg = trigger_workflow("run-model-predictions.yml")
    if status == 204:
        st.sidebar.success("Model prediction workflow started.")
    else:
        st.sidebar.error(f"Workflow failed: {status} {msg}")

if st.sidebar.button("Run Market Update"):
    status, msg = trigger_workflow("run-market-update.yml")
    if status == 204:
        st.sidebar.success("Market update workflow started.")
    else:
        st.sidebar.error(f"Workflow failed: {status} {msg}")

if st.sidebar.button("Run Betting Decision"):
    status, msg = trigger_workflow("run-betting-decision.yml")
    if status == 204:
        st.sidebar.success("Betting decision workflow started.")
    else:
        st.sidebar.error(f"Workflow failed: {status} {msg}")


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

selected_event = st.sidebar.selectbox(
    "Event",
    ["All Events"] + events,
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

selected_statuses = st.sidebar.multiselect(
    "Status",
    available_statuses,
    default=available_statuses,
)

show_only_actionable = st.sidebar.toggle(
    "Show only Official / Watchlist",
    value=False,
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

st.markdown('<div class="section-header">Primary Action Board</div>', unsafe_allow_html=True)

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

st.markdown(
    '<div class="section-header">Status Breakdown</div>',
    unsafe_allow_html=True,
)

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

st.markdown('<div class="section-header">Filter Diagnostics</div>', unsafe_allow_html=True)

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

st.markdown('<div class="section-header">Selected Fight Detail</div>', unsafe_allow_html=True)

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

st.markdown(
    '<div class="section-header">Selected Fight Line Movement</div>',
    unsafe_allow_html=True,
)

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

# ============================================================
# LINE MOVEMENT / CLV SECTION
# ============================================================

st.markdown(
    '<div class="section-header">Line Movement / CLV</div>',
    unsafe_allow_html=True,
)

snapshots = load_parquet("ufc_market_snapshots.parquet")

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

    st.subheader("Largest Market Movers")

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

    import streamlit.components.v1 as components

    components.html(
        cards_html,
        height=720,
        scrolling=True,
    )

    # --------------------------------------------------------
    # SELECTED FIGHT LINE CHART
    # --------------------------------------------------------

    st.subheader("Selected Fight Line Chart")

    fight_options_movement = (
        mover_display["fight"].dropna().tolist()
    )

    selected_movement_fight = st.selectbox(
        "Select fight for line movement",
        fight_options_movement,
        key="movement_fight_selector",
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

    chart_metric = st.selectbox(
        "Chart metric",
        [
            "American Odds",
            "Implied Probability",
        ],
        key="movement_chart_metric",
    )

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
        line=dict(color="#EF4444", width=3),
        marker=dict(size=6),
    ))
    
    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df[selected_movement_row["blue_fighter"]],
        mode="lines+markers",
        name=f'{selected_movement_row["blue_fighter"]} (Blue)',
        line=dict(color="#3B82F6", width=3),
        marker=dict(size=6),
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

    st.subheader("Selected Fight Snapshot History")

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
