# ============================================================
# UFC BETTING INTELLIGENCE DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests

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
    .stApp {
        background-color: #111827;
        color: #F9FAFB;
    }

    section[data-testid="stSidebar"] {
        background-color: #1F2937;
        border-right: 1px solid #374151;
    }

    h1, h2, h3 {
        color: #F9FAFB;
        font-family: Inter, sans-serif;
    }

    .metric-card {
        background-color: #1F2937;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #374151;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }

    .metric-label {
        color: #9CA3AF;
        font-size: 13px;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #F9FAFB;
        font-size: 28px;
        font-weight: 800;
    }

    .section-header {
        font-size: 22px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 10px;
        color: #F9FAFB;
    }

    .small-muted {
        color: #9CA3AF;
        font-size: 13px;
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


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
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
# RAW DEBUG VIEW
# ============================================================

with st.expander("Raw Betting Board Data"):
    st.dataframe(
        board,
        use_container_width=True,
    )