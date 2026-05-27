import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="UFC Betting Board",
    layout="wide",
)

st.title("UFC Betting Board")

@st.cache_data
def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Could not load {path}: {e}")
        return pd.DataFrame()
if st.button("Clear cache and reload"):
    st.cache_data.clear()
    st.rerun()
#
# ------------------------------------------------------------
# Buttons
# ------------------------------------------------------------

# ============================================================
# FEATURE AUDIT VIEWER
# ============================================================

import pandas as pd
import streamlit as st

st.header("Model Feature Audit")

feature_audit = pd.read_parquet(
    "ufc_live_feature_audit.parquet"
)

# ------------------------------------------------------------
# Summary Metrics
# ------------------------------------------------------------

total_fights = len(feature_audit)

failed_validation = (
    ~feature_audit["passes_feature_validation"]
).sum()

failed_match_quality = (
    ~feature_audit["passes_match_quality"]
).sum()

avg_zero_pct = feature_audit["zero_feature_pct"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Fights",
    total_fights
)

col2.metric(
    "Failed Validation",
    failed_validation
)

col3.metric(
    "Failed Match Quality",
    failed_match_quality
)

col4.metric(
    "Avg Zero %",
    f"{avg_zero_pct:.1f}%"
)

# ------------------------------------------------------------
# Full Audit Table
# ------------------------------------------------------------

st.subheader("Full Feature Audit")

display_cols = [
    "event_name",
    "red_fighter",
    "blue_fighter",

    "red_feature_match",
    "blue_feature_match",

    "feature_count_expected",
    "feature_count_actual",

    "missing_feature_count",
    "nonzero_feature_count",
    "zero_feature_pct",

    "passes_match_quality",
    "passes_feature_validation",
]

display_cols = [
    c for c in display_cols
    if c in feature_audit.columns
]

st.dataframe(
    feature_audit[display_cols].sort_values(
        [
            "passes_feature_validation",
            "zero_feature_pct"
        ],
        ascending=[True, False]
    ),
    use_container_width=True,
)

# ------------------------------------------------------------
# Failed Validation Rows
# ------------------------------------------------------------

st.subheader("Failed Validation Rows")

failed = feature_audit[
    ~feature_audit["passes_feature_validation"]
]

st.dataframe(
    failed[display_cols],
    use_container_width=True,
)

# ------------------------------------------------------------
# Missing Fighter Matches
# ------------------------------------------------------------

st.subheader("Missing Fighter Matches")

missing_matches = feature_audit[
    (
        feature_audit["red_feature_match"] == "missing"
    )
    |
    (
        feature_audit["blue_feature_match"] == "missing"
    )
]

st.dataframe(
    missing_matches[display_cols],
    use_container_width=True,
)

# ------------------------------------------------------------
# Match Type Counts
# ------------------------------------------------------------

st.subheader("Match Type Counts")

red_counts = (
    feature_audit["red_feature_match"]
    .value_counts()
    .rename("count")
)

blue_counts = (
    feature_audit["blue_feature_match"]
    .value_counts()
    .rename("count")
)

st.write("Red Fighter Match Types")
st.dataframe(red_counts)

st.write("Blue Fighter Match Types")
st.dataframe(blue_counts)

# ============================================================
# MARKET UPDATE VIEWER
# ============================================================

st.header("Market Update Review")

market_odds = load_parquet("ufc_market_odds.parquet")
market_audit = load_parquet("ufc_market_match_audit.parquet")
market_snapshots = load_parquet("ufc_market_snapshots.parquet")

# ------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Market Rows", len(market_odds))
col2.metric("Audit Rows", len(market_audit))
col3.metric("Snapshot Rows", len(market_snapshots))

low_confidence_count = (
    (market_audit["odds_match_type"] != "matched").sum()
    if not market_audit.empty and "odds_match_type" in market_audit.columns
    else 0
)

col4.metric("Low Confidence Matches", low_confidence_count)

# ------------------------------------------------------------
# Market Match Audit
# ------------------------------------------------------------

st.subheader("Market Match Audit")

audit_cols = [
    "event_name",
    "red_fighter",
    "blue_fighter",
    "matched_fighter_1",
    "matched_fighter_2",
    "odds_match_score",
    "odds_match_type",
]

audit_cols = [c for c in audit_cols if c in market_audit.columns]

st.dataframe(
    market_audit[audit_cols].sort_values(
        "odds_match_score",
        ascending=True,
    ),
    use_container_width=True,
)

# ------------------------------------------------------------
# Latest Market Odds
# ------------------------------------------------------------

st.subheader("Latest Market Odds")

market_cols = [
    "event_name",
    "red_fighter",
    "blue_fighter",
    "bookmaker",
    "red_american_odds",
    "blue_american_odds",
    "red_implied_prob",
    "blue_implied_prob",
    "odds_match_score",
    "odds_match_type",
    "snapshot_timestamp",
]

market_cols = [c for c in market_cols if c in market_odds.columns]

st.dataframe(
    market_odds[market_cols],
    use_container_width=True,
)

# ------------------------------------------------------------
# Market Snapshot History
# ------------------------------------------------------------

st.subheader("Market Snapshot History")

snapshot_cols = [
    "event_name",
    "red_fighter",
    "blue_fighter",
    "bookmaker",
    "red_american_odds",
    "blue_american_odds",
    "snapshot_timestamp",
]

snapshot_cols = [c for c in snapshot_cols if c in market_snapshots.columns]

st.dataframe(
    market_snapshots[snapshot_cols].tail(100),
    use_container_width=True,
)

# ============================================================
# BETTING BOARD VIEWER
# ============================================================

st.header("Betting Board Review")

betting_board = load_parquet("ufc_betting_board.parquet")

if betting_board.empty:
    st.warning("No betting board found.")
else:
    # --------------------------------------------------------
    # Summary cards
    # --------------------------------------------------------

    total_fights = len(betting_board)
    official_bets = betting_board["is_official_bet"].sum()
    watchlist = betting_board["is_watchlist_bet"].sum()

    best_ev = betting_board["best_ev"].max()
    avg_ev = betting_board["best_ev"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Fights", total_fights)
    col2.metric("Official Bets", int(official_bets))
    col3.metric("Watchlist", int(watchlist))
    col4.metric("Best EV", f"{best_ev:.2%}")
    col5.metric("Avg EV", f"{avg_ev:.2%}")

    # --------------------------------------------------------
    # Main board
    # --------------------------------------------------------

    st.subheader("Main Betting Board")

    display_cols = [
        "event_name",
        "red_fighter",
        "blue_fighter",
        "bet_status",
        "best_side",
        "best_american_odds",
        "best_prob",
        "best_implied_prob",
        "best_edge",
        "best_ev",
        "best_confidence",
        "recommended_stake",
        "failed_filters",
        "odds_match_score",
        "odds_match_type",
    ]

    display_cols = [c for c in display_cols if c in betting_board.columns]

    st.dataframe(
        betting_board[display_cols].sort_values(
            ["bet_status", "best_ev"],
            ascending=[True, False],
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Failed filter diagnostics
    # --------------------------------------------------------

    st.subheader("Filter Diagnostics")

    filter_cols = [
        "passes_model_quality_filter",
        "passes_feature_validation_filter",
        "passes_odds_match_filter",
        "passes_edge_filter",
        "passes_confidence_filter",
        "passes_odds_range_filter",
        "passes_positive_ev_filter",
        "passes_all_bet_filters",
    ]

    filter_cols = [c for c in filter_cols if c in betting_board.columns]

    diagnostic_cols = [
        "red_fighter",
        "blue_fighter",
        "bet_status",
        "failed_filters",
    ] + filter_cols

    st.dataframe(
        betting_board[diagnostic_cols],
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Official bets
    # --------------------------------------------------------

    st.subheader("Official Bets")

    official = betting_board[
        betting_board["is_official_bet"] == True
    ]

    st.dataframe(
        official[display_cols],
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Watchlist
    # --------------------------------------------------------

    st.subheader("Watchlist")

    watch = betting_board[
        betting_board["is_watchlist_bet"] == True
    ]

    st.dataframe(
        watch[display_cols],
        use_container_width=True,
    )
