import streamlit as st
import pandas as pd

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

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

action_board = load_parquet("ufc_live_action_board.parquet")
card_with_odds = load_parquet("ufc_live_card_with_odds.parquet")

# Fallback if card_with_odds parquet does not exist yet
if card_with_odds.empty:
    card_with_odds = load_parquet("ufc_live_card.parquet")

# ------------------------------------------------------------
# Event selector
# ------------------------------------------------------------

source_df = card_with_odds if not card_with_odds.empty else action_board

events = (
    sorted(source_df["event_name"].dropna().unique().tolist())
    if not source_df.empty and "event_name" in source_df.columns
    else []
)

selected_event = st.selectbox(
    "Select event",
    events,
)

# ------------------------------------------------------------
# Build display table
# ------------------------------------------------------------

event_card = card_with_odds[
    card_with_odds["event_name"] == selected_event
].copy()

event_action = action_board[
    action_board["event_name"] == selected_event
].copy()

# Merge action board recommendations onto full card
display_df = event_card.merge(
    event_action[
        [
            "fight_id",
            "best_side",
            "best_prob",
            "best_edge",
            "best_ev",
            "best_american_odds",
            "best_confidence",
            "recommended_stake",
            "watchlist_tier",
            "watchlist_reason",
            "is_official_bet",
            "is_watchlist_bet",
        ]
    ],
    how="left",
    on="fight_id",
)

display_df["recommended_action"] = "No Bet"

display_df.loc[
    display_df["is_watchlist_bet"] == True,
    "recommended_action"
] = "Watchlist"

display_df.loc[
    display_df["is_official_bet"] == True,
    "recommended_action"
] = "Official Bet"

# ------------------------------------------------------------
# Format display columns
# ------------------------------------------------------------

display_cols = [
    "red_fighter",
    "blue_fighter",

    "red_american_odds",
    "blue_american_odds",

    "red_model_prob",
    "blue_model_prob",

    "red_implied_prob",
    "blue_implied_prob",

    "best_side",
    "best_american_odds",
    "best_prob",
    "best_edge",
    "best_ev",
    "best_confidence",

    "recommended_action",
    "recommended_stake",
    "watchlist_tier",
    "watchlist_reason",
]

display_cols = [
    col for col in display_cols
    if col in display_df.columns
]

st.header(f"Betting Board — {selected_event}")

st.dataframe(
    display_df[display_cols].sort_values(
        "best_ev",
        ascending=False,
        na_position="last",
    ),
    use_container_width=True,
)
