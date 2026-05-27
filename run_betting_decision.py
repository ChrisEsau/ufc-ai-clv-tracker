# ============================================================
# run_betting_decision.py
# Dedicated EV / staking / bet qualification runner
# ============================================================

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline_config import *
from ufc_pipeline_utils import *

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "."

MODEL_PREDICTIONS_PATH = f"{BASE_PATH}/ufc_model_predictions.parquet"
MARKET_ODDS_PATH = f"{BASE_PATH}/ufc_market_odds.parquet"

BETTING_BOARD_OUTPUT = f"{BASE_PATH}/ufc_betting_board.parquet"
WATCHLIST_OUTPUT = f"{BASE_PATH}/ufc_live_watchlist.parquet"
OFFICIAL_BETS_OUTPUT = f"{BASE_PATH}/ufc_official_bets.parquet"

DECISION_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
DECISION_TIMESTAMP = datetime.now(timezone.utc).isoformat()

BANKROLL = 10000

# ============================================================
# LOAD INPUTS
# ============================================================

predictions_df = pd.read_parquet(MODEL_PREDICTIONS_PATH)
market_df = pd.read_parquet(MARKET_ODDS_PATH)

print("Prediction rows:", len(predictions_df))
print("Market rows:", len(market_df))

# ============================================================
# LOAD CONFIG / FILTERS
# ============================================================

paths = UFCPipelinePaths(
    base_path=BASE_PATH,
    model_version="UFC_Model_v5_Experiment",
)

artifacts = load_production_artifacts(paths)
production_config = artifacts["production_config"]

betting_filters = get_betting_filters(production_config)

MIN_EDGE = betting_filters["min_edge"]
MIN_CONFIDENCE = betting_filters["min_confidence"]
MIN_ODDS = betting_filters["min_odds"]
MAX_ODDS = betting_filters["max_odds"]

KELLY_FRACTION = production_config["staking"]["kelly_fraction"]
MAX_STAKE_PCT = production_config["staking"]["max_stake_pct"]

print("Filters:", betting_filters)
print("Staking:", production_config["staking"])

# ============================================================
# MERGE MODEL + MARKET
# ============================================================

df = predictions_df.merge(
    market_df[
        [
            "fight_id",
            "snapshot_run_id",
            "snapshot_timestamp",
            "bookmaker",
            "red_american_odds",
            "blue_american_odds",
            "red_decimal_odds",
            "blue_decimal_odds",
            "red_implied_prob",
            "blue_implied_prob",
            "odds_match_score",
            "odds_match_type",
        ]
    ],
    how="left",
    on="fight_id",
)

df["decision_run_id"] = DECISION_RUN_ID
df["decision_timestamp"] = DECISION_TIMESTAMP

# ============================================================
# CLEAN ODDS
# ============================================================

for col in [
    "red_american_odds",
    "blue_american_odds",
    "red_decimal_odds",
    "blue_decimal_odds",
    "red_implied_prob",
    "blue_implied_prob",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# CALCULATE EDGE + EV
# ============================================================

df["red_edge"] = df["red_model_prob"] - df["red_implied_prob"]
df["blue_edge"] = df["blue_model_prob"] - df["blue_implied_prob"]

df["red_ev"] = df.apply(
    lambda row: decimal_ev(
        row["red_model_prob"],
        row["red_decimal_odds"],
    )
    if pd.notna(row["red_decimal_odds"])
    else np.nan,
    axis=1,
)

df["blue_ev"] = df.apply(
    lambda row: decimal_ev(
        row["blue_model_prob"],
        row["blue_decimal_odds"],
    )
    if pd.notna(row["blue_decimal_odds"])
    else np.nan,
    axis=1,
)

# ============================================================
# EV DISPLAY COLUMNS
# Stored as percentage-style values for dashboard readability
# ============================================================

df["red_ev_pct"] = df["red_ev"]
df["blue_ev_pct"] = df["blue_ev"]

df["red_is_best"] = df["red_ev"] >= df["blue_ev"]

df["best_side"] = np.where(
    df["red_is_best"],
    df["red_fighter"],
    df["blue_fighter"],
)

df["best_side_fighter_id"] = np.where(
    df["red_is_best"],
    df["red_fighter_id"],
    df["blue_fighter_id"],
)

df["best_side_opponent_id"] = np.where(
    df["red_is_best"],
    df["blue_fighter_id"],
    df["red_fighter_id"],
)

df["best_prob"] = np.where(
    df["red_is_best"],
    df["red_model_prob"],
    df["blue_model_prob"],
)

df["best_implied_prob"] = np.where(
    df["red_is_best"],
    df["red_implied_prob"],
    df["blue_implied_prob"],
)

df["best_edge"] = np.where(
    df["red_is_best"],
    df["red_edge"],
    df["blue_edge"],
)

df["best_ev"] = np.where(
    df["red_is_best"],
    df["red_ev"],
    df["blue_ev"],
)

df["best_ev_pct"] = df["best_ev"]

df["best_american_odds"] = np.where(
    df["red_is_best"],
    df["red_american_odds"],
    df["blue_american_odds"],
)

df["best_decimal_odds"] = np.where(
    df["red_is_best"],
    df["red_decimal_odds"],
    df["blue_decimal_odds"],
)

df["best_confidence"] = df["best_prob"] * 100

# ============================================================
# BET FILTERS
# ============================================================

df["passes_model_quality_filter"] = (
    df["passes_model_data_quality"] == True
)

df["passes_feature_validation_filter"] = (
    df["passes_feature_validation"] == True
)

df["passes_odds_match_filter"] = (
    df["odds_match_type"] == "matched"
)

df["passes_edge_filter"] = (
    df["best_edge"] >= MIN_EDGE
)

df["passes_confidence_filter"] = (
    df["best_confidence"] >= MIN_CONFIDENCE
)

df["passes_odds_range_filter"] = (
    df["best_american_odds"] >= MIN_ODDS
) & (
    df["best_american_odds"] <= MAX_ODDS
)

df["passes_positive_ev_filter"] = (
    df["best_ev"] > 0
)

df["passes_all_bet_filters"] = (
    df["passes_model_quality_filter"]
    &
    df["passes_feature_validation_filter"]
    &
    df["passes_odds_match_filter"]
    &
    df["passes_edge_filter"]
    &
    df["passes_confidence_filter"]
    &
    df["passes_odds_range_filter"]
    &
    df["passes_positive_ev_filter"]
)

# ============================================================
# STAKING
# ============================================================

df["recommended_stake"] = df.apply(
    lambda row: scaled_kelly_stake(
        bankroll=BANKROLL,
        model_prob=row["best_prob"],
        american_odds=row["best_american_odds"],
        kelly_multiplier=KELLY_FRACTION,
        max_stake_pct=MAX_STAKE_PCT,
    )
    if row["passes_all_bet_filters"]
    else 0,
    axis=1,
)

# ============================================================
# BET STATUS / WATCHLIST
# ============================================================

filter_cols = [
    "passes_model_quality_filter",
    "passes_feature_validation_filter",
    "passes_odds_match_filter",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_range_filter",
    "passes_positive_ev_filter",
]

df["failed_filter_count"] = (
    len(filter_cols) - df[filter_cols].sum(axis=1)
)

def build_failed_filter_reason(row):
    failed = []

    for col in filter_cols:
        if not row[col]:
            failed.append(col.replace("passes_", "").replace("_filter", ""))

    return ", ".join(failed)


df["failed_filters"] = df.apply(
    build_failed_filter_reason,
    axis=1,
)

# ------------------------------------------------------------
# Hard invalid gates
# These must pass before a fight can be actionable.
# ------------------------------------------------------------

df["passes_core_data_filters"] = (
    df["passes_model_quality_filter"]
    &
    df["passes_feature_validation_filter"]
    &
    df["passes_odds_match_filter"]
)

# ------------------------------------------------------------
# Official bet
# ------------------------------------------------------------

df["is_official_bet"] = df["passes_all_bet_filters"]

# ------------------------------------------------------------
# Watchlist
# Only valid-data fights can be watchlist.
# Near-miss only on betting thresholds, not data quality.
# ------------------------------------------------------------

betting_threshold_cols = [
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_range_filter",
    "passes_positive_ev_filter",
]

df["failed_betting_threshold_count"] = (
    len(betting_threshold_cols)
    -
    df[betting_threshold_cols].sum(axis=1)
)

df["is_watchlist_bet"] = (
    (~df["is_official_bet"])
    &
    df["passes_core_data_filters"]
    &
    (
        (df["failed_betting_threshold_count"] <= 2)
        |
        (df["best_ev"] > 0.25)
    )
)

# ------------------------------------------------------------
# Human-readable status
# ------------------------------------------------------------

df["bet_status"] = "NO BET"

df.loc[
    ~df["passes_model_quality_filter"],
    "bet_status",
] = "INVALID MODEL DATA"

df.loc[
    df["passes_model_quality_filter"]
    &
    ~df["passes_feature_validation_filter"],
    "bet_status",
] = "SPARSE FEATURES"

df.loc[
    df["passes_model_quality_filter"]
    &
    df["passes_feature_validation_filter"]
    &
    ~df["passes_odds_match_filter"],
    "bet_status",
] = "LOW ODDS MATCH"

df.loc[
    df["is_watchlist_bet"],
    "bet_status",
] = "WATCHLIST"

df.loc[
    df["is_official_bet"],
    "bet_status",
] = "OFFICIAL BET"

df["bet_reason"] = np.where(
    df["is_official_bet"],
    "All betting filters passed",
    df["failed_filters"],
)

# ============================================================
# FINAL OUTPUT COLUMNS
# ============================================================

output_cols = [
    "decision_run_id",
    "decision_timestamp",

    "event_name",
    "commence_time",
    "fight_id",

    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",

    "bookmaker",
    "snapshot_timestamp",

    "red_model_prob",
    "blue_model_prob",
    "model_pick",
    "model_confidence",

    "red_american_odds",
    "blue_american_odds",
    "red_implied_prob",
    "blue_implied_prob",

    "red_edge",
    "blue_edge",
    "red_ev",
    "blue_ev",

    "best_side",
    "best_side_fighter_id",
    "best_side_opponent_id",
    "best_prob",
    "best_implied_prob",
    "best_edge",
    "best_ev",
    "best_american_odds",
    "best_confidence",

    "recommended_stake",

    "bet_status",
    "bet_reason",
    "failed_filters",
    "failed_filter_count",

    "is_official_bet",
    "is_watchlist_bet",

    "passes_model_quality_filter",
    "passes_feature_validation_filter",
    "passes_odds_match_filter",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_range_filter",
    "passes_positive_ev_filter",
    "passes_all_bet_filters",

    "red_feature_match",
    "blue_feature_match",
    "nonzero_feature_count",
    "zero_feature_pct",
    "passes_feature_validation",

    "odds_match_score",
    "odds_match_type",
]

output_cols = [
    col for col in output_cols
    if col in df.columns
]

betting_board_df = df[output_cols].copy()

# ============================================================
# SAVE OUTPUTS
# ============================================================

betting_board_df.to_parquet(
    BETTING_BOARD_OUTPUT,
    index=False,
)

watchlist_df = betting_board_df[
    betting_board_df["is_watchlist_bet"] == True
].copy()

official_bets_df = betting_board_df[
    betting_board_df["is_official_bet"] == True
].copy()

watchlist_df.to_parquet(
    WATCHLIST_OUTPUT,
    index=False,
)

official_bets_df.to_parquet(
    OFFICIAL_BETS_OUTPUT,
    index=False,
)

print("========== BETTING DECISION SUMMARY ==========")
print("Board rows:", len(betting_board_df))
print("Official bets:", len(official_bets_df))
print("Watchlist:", len(watchlist_df))

print("Saved:")
print(BETTING_BOARD_OUTPUT)
print(WATCHLIST_OUTPUT)
print(OFFICIAL_BETS_OUTPUT)