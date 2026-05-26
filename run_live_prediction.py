# ============================================================
# UFC LIVE PREDICTION PIPELINE V5 — GITHUB ACTIONS RUNNER
# ============================================================

import os
import re
import sys
import time
import joblib
import requests
import unicodedata
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from datetime import datetime, timezone

# Repo-local imports for GitHub Actions / Streamlit project layout
PROJECT_ROOT = "."
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ufc_pipeline_utils import *
from ufc_odds_utils import *
from pipeline_config import *

try:
    from ufc_clv_utils import *
except Exception:
    pass

# Notebook compatibility: no-op display() when running as .py
try:
    display
except NameError:
    def display(*args, **kwargs):
        return None

SNAPSHOT_RUN_ID = datetime.now(
    timezone.utc
).strftime("%Y%m%d_%H%M%S")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

# ============================================================
# SECTION 1 — PATHS / CONFIG
# ============================================================

# Centralized path config from ufc_pipeline_utils.py
# Change model_version here once, instead of hardcoding paths throughout the notebook.
paths = UFCPipelinePaths(
    base_path=".",
    model_version="UFC_Model_v5_Experiment"
)

BASE_PATH = paths.base_path
PRODUCTION_DIR = paths.production_dir
ROLLING_FEATURES_PATH = paths.rolling_features_path

LIVE_CARD_OUTPUT = paths.live_card_output
LIVE_PREDICTIONS_OUTPUT = paths.live_predictions_output
LIVE_BETTING_CARD_OUTPUT = paths.live_betting_card_output
LIVE_FEATURE_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_live_feature_audit.csv"
LIVE_MATCH_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_live_match_audit.csv"
LIVE_ODDS_AUDIT_OUTPUT = f"{BASE_PATH}/ufc_live_odds_audit.csv"
ACTION_BOARD_PARQUET_OUTPUT = f"{BASE_PATH}/ufc_live_action_board.parquet"
WATCHLIST_OUTPUT = paths.watchlist_output
ACTION_BOARD_OUTPUT = paths.action_board_output
CLV_LOG_PATH = paths.clv_log_path

PREFERRED_BOOKMAKER = "DraftKings"
BANKROLL = 10000

print("BASE_PATH:", BASE_PATH)
print("PRODUCTION_DIR:", PRODUCTION_DIR)


# ============================================================
# SECTION 2 — LOAD FROZEN PRODUCTION ARTIFACTS
# ============================================================

artifacts = load_production_artifacts(paths)

model = artifacts["model"]
feature_columns = artifacts["feature_columns"]
BEST_THRESHOLD = artifacts["best_threshold"]
production_config = artifacts["production_config"]

betting_filters = get_betting_filters(production_config)
MIN_EDGE = betting_filters["min_edge"]
MIN_CONFIDENCE = betting_filters["min_confidence"]
MIN_ODDS = betting_filters["min_odds"]
MAX_ODDS = betting_filters["max_odds"]

KELLY_FRACTION = production_config["staking"]["kelly_fraction"]
KELLY_MULTIPLIER = KELLY_FRACTION
MAX_STAKE_PCT = production_config["staking"]["max_stake_pct"]

print("Model version:", production_config.get("version"))
print("Feature count:", len(feature_columns))
print("Best threshold:", BEST_THRESHOLD)
print("Filters:", production_config["betting_filters"])
print("Staking:", production_config["staking"])


# ============================================================
# SECTION 3 — HELPER FUNCTIONS
# ============================================================

# Shared helpers are imported from ufc_pipeline_utils.py above.
# Verified available:
# - normalize_name / token_set_score / fuzzy_score
# - safe_value / to_decimal_rate
# - american_to_decimal / american_to_implied_prob
# - decimal_ev / calculate_ev
# - kelly_fraction / scaled_kelly_stake
# - decimal_kelly_fraction / scaled_decimal_kelly_stake
# - align_features / clip_probability_series

print("Shared helper functions loaded from ufc_pipeline_utils.py")

# ============================================================
# SECTION 3A — VERIFY SHARED HELPER AVAILABILITY
# ============================================================

required_helpers = [
    "normalize_name",
    "token_set_score",
    "fuzzy_score",
    "safe_value",
    "to_decimal_rate",
    "american_to_decimal",
    "american_to_implied_prob",
    "decimal_ev",
    "calculate_ev",
    "kelly_fraction",
    "scaled_kelly_stake",
    "decimal_kelly_fraction",
    "scaled_decimal_kelly_stake",
    "align_features",
    "clip_probability_series",
]

missing_helpers = [
    fn for fn in required_helpers
    if fn not in globals()
]

if missing_helpers:
    raise NameError(
        f"Missing shared helper functions from ufc_pipeline_utils.py: {missing_helpers}"
    )

print("All required shared helpers are available.")

# ============================================================
# SECTION 4 — LOAD ROLLING FEATURE DATABASE
# ============================================================

rolling_df = pd.read_parquet(
    "UFC_enhanced_rolling_features_EWM.parquet"
)

rolling_df["date"] = pd.to_datetime(
    rolling_df["date"],
    errors="coerce"
)

rolling_df = rolling_df.sort_values("date").reset_index(drop=True)

print("Rolling rows:", len(rolling_df))
print("Date range:", rolling_df["date"].min(), "to", rolling_df["date"].max())
print("Columns:", len(rolling_df.columns))

# ============================================================
# SECTION 5 — BUILD NEUTRAL FIGHTER LOOKUP
# ============================================================

possible_red_id_cols = [
    "r_fighter_id",
    "red_fighter_id",
    "R_fighter_id",
    "R_ID",
    "r_id"
]

possible_blue_id_cols = [
    "b_fighter_id",
    "blue_fighter_id",
    "B_fighter_id",
    "B_ID",
    "b_id"
]

possible_red_name_cols = [
    "r_name",
    "red_fighter",
    "R_fighter",
    "R"
]

possible_blue_name_cols = [
    "b_name",
    "blue_fighter",
    "B_fighter",
    "B"
]


def first_existing_column(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col

    return None


red_id_col = first_existing_column(rolling_df, possible_red_id_cols)
blue_id_col = first_existing_column(rolling_df, possible_blue_id_cols)

red_name_col = first_existing_column(rolling_df, possible_red_name_cols)
blue_name_col = first_existing_column(rolling_df, possible_blue_name_cols)

print("Red ID column:", red_id_col)
print("Blue ID column:", blue_id_col)
print("Red name column:", red_name_col)
print("Blue name column:", blue_name_col)

if red_name_col is None or blue_name_col is None:
    raise ValueError("Could not find fighter name columns in rolling dataset.")

# ============================================================
# SECTION 6 — CREATE LONG FIGHTER-LEVEL FEATURE TABLE
# ============================================================

long_rows = []

for _, row in rolling_df.iterrows():

    red_long = {
        "fighter_name": row.get(red_name_col),
        "fighter_norm": normalize_name(row.get(red_name_col)),
        "fighter_id": str(row.get(red_id_col, "")) if red_id_col else "",
        "date": row["date"]
    }

    blue_long = {
        "fighter_name": row.get(blue_name_col),
        "fighter_norm": normalize_name(row.get(blue_name_col)),
        "fighter_id": str(row.get(blue_id_col, "")) if blue_id_col else "",
        "date": row["date"]
    }

    for col in rolling_df.columns:

        if col.startswith("r_pre_"):
            neutral = col.replace("r_pre_", "")
            red_long[neutral] = row[col]

        elif col.startswith("b_pre_"):
            neutral = col.replace("b_pre_", "")
            blue_long[neutral] = row[col]

        elif col.startswith("r_ewm_"):
            neutral = col.replace("r_ewm_", "ewm_")
            red_long[neutral] = row[col]

        elif col.startswith("b_ewm_"):
            neutral = col.replace("b_ewm_", "ewm_")
            blue_long[neutral] = row[col]

        elif col.startswith("r_recent_form_"):
            neutral = col.replace("r_recent_form_", "recent_form_")
            red_long[neutral] = row[col]

        elif col.startswith("b_recent_form_"):
            neutral = col.replace("b_recent_form_", "recent_form_")
            blue_long[neutral] = row[col]

        elif col.startswith("r_") and not col.startswith("r_pre_"):
            neutral = col.replace("r_", "")
            red_long[neutral] = row[col]

        elif col.startswith("b_") and not col.startswith("b_pre_"):
            neutral = col.replace("b_", "")
            blue_long[neutral] = row[col]

    long_rows.append(red_long)
    long_rows.append(blue_long)

fighter_long_df = pd.DataFrame(long_rows)

fighter_latest = (
    fighter_long_df
    .sort_values("date")
    .groupby(["fighter_id", "fighter_norm"], as_index=False)
    .tail(1)
    .reset_index(drop=True)
)

print("Fighter lookup rows:", len(fighter_latest))

display(
    fighter_latest[[
        "fighter_name",
        "fighter_norm",
        "fighter_id",
        "date"
    ]].head()
)

# ============================================================
# SECTION 7 — LOAD CACHED LIVE CARD
# ============================================================

LIVE_CARD_BASE_OUTPUT = f"{BASE_PATH}/ufc_live_card.parquet"

ufcstats_card_df = pd.read_parquet(
    LIVE_CARD_BASE_OUTPUT
)

print("Loaded cached UFC live card:")
print(LIVE_CARD_BASE_OUTPUT)
print("Fights loaded:", len(ufcstats_card_df))

display(ufcstats_card_df)

# ============================================================
# SECTION 8 — MATCH CARD FIGHTERS TO FEATURE DATABASE
# ============================================================

def find_fighter_row(fighter_id, fighter_name, fighter_latest):

    fighter_id = str(fighter_id).strip()
    fighter_norm = normalize_name(fighter_name)

    if fighter_id:
        id_match = fighter_latest[
            fighter_latest["fighter_id"].astype(str) == fighter_id
        ]

        if len(id_match) > 0:
            return id_match.iloc[0], "id_match"

    exact_name_match = fighter_latest[
        fighter_latest["fighter_norm"] == fighter_norm
    ]

    if len(exact_name_match) > 0:
        return exact_name_match.iloc[0], "exact_name_match"

    temp = fighter_latest.copy()

    temp["name_score"] = temp["fighter_norm"].apply(
    lambda x: token_set_score(x, fighter_norm)
)

    temp = temp.sort_values("name_score", ascending=False)

    best = temp.iloc[0]

    if best["name_score"] >= 88:
        return best, f"fuzzy_name_match_{round(best['name_score'], 3)}"

    return None, "missing"


match_audit_rows = []

for _, fight in ufcstats_card_df.iterrows():

    red_row, red_source = find_fighter_row(
        fight["red_fighter_id"],
        fight["red_fighter"],
        fighter_latest
    )

    blue_row, blue_source = find_fighter_row(
        fight["blue_fighter_id"],
        fight["blue_fighter"],
        fighter_latest
    )

    match_audit_rows.append({
        "red_fighter": fight["red_fighter"],
        "blue_fighter": fight["blue_fighter"],
        "red_fighter_id": fight["red_fighter_id"],
        "blue_fighter_id": fight["blue_fighter_id"],
        "red_feature_match": red_source,
        "blue_feature_match": blue_source,
        "matched_red_name": None if red_row is None else red_row["fighter_name"],
        "matched_blue_name": None if blue_row is None else blue_row["fighter_name"],
        "red_latest_date": None if red_row is None else red_row["date"],
        "blue_latest_date": None if blue_row is None else blue_row["date"]
    })

match_audit_df = pd.DataFrame(match_audit_rows)
match_audit_df.to_parquet(LIVE_MATCH_AUDIT_OUTPUT, index=False)

display(match_audit_df)

missing = match_audit_df[
    (match_audit_df["red_feature_match"] == "missing")
    |
    (match_audit_df["blue_feature_match"] == "missing")
]

if len(missing) > 0:
    print("WARNING: Some fighters were not matched to historical features.")
    display(missing)

# ============================================================
# SECTION 9 — BUILD LIVE MODEL FEATURE ROWS
# ============================================================

from ufc_feature_engineering import (
    add_v5_engineered_features,
    get_engineered_feature_list,
)

ENGINEERED_FEATURE_SET = set(get_engineered_feature_list())


def build_live_feature_row(fight):

    red_row, red_match = find_fighter_row(
        fight["red_fighter_id"],
        fight["red_fighter"],
        fighter_latest,
    )

    blue_row, blue_match = find_fighter_row(
        fight["blue_fighter_id"],
        fight["blue_fighter"],
        fighter_latest,
    )

    out = fight.to_dict()

    out["red_feature_match"] = red_match
    out["blue_feature_match"] = blue_match

    # --------------------------------------------------------
    # Build standard *_diff features only.
    # Skip engineered features because add_v5_engineered_features()
    # creates those centrally.
    # --------------------------------------------------------

    for feature in feature_columns:

        if feature in ENGINEERED_FEATURE_SET:
            continue

        if feature.endswith("_diff"):

            base = feature[:-5]

            red_value = safe_value(red_row, base)
            blue_value = safe_value(blue_row, base)

            out[feature] = red_value - blue_value

    # --------------------------------------------------------
    # Add prefixed raw columns needed by add_v5_engineered_features()
    # --------------------------------------------------------

    raw_feature_map = {
        "age": "age",
        "height": "height",
        "reach": "reach",
        "weight": "weight",

        "pre_str_def": "str_def",
        "pre_td_def": "td_def",
        "pre_sapm": "sapm",
        "pre_splm": "splm",
        "pre_td_avg": "td_avg",
        "pre_sub_avg": "sub_avg",
        "pre_fights": "fights",
        "pre_losses": "losses",
        "pre_ctrl_against_per_min": "ctrl_against_per_min",
    }

    for prefixed_name, source_name in raw_feature_map.items():

        out[f"r_{prefixed_name}"] = safe_value(
            red_row,
            source_name,
        )

        out[f"b_{prefixed_name}"] = safe_value(
            blue_row,
            source_name,
        )

    return out


live_rows = []

for _, fight in ufcstats_card_df.iterrows():
    live_rows.append(
        build_live_feature_row(fight)
    )

live_df = pd.DataFrame(live_rows)

# ------------------------------------------------------------
# Add centralized Champion Clean Set engineered features
# ------------------------------------------------------------

live_df = add_v5_engineered_features(live_df)

# ------------------------------------------------------------
# Safety: remove any accidental duplicate columns
# ------------------------------------------------------------

duplicate_cols = live_df.columns[
    live_df.columns.duplicated()
].tolist()

print("Duplicate columns after Section 9:", len(duplicate_cols))
print(duplicate_cols)

live_df = live_df.loc[
    :,
    ~live_df.columns.duplicated()
].copy()

print("Live feature rows:", len(live_df))

display(live_df.head())

# ============================================================
# SECTION 10 — FEATURE VALIDATION
# ============================================================

missing_features = [
    col for col in feature_columns
    if col not in live_df.columns
]

extra_model_like_cols = [
    col for col in live_df.columns
    if col.endswith("_diff") and col not in feature_columns
]

print("Missing model features:", len(missing_features))
print(missing_features[:50])

print("Extra diff columns not used by model:", len(extra_model_like_cols))
print(extra_model_like_cols[:50])

# Add missing columns as 0.
# This keeps inference from crashing, but the audit will show if many are missing.
for col in missing_features:
    live_df[col] = 0

X_live = live_df[feature_columns].copy()

for col in X_live.columns:
    X_live[col] = pd.to_numeric(X_live[col], errors="coerce").fillna(0)

feature_audit_rows = []

for idx, row in X_live.iterrows():

    zero_pct = (row == 0).mean() * 100
    nonzero_count = (row != 0).sum()

    feature_audit_rows.append({
        "red_fighter": live_df.loc[idx, "red_fighter"],
        "blue_fighter": live_df.loc[idx, "blue_fighter"],
        "red_feature_match": live_df.loc[idx, "red_feature_match"],
        "blue_feature_match": live_df.loc[idx, "blue_feature_match"],
        "missing_feature_count": len(missing_features),
        "nonzero_feature_count": nonzero_count,
        "zero_feature_pct": zero_pct
    })

feature_audit_df = pd.DataFrame(feature_audit_rows)
feature_audit_df.to_parquet(LIVE_FEATURE_AUDIT_OUTPUT, index=False)

display(feature_audit_df.sort_values("zero_feature_pct", ascending=False))

print("X_live shape:", X_live.shape)

# ============================================================
# SAVE NORMALIZED LIVE CARD FOR DOWNSTREAM SYSTEMS
# ============================================================

LIVE_CARD_BASE_OUTPUT = f"{BASE_PATH}/ufc_live_card.parquet"

live_card_base_cols = [
    "event_name",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
]

if "commence_time" in live_df.columns:
    live_card_base_cols.append("commence_time")

live_card_df = live_df[
    [col for col in live_card_base_cols if col in live_df.columns]
].copy()

live_card_df["fight_id"] = (
    live_card_df["red_fighter_id"].astype(str)
    + "_vs_"
    + live_card_df["blue_fighter_id"].astype(str)
)

live_card_df.to_parquet(
    LIVE_CARD_BASE_OUTPUT,
    index=False
)

print("Normalized live card saved:")
print(LIVE_CARD_BASE_OUTPUT)

display(live_card_df.head())

# ============================================================
# SECTION 11 — RUN MODEL PREDICTIONS
# ============================================================

# Raw model probabilities
live_probs = model.predict_proba(X_live)[:, 1]

# Probability clipping
CLIP_LOW = 0.02
CLIP_HIGH = 0.98

live_probs = np.clip(
    live_probs,
    CLIP_LOW,
    CLIP_HIGH
)

live_df["red_model_prob"] = live_probs
live_df["blue_model_prob"] = 1 - live_probs

live_df["red_model_prob_pct"] = live_df["red_model_prob"] * 100
live_df["blue_model_prob_pct"] = live_df["blue_model_prob"] * 100

live_df["model_pick"] = np.where(
    live_df["red_model_prob"] >= BEST_THRESHOLD,
    live_df["red_fighter"],
    live_df["blue_fighter"]
)

live_df["model_confidence"] = np.maximum(
    live_df["red_model_prob"],
    live_df["blue_model_prob"]
) * 100

prediction_display = live_df[[
    "event_name",
    "red_fighter",
    "blue_fighter",
    "red_model_prob_pct",
    "blue_model_prob_pct",
    "model_pick",
    "model_confidence",
    "red_feature_match",
    "blue_feature_match"
]].sort_values("model_confidence", ascending=False)

display(prediction_display)

# ============================================================
# SECTION 12 — PULL ODDS FROM THE ODDS API
# ============================================================

from pipeline_config import *

odds_data = fetch_the_odds_api_events(
    api_key=ODDS_API_KEY,
    sport=SPORT,
    regions=REGIONS,
    markets=MARKETS,
    odds_format=ODDS_FORMAT,
)

print("Odds events returned:", len(odds_data))

odds_df = flatten_h2h_odds(
    odds_data,
    preferred_bookmaker=PREFERRED_BOOKMAKER,
)

print("Flattened odds rows:", len(odds_df))

if odds_df.empty:
    raise ValueError(
        f"No odds rows found for bookmaker: {PREFERRED_BOOKMAKER}. "
        "Check that PREFERRED_BOOKMAKER matches the bookmaker title returned by the API."
    )

odds_df["odds_fighter_1"] = odds_df["fighter_1"]
odds_df["odds_fighter_2"] = odds_df["fighter_2"]

odds_df["fighter_1_odds"] = odds_df["fighter_1_american_odds"]
odds_df["fighter_2_odds"] = odds_df["fighter_2_american_odds"]

print("Odds rows created:", len(odds_df))
print("Odds columns:")
print(odds_df.columns.tolist())

display(odds_df.head(20))

# ============================================================
# SECTION 13 — MATCH ODDS TO UFCSTATS CARD
# ============================================================

live_df = attach_h2h_odds_to_live_df(
    live_df=live_df.copy(),
    odds_pool=odds_df,
    min_match_score=MIN_ODDS_MATCH_SCORE,
)

live_df["red_odds"] = live_df["red_american_odds"]
live_df["blue_odds"] = live_df["blue_american_odds"]

odds_audit_columns = [
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "matched_odds_fighter_1",
    "matched_odds_fighter_2",
    "odds_fighter_1_id",
    "odds_fighter_2_id",
    "odds_fighter_1_score",
    "odds_fighter_2_score",
    "odds_min_single_score",
    "red_odds",
    "blue_odds",
    "bookmaker",
    "commence_time",
    "odds_match_type",
    "odds_match_score",
]

odds_audit_df = live_df[
    [col for col in odds_audit_columns if col in live_df.columns]
].copy()

odds_audit_df.to_parquet(
    LIVE_ODDS_AUDIT_OUTPUT,
    index=False,
)

print("Odds matching complete.")
print(f"Odds audit saved to: {LIVE_ODDS_AUDIT_OUTPUT}")

display(odds_audit_df)

# ============================================================
# SECTION 14 — DATA QUALITY FILTERS
# ============================================================

MIN_NONZERO_FEATURES = 90
MAX_ZERO_FEATURE_PCT = 35
MIN_ODDS_MATCH_SCORE = 0.90

# Merge feature-audit quality columns back into live_df
quality_cols = [
    "red_fighter",
    "blue_fighter",
    "nonzero_feature_count",
    "zero_feature_pct"
]

live_df = live_df.merge(
    feature_audit_df[quality_cols],
    on=["red_fighter", "blue_fighter"],
    how="left"
)

# Feature coverage filters
live_df["passes_feature_count_filter"] = (
    live_df["nonzero_feature_count"] >= MIN_NONZERO_FEATURES
)

live_df["passes_zero_pct_filter"] = (
    live_df["zero_feature_pct"] <= MAX_ZERO_FEATURE_PCT
)

# Fighter match filters
live_df["passes_fighter_match_filter"] = (
    (live_df["red_feature_match"] != "missing")
    &
    (live_df["blue_feature_match"] != "missing")
)

# Odds quality filters
live_df["passes_odds_match_filter"] = (
    (live_df["odds_match_score"] >= MIN_ODDS_MATCH_SCORE)
    &
    (~live_df["odds_match_type"].isin([
        "missing",
        "low_confidence_missing"
    ]))
)

# Combined data quality filter
live_df["passes_data_quality_filter"] = (
    live_df["passes_feature_count_filter"]
    &
    live_df["passes_zero_pct_filter"]
    &
    live_df["passes_fighter_match_filter"]
    &
    live_df["passes_odds_match_filter"]
)

display(live_df[[
    "red_fighter",
    "blue_fighter",
    "nonzero_feature_count",
    "zero_feature_pct",
    "red_feature_match",
    "blue_feature_match",
    "odds_match_type",
    "odds_match_score",
    "passes_feature_count_filter",
    "passes_zero_pct_filter",
    "passes_fighter_match_filter",
    "passes_odds_match_filter",
    "passes_data_quality_filter"
]].sort_values("passes_data_quality_filter"))

# ============================================================
# SECTION 14 — CLEAN ODDS + IMPLIED PROBABILITIES
# ============================================================

live_df["commence_time"] = pd.to_datetime(
    live_df["commence_time"],
    utc=True,
    errors="coerce"
)

live_df["red_odds"] = pd.to_numeric(
    live_df["red_odds"],
    errors="coerce"
)

live_df["blue_odds"] = pd.to_numeric(
    live_df["blue_odds"],
    errors="coerce"
)

live_df = live_df.dropna(
    subset=[
        "red_odds",
        "blue_odds"
    ]
).copy()

live_df["red_decimal_odds"] = live_df["red_odds"].apply(american_to_decimal)
live_df["blue_decimal_odds"] = live_df["blue_odds"].apply(american_to_decimal)

live_df["red_implied_prob"] = live_df["red_odds"].apply(american_to_implied_prob)
live_df["blue_implied_prob"] = live_df["blue_odds"].apply(american_to_implied_prob)

live_df.to_parquet(LIVE_CARD_OUTPUT, index=False)

print("Saved live card with odds:")
print(LIVE_CARD_OUTPUT)

display(
    live_df[[
        "red_fighter",
        "blue_fighter",
        "red_odds",
        "blue_odds",
        "bookmaker",
        "odds_match_type",
        "odds_match_score"
    ]]
)

# ============================================================
# SECTION 15 — CALCULATE EDGE + EV
# ============================================================

live_df["red_edge"] = (
    live_df["red_model_prob"]
    -
    live_df["red_implied_prob"]
)

live_df["blue_edge"] = (
    live_df["blue_model_prob"]
    -
    live_df["blue_implied_prob"]
)

live_df["red_ev"] = live_df.apply(
    lambda row: decimal_ev(
        row["red_model_prob"],
        row["red_decimal_odds"]
    ),
    axis=1
)

live_df["blue_ev"] = live_df.apply(
    lambda row: decimal_ev(
        row["blue_model_prob"],
        row["blue_decimal_odds"]
    ),
    axis=1
)

live_df["red_is_best"] = live_df["red_ev"] >= live_df["blue_ev"]

live_df["best_side"] = np.where(
    live_df["red_is_best"],
    live_df["red_fighter"],
    live_df["blue_fighter"]
)

live_df["fight_id"] = (
    live_df["red_fighter_id"].astype(str)
    + "_vs_"
    + live_df["blue_fighter_id"].astype(str)
)

live_df["best_side_fighter_id"] = np.where(
    live_df["red_is_best"],
    live_df["red_fighter_id"],
    live_df["blue_fighter_id"]
)

live_df["best_side_opponent_id"] = np.where(
    live_df["red_is_best"],
    live_df["blue_fighter_id"],
    live_df["red_fighter_id"]
)
live_df["best_prob"] = np.where(
    live_df["red_is_best"],
    live_df["red_model_prob"],
    live_df["blue_model_prob"]
)

live_df["best_implied_prob"] = np.where(
    live_df["red_is_best"],
    live_df["red_implied_prob"],
    live_df["blue_implied_prob"]
)

live_df["best_edge"] = np.where(
    live_df["red_is_best"],
    live_df["red_edge"],
    live_df["blue_edge"]
)

live_df["best_ev"] = np.where(
    live_df["red_is_best"],
    live_df["red_ev"],
    live_df["blue_ev"]
)

live_df["best_american_odds"] = np.where(
    live_df["red_is_best"],
    live_df["red_odds"],
    live_df["blue_odds"]
)

live_df["best_decimal_odds"] = np.where(
    live_df["red_is_best"],
    live_df["red_decimal_odds"],
    live_df["blue_decimal_odds"]
)

live_df["best_confidence"] = live_df["best_prob"] * 100

print("EV calculated.")

# ============================================================
# SECTION 16 — APPLY FILTERS + KELLY STAKING
# ============================================================

live_df = apply_standard_bet_filters(
    live_df,
    edge_col="best_edge",
    confidence_col="best_confidence",
    odds_col="best_american_odds",
    ev_col="best_ev",
    min_edge=MIN_EDGE,
    min_confidence=MIN_CONFIDENCE,
    min_odds=MIN_ODDS,
    max_odds=MAX_ODDS
)

# Preserve the live pipeline's additional data-quality gate.
live_df["is_official_bet"] = (
    live_df["is_official_bet"]
    &
    live_df["passes_data_quality_filter"]
)

live_df["recommended_stake"] = live_df.apply(
    lambda row: scaled_kelly_stake(
        bankroll=BANKROLL,
        model_prob=row["best_prob"],
        american_odds=row["best_american_odds"],
        kelly_multiplier=KELLY_MULTIPLIER,
        max_stake_pct=MAX_STAKE_PCT,
    ),
    axis=1,
)

print("Official bets:", int(live_df["is_official_bet"].sum()))

display(live_df[[
    "red_fighter",
    "blue_fighter",
    "best_side",
    "best_prob",
    "best_implied_prob",
    "best_edge",
    "best_ev",
    "best_american_odds",
    "best_confidence",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_filter",
    "passes_positive_ev_filter",
    "is_official_bet",
    "passes_data_quality_filter",
    "recommended_stake"
]].sort_values("best_ev", ascending=False))


# ============================================================
# SECTION 17 — WATCHLIST / NEAR-MISS ENGINE
# ============================================================

# Count failed production filters
filter_cols = [
    "passes_data_quality_filter",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_filter",
    "passes_positive_ev_filter"
]

live_df["failed_filter_count"] = (
    len(filter_cols)
    -
    live_df[filter_cols].sum(axis=1)
)

def get_watchlist_reason(row):
    failed = []

    if not row["passes_data_quality_filter"]:
        failed.append("data_quality")

    if not row["passes_edge_filter"]:
        failed.append("edge")

    if not row["passes_confidence_filter"]:
        failed.append("confidence")

    if not row["passes_odds_filter"]:
        failed.append("odds_range")

    if not row["passes_positive_ev_filter"]:
        failed.append("positive_ev")

    return ", ".join(failed)


def get_watchlist_tier(row):
    if row["is_official_bet"]:
        return "official_bet"

    if row["passes_data_quality_filter"] and row["failed_filter_count"] == 1:
        return "tier_1_near_miss"

    if row["passes_data_quality_filter"] and row["best_ev"] > 0.50:
        return "tier_2_high_ev_review"

    if not row["passes_data_quality_filter"] and row["best_ev"] > 0.25:
        return "tier_3_sparse_data_review"

    return "rejected"


live_df["watchlist_reason"] = live_df.apply(get_watchlist_reason, axis=1)
live_df["watchlist_tier"] = live_df.apply(get_watchlist_tier, axis=1)

live_df["is_watchlist_bet"] = live_df["watchlist_tier"].isin([
    "tier_1_near_miss",
    "tier_2_high_ev_review",
    "tier_3_sparse_data_review"
])

watchlist_df = live_df[
    live_df["is_watchlist_bet"]
].copy()

watchlist_display_cols = [
    "event_name",
    "red_fighter",
    "blue_fighter",
    "best_side",
    "best_prob",
    "best_implied_prob",
    "best_edge",
    "best_ev",
    "best_american_odds",
    "best_confidence",
    "watchlist_tier",
    "watchlist_reason",
    "passes_data_quality_filter",
    "nonzero_feature_count",
    "zero_feature_pct",
    "red_feature_match",
    "blue_feature_match",
    "odds_match_type",
    "odds_match_score",
    "bookmaker"
]

watchlist_df = watchlist_df[watchlist_display_cols].sort_values(
    ["watchlist_tier", "best_ev"],
    ascending=[True, False]
)

watchlist_df.to_parquet(WATCHLIST_OUTPUT, index=False)

print("Watchlist bets:", len(watchlist_df))
print("Saved watchlist:")
print(WATCHLIST_OUTPUT)

display(watchlist_df)

# ============================================================
# SECTION 17 — DISPLAY FULL LIVE CARD
# ============================================================

full_card_display = live_df.copy()

percent_cols = [
    "red_model_prob",
    "blue_model_prob",
    "red_implied_prob",
    "blue_implied_prob",
    "red_edge",
    "blue_edge",
    "red_ev",
    "blue_ev",
    "best_prob",
    "best_implied_prob",
    "best_edge",
    "best_ev"
]

for col in percent_cols:
    full_card_display[col] = full_card_display[col] * 100

full_card_display = full_card_display[[
    "event_name",
    "red_fighter",
    "blue_fighter",
    "model_pick",
    "model_confidence",
    "red_model_prob",
    "blue_model_prob",
    "red_odds",
    "blue_odds",
    "best_side",
    "best_prob",
    "best_implied_prob",
    "best_edge",
    "best_ev",
    "best_american_odds",
    "is_official_bet",
    "recommended_stake",
    "bookmaker",
    "odds_match_type",
    "red_feature_match",
    "blue_feature_match"
]].sort_values(
    "best_ev",
    ascending=False
)

display(full_card_display)

# ============================================================
# SECTION 17B — ADD TWO-SIDED MARKET ODDS FOR CLV
# ============================================================

live_df["red_american_odds"] = live_df["red_odds"]
live_df["blue_american_odds"] = live_df["blue_odds"]

live_df["red_market_implied_prob"] = live_df["red_implied_prob"]
live_df["blue_market_implied_prob"] = live_df["blue_implied_prob"]

# ============================================================
# SECTION 18 — FINAL ACTION BOARD
# ============================================================

live_df["snapshot_run_id"] = SNAPSHOT_RUN_ID

action_board_df = live_df[
    (
        live_df["is_official_bet"]
    )
    |
    (
        live_df["is_watchlist_bet"]
    )
].copy()

action_board_display_cols = [
    # =========================================================
    # SNAPSHOT METADATA
    # =========================================================
    "snapshot_run_id",

    # =========================================================
    # IDENTIFIERS
    # =========================================================
    "event_name",
    "commence_time",
    "fight_id",

    "red_fighter_id",
    "blue_fighter_id",

    "red_fighter",
    "blue_fighter",

    "best_side",
    "best_side_fighter_id",
    "best_side_opponent_id",


    # =========================================================
    # MODEL OUTPUTS
    # =========================================================
    "best_prob",
    "best_implied_prob",
    "best_edge",
    "best_ev",

    # =========================================================
    # MARKET INFO
    # =========================================================
    "best_american_odds",
    "bookmaker",
    "red_american_odds",
    "blue_american_odds",

    "red_market_implied_prob",
    "blue_market_implied_prob",

    # =========================================================
    # CONFIDENCE + STAKING
    # =========================================================
    "best_confidence",
    "recommended_stake",

    # =========================================================
    # WATCHLIST / BET STATUS
    # =========================================================
    "watchlist_tier",
    "watchlist_reason",

    "is_official_bet",
    "is_watchlist_bet",

    # =========================================================
    # DATA QUALITY
    # =========================================================
    "passes_data_quality_filter",

    "nonzero_feature_count",
    "zero_feature_pct",

    # =========================================================
    # MATCH QUALITY
    # =========================================================
    "red_feature_match",
    "blue_feature_match",

    "odds_match_type",
    "odds_match_score",
]

action_board_df = action_board_df[
    action_board_display_cols
].sort_values(
    [
        "is_official_bet",
        "watchlist_tier",
        "best_ev"
    ],
    ascending=[False, True, False]
)

action_board_df.to_parquet(
    ACTION_BOARD_OUTPUT,
    index=False
)

print("Action board saved:")
print(ACTION_BOARD_OUTPUT)

display(action_board_df)

# Save parquet mirror for dashboard / automation

action_board_df.to_parquet(ACTION_BOARD_PARQUET_OUTPUT, index=False)

# Update official bets ledger if CLV helpers are available
try:
    from ufc_clv_utils import build_official_bets_from_action_board, append_official_bets_log

    OFFICIAL_BETS_LOG_PATH = f"{BASE_PATH}/ufc_official_bets_log.csv"

    official_bets_df = build_official_bets_from_action_board(action_board_df)

    append_official_bets_log(
        official_bets_df=official_bets_df,
        output_path=OFFICIAL_BETS_LOG_PATH,
    )

    print("Official bets ledger updated:")
    print(OFFICIAL_BETS_LOG_PATH)

except Exception as exc:
    print(f"Official bets ledger update skipped: {exc}")

# ============================================================
# SECTION 19 — FINAL AUDIT SUMMARY
# ============================================================

print("========== LIVE PIPELINE AUDIT ==========")
print("Fights scraped from UFCStats:", len(ufcstats_card_df))
print("Fights with odds matched:", len(live_df))
print("Official bets:", int(live_df["is_official_bet"].sum()))
print("Feature count used:", len(feature_columns))
print("Missing model features added as zero:", len(missing_features))
print()

print("Files saved:")
print("Card with odds:", LIVE_CARD_OUTPUT)
print("Predictions:", LIVE_PREDICTIONS_OUTPUT)
print("Betting card:", LIVE_BETTING_CARD_OUTPUT)
print("Feature audit:", LIVE_FEATURE_AUDIT_OUTPUT)
print("Match audit:", LIVE_MATCH_AUDIT_OUTPUT)
print("Odds audit:", LIVE_ODDS_AUDIT_OUTPUT)
print("Action board:", ACTION_BOARD_OUTPUT)

display(feature_audit_df.sort_values("zero_feature_pct", ascending=False))
display(odds_audit_df)
