# ============================================================
# run_market_update.py
# Dedicated market / odds update runner
# ============================================================

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    MARKET_MATCH_AUDIT_PATH,
    MARKET_ODDS_PATH,
    MARKET_SNAPSHOTS_PATH,
    MODEL_PREDICTIONS_PATH,
    ensure_data_dirs,
)
from pipeline_config import *
from ufc_pipeline_utils import *
from ufc_odds_utils import *

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "."

MARKET_ODDS_OUTPUT = MARKET_ODDS_PATH
MARKET_SNAPSHOTS_OUTPUT = MARKET_SNAPSHOTS_PATH
MARKET_MATCH_AUDIT_OUTPUT = MARKET_MATCH_AUDIT_PATH

SNAPSHOT_TIMESTAMP = datetime.now(
    timezone.utc
).isoformat()

SNAPSHOT_RUN_ID = datetime.now(
    timezone.utc
).strftime("%Y%m%d_%H%M%S")

PREFERRED_BOOKMAKER = "DraftKings"

# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

ensure_data_dirs()

predictions_df = pd.read_parquet(
    MODEL_PREDICTIONS_PATH
)

print("Prediction rows:", len(predictions_df))

# ============================================================
# PULL ODDS
# ============================================================

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
        "No odds returned from bookmaker."
    )

# ============================================================
# STANDARDIZE ODDS DATA
# ============================================================

odds_df["snapshot_run_id"] = SNAPSHOT_RUN_ID
odds_df["snapshot_timestamp"] = SNAPSHOT_TIMESTAMP

odds_df["fighter_1_norm"] = odds_df[
    "fighter_1"
].apply(normalize_name)

odds_df["fighter_2_norm"] = odds_df[
    "fighter_2"
].apply(normalize_name)

# ============================================================
# MATCH ODDS TO MODEL PREDICTIONS
# ============================================================

market_rows = []
market_audit_rows = []

MIN_ODDS_MATCH_SCORE = 90

MARKET_OUTPUT_COLUMNS = [
    "fight_id",
    "event_name",
    "red_fighter",
    "blue_fighter",
    "snapshot_run_id",
    "snapshot_timestamp",
    "bookmaker",
    "commence_time",
    "red_american_odds",
    "blue_american_odds",
    "red_decimal_odds",
    "blue_decimal_odds",
    "red_implied_prob",
    "blue_implied_prob",
    "odds_match_score",
    "odds_min_single_score",
    "odds_match_type",
    "odds_match_order",
    "matched_fighter_1",
    "matched_fighter_2",
    "odds_fighter_1_id",
    "odds_fighter_2_id",
    "odds_fighter_1_score",
    "odds_fighter_2_score",
]


def is_blank_name(value):
    return pd.isna(value) or str(value).strip() == ""


for _, pred in predictions_df.iterrows():

    red_fighter = pred.get("red_fighter")
    blue_fighter = pred.get("blue_fighter")

    audit_row = {
        "snapshot_run_id": SNAPSHOT_RUN_ID,
        "snapshot_timestamp": SNAPSHOT_TIMESTAMP,
        "event_name": pred.get("event_name"),
        "fight_id": pred.get("fight_id"),
        "red_fighter": red_fighter,
        "blue_fighter": blue_fighter,
        "matched_fighter_1": None,
        "matched_fighter_2": None,
        "odds_match_score": np.nan,
        "odds_min_single_score": np.nan,
        "odds_match_type": "low_confidence",
        "odds_match_order": None,
        "odds_fighter_1_score": np.nan,
        "odds_fighter_2_score": np.nan,
    }

    if is_blank_name(red_fighter) or is_blank_name(blue_fighter):
        audit_row["odds_match_type"] = "invalid_prediction_fighters"
        market_audit_rows.append(audit_row)
        continue

    odds_match = match_live_fight_to_odds_row(
        live_row=pred,
        odds_pool=odds_df,
        min_single_score=MIN_ODDS_MATCH_SCORE,
    )

    if odds_match is None:
        market_audit_rows.append(audit_row)
        continue

    # match_live_fight_to_odds_row() detects whether The Odds API row
    # is in the same order as UFCStats red/blue or reversed.  Use its
    # already-side-mapped red/blue odds rather than blindly assigning
    # fighter_1 odds to red and fighter_2 odds to blue.
    odds_match_order = odds_match.get("odds_match_type")

    audit_row.update({
        "matched_fighter_1": odds_match.get("matched_odds_fighter_1"),
        "matched_fighter_2": odds_match.get("matched_odds_fighter_2"),
        "odds_match_score": odds_match.get("odds_match_score"),
        "odds_min_single_score": odds_match.get("odds_min_single_score"),
        "odds_match_type": "matched",
        "odds_match_order": odds_match_order,
        "odds_fighter_1_id": odds_match.get("odds_fighter_1_id"),
        "odds_fighter_2_id": odds_match.get("odds_fighter_2_id"),
        "odds_fighter_1_score": odds_match.get("odds_fighter_1_score"),
        "odds_fighter_2_score": odds_match.get("odds_fighter_2_score"),
    })
    market_audit_rows.append(audit_row)

    row = pred.to_dict()
    row.update({
        "snapshot_run_id": SNAPSHOT_RUN_ID,
        "snapshot_timestamp": SNAPSHOT_TIMESTAMP,
        "bookmaker": odds_match.get("bookmaker"),
        "commence_time": odds_match.get("commence_time"),
        "red_american_odds": odds_match.get("red_american_odds"),
        "blue_american_odds": odds_match.get("blue_american_odds"),
        "red_decimal_odds": odds_match.get("red_decimal_odds"),
        "blue_decimal_odds": odds_match.get("blue_decimal_odds"),
        "red_implied_prob": odds_match.get("red_implied_prob"),
        "blue_implied_prob": odds_match.get("blue_implied_prob"),
        "odds_match_score": odds_match.get("odds_match_score"),
        "odds_min_single_score": odds_match.get("odds_min_single_score"),
        "odds_match_type": "matched",
        "odds_match_order": odds_match_order,
        "matched_fighter_1": odds_match.get("matched_odds_fighter_1"),
        "matched_fighter_2": odds_match.get("matched_odds_fighter_2"),
        "odds_fighter_1_id": odds_match.get("odds_fighter_1_id"),
        "odds_fighter_2_id": odds_match.get("odds_fighter_2_id"),
        "odds_fighter_1_score": odds_match.get("odds_fighter_1_score"),
        "odds_fighter_2_score": odds_match.get("odds_fighter_2_score"),
    })

    market_rows.append(row)

# ============================================================
# BUILD OUTPUTS
# ============================================================

market_df = pd.DataFrame(market_rows)
for column in MARKET_OUTPUT_COLUMNS:
    if column not in market_df.columns:
        market_df[column] = pd.Series(dtype="object")

market_audit_df = pd.DataFrame(
    market_audit_rows
)

# ============================================================
# SAVE MARKET ODDS
# ============================================================

market_df.to_parquet(
    MARKET_ODDS_OUTPUT,
    index=False,
)

print("Market odds saved:")
print(MARKET_ODDS_OUTPUT)

# ============================================================
# CLEAN MARKET SNAPSHOT ROWS
# ============================================================

market_df = market_df.dropna(
    subset=[
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_american_odds",
        "blue_american_odds",
        "bookmaker",
    ]
).copy()

market_df = market_df[
    market_df["odds_match_type"] == "matched"
].copy()

# ============================================================
# APPEND SNAPSHOT HISTORY
# ============================================================

try:
    existing_snapshots = pd.read_parquet(
        MARKET_SNAPSHOTS_OUTPUT
    )

    snapshot_df = pd.concat(
        [
            existing_snapshots,
            market_df,
        ],
        ignore_index=True,
    )

except Exception:
    snapshot_df = market_df.copy()

snapshot_df.to_parquet(
    MARKET_SNAPSHOTS_OUTPUT,
    index=False,
)

print("Market snapshots updated:")
print(MARKET_SNAPSHOTS_OUTPUT)

# ============================================================
# SAVE MATCH AUDIT
# ============================================================

market_audit_df.to_parquet(
    MARKET_MATCH_AUDIT_OUTPUT,
    index=False,
)

print("Market match audit saved:")
print(MARKET_MATCH_AUDIT_OUTPUT)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("========== MARKET UPDATE SUMMARY ==========")

print("Prediction rows:", len(predictions_df))
print("Market rows:", len(market_df))
print("Audit rows:", len(market_audit_df))

print()

print("Files saved:")
print(MARKET_ODDS_OUTPUT)
print(MARKET_SNAPSHOTS_OUTPUT)
print(MARKET_MATCH_AUDIT_OUTPUT)
