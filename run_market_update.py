# ============================================================
# run_market_update.py
# Dedicated market / odds update runner
# ============================================================

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline_config import *
from ufc_pipeline_utils import *
from ufc_odds_utils import *

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = "."

MODEL_PREDICTIONS_PATH = (
    f"{BASE_PATH}/ufc_model_predictions.parquet"
)

MARKET_ODDS_OUTPUT = (
    f"{BASE_PATH}/ufc_market_odds.parquet"
)

MARKET_SNAPSHOTS_OUTPUT = (
    f"{BASE_PATH}/ufc_market_snapshots.parquet"
)

MARKET_MATCH_AUDIT_OUTPUT = (
    f"{BASE_PATH}/ufc_market_match_audit.parquet"
)

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

for _, pred in predictions_df.iterrows():

    red_norm = normalize_name(
        pred["red_fighter"]
    )

    blue_norm = normalize_name(
        pred["blue_fighter"]
    )

    best_match = None
    best_score = -1

    for _, odds in odds_df.iterrows():

        f1 = odds["fighter_1_norm"]
        f2 = odds["fighter_2_norm"]

        score_a = (
            token_set_score(red_norm, f1)
            +
            token_set_score(blue_norm, f2)
        ) / 2

        score_b = (
            token_set_score(red_norm, f2)
            +
            token_set_score(blue_norm, f1)
        ) / 2

        score = max(score_a, score_b)

        if score > best_score:
            best_score = score
            best_match = odds

    match_type = (
        "matched"
        if best_score >= 0.90
        else "low_confidence"
    )

    market_audit_rows.append({
        "snapshot_run_id": SNAPSHOT_RUN_ID,
        "snapshot_timestamp": SNAPSHOT_TIMESTAMP,

        "event_name": pred["event_name"],

        "red_fighter": pred["red_fighter"],
        "blue_fighter": pred["blue_fighter"],

        "matched_fighter_1": (
            None if best_match is None
            else best_match["fighter_1"]
        ),

        "matched_fighter_2": (
            None if best_match is None
            else best_match["fighter_2"]
        ),

        "odds_match_score": best_score,
        "odds_match_type": match_type,
    })

    if best_match is None:
        continue

    row = pred.to_dict()

    row["snapshot_run_id"] = SNAPSHOT_RUN_ID
    row["snapshot_timestamp"] = SNAPSHOT_TIMESTAMP

    row["bookmaker"] = best_match["bookmaker"]

    row["red_american_odds"] = (
        best_match["fighter_1_american_odds"]
    )

    row["blue_american_odds"] = (
        best_match["fighter_2_american_odds"]
    )

    row["red_decimal_odds"] = (
        best_match["fighter_1_decimal_odds"]
    )

    row["blue_decimal_odds"] = (
        best_match["fighter_2_decimal_odds"]
    )

    row["red_implied_prob"] = (
        best_match["fighter_1_implied_prob"]
    )

    row["blue_implied_prob"] = (
        best_match["fighter_2_implied_prob"]
    )

    row["odds_match_score"] = best_score
    row["odds_match_type"] = match_type

    market_rows.append(row)

# ============================================================
# BUILD OUTPUTS
# ============================================================

market_df = pd.DataFrame(market_rows)

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