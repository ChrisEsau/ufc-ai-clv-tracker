from __future__ import annotations

import pandas as pd

from pipeline.betting.betting_joiner import prepare_market_outcomes, prepare_model_predictions
from pipeline.betting.betting_schema import AUDIT_COLUMNS, JOIN_KEYS


def build_betting_audit(
    *,
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    betting_df: pd.DataFrame,
    betting_run_id: str,
    betting_timestamp: str,
) -> pd.DataFrame:
    """Build Betting Outcomes V2 audit summary."""

    model_keys = prepare_model_predictions(model_df)[JOIN_KEYS].drop_duplicates()
    market_keys = prepare_market_outcomes(market_df)[JOIN_KEYS].drop_duplicates()

    missing_market = model_keys.merge(market_keys, on=JOIN_KEYS, how="left", indicator=True)
    missing_market_count = int((missing_market["_merge"] == "left_only").sum())

    missing_prediction = market_keys.merge(model_keys, on=JOIN_KEYS, how="left", indicator=True)
    missing_prediction_count = int((missing_prediction["_merge"] == "left_only").sum())

    bet_candidates = int(betting_df["is_bet_candidate"].fillna(False).sum()) if "is_bet_candidate" in betting_df else 0
    joined_rows = int(len(betting_df))

    row = {
        "betting_run_id": betting_run_id,
        "betting_timestamp": betting_timestamp,
        "prediction_rows": int(len(model_df)),
        "market_rows": int(len(market_df)),
        "joined_rows": joined_rows,
        "unique_fights_joined": int(betting_df["fight_id"].nunique()) if "fight_id" in betting_df else 0,
        "unique_bookmakers": int(betting_df["bookmaker"].nunique()) if "bookmaker" in betting_df else 0,
        "unique_markets": int(betting_df["market_key"].nunique()) if "market_key" in betting_df else 0,
        "missing_prediction_market_rows": missing_market_count,
        "missing_market_prediction_rows": missing_prediction_count,
        "bet_candidates": bet_candidates,
        "filtered_by_edge": int((betting_df.get("bet_status", pd.Series(dtype=str)) == "FILTERED_EDGE").sum()),
        "filtered_by_confidence": int((betting_df.get("bet_status", pd.Series(dtype=str)) == "FILTERED_CONFIDENCE").sum()),
        "filtered_by_odds": int((betting_df.get("bet_status", pd.Series(dtype=str)) == "FILTERED_ODDS").sum()),
        "filtered_by_market_data": int((betting_df.get("bet_status", pd.Series(dtype=str)) == "NO_MARKET_DATA").sum()),
        "passes_validation": bool(joined_rows == len(market_df) and missing_prediction_count == 0),
    }

    audit = pd.DataFrame([row])
    for column in AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = pd.NA
    return audit[AUDIT_COLUMNS]
