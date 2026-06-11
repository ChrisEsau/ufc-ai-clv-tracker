# ============================================================
# pipeline/betting/run_betting_outcomes_v2.py
# ============================================================

"""Build generic outcome-level betting opportunities.

Betting Outcomes V2 joins prediction outcomes to market outcomes using the
canonical ID-based key:

    fight_id + market_key + outcome_join_key

The output is market-type agnostic. Moneyline is the first supported market,
but the schema is designed to also support props such as KO/TKO, submission,
decision, goes distance, totals, and round props.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import (
    BETTING_OUTCOMES_AUDIT_PATH,
    BETTING_OUTCOMES_PATH,
    MARKET_OUTCOMES_PATH,
    PREDICTIONS_DIR,
    ensure_data_dirs,
)
from pipeline.common.risk_settings import RiskSettings, load_risk_settings


MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
JOIN_KEYS = ["fight_id", "market_key", "outcome_join_key"]

OUTPUT_COLUMNS = [
    "betting_run_id",
    "betting_timestamp",
    "prediction_run_id",
    "prediction_timestamp",
    "snapshot_run_id",
    "snapshot_timestamp",
    "model_id",
    "model_family",
    "algorithm",
    "prediction_type",
    "event_id",
    "event_name",
    "commence_time",
    "fight_id",
    "fight_display",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "market_key",
    "market_display",
    "bookmaker",
    "source",
    "outcome_label",
    "outcome_display",
    "outcome_fighter_id",
    "outcome_join_key",
    "outcome_side",
    "model_probability",
    "model_pick_probability",
    "is_model_pick",
    "model_pick",
    "model_confidence",
    "confidence_score",
    "confidence_pct",
    "confidence_tier",
    "confidence_data_quality",
    "confidence_feature_coverage",
    "confidence_family_coverage",
    "confidence_history_depth",
    "confidence_calibration_reliability",
    "confidence_bucket",
    "confidence_bucket_accuracy",
    "confidence_bucket_fight_count",
    "confidence_bucket_calibration_error",
    "confidence_penalty_reason",
    "american_odds",
    "decimal_odds",
    "implied_probability",
    "edge",
    "edge_pct",
    "ev",
    "ev_pct",
    "ev_dollars_at_100",
    "full_kelly_fraction",
    "fractional_kelly_fraction",
    "recommended_stake",
    "max_stake",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_filter",
    "passes_market_data_filter",
    "is_bet_candidate",
    "bet_status",
    "risk_starting_bankroll",
    "risk_kelly_fraction",
    "risk_max_stake_pct",
    "risk_min_edge",
    "risk_min_confidence",
    "risk_min_odds",
    "risk_max_odds",
]

AUDIT_COLUMNS = [
    "betting_run_id",
    "betting_timestamp",
    "prediction_rows",
    "market_rows",
    "joined_rows",
    "unique_fights_joined",
    "unique_bookmakers",
    "unique_markets",
    "missing_prediction_market_rows",
    "missing_market_prediction_rows",
    "bet_candidates",
    "filtered_by_edge",
    "filtered_by_confidence",
    "filtered_by_odds",
    "filtered_by_market_data",
    "passes_validation",
]


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("betting_%Y%m%d_%H%M%S"), now.isoformat()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _american_to_decimal(american_odds: Any) -> float | None:
    odds = pd.to_numeric(pd.Series([american_odds]), errors="coerce").iloc[0]
    if pd.isna(odds) or float(odds) == 0:
        return None
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def _kelly_fraction(probability: Any, decimal_odds: Any) -> float:
    p = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    d = pd.to_numeric(pd.Series([decimal_odds]), errors="coerce").iloc[0]
    if pd.isna(p) or pd.isna(d) or d <= 1.0:
        return 0.0
    b = float(d) - 1.0
    q = 1.0 - float(p)
    kelly = (b * float(p) - q) / b
    return float(max(kelly, 0.0))


def _market_display(market_key: Any) -> str:
    key = str(market_key or "").strip().lower()
    mapping = {
        "h2h": "Moneyline",
        "moneyline": "Moneyline",
        "method": "Method of Victory",
        "goes_distance": "Goes Distance",
        "totals": "Totals",
        "round": "Round Props",
    }
    return mapping.get(key, str(market_key or "").replace("_", " ").title())


def _bet_status(row: pd.Series) -> str:
    if not bool(row.get("passes_market_data_filter", False)):
        return "NO_MARKET_DATA"
    if not bool(row.get("passes_edge_filter", False)):
        return "FILTERED_EDGE"
    if not bool(row.get("passes_confidence_filter", False)):
        return "FILTERED_CONFIDENCE"
    if not bool(row.get("passes_odds_filter", False)):
        return "FILTERED_ODDS"
    return "BET_CANDIDATE"


def _prepare_model_predictions(model_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in JOIN_KEYS if column not in model_df.columns]
    if missing:
        raise ValueError(f"Model outcomes missing join keys: {missing}")

    out = model_df.copy()
    for column in JOIN_KEYS:
        out[column] = out[column].astype(str)
    return out


def _prepare_market_outcomes(market_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in JOIN_KEYS if column not in market_df.columns]
    if missing:
        raise ValueError(f"Market outcomes missing join keys: {missing}")

    out = market_df.copy()
    for column in JOIN_KEYS:
        out[column] = out[column].astype(str)

    # Ensure odds math fields are available even if a provider adapter changes.
    if "decimal_odds" not in out.columns:
        out["decimal_odds"] = out["american_odds"].apply(_american_to_decimal)
    if "implied_probability" not in out.columns:
        out["implied_probability"] = pd.to_numeric(out["decimal_odds"], errors="coerce").rdiv(1.0)

    return out


def _build_betting_outcomes(
    *,
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    settings: RiskSettings,
    betting_run_id: str,
    betting_timestamp: str,
) -> pd.DataFrame:
    model_df = _prepare_model_predictions(model_df)
    market_df = _prepare_market_outcomes(market_df)

    joined = model_df.merge(
        market_df,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_model", "_market"),
    )

    if joined.empty:
        return _ensure_output_columns(pd.DataFrame())

    out = pd.DataFrame()
    out["betting_run_id"] = betting_run_id
    out["betting_timestamp"] = betting_timestamp

    for column in [
        "prediction_run_id",
        "prediction_timestamp",
        "model_id",
        "model_family",
        "algorithm",
        "prediction_type",
        "event_id",
        "event_name",
        "commence_time",
        "fight_id",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
        "market_key",
        "outcome_label",
        "outcome_fighter_id",
        "outcome_join_key",
        "outcome_side",
        "model_probability",
        "model_pick_probability",
        "is_model_pick",
        "model_pick",
        "model_confidence",
        "confidence_score",
        "confidence_pct",
        "confidence_tier",
        "confidence_data_quality",
        "confidence_feature_coverage",
        "confidence_family_coverage",
        "confidence_history_depth",
        "confidence_calibration_reliability",
        "confidence_bucket",
        "confidence_bucket_accuracy",
        "confidence_bucket_fight_count",
        "confidence_bucket_calibration_error",
        "confidence_penalty_reason",
    ]:
        source_column = f"{column}_model" if f"{column}_model" in joined.columns else column
        out[column] = joined[source_column] if source_column in joined.columns else pd.NA

    for column in [
        "snapshot_run_id",
        "snapshot_timestamp",
        "bookmaker",
        "source",
        "american_odds",
        "decimal_odds",
        "implied_probability",
    ]:
        source_column = f"{column}_market" if f"{column}_market" in joined.columns else column
        out[column] = joined[source_column] if source_column in joined.columns else pd.NA

    out["fight_display"] = out["red_fighter"].astype(str) + " vs " + out["blue_fighter"].astype(str)
    out["market_display"] = out["market_key"].apply(_market_display)
    out["outcome_display"] = out["outcome_label"].astype(str)

    model_probability = pd.to_numeric(out["model_probability"], errors="coerce")
    implied_probability = pd.to_numeric(out["implied_probability"], errors="coerce")
    decimal_odds = pd.to_numeric(out["decimal_odds"], errors="coerce")
    american_odds = pd.to_numeric(out["american_odds"], errors="coerce")
    confidence_pct = pd.to_numeric(out["confidence_pct"], errors="coerce")

    out["edge"] = model_probability - implied_probability
    out["edge_pct"] = out["edge"] * 100.0
    out["ev"] = model_probability * (decimal_odds - 1.0) - (1.0 - model_probability)
    out["ev_pct"] = out["ev"] * 100.0
    out["ev_dollars_at_100"] = out["ev"] * 100.0

    out["full_kelly_fraction"] = [
        _kelly_fraction(probability, odds)
        for probability, odds in zip(model_probability, decimal_odds)
    ]
    out["fractional_kelly_fraction"] = out["full_kelly_fraction"] * float(settings.kelly_fraction)
    out["max_stake"] = float(settings.starting_bankroll) * float(settings.max_stake_pct)
    out["recommended_stake"] = (
        float(settings.starting_bankroll) * out["fractional_kelly_fraction"]
    ).clip(lower=0.0, upper=out["max_stake"])

    out["passes_market_data_filter"] = (
        model_probability.notna()
        & implied_probability.notna()
        & decimal_odds.notna()
        & american_odds.notna()
        & decimal_odds.gt(1.0)
    )
    out["passes_edge_filter"] = pd.to_numeric(out["edge"], errors="coerce").ge(float(settings.min_edge))
    out["passes_confidence_filter"] = confidence_pct.ge(float(settings.min_confidence))
    out["passes_odds_filter"] = american_odds.between(int(settings.min_odds), int(settings.max_odds), inclusive="both")
    out["is_bet_candidate"] = (
        out["passes_market_data_filter"]
        & out["passes_edge_filter"]
        & out["passes_confidence_filter"]
        & out["passes_odds_filter"]
    )
    out["bet_status"] = out.apply(_bet_status, axis=1)

    out["risk_starting_bankroll"] = float(settings.starting_bankroll)
    out["risk_kelly_fraction"] = float(settings.kelly_fraction)
    out["risk_max_stake_pct"] = float(settings.max_stake_pct)
    out["risk_min_edge"] = float(settings.min_edge)
    out["risk_min_confidence"] = float(settings.min_confidence)
    out["risk_min_odds"] = int(settings.min_odds)
    out["risk_max_odds"] = int(settings.max_odds)

    return _ensure_output_columns(out)


def _ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[OUTPUT_COLUMNS]


def _build_audit(
    *,
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    betting_df: pd.DataFrame,
    betting_run_id: str,
    betting_timestamp: str,
) -> pd.DataFrame:
    model_keys = _prepare_model_predictions(model_df)[JOIN_KEYS].drop_duplicates()
    market_keys = _prepare_market_outcomes(market_df)[JOIN_KEYS].drop_duplicates()

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


def main() -> None:
    print("=" * 80)
    print("UFC BETTING OUTCOMES V2")
    print("=" * 80)

    ensure_data_dirs()
    betting_run_id, betting_timestamp = _utc_run()

    model_df = _load_required_parquet(MODEL_OUTCOMES_PATH, "Model outcomes")
    market_df = _load_required_parquet(MARKET_OUTCOMES_PATH, "Market outcomes")
    settings = load_risk_settings()

    print("Betting run ID:", betting_run_id)
    print("Model rows:", len(model_df))
    print("Market rows:", len(market_df))
    print("Risk settings:", asdict(settings))

    betting_df = _build_betting_outcomes(
        model_df=model_df,
        market_df=market_df,
        settings=settings,
        betting_run_id=betting_run_id,
        betting_timestamp=betting_timestamp,
    )
    audit_df = _build_audit(
        model_df=model_df,
        market_df=market_df,
        betting_df=betting_df,
        betting_run_id=betting_run_id,
        betting_timestamp=betting_timestamp,
    )

    betting_df.to_parquet(BETTING_OUTCOMES_PATH, index=False)
    audit_df.to_parquet(BETTING_OUTCOMES_AUDIT_PATH, index=False)

    print()
    print("========== BETTING OUTCOMES V2 SUMMARY ==========")
    print("Joined rows:", len(betting_df))
    print("Bet candidates:", int(betting_df["is_bet_candidate"].fillna(False).sum()) if not betting_df.empty else 0)
    print("Validation passes:", bool(audit_df["passes_validation"].iloc[0]))
    print()
    print("Files saved:")
    print(BETTING_OUTCOMES_PATH)
    print(BETTING_OUTCOMES_AUDIT_PATH)


if __name__ == "__main__":
    main()
