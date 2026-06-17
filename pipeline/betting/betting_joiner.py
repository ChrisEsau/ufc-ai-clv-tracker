from __future__ import annotations

import pandas as pd

from pipeline.common.outcome_join import build_outcome_join_key
from pipeline.common.risk_settings import RiskSettings
from pipeline.betting.betting_math import (
    american_to_decimal,
    bet_status,
    kelly_fraction,
    market_display,
)
from pipeline.betting.betting_schema import JOIN_KEYS, OUTPUT_COLUMNS


def backfill_outcome_join_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add outcome_join_key for artifacts that predate the V2 join contract."""

    out = df.copy()
    if "outcome_join_key" not in out.columns:
        out["outcome_join_key"] = pd.NA

    missing_mask = out["outcome_join_key"].isna() | out["outcome_join_key"].astype(str).str.strip().isin(
        {"", "nan", "None", "<NA>"}
    )

    if missing_mask.any():
        out.loc[missing_mask, "outcome_join_key"] = out.loc[missing_mask].apply(
            lambda row: build_outcome_join_key(
                market_key=row.get("market_key"),
                outcome_label=row.get("outcome_label"),
                outcome_fighter_id=row.get("outcome_fighter_id"),
                outcome_key=row.get("outcome_key"),
                side=row.get("side", row.get("outcome_side")),
                line=row.get("line"),
            ),
            axis=1,
        )

    return out


def prepare_model_predictions(model_df: pd.DataFrame) -> pd.DataFrame:
    out = backfill_outcome_join_key(model_df)
    missing = [column for column in JOIN_KEYS if column not in out.columns]
    if missing:
        raise ValueError(f"Model outcomes missing join keys: {missing}")

    for column in JOIN_KEYS:
        out[column] = out[column].astype(str)
    return out


def prepare_market_outcomes(market_df: pd.DataFrame) -> pd.DataFrame:
    out = backfill_outcome_join_key(market_df)
    missing = [column for column in JOIN_KEYS if column not in out.columns]
    if missing:
        raise ValueError(f"Market outcomes missing join keys: {missing}")

    for column in JOIN_KEYS:
        out[column] = out[column].astype(str)

    if "decimal_odds" not in out.columns:
        out["decimal_odds"] = out["american_odds"].apply(american_to_decimal)
    if "implied_probability" not in out.columns:
        out["implied_probability"] = pd.to_numeric(out["decimal_odds"], errors="coerce").rdiv(1.0)

    return out


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[OUTPUT_COLUMNS]


def build_betting_outcomes(
    *,
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    settings: RiskSettings,
    betting_run_id: str,
    betting_timestamp: str,
) -> pd.DataFrame:
    model_df = prepare_model_predictions(model_df)
    market_df = prepare_market_outcomes(market_df)

    joined = model_df.merge(
        market_df,
        on=JOIN_KEYS,
        how="inner",
        suffixes=("_model", "_market"),
    )

    if joined.empty:
        return ensure_output_columns(pd.DataFrame())

    out = pd.DataFrame()
    out["betting_run_id"] = betting_run_id
    out["betting_timestamp"] = betting_timestamp

    for column in [
        "prediction_run_id",
        "prediction_timestamp",
        "model_id",
        "model_family",
        "model_registry_status",
        "model_outcomes_path",
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
    out["market_display"] = out["market_key"].apply(market_display)
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
        kelly_fraction(probability, odds)
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
    out["bet_status"] = out.apply(bet_status, axis=1)

    out["risk_starting_bankroll"] = float(settings.starting_bankroll)
    out["risk_kelly_fraction"] = float(settings.kelly_fraction)
    out["risk_max_stake_pct"] = float(settings.max_stake_pct)
    out["risk_min_edge"] = float(settings.min_edge)
    out["risk_min_confidence"] = float(settings.min_confidence)
    out["risk_min_odds"] = int(settings.min_odds)
    out["risk_max_odds"] = int(settings.max_odds)

    return ensure_output_columns(out)
