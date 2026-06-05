"""CLV result artifact builder using bankroll ledger bets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clv.closing_lines import CLOSING_LINE_COLUMNS
from pipeline.clv.utils import (
    MONEYLINE,
    american_to_implied_prob,
    clv_pct,
    confidence_tier,
    empty_frame,
    normalize_market_type,
    odds_bucket,
)

CLV_RESULT_COLUMNS = [
    "bet_id",
    "event_name",
    "event_date",
    "fight_id",
    "fighter",
    "fighter_id",
    "opponent",
    "opponent_id",
    "market_type",
    "sportsbook",
    "odds_taken",
    "bet_implied_prob",
    "closing_odds",
    "closing_implied_prob",
    "clv_pct",
    "clv_implied_prob_delta",
    "beat_closing_line",
    "stake",
    "result",
    "profit_loss",
    "roi",
    "model_probability",
    "implied_probability",
    "edge",
    "ev",
    "confidence_tier",
    "odds_bucket",
    "placed_timestamp",
    "closing_timestamp",
    "closing_line_status",
    "source_workflow",
]


def _column(df: pd.DataFrame, column: str, default=pd.NA) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _prepare_ledger(ledger: pd.DataFrame | None) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return pd.DataFrame()
    bets = ledger.copy()
    for column in ["bet_id", "fight_id", "fighter_id", "market_type", "odds_taken"]:
        if column not in bets.columns:
            bets[column] = pd.NA
    bets = bets.dropna(subset=["bet_id", "fight_id", "fighter_id", "odds_taken"]).copy()
    if bets.empty:
        return bets
    bets["market_type"] = bets["market_type"].fillna(MONEYLINE).apply(normalize_market_type)
    if "sportsbook" not in bets.columns:
        bets["sportsbook"] = _column(bets, "bookmaker", pd.NA)
    bets["sportsbook"] = bets["sportsbook"].replace("", pd.NA)
    bets["odds_taken"] = pd.to_numeric(bets["odds_taken"], errors="coerce")
    bets["bet_implied_prob"] = pd.to_numeric(_column(bets, "implied_probability", pd.NA), errors="coerce")
    bets.loc[bets["bet_implied_prob"].isna(), "bet_implied_prob"] = bets.loc[
        bets["bet_implied_prob"].isna(), "odds_taken"
    ].apply(american_to_implied_prob)
    return bets


def _prepare_closing(closing_lines: pd.DataFrame | None) -> pd.DataFrame:
    if closing_lines is None or closing_lines.empty:
        return pd.DataFrame()
    closing = closing_lines.copy()
    for column in CLOSING_LINE_COLUMNS:
        if column not in closing.columns:
            closing[column] = pd.NA
    closing["market_type"] = closing["market_type"].fillna(MONEYLINE).apply(normalize_market_type)
    closing["closing_timestamp"] = pd.to_datetime(closing["closing_timestamp"], utc=True, errors="coerce")
    closing["closing_odds"] = pd.to_numeric(closing["closing_odds"], errors="coerce")
    closing["closing_implied_prob"] = pd.to_numeric(closing["closing_implied_prob"], errors="coerce")
    return closing


def _merge_bets_to_closing(bets: pd.DataFrame, closing: pd.DataFrame) -> pd.DataFrame:
    join_cols = ["fight_id", "fighter_id", "market_type"]
    closing_cols = [
        "fight_id",
        "fighter_id",
        "market_type",
        "sportsbook",
        "closing_timestamp",
        "closing_odds",
        "closing_implied_prob",
        "closing_line_status",
    ]
    closing = closing[closing_cols].copy()

    if "sportsbook" in bets.columns and bets["sportsbook"].notna().any():
        exact_bets = bets[bets["sportsbook"].notna()].copy()
        missing_book_bets = bets[bets["sportsbook"].isna()].copy()
        exact = exact_bets.merge(closing, how="left", on=[*join_cols, "sportsbook"], suffixes=("", "_closing"))
    else:
        exact = pd.DataFrame()
        missing_book_bets = bets.copy()

    if not missing_book_bets.empty:
        fallback_closing = (
            closing.sort_values("closing_timestamp")
            .groupby(join_cols, dropna=False)
            .tail(1)
            .copy()
        )
        fallback = missing_book_bets.drop(columns=["sportsbook"], errors="ignore").merge(
            fallback_closing, how="left", on=join_cols
        )
    else:
        fallback = pd.DataFrame()

    merged = pd.concat([exact, fallback], ignore_index=True, sort=False)
    return merged


def build_clv_results(ledger: pd.DataFrame | None, closing_lines: pd.DataFrame | None) -> pd.DataFrame:
    """Build one CLV result row per logged bankroll bet."""

    bets = _prepare_ledger(ledger)
    if bets.empty:
        return empty_frame(CLV_RESULT_COLUMNS)

    closing = _prepare_closing(closing_lines)
    if closing.empty:
        results = bets.copy()
        for column in ["closing_timestamp", "closing_odds", "closing_implied_prob", "closing_line_status"]:
            results[column] = pd.NA
    else:
        results = _merge_bets_to_closing(bets, closing)

    results["clv_pct"] = results.apply(lambda row: clv_pct(row.get("odds_taken"), row.get("closing_odds")), axis=1)
    results["clv_implied_prob_delta"] = pd.to_numeric(results.get("closing_implied_prob"), errors="coerce") - pd.to_numeric(
        results.get("bet_implied_prob"), errors="coerce"
    )
    results["beat_closing_line"] = (results["clv_pct"] >= 0).astype("boolean")
    results.loc[results["clv_pct"].isna(), "beat_closing_line"] = pd.NA
    results["stake"] = pd.to_numeric(_column(results, "stake", 0), errors="coerce").fillna(0.0)
    results["profit_loss"] = pd.to_numeric(_column(results, "profit_loss", 0), errors="coerce").fillna(0.0)
    results["roi"] = np.where(results["stake"] > 0, results["profit_loss"] / results["stake"], 0.0)
    results["confidence_tier"] = _column(results, "model_probability", pd.NA).apply(confidence_tier)
    results["odds_bucket"] = results["odds_taken"].apply(odds_bucket)

    rename_map = {
        "fighter_name": "fighter",
        "opponent_name": "opponent",
    }
    results = results.rename(columns=rename_map)
    for column in CLV_RESULT_COLUMNS:
        if column not in results.columns:
            if column == "sportsbook":
                results[column] = _column(results, "bookmaker", pd.NA)
            else:
                results[column] = pd.NA
    return results[CLV_RESULT_COLUMNS].sort_values(["placed_timestamp", "event_name", "fighter"], ascending=[False, True, True]).reset_index(drop=True)
