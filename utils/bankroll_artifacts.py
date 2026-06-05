from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    BANKROLL_SNAPSHOTS_PATH,
    BET_LEDGER_PATH,
    OPEN_BETS_PATH,
    ensure_data_dirs,
)
from pipeline.common.risk_settings import RiskSettings, load_risk_settings


LEDGER_COLUMNS = [
    "bet_id",
    "event_name",
    "event_date",
    "fight_id",
    "fighter",
    "opponent",
    "market_type",
    "odds_taken",
    "stake",
    "result",
    "profit_loss",
    "model_probability",
    "implied_probability",
    "edge",
    "ev",
    "clv",
    "closing_odds",
    "bet_status",
    "placed_timestamp",
    "settled_timestamp",
    "source_workflow",
    "source_prediction_run_id",
    "notes",
]

OPEN_RESULTS = {"open", "pending", ""}
SETTLED_RESULTS = {"win", "loss", "push", "void"}


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _read_parquet(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return _empty_frame(columns or [])
    try:
        df = pd.read_parquet(path)
    except Exception:
        return _empty_frame(columns or [])
    if columns:
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan
    return df


def load_bet_ledger() -> pd.DataFrame:
    return _read_parquet(BET_LEDGER_PATH, LEDGER_COLUMNS)[LEDGER_COLUMNS]


def save_bet_ledger(ledger: pd.DataFrame) -> None:
    ensure_data_dirs()
    clean = ledger.copy()
    for column in LEDGER_COLUMNS:
        if column not in clean.columns:
            clean[column] = np.nan
    clean = clean[LEDGER_COLUMNS]
    clean.to_parquet(BET_LEDGER_PATH, index=False)
    derive_open_bets(clean).to_parquet(OPEN_BETS_PATH, index=False)
    build_bankroll_snapshot(clean).to_parquet(BANKROLL_SNAPSHOTS_PATH, index=False)


def _clean_result(value) -> str:
    if pd.isna(value):
        return "Open"
    text = str(value).strip().title()
    return text or "Open"


def is_open_result(value) -> bool:
    return _clean_result(value).lower() in OPEN_RESULTS


def american_profit(stake, odds, result) -> float:
    stake = pd.to_numeric(pd.Series([stake]), errors="coerce").iloc[0]
    odds = pd.to_numeric(pd.Series([odds]), errors="coerce").iloc[0]
    result_text = _clean_result(result).lower()

    if pd.isna(stake):
        stake = 0.0
    if result_text == "win" and pd.notna(odds) and odds != 0:
        return float(stake * odds / 100) if odds > 0 else float(stake * 100 / abs(odds))
    if result_text == "loss":
        return float(-stake)
    if result_text in {"push", "void"}:
        return 0.0
    return 0.0


def normalize_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    clean = ledger.copy()
    for column in LEDGER_COLUMNS:
        if column not in clean.columns:
            clean[column] = np.nan

    clean["result"] = clean["result"].apply(_clean_result)
    clean["stake"] = pd.to_numeric(clean["stake"], errors="coerce").fillna(0.0)
    clean["odds_taken"] = pd.to_numeric(clean["odds_taken"], errors="coerce")
    clean["profit_loss"] = clean.apply(
        lambda row: american_profit(row.get("stake"), row.get("odds_taken"), row.get("result"))
        if not is_open_result(row.get("result"))
        else 0.0,
        axis=1,
    )
    return clean[LEDGER_COLUMNS]


def derive_open_bets(ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    ledger = load_bet_ledger() if ledger is None else normalize_ledger(ledger)
    if ledger.empty:
        return _empty_frame(LEDGER_COLUMNS)
    return ledger[ledger["result"].apply(is_open_result)].copy()


def _stable_bet_id(row: pd.Series) -> str:
    parts = [
        row.get("event_name", ""),
        row.get("fight_id", ""),
        row.get("fighter", ""),
        row.get("market_type", "Moneyline"),
        row.get("odds_taken", ""),
        row.get("stake", ""),
        row.get("source_prediction_run_id", ""),
    ]
    raw = "|".join("" if pd.isna(part) else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def official_bets_to_ledger_rows(official_bets: pd.DataFrame, source_workflow: str = "Betting Board") -> pd.DataFrame:
    if official_bets.empty:
        return _empty_frame(LEDGER_COLUMNS)

    rows = pd.DataFrame(index=official_bets.index)
    rows["event_name"] = official_bets.get("event_name", "")
    rows["event_date"] = official_bets.get("event_date", official_bets.get("date", ""))
    rows["fight_id"] = official_bets.get("fight_id", "")
    rows["fighter"] = official_bets.get("best_side", official_bets.get("fighter", ""))
    rows["opponent"] = np.where(
        official_bets.get("best_side", "") == official_bets.get("red_fighter", ""),
        official_bets.get("blue_fighter", ""),
        official_bets.get("red_fighter", ""),
    )
    rows["market_type"] = "Moneyline"
    rows["odds_taken"] = official_bets.get("best_american_odds", np.nan)
    rows["stake"] = official_bets.get(
        "scenario_recommended_stake",
        official_bets.get("recommended_stake", 0.0),
    )
    rows["result"] = "Open"
    rows["profit_loss"] = 0.0
    rows["model_probability"] = official_bets.get("best_prob", np.nan)
    rows["implied_probability"] = official_bets.get("best_implied_prob", np.nan)
    rows["edge"] = official_bets.get("best_edge", np.nan)
    rows["ev"] = official_bets.get("best_ev", np.nan)
    rows["clv"] = np.nan
    rows["closing_odds"] = np.nan
    rows["bet_status"] = official_bets.get("scenario_bet_status", official_bets.get("bet_status", "OFFICIAL BET"))
    rows["placed_timestamp"] = datetime.now(timezone.utc).isoformat()
    rows["settled_timestamp"] = ""
    rows["source_workflow"] = source_workflow
    rows["source_prediction_run_id"] = official_bets.get(
        "decision_run_id",
        official_bets.get("prediction_run_id", ""),
    )
    rows["notes"] = official_bets.get("scenario_bet_reason", official_bets.get("bet_reason", ""))
    rows["bet_id"] = rows.apply(_stable_bet_id, axis=1)
    return normalize_ledger(rows)


def append_official_bets(official_bets: pd.DataFrame, source_workflow: str = "Betting Board") -> tuple[int, int]:
    ledger = load_bet_ledger()
    new_rows = official_bets_to_ledger_rows(official_bets, source_workflow=source_workflow)
    if new_rows.empty:
        return 0, 0

    existing_ids = set(ledger["bet_id"].dropna().astype(str)) if not ledger.empty else set()
    deduped = new_rows[~new_rows["bet_id"].astype(str).isin(existing_ids)].copy()
    combined = pd.concat([ledger, deduped], ignore_index=True)
    save_bet_ledger(normalize_ledger(combined))
    return len(deduped), len(new_rows) - len(deduped)


def settle_bet(bet_id: str, result: str, closing_odds=None, clv=None, notes: str | None = None) -> bool:
    ledger = load_bet_ledger()
    if ledger.empty or "bet_id" not in ledger.columns:
        return False
    mask = ledger["bet_id"].astype(str) == str(bet_id)
    if not mask.any():
        return False

    clean_result = _clean_result(result)
    ledger.loc[mask, "result"] = clean_result
    ledger.loc[mask, "settled_timestamp"] = datetime.now(timezone.utc).isoformat()
    if closing_odds is not None:
        ledger.loc[mask, "closing_odds"] = closing_odds
    if clv is not None:
        ledger.loc[mask, "clv"] = clv
    if notes is not None:
        ledger.loc[mask, "notes"] = notes
    save_bet_ledger(normalize_ledger(ledger))
    return True


def bankroll_summary(ledger: pd.DataFrame | None = None, settings: RiskSettings | None = None) -> dict:
    settings = settings or load_risk_settings()
    ledger = load_bet_ledger() if ledger is None else normalize_ledger(ledger)
    open_bets = derive_open_bets(ledger)

    realized_profit = float(pd.to_numeric(ledger.get("profit_loss", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    open_risk = float(pd.to_numeric(open_bets.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    current_bankroll = float(settings.starting_bankroll + realized_profit)
    available_bankroll = float(current_bankroll - open_risk)
    total_staked = float(pd.to_numeric(ledger.get("stake", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    roi = float(realized_profit / total_staked) if total_staked else 0.0

    return {
        "starting_bankroll": float(settings.starting_bankroll),
        "current_bankroll": current_bankroll,
        "available_bankroll": available_bankroll,
        "open_risk": open_risk,
        "total_profit": realized_profit,
        "roi": roi,
        "open_bets": int(len(open_bets)),
        "total_bets": int(len(ledger)),
    }


def build_bankroll_snapshot(ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    summary = bankroll_summary(ledger=ledger)
    summary["snapshot_timestamp"] = datetime.now(timezone.utc).isoformat()
    return pd.DataFrame([summary])


def exposure_by_event(open_bets: pd.DataFrame | None = None) -> pd.DataFrame:
    open_bets = derive_open_bets() if open_bets is None else open_bets
    if open_bets.empty or "event_name" not in open_bets.columns:
        return pd.DataFrame(columns=["event_name", "open_bets", "open_risk", "potential_profit"])

    work = open_bets.copy()
    work["stake"] = pd.to_numeric(work["stake"], errors="coerce").fillna(0)
    work["potential_profit"] = work.apply(
        lambda row: max(american_profit(row.get("stake"), row.get("odds_taken"), "Win"), 0.0),
        axis=1,
    )
    return (
        work.groupby("event_name", dropna=False)
        .agg(open_bets=("bet_id", "count"), open_risk=("stake", "sum"), potential_profit=("potential_profit", "sum"))
        .reset_index()
        .sort_values("open_risk", ascending=False)
    )


def performance_by_event(ledger: pd.DataFrame | None = None) -> pd.DataFrame:
    ledger = load_bet_ledger() if ledger is None else normalize_ledger(ledger)
    if ledger.empty or "event_name" not in ledger.columns:
        return pd.DataFrame(columns=["event_name", "bets", "profit_loss", "stake", "roi"])

    settled = ledger[~ledger["result"].apply(is_open_result)].copy()
    if settled.empty:
        return pd.DataFrame(columns=["event_name", "bets", "profit_loss", "stake", "roi"])
    settled["stake"] = pd.to_numeric(settled["stake"], errors="coerce").fillna(0)
    settled["profit_loss"] = pd.to_numeric(settled["profit_loss"], errors="coerce").fillna(0)
    grouped = (
        settled.groupby("event_name", dropna=False)
        .agg(bets=("bet_id", "count"), profit_loss=("profit_loss", "sum"), stake=("stake", "sum"))
        .reset_index()
    )
    grouped["roi"] = np.where(grouped["stake"] > 0, grouped["profit_loss"] / grouped["stake"], 0.0)
    return grouped.sort_values("profit_loss", ascending=False)
