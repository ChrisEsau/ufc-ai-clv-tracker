"""Append one manually confirmed wager to the bankroll ledger.

This runner is intended for GitHub Actions workflow_dispatch from the Bankroll
workspace.  The bankroll ledger is the source of truth, so manual UI entries are
persisted by committing the ledger and its derived bankroll artifacts from CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import ensure_data_dirs
from utils.bankroll_artifacts import load_bet_ledger, normalize_ledger, save_bet_ledger


def _manual_bet_id(row: dict) -> str:
    raw = "|".join(
        str(row.get(key, ""))
        for key in [
            "event_name",
            "event_id",
            "fight_id",
            "fighter",
            "fighter_id",
            "odds_taken",
            "stake",
            "placed_timestamp",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _float_value(value, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(parsed) else float(parsed)


def _row_from_payload(payload: dict) -> dict:
    placed_timestamp = payload.get("placed_timestamp") or datetime.now(timezone.utc).isoformat()
    row = {
        "event_name": payload.get("event_name", ""),
        "event_id": payload.get("event_id", ""),
        "event_date": payload.get("event_date", ""),
        "fight_id": payload.get("fight_id", ""),
        "fighter": payload.get("fighter", ""),
        "fighter_id": payload.get("fighter_id", ""),
        "opponent": payload.get("opponent", ""),
        "opponent_id": payload.get("opponent_id", ""),
        "market_type": payload.get("market_type") or "Moneyline",
        "sportsbook": payload.get("sportsbook", ""),
        "odds_taken": _float_value(payload.get("odds_taken")),
        "stake": _float_value(payload.get("stake")),
        "result": "Open",
        "profit_loss": 0.0,
        "model_probability": _float_value(payload.get("model_probability"), default=float("nan")),
        "implied_probability": _float_value(payload.get("implied_probability"), default=float("nan")),
        "edge": _float_value(payload.get("edge"), default=float("nan")),
        "ev": _float_value(payload.get("ev"), default=float("nan")),
        "clv": pd.NA,
        "closing_odds": pd.NA,
        "bet_status": "MANUAL",
        "placed_timestamp": placed_timestamp,
        "settled_timestamp": "",
        "source_workflow": "Bankroll Manual Entry",
        "source_prediction_run_id": payload.get("source_prediction_run_id", ""),
        "notes": payload.get("notes") or "Manual bankroll entry",
    }
    row["bet_id"] = payload.get("bet_id") or _manual_bet_id(row)
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append one manual wager to the bankroll ledger.")
    parser.add_argument("--bet-json", required=True, help="JSON object containing manual bet fields.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.bet_json)
    ensure_data_dirs()

    row = _row_from_payload(payload)
    if not row["event_name"] or not row["fight_id"] or not row["fighter"]:
        raise ValueError("event_name, fight_id, and fighter are required for manual ledger append.")
    if row["stake"] <= 0 or row["odds_taken"] == 0:
        raise ValueError("stake must be positive and odds_taken must be non-zero.")

    ledger = load_bet_ledger()
    existing_ids = set(ledger["bet_id"].dropna().astype(str)) if not ledger.empty else set()
    if str(row["bet_id"]) in existing_ids:
        print(f"Manual bet already exists in ledger: {row['bet_id']}")
        save_bet_ledger(normalize_ledger(ledger))
        return

    updated = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    save_bet_ledger(normalize_ledger(updated))
    print("========== MANUAL BET APPENDED ==========")
    print(f"Bet ID: {row['bet_id']}")
    print(f"Event: {row['event_name']}")
    print(f"Fighter: {row['fighter']}")
    print(f"Stake: {row['stake']}")
    print(f"Odds: {row['odds_taken']}")
    print(f"Ledger rows: {len(updated)}")


if __name__ == "__main__":
    main()
