from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import ensure_data_dirs
from utils.bankroll_artifacts import load_bet_ledger, normalize_ledger, save_bet_ledger

RESULTS = {"Open", "Win", "Loss", "Push", "Void"}
FIELDS = {"result", "closing_odds", "clv", "notes", "odds_taken", "stake", "sportsbook", "market_type", "bet_status"}
NUMERIC = {"closing_odds", "clv", "odds_taken", "stake"}


def _num(value):
    if value is None or value == "":
        return pd.NA
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        raise ValueError(f"Invalid numeric value: {value!r}")
    return float(parsed)


def _result(value):
    result = str(value or "").strip().title()
    if result not in RESULTS:
        raise ValueError(f"Invalid result: {result}")
    return result


def _parse_payload(payload):
    row_id = str(payload.get("bet_id") or "").strip()
    if not row_id:
        raise ValueError("bet_id is required")
    updates = payload.get("updates", payload)
    if not isinstance(updates, dict):
        raise ValueError("updates must be a JSON object")
    clean = {}
    for key, value in updates.items():
        if key == "bet_id" or key not in FIELDS or value is None:
            continue
        if key == "result":
            clean[key] = _result(value)
        elif key in NUMERIC:
            clean[key] = _num(value)
        else:
            clean[key] = "" if pd.isna(value) else str(value)
    if not clean:
        raise ValueError("No editable fields supplied")
    if "stake" in clean and pd.notna(clean["stake"]) and float(clean["stake"]) < 0:
        raise ValueError("stake cannot be negative")
    if "odds_taken" in clean and pd.notna(clean["odds_taken"]) and float(clean["odds_taken"]) == 0:
        raise ValueError("odds_taken cannot be zero")
    return row_id, clean


def apply_edit(row_id, updates):
    ledger = load_bet_ledger()
    if ledger.empty or "bet_id" not in ledger.columns:
        return False
    mask = ledger["bet_id"].astype(str) == str(row_id)
    if not mask.any():
        return False
    for field, value in updates.items():
        ledger.loc[mask, field] = value
    if updates.get("result") == "Open":
        ledger.loc[mask, "settled_timestamp"] = ""
    elif "result" in updates:
        current = ledger.loc[mask, "settled_timestamp"].astype(str).str.strip()
        missing = current.eq("") | current.str.lower().isin({"nan", "nat", "none"})
        if bool(missing.any()):
            ledger.loc[mask, "settled_timestamp"] = datetime.now(timezone.utc).isoformat()
    save_bet_ledger(normalize_ledger(ledger))
    return True


def main():
    parser = argparse.ArgumentParser(description="Edit one row in the bankroll ledger")
    parser.add_argument("--edit-json", required=True)
    args = parser.parse_args()
    payload = json.loads(args.edit_json)
    if not isinstance(payload, dict):
        raise ValueError("edit-json must be a JSON object")
    ensure_data_dirs()
    row_id, updates = _parse_payload(payload)
    if not apply_edit(row_id, updates):
        raise ValueError(f"Could not find bet_id={row_id}")
    print("BANKROLL LEDGER ROW EDITED")
    print(f"Bet ID: {row_id}")
    print(f"Updated fields: {sorted(updates)}")


if __name__ == "__main__":
    main()
