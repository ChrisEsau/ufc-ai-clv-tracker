"""Settle one manually tracked wager in the canonical bankroll ledger.

This runner is intended for GitHub Actions ``workflow_dispatch`` from the
Bankroll workspace.  Settlements mutate the bankroll ledger source of truth, so
they are persisted from CI rather than only writing to the local Streamlit
filesystem.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from pipeline.common.paths import ensure_data_dirs
from utils.bankroll_artifacts import settle_bet


VALID_RESULTS = {"Win", "Loss", "Push", "Void"}


def _float_or_none(value: Any):
    if value is None or value == "":
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _settlement_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    bet_id = str(payload.get("bet_id") or "").strip()
    result = str(payload.get("result") or "").strip().title()
    if not bet_id:
        raise ValueError("bet_id is required for settlement.")
    if result not in VALID_RESULTS:
        raise ValueError(f"result must be one of: {', '.join(sorted(VALID_RESULTS))}.")

    return {
        "bet_id": bet_id,
        "result": result,
        "closing_odds": _float_or_none(payload.get("closing_odds")),
        "clv": _float_or_none(payload.get("clv")),
        "notes": payload.get("notes"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Settle one wager in the canonical bankroll ledger.")
    parser.add_argument("--settlement-json", required=True, help="JSON object containing settlement fields.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.settlement_json)
    if not isinstance(payload, dict):
        raise ValueError("settlement-json must decode to a JSON object.")

    ensure_data_dirs()
    settlement = _settlement_from_payload(payload)
    ok = settle_bet(**settlement)
    if not ok:
        raise ValueError(f"Could not find bet_id={settlement['bet_id']} in the bankroll ledger.")

    print("========== MANUAL BET SETTLED ==========")
    print(f"Bet ID: {settlement['bet_id']}")
    print(f"Result: {settlement['result']}")
    if settlement.get("closing_odds") is not None:
        print(f"Closing odds: {settlement['closing_odds']}")
    if settlement.get("clv") is not None:
        print(f"CLV: {settlement['clv']}")


if __name__ == "__main__":
    main()
