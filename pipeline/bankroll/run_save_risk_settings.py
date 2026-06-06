"""Persist manually confirmed bankroll risk settings.

This runner is intended for GitHub Actions ``workflow_dispatch`` from the
Bankroll workspace.  Risk settings affect betting-board filters and bankroll
summaries, so the Streamlit UI dispatches this runner rather than only writing
settings to the local app filesystem.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from pipeline.common.paths import ensure_data_dirs
from pipeline.common.risk_settings import RiskSettings, save_risk_settings
from utils.bankroll_artifacts import load_bet_ledger, save_bet_ledger


REQUIRED_SETTING_KEYS = {
    "starting_bankroll",
    "kelly_fraction",
    "max_stake_pct",
    "max_event_exposure_pct",
    "min_edge",
    "min_confidence",
    "min_odds",
    "max_odds",
}


def _float_value(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required.")
    return float(value)


def _int_value(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"{key} is required.")
    return int(float(value))


def settings_from_payload(payload: dict[str, Any]) -> RiskSettings:
    """Build validated risk settings from a workflow JSON payload."""

    missing = sorted(key for key in REQUIRED_SETTING_KEYS if key not in payload)
    if missing:
        raise ValueError(f"Missing required risk setting keys: {', '.join(missing)}")

    settings = RiskSettings(
        starting_bankroll=_float_value(payload, "starting_bankroll"),
        kelly_fraction=_float_value(payload, "kelly_fraction"),
        max_stake_pct=_float_value(payload, "max_stake_pct"),
        max_event_exposure_pct=_float_value(payload, "max_event_exposure_pct"),
        min_edge=_float_value(payload, "min_edge"),
        min_confidence=_float_value(payload, "min_confidence"),
        min_odds=_int_value(payload, "min_odds"),
        max_odds=_int_value(payload, "max_odds"),
    )

    if settings.starting_bankroll < 0:
        raise ValueError("starting_bankroll must be non-negative.")
    if not 0 <= settings.max_stake_pct <= 1:
        raise ValueError("max_stake_pct must be a decimal between 0 and 1.")
    if not 0 <= settings.max_event_exposure_pct <= 1:
        raise ValueError("max_event_exposure_pct must be a decimal between 0 and 1.")
    if not 0 <= settings.min_confidence <= 100:
        raise ValueError("min_confidence must be between 0 and 100.")
    if settings.min_odds > settings.max_odds:
        raise ValueError("min_odds cannot be greater than max_odds.")

    return settings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist bankroll risk settings from a JSON payload.")
    parser.add_argument("--settings-json", required=True, help="JSON object containing risk setting fields.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = json.loads(args.settings_json)
    if not isinstance(payload, dict):
        raise ValueError("settings-json must decode to a JSON object.")

    ensure_data_dirs()
    settings = settings_from_payload(payload)
    save_risk_settings(settings)

    # Refresh the canonical ledger and derived bankroll artifacts under the new
    # settings so the committed snapshot reflects the same risk configuration.
    ledger = load_bet_ledger()
    save_bet_ledger(ledger)

    print("========== RISK SETTINGS SAVED ==========")
    for key, value in asdict(settings).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
