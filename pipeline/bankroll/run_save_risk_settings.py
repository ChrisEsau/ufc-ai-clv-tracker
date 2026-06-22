"""Persist manually confirmed bankroll risk settings.

This runner is intended for GitHub Actions ``workflow_dispatch`` from the
Bankroll workspace. Risk settings affect betting-board filters and bankroll
summaries, so the Streamlit UI dispatches this runner rather than only writing
settings to the local app filesystem.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from pipeline.common.paths import ensure_data_dirs
from pipeline.common.risk_settings import (
    DEFAULT_MARKET_KEY,
    MarketRiskFilter,
    RiskSettings,
    default_market_filters,
    normalize_market_key,
    save_risk_settings,
)
from utils.bankroll_artifacts import load_bet_ledger, save_bet_ledger


REQUIRED_SETTING_KEYS = {
    "starting_bankroll",
    "kelly_fraction",
    "max_stake_pct",
    "max_event_exposure_pct",
}
FILTER_KEYS = {"min_edge", "min_confidence", "min_odds", "max_odds"}


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


def _filter_from_payload(payload: dict[str, Any], fallback: MarketRiskFilter) -> MarketRiskFilter:
    return MarketRiskFilter(
        min_edge=float(payload.get("min_edge", fallback.min_edge)),
        min_confidence=float(payload.get("min_confidence", fallback.min_confidence)),
        min_odds=int(float(payload.get("min_odds", fallback.min_odds))),
        max_odds=int(float(payload.get("max_odds", fallback.max_odds))),
    )


def _validate_filter(market_key: str, market_filter: MarketRiskFilter) -> None:
    if not 0 <= market_filter.min_confidence <= 100:
        raise ValueError(f"{market_key} min_confidence must be between 0 and 100.")
    if market_filter.min_odds > market_filter.max_odds:
        raise ValueError(f"{market_key} min_odds cannot be greater than max_odds.")


def _market_filters_from_payload(payload: dict[str, Any]) -> dict[str, MarketRiskFilter]:
    defaults = default_market_filters()
    raw_markets = payload.get("market_filters") or payload.get("markets")
    if isinstance(raw_markets, dict) and raw_markets:
        filters = defaults.copy()
        for market_key, raw_filter in raw_markets.items():
            if not isinstance(raw_filter, dict):
                raise ValueError(f"Market filter for {market_key} must be an object.")
            normalized_key = normalize_market_key(market_key)
            filters[normalized_key] = _filter_from_payload(raw_filter, filters.get(normalized_key, filters[DEFAULT_MARKET_KEY]))
        return filters

    # Backward-compatible single-filter payload.
    fallback = defaults[DEFAULT_MARKET_KEY]
    filters = defaults.copy()
    filters[DEFAULT_MARKET_KEY] = _filter_from_payload(payload, fallback)
    return filters


def settings_from_payload(payload: dict[str, Any]) -> RiskSettings:
    """Build validated risk settings from a workflow JSON payload."""

    missing = sorted(key for key in REQUIRED_SETTING_KEYS if key not in payload)
    if missing:
        raise ValueError(f"Missing required risk setting keys: {', '.join(missing)}")

    market_filters = _market_filters_from_payload(payload)
    for market_key, market_filter in market_filters.items():
        _validate_filter(market_key, market_filter)

    moneyline_filter = market_filters.get(DEFAULT_MARKET_KEY, MarketRiskFilter())
    settings = RiskSettings(
        starting_bankroll=_float_value(payload, "starting_bankroll"),
        kelly_fraction=_float_value(payload, "kelly_fraction"),
        max_stake_pct=_float_value(payload, "max_stake_pct"),
        max_event_exposure_pct=_float_value(payload, "max_event_exposure_pct"),
        min_edge=moneyline_filter.min_edge,
        min_confidence=moneyline_filter.min_confidence,
        min_odds=moneyline_filter.min_odds,
        max_odds=moneyline_filter.max_odds,
        market_filters=market_filters,
    )

    if settings.starting_bankroll < 0:
        raise ValueError("starting_bankroll must be non-negative.")
    if not 0 <= settings.max_stake_pct <= 1:
        raise ValueError("max_stake_pct must be a decimal between 0 and 1.")
    if not 0 <= settings.max_event_exposure_pct <= 1:
        raise ValueError("max_event_exposure_pct must be a decimal between 0 and 1.")

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
