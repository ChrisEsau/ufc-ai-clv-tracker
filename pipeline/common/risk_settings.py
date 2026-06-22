from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import BANKROLL_SETTINGS_PATH, ensure_data_dirs


ALLOWED_KELLY_FRACTIONS = (0.25, 0.50)
DEFAULT_MARKET_KEY = "moneyline"


@dataclass(frozen=True)
class MarketRiskFilter:
    """Market-specific filter thresholds for action-board qualification."""

    min_edge: float = 0.05
    min_confidence: float = 70.0
    min_odds: int = -250
    max_odds: int = 400


@dataclass(frozen=True)
class RiskSettings:
    """Single source of truth for UFC bankroll, staking, and market filters."""

    starting_bankroll: float = 10000.0
    kelly_fraction: float = 0.50
    max_stake_pct: float = 0.03
    max_event_exposure_pct: float = 0.10
    min_edge: float = 0.05
    min_confidence: float = 70.0
    min_odds: int = -250
    max_odds: int = 400
    market_filters: dict[str, MarketRiskFilter] = field(default_factory=dict)

    def filter_for_market(self, market_key: Any) -> MarketRiskFilter:
        key = normalize_market_key(market_key)
        if key in self.market_filters:
            return self.market_filters[key]
        if DEFAULT_MARKET_KEY in self.market_filters:
            return self.market_filters[DEFAULT_MARKET_KEY]
        return MarketRiskFilter(
            min_edge=self.min_edge,
            min_confidence=self.min_confidence,
            min_odds=self.min_odds,
            max_odds=self.max_odds,
        )


def normalize_market_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return DEFAULT_MARKET_KEY
    key = str(value).strip().lower()
    return key or DEFAULT_MARKET_KEY


def default_market_filters() -> dict[str, MarketRiskFilter]:
    return {
        "moneyline": MarketRiskFilter(0.05, 70.0, -250, 400),
        "goes_distance": MarketRiskFilter(0.10, 0.0, -1000, 3000),
    }


def _read_settings_frame(path: Path = BANKROLL_SETTINGS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _coerce_float(value: Any, default: float) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(default) if pd.isna(parsed) else float(parsed)


def _coerce_int(value: Any, default: int) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(default) if pd.isna(parsed) else int(parsed)


def _coerce_kelly_fraction(value: Any, default: float) -> float:
    parsed = _coerce_float(value, default)
    return min(ALLOWED_KELLY_FRACTIONS, key=lambda allowed: abs(allowed - parsed))


def _filter_from_row(row: dict[str, Any], default: MarketRiskFilter) -> MarketRiskFilter:
    return MarketRiskFilter(
        min_edge=_coerce_float(row.get("min_edge"), default.min_edge),
        min_confidence=_coerce_float(row.get("min_confidence"), default.min_confidence),
        min_odds=_coerce_int(row.get("min_odds"), default.min_odds),
        max_odds=_coerce_int(row.get("max_odds"), default.max_odds),
    )


def load_risk_settings(path: Path = BANKROLL_SETTINGS_PATH) -> RiskSettings:
    """Load persisted risk settings, falling back to locked production defaults."""

    defaults = asdict(RiskSettings())
    df = _read_settings_frame(path)
    market_filters = default_market_filters()

    if df.empty:
        values = defaults
    elif "market_key" not in df.columns:
        row = df.iloc[-1].to_dict()
        values = {key: row.get(key, default) for key, default in defaults.items() if key != "market_filters"}
        market_filters[DEFAULT_MARKET_KEY] = _filter_from_row(row, market_filters[DEFAULT_MARKET_KEY])
    else:
        work = df.copy()
        if "updated_timestamp" in work.columns:
            work["_updated_sort"] = pd.to_datetime(work["updated_timestamp"], utc=True, errors="coerce")
            work = work.sort_values("_updated_sort")
        latest = work.groupby("market_key", dropna=False).tail(1)
        first_row = latest.iloc[-1].to_dict()
        values = {key: first_row.get(key, default) for key, default in defaults.items() if key != "market_filters"}
        for _, market_row in latest.iterrows():
            key = normalize_market_key(market_row.get("market_key"))
            base = market_filters.get(key, market_filters[DEFAULT_MARKET_KEY])
            market_filters[key] = _filter_from_row(market_row.to_dict(), base)

    default_filter = market_filters.get(DEFAULT_MARKET_KEY, MarketRiskFilter())
    return RiskSettings(
        starting_bankroll=_coerce_float(values.get("starting_bankroll"), defaults["starting_bankroll"]),
        kelly_fraction=_coerce_kelly_fraction(values.get("kelly_fraction"), defaults["kelly_fraction"]),
        max_stake_pct=_coerce_float(values.get("max_stake_pct"), defaults["max_stake_pct"]),
        max_event_exposure_pct=_coerce_float(values.get("max_event_exposure_pct"), defaults["max_event_exposure_pct"]),
        min_edge=default_filter.min_edge,
        min_confidence=default_filter.min_confidence,
        min_odds=default_filter.min_odds,
        max_odds=default_filter.max_odds,
        market_filters=market_filters,
    )


def save_risk_settings(settings: RiskSettings, path: Path = BANKROLL_SETTINGS_PATH) -> None:
    """Persist risk settings to the canonical bankroll settings artifact."""

    ensure_data_dirs()
    market_filters = settings.market_filters or {
        DEFAULT_MARKET_KEY: MarketRiskFilter(settings.min_edge, settings.min_confidence, settings.min_odds, settings.max_odds)
    }
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for market_key, market_filter in sorted(market_filters.items()):
        rows.append(
            {
                "market_key": normalize_market_key(market_key),
                "starting_bankroll": settings.starting_bankroll,
                "kelly_fraction": _coerce_kelly_fraction(settings.kelly_fraction, RiskSettings().kelly_fraction),
                "max_stake_pct": settings.max_stake_pct,
                "max_event_exposure_pct": settings.max_event_exposure_pct,
                "min_edge": market_filter.min_edge,
                "min_confidence": market_filter.min_confidence,
                "min_odds": market_filter.min_odds,
                "max_odds": market_filter.max_odds,
                "updated_timestamp": timestamp,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def risk_settings_to_betting_filters(settings: RiskSettings | None = None, market_key: Any = DEFAULT_MARKET_KEY) -> dict[str, float | int]:
    settings = settings or load_risk_settings()
    market_filter = settings.filter_for_market(market_key)
    return {
        "min_edge": market_filter.min_edge,
        "min_confidence": market_filter.min_confidence,
        "min_odds": market_filter.min_odds,
        "max_odds": market_filter.max_odds,
    }


def risk_settings_to_staking_config(settings: RiskSettings | None = None) -> dict[str, float | str]:
    settings = settings or load_risk_settings()
    return {
        "method": "fractional_kelly",
        "kelly_fraction": settings.kelly_fraction,
        "max_stake_pct": settings.max_stake_pct,
        "starting_bankroll": settings.starting_bankroll,
        "max_event_exposure_pct": settings.max_event_exposure_pct,
    }
