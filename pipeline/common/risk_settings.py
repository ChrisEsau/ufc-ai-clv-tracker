from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import BANKROLL_SETTINGS_PATH, ensure_data_dirs


@dataclass(frozen=True)
class RiskSettings:
    """Single source of truth for UFC betting risk and staking settings."""

    starting_bankroll: float = 10000.0
    kelly_fraction: float = 0.50
    max_stake_pct: float = 0.03
    max_event_exposure_pct: float = 0.10
    min_edge: float = 0.05
    min_confidence: float = 70.0
    min_odds: int = -250
    max_odds: int = 400


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


def load_risk_settings(path: Path = BANKROLL_SETTINGS_PATH) -> RiskSettings:
    """Load persisted risk settings, falling back to locked production defaults."""

    defaults = asdict(RiskSettings())
    df = _read_settings_frame(path)
    if df.empty:
        values = defaults
    else:
        row = df.iloc[-1].to_dict()
        values = {key: row.get(key, default) for key, default in defaults.items()}

    return RiskSettings(
        starting_bankroll=_coerce_float(values.get("starting_bankroll"), defaults["starting_bankroll"]),
        kelly_fraction=_coerce_float(values.get("kelly_fraction"), defaults["kelly_fraction"]),
        max_stake_pct=_coerce_float(values.get("max_stake_pct"), defaults["max_stake_pct"]),
        max_event_exposure_pct=_coerce_float(
            values.get("max_event_exposure_pct"),
            defaults["max_event_exposure_pct"],
        ),
        min_edge=_coerce_float(values.get("min_edge"), defaults["min_edge"]),
        min_confidence=_coerce_float(values.get("min_confidence"), defaults["min_confidence"]),
        min_odds=_coerce_int(values.get("min_odds"), defaults["min_odds"]),
        max_odds=_coerce_int(values.get("max_odds"), defaults["max_odds"]),
    )


def save_risk_settings(settings: RiskSettings, path: Path = BANKROLL_SETTINGS_PATH) -> None:
    """Persist risk settings to the canonical bankroll settings artifact."""

    ensure_data_dirs()
    row = asdict(settings)
    row["updated_timestamp"] = datetime.now(timezone.utc).isoformat()
    pd.DataFrame([row]).to_parquet(path, index=False)


def risk_settings_to_betting_filters(settings: RiskSettings | None = None) -> dict[str, float | int]:
    settings = settings or load_risk_settings()
    return {
        "min_edge": settings.min_edge,
        "min_confidence": settings.min_confidence,
        "min_odds": settings.min_odds,
        "max_odds": settings.max_odds,
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

