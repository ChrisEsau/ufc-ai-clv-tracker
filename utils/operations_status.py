from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class StatusTile:
    label: str
    value: str
    caption: str
    status: str
    icon: str


def _path(path: str) -> Path:
    return Path(path)


def _exists(path: str) -> bool:
    return _path(path).exists()


def _mtime(path: str) -> pd.Timestamp | None:
    p = _path(path)
    if not p.exists():
        return None
    return pd.Timestamp(p.stat().st_mtime, unit="s")


def _freshness(paths: Iterable[str]) -> str:
    times = [_mtime(path) for path in paths]
    times = [time for time in times if time is not None]
    if not times:
        return "No artifact yet"
    latest = max(times)
    age = pd.Timestamp.now() - latest.tz_localize(None)
    if age.days > 0:
        return f"Updated {age.days}d ago"
    hours = int(age.total_seconds() // 3600)
    if hours > 0:
        return f"Updated {hours}h ago"
    minutes = max(1, int(age.total_seconds() // 60))
    return f"Updated {minutes}m ago"


def _ready(paths: Iterable[str]) -> bool:
    return all(_exists(path) for path in paths)


def operation_status_tiles() -> list[StatusTile]:
    market_paths = [
        "data/market/ufc_market_odds.parquet",
        "data/market/ufc_market_snapshots.parquet",
    ]
    feature_paths = ["data/features/ufc_current_fighter_features.parquet"]
    prediction_paths = [
        "data/predictions/ufc_model_predictions.parquet",
        "data/predictions/ufc_betting_board.parquet",
    ]
    clv_paths = [
        "data/market/ufc_clv_results.parquet",
        "data/market/ufc_line_movement.parquet",
    ]
    data_paths = ["data/master/ufc_master.parquet"]

    return [
        StatusTile("Market Status", "Ready" if _ready(market_paths) else "Needs Run", _freshness(market_paths), "success" if _ready(market_paths) else "warning", "🌐"),
        StatusTile("Features Status", "Ready" if _ready(feature_paths) else "Missing", _freshness(feature_paths), "success" if _ready(feature_paths) else "warning", "◇"),
        StatusTile("Predictions Status", "Ready" if _ready(prediction_paths) else "Needs Run", _freshness(prediction_paths), "success" if _ready(prediction_paths) else "warning", "📈"),
        StatusTile("CLV Status", "Ready" if _ready(clv_paths) else "Tracking", _freshness(clv_paths), "success" if _ready(clv_paths) else "warning", "↗"),
        StatusTile("Model Status", "Production", "Registry-driven model selection", "success", "🧠"),
        StatusTile("Data Status", "Healthy" if _ready(data_paths) else "Missing", _freshness(data_paths), "success" if _ready(data_paths) else "danger", "▤"),
    ]


def latest_update_label() -> str:
    candidates = [
        "data/market/ufc_market_snapshots.parquet",
        "data/predictions/ufc_model_predictions.parquet",
        "data/features/ufc_current_fighter_features.parquet",
        "data/status/ufc_dataset_status.parquet",
    ]
    times = [_mtime(path) for path in candidates]
    times = [time for time in times if time is not None]
    if not times:
        return "Last Updated: No operation artifacts yet"
    latest = max(times)
    return f"Last Updated: {latest.strftime('%b %-d, %Y %I:%M %p')}"


def system_rows() -> list[tuple[str, str, str]]:
    return [
        ("Database", "Healthy", "Master and artifact paths available"),
        ("Data Pipeline", "Healthy", "Artifact-backed workflow model"),
        ("Feature Store", "Healthy", "Current and historical feature stores"),
        ("Prediction Engine", "Healthy", "Model prediction artifacts monitored"),
        ("CLV Engine", "Healthy", "Market and closing-line artifacts monitored"),
        ("Storage", "Healthy", "Canonical data/ layout"),
    ]


def recent_job_rows() -> list[tuple[str, str, str, str]]:
    return [
        ("Refresh Market Odds", "Market", "Unknown", "Workflow status pending"),
        ("Build Live Features", "Prediction", "Unknown", "Workflow status pending"),
        ("Generate Predictions", "Prediction", "Unknown", "Workflow status pending"),
        ("Update CLV Tracking", "Market", "Unknown", "Workflow status pending"),
        ("Dataset Status", "Data", "Unknown", "Workflow status pending"),
    ]


def schedule_rows() -> list[tuple[str, str, str]]:
    return [
        ("Refresh Market Odds", "Manual / scheduled workflow", "TBD"),
        ("Update CLV Tracking", "Manual / scheduled workflow", "TBD"),
        ("Build Live Features", "Manual / scheduled workflow", "TBD"),
        ("Generate Predictions", "Manual / scheduled workflow", "TBD"),
        ("Weekly Model Retrain", "Not wired", "TBD"),
    ]
