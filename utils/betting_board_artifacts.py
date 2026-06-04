from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pipeline.common.paths import (
    BETTING_BOARD_PATH,
    CLV_RESULTS_PATH,
    LIVE_ACTION_BOARD_PATH,
    LIVE_CARD_PATH,
    LIVE_WATCHLIST_PATH,
    MARKET_MATCH_AUDIT_PATH,
    MARKET_ODDS_PATH,
    MARKET_SNAPSHOTS_PATH,
    MODEL_PREDICTIONS_PATH,
    OFFICIAL_BETS_PATH,
    SELECTED_LIVE_CARD_EVENT_PATH,
    UPCOMING_EVENTS_PATH,
    UPCOMING_FIGHTS_PATH,
)


UPCOMING_ARTIFACTS = {
    "Upcoming events": {
        "path": UPCOMING_EVENTS_PATH,
        "required_for": "Event selection",
    },
    "Upcoming fights": {
        "path": UPCOMING_FIGHTS_PATH,
        "required_for": "Event preview",
    },
    "Selected live card event": {
        "path": SELECTED_LIVE_CARD_EVENT_PATH,
        "required_for": "Selected-event diagnostics",
    },
    "Live card": {
        "path": LIVE_CARD_PATH,
        "required_for": "Prediction input",
    },
}

BETTING_ARTIFACTS = {
    "Model predictions": {
        "path": MODEL_PREDICTIONS_PATH,
        "required_for": "Betting Board output",
    },
    "Market odds": {
        "path": MARKET_ODDS_PATH,
        "required_for": "Betting Board output",
    },
    "Market snapshots": {
        "path": MARKET_SNAPSHOTS_PATH,
        "required_for": "Market diagnostics",
    },
    "Market match audit": {
        "path": MARKET_MATCH_AUDIT_PATH,
        "required_for": "Odds matching diagnostics",
    },
    "Betting board": {
        "path": BETTING_BOARD_PATH,
        "required_for": "Betting Board output",
    },
    "Action board": {
        "path": LIVE_ACTION_BOARD_PATH,
        "required_for": "Legacy action-board output",
        "optional": True,
    },
    "Watchlist": {
        "path": LIVE_WATCHLIST_PATH,
        "required_for": "Watchlist output",
    },
    "Official bets": {
        "path": OFFICIAL_BETS_PATH,
        "required_for": "Official-bets output",
    },
    "CLV results": {
        "path": CLV_RESULTS_PATH,
        "required_for": "CLV output",
        "optional": True,
    },
}


def format_file_size(path):
    size = Path(path).stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def parquet_row_count(path):
    """Return a lightweight row count from parquet metadata when possible."""

    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def artifact_status_rows(artifacts):
    rows = []
    now = pd.Timestamp.utcnow()

    for label, metadata in artifacts.items():
        path = Path(metadata["path"])
        exists = path.exists()
        optional = bool(metadata.get("optional", False))
        row_count = parquet_row_count(path) if exists and path.suffix == ".parquet" else None
        modified_utc = pd.to_datetime(path.stat().st_mtime, unit="s", utc=True) if exists else None

        if not exists:
            health = "optional_missing" if optional else "missing"
        elif row_count == 0 and not optional:
            health = "empty"
        else:
            health = "ready"

        rows.append(
            {
                "artifact": label,
                "required_for": metadata.get("required_for", ""),
                "optional": optional,
                "path": str(path),
                "exists": exists,
                "health": health,
                "rows": row_count,
                "size": format_file_size(path) if exists else "missing",
                "modified_utc": modified_utc.isoformat() if modified_utc is not None else None,
                "age_hours": round((now - modified_utc).total_seconds() / 3600, 2) if modified_utc is not None else None,
            }
        )
    return pd.DataFrame(rows)


def artifact_readiness_summary(status):
    if status.empty:
        return {
            "required_ready": 0,
            "required_total": 0,
            "missing_required": 0,
            "empty_required": 0,
            "optional_missing": 0,
            "ready_to_review": False,
        }

    required = status[~status["optional"]]
    return {
        "required_ready": int((required["health"] == "ready").sum()),
        "required_total": int(len(required)),
        "missing_required": int((required["health"] == "missing").sum()),
        "empty_required": int((required["health"] == "empty").sum()),
        "optional_missing": int((status["health"] == "optional_missing").sum()),
        "ready_to_review": bool((required["health"] == "ready").all()) if len(required) else False,
    }


def load_parquet_with_error(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), f"Missing artifact: {path}"
    try:
        return pd.read_parquet(path), None
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {path}: {exc}"


def get_upcoming_artifact_status():
    return artifact_status_rows(UPCOMING_ARTIFACTS)


def get_betting_artifact_status():
    return artifact_status_rows(BETTING_ARTIFACTS)


def load_upcoming_events():
    return load_parquet_with_error(UPCOMING_EVENTS_PATH)


def load_upcoming_fights():
    return load_parquet_with_error(UPCOMING_FIGHTS_PATH)


def load_selected_live_card_event():
    return load_parquet_with_error(SELECTED_LIVE_CARD_EVENT_PATH)


def event_label(event_row):
    name = event_row.get("ufcstats_event_name") or event_row.get("event_name") or "Unknown event"
    date = event_row.get("ufcstats_event_date") or event_row.get("event_date") or "date unknown"
    location = event_row.get("ufcstats_event_location") or event_row.get("event_location") or "location unknown"
    event_id = event_row.get("ufcstats_event_id") or event_row.get("event_id") or "missing-id"
    return f"{date} — {name} — {location} ({event_id})"
