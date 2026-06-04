from pathlib import Path

import pandas as pd

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
    "Upcoming events": UPCOMING_EVENTS_PATH,
    "Upcoming fights": UPCOMING_FIGHTS_PATH,
    "Selected live card event": SELECTED_LIVE_CARD_EVENT_PATH,
    "Live card": LIVE_CARD_PATH,
}

BETTING_ARTIFACTS = {
    "Model predictions": MODEL_PREDICTIONS_PATH,
    "Market odds": MARKET_ODDS_PATH,
    "Market snapshots": MARKET_SNAPSHOTS_PATH,
    "Market match audit": MARKET_MATCH_AUDIT_PATH,
    "Betting board": BETTING_BOARD_PATH,
    "Action board": LIVE_ACTION_BOARD_PATH,
    "Watchlist": LIVE_WATCHLIST_PATH,
    "Official bets": OFFICIAL_BETS_PATH,
    "CLV results": CLV_RESULTS_PATH,
}


def format_file_size(path):
    size = Path(path).stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def artifact_status_rows(artifacts):
    rows = []
    for label, path in artifacts.items():
        path = Path(path)
        exists = path.exists()
        rows.append(
            {
                "artifact": label,
                "path": str(path),
                "exists": exists,
                "size": format_file_size(path) if exists else "missing",
                "modified_utc": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True).isoformat() if exists else None,
            }
        )
    return pd.DataFrame(rows)


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
