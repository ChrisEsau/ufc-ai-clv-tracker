from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pipeline.common.paths import (
    BET_LEDGER_PATH,
    CLOSING_LINES_PATH,
    CLV_RESULTS_PATH,
    LINE_MOVEMENT_PATH,
    CLV_MARKET_NORMALIZATION_AUDIT_PATH,
    MARKET_MATCH_AUDIT_PATH,
    MARKET_ODDS_PATH,
    MARKET_SNAPSHOTS_PATH,
    NORMALIZED_MARKET_SNAPSHOTS_PATH,
    OFFICIAL_BETS_PATH,
)


CLV_ARTIFACTS = {
    "Market odds": {
        "path": MARKET_ODDS_PATH,
        "required_for": "Current market prices",
    },
    "Market snapshots": {
        "path": MARKET_SNAPSHOTS_PATH,
        "required_for": "Line movement history",
    },
    "Normalized market snapshots": {
        "path": NORMALIZED_MARKET_SNAPSHOTS_PATH,
        "required_for": "Side-level CLV calculations",
    },
    "CLV normalization audit": {
        "path": CLV_MARKET_NORMALIZATION_AUDIT_PATH,
        "required_for": "CLV input validation",
    },
    "Market match audit": {
        "path": MARKET_MATCH_AUDIT_PATH,
        "required_for": "Odds matching diagnostics",
    },
    "Closing lines": {
        "path": CLOSING_LINES_PATH,
        "required_for": "Closing-line capture",
    },
    "Line movement": {
        "path": LINE_MOVEMENT_PATH,
        "required_for": "Line movement analytics",
    },
    "CLV results": {
        "path": CLV_RESULTS_PATH,
        "required_for": "CLV reporting",
    },
    "Official bets": {
        "path": OFFICIAL_BETS_PATH,
        "required_for": "Recommended official bets",
        "optional": True,
    },
    "Bet ledger": {
        "path": BET_LEDGER_PATH,
        "required_for": "Placed wager source of truth",
        "optional": True,
    },
}


FRESH_HOURS = 6
AGING_HOURS = 24


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


def freshness_status(age_hours):
    if age_hours is None:
        return "missing"
    if age_hours <= FRESH_HOURS:
        return "fresh"
    if age_hours <= AGING_HOURS:
        return "aging"
    return "stale"


def artifact_status_rows(artifacts=CLV_ARTIFACTS):
    rows = []
    now = pd.Timestamp.utcnow()

    for label, metadata in artifacts.items():
        path = Path(metadata["path"])
        exists = path.exists()
        optional = bool(metadata.get("optional", False))
        row_count = parquet_row_count(path) if exists and path.suffix == ".parquet" else None
        modified_utc = pd.to_datetime(path.stat().st_mtime, unit="s", utc=True) if exists else None
        age_hours = round((now - modified_utc).total_seconds() / 3600, 2) if modified_utc is not None else None

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
                "freshness": freshness_status(age_hours),
                "rows": row_count,
                "size": format_file_size(path) if exists else "missing",
                "modified_utc": modified_utc.isoformat() if modified_utc is not None else None,
                "age_hours": age_hours,
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
            "stale_ready_artifacts": 0,
            "ready_to_review": False,
        }

    required = status[~status["optional"]]
    ready_required = required[required["health"] == "ready"]

    return {
        "required_ready": int(len(ready_required)),
        "required_total": int(len(required)),
        "missing_required": int((required["health"] == "missing").sum()),
        "empty_required": int((required["health"] == "empty").sum()),
        "optional_missing": int((status["health"] == "optional_missing").sum()),
        "stale_ready_artifacts": int((ready_required["freshness"] == "stale").sum()),
        "ready_to_review": bool((required["health"] == "ready").all()) if len(required) else False,
    }


def _safe_nunique(df, column):
    if column not in df.columns:
        return 0
    return int(df[column].dropna().nunique())


def _missing_count(df, column):
    if column not in df.columns:
        return None
    return int(df[column].isna().sum())


def market_snapshot_coverage_summary(snapshots_df):
    """Summarize market snapshot freshness and fight/card coverage."""

    if snapshots_df is None or snapshots_df.empty:
        return {
            "snapshot_rows": 0,
            "latest_snapshot": None,
            "snapshot_age_hours": None,
            "freshness": "missing",
            "unique_events": 0,
            "unique_fights": 0,
            "unique_sportsbooks": 0,
            "matched_rows": 0,
            "unmatched_rows": 0,
            "missing_red_odds": None,
            "missing_blue_odds": None,
        }

    snapshots = snapshots_df.copy()
    if "snapshot_timestamp" in snapshots.columns:
        snapshots["snapshot_timestamp"] = pd.to_datetime(
            snapshots["snapshot_timestamp"],
            utc=True,
            errors="coerce",
        )
        latest_snapshot = snapshots["snapshot_timestamp"].max()
    else:
        latest_snapshot = None

    if pd.notna(latest_snapshot):
        age_hours = round(
            (pd.Timestamp.utcnow() - latest_snapshot).total_seconds() / 3600,
            2,
        )
        latest_snapshot_value = latest_snapshot.isoformat()
    else:
        age_hours = None
        latest_snapshot_value = None

    if "odds_match_type" in snapshots.columns:
        matched_rows = int((snapshots["odds_match_type"] == "matched").sum())
        unmatched_rows = int((snapshots["odds_match_type"] != "matched").sum())
    else:
        matched_rows = 0
        unmatched_rows = 0

    sportsbook_column = "bookmaker" if "bookmaker" in snapshots.columns else "sportsbook"

    return {
        "snapshot_rows": int(len(snapshots)),
        "latest_snapshot": latest_snapshot_value,
        "snapshot_age_hours": age_hours,
        "freshness": freshness_status(age_hours),
        "unique_events": _safe_nunique(snapshots, "event_name"),
        "unique_fights": _safe_nunique(snapshots, "fight_id"),
        "unique_sportsbooks": _safe_nunique(snapshots, sportsbook_column),
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "missing_red_odds": _missing_count(snapshots, "red_american_odds"),
        "missing_blue_odds": _missing_count(snapshots, "blue_american_odds"),
    }


def get_clv_artifact_status():
    return artifact_status_rows(CLV_ARTIFACTS)
