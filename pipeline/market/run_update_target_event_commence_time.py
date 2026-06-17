from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.common.paths import DRAFTKINGS_EVENT_CARD_MATCH_PATH, SELECTED_LIVE_CARD_EVENT_PATH, ensure_data_dirs

CENTRAL_TZ = ZoneInfo("America/Chicago")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_central(value: str) -> str:
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.to_pydatetime().astimezone(CENTRAL_TZ).isoformat()


def _load_required(path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def update_target_event_commence_time(
    *,
    target_event_path=SELECTED_LIVE_CARD_EVENT_PATH,
    card_match_path=DRAFTKINGS_EVENT_CARD_MATCH_PATH,
) -> pd.DataFrame:
    print("=" * 80)
    print("UPDATE TARGET EVENT COMMENCE TIME FROM DRAFTKINGS")
    print("=" * 80)

    ensure_data_dirs()
    target = _load_required(target_event_path, "Selected target event")
    matches = _load_required(card_match_path, "DraftKings card-match artifact")

    required = {"is_matched", "provider_start_time"}
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"DraftKings card-match artifact missing required columns: {missing}")

    matched = matches[matches["is_matched"].fillna(False)].copy()
    if matched.empty:
        target["commence_time_source"] = "draftkings_no_matched_events"
        target["commence_time_match_count"] = 0
        target["commence_time_updated_at"] = _now_iso()
        target.to_parquet(target_event_path, index=False)
        print("No matched DraftKings events found. Target event updated with pending commence time status.")
        return target

    parsed = pd.to_datetime(matched["provider_start_time"], errors="coerce", utc=True).dropna()
    if parsed.empty:
        target["commence_time_source"] = "draftkings_unparseable_provider_start_time"
        target["commence_time_match_count"] = int(len(matched))
        target["commence_time_updated_at"] = _now_iso()
        target.to_parquet(target_event_path, index=False)
        print("Matched DraftKings rows had no parseable provider_start_time values.")
        return target

    commence_time_utc = parsed.min().to_pydatetime().isoformat()
    target["commence_time_utc"] = commence_time_utc
    target["commence_time_cdt"] = _format_central(commence_time_utc)
    target["commence_time_source"] = "draftkings_provider_start_time"
    target["commence_time_match_count"] = int(len(parsed))
    target["commence_time_updated_at"] = _now_iso()
    target.to_parquet(target_event_path, index=False)

    print("Commence time UTC:", commence_time_utc)
    print("Commence time CDT:", target.iloc[0].get("commence_time_cdt"))
    print("Matched rows used:", len(parsed))
    print("Saved target event:", target_event_path)
    return target


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Update selected target UFC event commence time from matched DraftKings provider start times.").parse_args()


def main() -> None:
    parse_args()
    update_target_event_commence_time()


if __name__ == "__main__":
    main()
