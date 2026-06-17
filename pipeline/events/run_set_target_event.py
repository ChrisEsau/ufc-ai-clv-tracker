from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline.common.paths import (
    LIVE_CARD_PATH,
    SELECTED_LIVE_CARD_EVENT_PATH,
    UPCOMING_EVENTS_PATH,
    UPCOMING_FIGHTS_PATH,
    ensure_data_dirs,
)
from pipeline.prediction.run_build_live_card import build_live_card
from pipeline.prediction.run_refresh_upcoming_events import refresh_upcoming_events
from pipeline_config import ODDS_API_KEY, ODDS_FORMAT, PREFERRED_BOOKMAKER, REGIONS, SPORT
from ufc_odds_utils import fetch_the_odds_api_events, flatten_h2h_odds, match_live_fight_to_odds_row

CENTRAL_TZ = ZoneInfo("America/Chicago")

TARGET_EVENT_COLUMNS = [
    "event_id",
    "event_name",
    "event_date",
    "event_location",
    "event_url",
    "event_status",
    "selection_scope",
    "commence_time_utc",
    "commence_time_cdt",
    "commence_time_source",
    "commence_time_match_count",
    "created_at",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _safe_str(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _format_central(value) -> str:
    if not value:
        return ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.to_pydatetime().astimezone(CENTRAL_TZ).isoformat()


def _select_nearest_future_event(events: pd.DataFrame) -> pd.Series:
    if events.empty:
        raise ValueError("No upcoming UFCStats events are available.")

    required = {"ufcstats_event_id", "ufcstats_event_name", "ufcstats_event_date", "ufcstats_event_url"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"Upcoming events artifact missing required columns: {missing}")

    work = events.copy()
    work["_event_date"] = pd.to_datetime(work["ufcstats_event_date"], errors="coerce")
    valid = work[work["_event_date"].notna()].copy()
    if valid.empty:
        raise ValueError("No upcoming UFCStats events have parseable dates.")

    today = pd.Timestamp(_now_utc().date())
    future = valid[valid["_event_date"] >= today].copy()
    selected_pool = future if not future.empty else valid
    selected_pool = selected_pool.sort_values("_event_date", na_position="last").reset_index(drop=True)
    return selected_pool.iloc[0]


def _resolve_commence_time_from_odds(live_card: pd.DataFrame) -> tuple[str, str, int]:
    if not ODDS_API_KEY:
        return "", "odds_api_key_missing", 0

    odds_json = fetch_the_odds_api_events(
        api_key=ODDS_API_KEY,
        sport=SPORT,
        regions=REGIONS,
        markets="h2h",
        odds_format=ODDS_FORMAT,
    )
    odds_df = flatten_h2h_odds(odds_json, preferred_bookmaker=PREFERRED_BOOKMAKER)
    if odds_df.empty:
        odds_df = flatten_h2h_odds(odds_json, preferred_bookmaker=None)
    if odds_df.empty or "commence_time" not in odds_df.columns:
        return "", "odds_api_no_commence_time", 0

    matched_times: list[str] = []
    for _, live_row in live_card.iterrows():
        match = match_live_fight_to_odds_row(live_row, odds_df, min_single_score=70)
        if not match:
            continue
        commence_time = _safe_str(match.get("commence_time"))
        if commence_time:
            matched_times.append(commence_time)

    if not matched_times:
        return "", "odds_api_no_live_card_match", 0

    parsed = pd.to_datetime(pd.Series(matched_times), errors="coerce", utc=True).dropna()
    if parsed.empty:
        return "", "odds_api_unparseable_commence_time", 0

    earliest = parsed.min().to_pydatetime().isoformat()
    return earliest, "the_odds_api", len(parsed)


def _target_event_row(selected_event: pd.Series, live_card: pd.DataFrame, *, run_timestamp: str) -> pd.DataFrame:
    commence_time_utc, source, match_count = _resolve_commence_time_from_odds(live_card)
    row = {
        "event_id": _safe_str(selected_event.get("ufcstats_event_id")),
        "event_name": _safe_str(selected_event.get("ufcstats_event_name")),
        "event_date": _safe_str(selected_event.get("ufcstats_event_date")),
        "event_location": _safe_str(selected_event.get("ufcstats_event_location")),
        "event_url": _safe_str(selected_event.get("ufcstats_event_url")),
        "event_status": "active_target",
        "selection_scope": "current_target_event",
        "commence_time_utc": commence_time_utc,
        "commence_time_cdt": _format_central(commence_time_utc),
        "commence_time_source": source,
        "commence_time_match_count": int(match_count),
        "created_at": run_timestamp,
    }
    return pd.DataFrame([row], columns=TARGET_EVENT_COLUMNS)


def run_set_target_event(*, refresh_upcoming: bool = True, max_events: int | None = None) -> pd.DataFrame:
    print("=" * 80)
    print("SET CURRENT TARGET UFC EVENT")
    print("=" * 80)
    print("Refresh upcoming:", refresh_upcoming)
    print("Max events:", "all" if max_events is None else max_events)

    ensure_data_dirs()
    run_timestamp = _now_utc().isoformat()

    if refresh_upcoming or not UPCOMING_EVENTS_PATH.exists() or not UPCOMING_FIGHTS_PATH.exists():
        events, _upcoming_fights = refresh_upcoming_events(max_events=max_events)
    else:
        events = pd.read_parquet(UPCOMING_EVENTS_PATH)

    selected_event = _select_nearest_future_event(events)
    event_id = _safe_str(selected_event.get("ufcstats_event_id"))
    if not event_id:
        raise ValueError("Selected target event is missing ufcstats_event_id.")

    live_card = build_live_card(event_id)
    target_event = _target_event_row(selected_event, live_card, run_timestamp=run_timestamp)
    SELECTED_LIVE_CARD_EVENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_event.to_parquet(SELECTED_LIVE_CARD_EVENT_PATH, index=False)

    print()
    print("========== CURRENT TARGET EVENT ==========")
    print(target_event.to_string(index=False))
    print("Saved target event:", SELECTED_LIVE_CARD_EVENT_PATH)
    print("Saved live card:", LIVE_CARD_PATH)

    return target_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set the current target UFC event for Market Refresh and Fight Day flows.")
    parser.add_argument("--no-refresh-upcoming", action="store_true", help="Use existing upcoming artifacts instead of refreshing UFCStats.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap for upcoming event fight-card scraping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_set_target_event(
        refresh_upcoming=not bool(args.no_refresh_upcoming),
        max_events=args.max_events,
    )


if __name__ == "__main__":
    main()
