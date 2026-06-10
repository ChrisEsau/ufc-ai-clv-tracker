import argparse
from datetime import datetime, timezone

import pandas as pd

from pipeline.common.fight_context import (
    clean_division,
    title_fight_flag,
    total_rounds_from_time_format,
)
from pipeline.common.paths import (
    UPCOMING_EVENTS_PATH,
    UPCOMING_FIGHTS_PATH,
    ensure_data_dirs,
)
from scrapers.ufcstats_events import scrape_upcoming_events
from scrapers.ufcstats_fights import scrape_event_fights


def extract_id_from_url(url):
    if not url:
        return None
    return str(url).rstrip("/").split("/")[-1]


def build_fight_id(red_fighter_id, blue_fighter_id):
    if not red_fighter_id or not blue_fighter_id:
        return None
    return "__".join(sorted([str(red_fighter_id), str(blue_fighter_id)]))


def normalize_upcoming_fights(fights, event_row, run_id, run_timestamp):
    normalized = fights.copy()
    normalized["run_id"] = run_id
    normalized["run_timestamp"] = run_timestamp
    normalized["event_id"] = event_row["ufcstats_event_id"]
    normalized["event_name"] = event_row["ufcstats_event_name"]
    normalized["event_date"] = event_row["ufcstats_event_date"]
    normalized["event_location"] = event_row.get("ufcstats_event_location")
    normalized["event_url"] = event_row["ufcstats_event_url"]
    normalized["event_state"] = "upcoming"

    weight_class = normalized["weight_class"] if "weight_class" in normalized.columns else pd.Series(pd.NA, index=normalized.index)
    time_format = normalized["time_format"] if "time_format" in normalized.columns else pd.Series(pd.NA, index=normalized.index)

    normalized["division"] = weight_class.apply(clean_division)
    normalized["title_fight"] = weight_class.apply(title_fight_flag)
    normalized["total_rounds"] = [
        total_rounds_from_time_format(value, title_fight)
        for value, title_fight in zip(time_format, normalized["title_fight"])
    ]

    normalized["red_fighter_id"] = normalized["red_fighter_url"].apply(extract_id_from_url)
    normalized["blue_fighter_id"] = normalized["blue_fighter_url"].apply(extract_id_from_url)
    normalized["fight_id"] = normalized.apply(
        lambda row: build_fight_id(row.get("red_fighter_id"), row.get("blue_fighter_id")),
        axis=1,
    )
    return normalized


def refresh_upcoming_events(max_events=None):
    ensure_data_dirs()

    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    events = scrape_upcoming_events(run_id=run_id, run_timestamp=run_timestamp)
    events = events.sort_values("ufcstats_event_date", na_position="last").reset_index(drop=True)

    if max_events is not None:
        events_to_scrape = events.head(max_events)
    else:
        events_to_scrape = events

    fight_frames = []

    for _, event_row in events_to_scrape.iterrows():
        fights = scrape_event_fights(
            event_row["ufcstats_event_url"],
            event_name=event_row["ufcstats_event_name"],
            event_date=event_row["ufcstats_event_date"],
        )
        if fights.empty:
            continue
        fight_frames.append(normalize_upcoming_fights(fights, event_row, run_id, run_timestamp))

    upcoming_fights = pd.concat(fight_frames, ignore_index=True) if fight_frames else pd.DataFrame()

    events.to_parquet(UPCOMING_EVENTS_PATH, index=False)
    upcoming_fights.to_parquet(UPCOMING_FIGHTS_PATH, index=False)

    print(f"Upcoming events saved: {UPCOMING_EVENTS_PATH} ({len(events)} rows)")
    print(f"Upcoming fights saved: {UPCOMING_FIGHTS_PATH} ({len(upcoming_fights)} rows)")

    return events, upcoming_fights


def parse_args():
    parser = argparse.ArgumentParser(description="Refresh UFCStats upcoming event and fight card artifacts.")
    parser.add_argument("--max-events", type=int, default=None, help="Optional limit for event fight-card scraping.")
    return parser.parse_args()


def main():
    args = parse_args()
    refresh_upcoming_events(max_events=args.max_events)


if __name__ == "__main__":
    main()
