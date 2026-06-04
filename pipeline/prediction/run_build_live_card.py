import argparse
import os

import pandas as pd

from pipeline.common.paths import (
    LIVE_CARD_PATH,
    SELECTED_LIVE_CARD_EVENT_PATH,
    UPCOMING_FIGHTS_PATH,
    ensure_data_dirs,
)


LIVE_CARD_COLUMNS = [
    "event_name",
    "event_id",
    "event_date",
    "event_location",
    "event_url",
    "fight_order",
    "fight_id",
    "fight_url",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "red_fighter_url",
    "blue_fighter_url",
    "weight_class",
]


def build_live_card(event_id):
    ensure_data_dirs()

    if not event_id:
        raise ValueError("event_id is required to build a selected live card.")

    upcoming_fights = pd.read_parquet(UPCOMING_FIGHTS_PATH)

    if "event_id" not in upcoming_fights.columns:
        raise ValueError(f"Upcoming fights artifact is missing event_id: {UPCOMING_FIGHTS_PATH}")

    live_card = upcoming_fights[upcoming_fights["event_id"].astype(str) == str(event_id)].copy()

    if live_card.empty:
        raise ValueError(f"No upcoming fights found for event_id={event_id} in {UPCOMING_FIGHTS_PATH}")

    for column in LIVE_CARD_COLUMNS:
        if column not in live_card.columns:
            live_card[column] = pd.NA

    live_card = live_card[LIVE_CARD_COLUMNS]
    live_card = live_card.sort_values("fight_order", na_position="last").reset_index(drop=True)
    live_card.to_parquet(LIVE_CARD_PATH, index=False)

    event_columns = [
        "event_id",
        "event_name",
        "event_date",
        "event_location",
        "event_url",
    ]
    selected_event = live_card[event_columns].drop_duplicates().head(1)
    selected_event.to_parquet(SELECTED_LIVE_CARD_EVENT_PATH, index=False)

    print(f"Selected live card saved: {LIVE_CARD_PATH} ({len(live_card)} rows)")
    print(f"Selected event saved: {SELECTED_LIVE_CARD_EVENT_PATH}")

    return live_card


def parse_args():
    parser = argparse.ArgumentParser(description="Build data/predictions/ufc_live_card.parquet from a selected upcoming UFCStats event.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID") or os.getenv("BETTING_EVENT_ID"), help="UFCStats event id to select.")
    return parser.parse_args()


def main():
    args = parse_args()
    build_live_card(args.event_id)


if __name__ == "__main__":
    main()
