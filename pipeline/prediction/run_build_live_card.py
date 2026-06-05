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


def _prepare_live_card(upcoming_fights):
    """Normalize an upcoming-fights slice into the canonical live-card shape."""

    live_card = upcoming_fights.copy()

    for column in LIVE_CARD_COLUMNS:
        if column not in live_card.columns:
            live_card[column] = pd.NA

    live_card = live_card[LIVE_CARD_COLUMNS]
    sort_columns = [column for column in ["event_date", "event_name", "fight_order"] if column in live_card.columns]
    if sort_columns:
        live_card = live_card.sort_values(sort_columns, na_position="last")

    return live_card.reset_index(drop=True)


def _write_live_card(live_card, status_label):
    """Persist the live-card artifact plus event metadata used by the dashboard."""

    live_card.to_parquet(LIVE_CARD_PATH, index=False)

    event_columns = [
        "event_id",
        "event_name",
        "event_date",
        "event_location",
        "event_url",
    ]
    selected_events = live_card[event_columns].drop_duplicates().reset_index(drop=True)
    selected_events["selection_scope"] = status_label
    selected_events.to_parquet(SELECTED_LIVE_CARD_EVENT_PATH, index=False)

    print(f"{status_label.title().replace('_', ' ')} live card saved: {LIVE_CARD_PATH} ({len(live_card)} rows)")
    print(f"Live-card event metadata saved: {SELECTED_LIVE_CARD_EVENT_PATH} ({len(selected_events)} event rows)")


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

    live_card = _prepare_live_card(live_card)
    _write_live_card(live_card, "selected_event")

    return live_card


def build_all_upcoming_live_card():
    ensure_data_dirs()

    upcoming_fights = pd.read_parquet(UPCOMING_FIGHTS_PATH)

    if upcoming_fights.empty:
        raise ValueError(f"No upcoming fights found in {UPCOMING_FIGHTS_PATH}")

    live_card = _prepare_live_card(upcoming_fights)
    _write_live_card(live_card, "all_upcoming")

    return live_card


def parse_args():
    parser = argparse.ArgumentParser(description="Build data/predictions/ufc_live_card.parquet from upcoming UFCStats fights.")
    parser.add_argument("--event-id", default=os.getenv("EVENT_ID") or os.getenv("BETTING_EVENT_ID"), help="UFCStats event id to select.")
    parser.add_argument(
        "--all-upcoming",
        action="store_true",
        help="Build the live-card artifact from every fight in the upcoming-fights artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_upcoming:
        build_all_upcoming_live_card()
    else:
        build_live_card(args.event_id)


if __name__ == "__main__":
    main()
