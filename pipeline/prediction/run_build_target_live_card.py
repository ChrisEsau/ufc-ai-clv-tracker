from __future__ import annotations

import argparse

import pandas as pd

from pipeline.common.paths import LIVE_CARD_PATH, SELECTED_LIVE_CARD_EVENT_PATH, ensure_data_dirs
from pipeline.prediction.run_build_live_card import build_live_card


def _load_target_event(path=SELECTED_LIVE_CARD_EVENT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Selected target event artifact not found: {path}")
    target = pd.read_parquet(path)
    if target.empty or "event_id" not in target.columns:
        raise ValueError(f"Selected target event artifact missing event_id: {path}")
    return target


def run_build_target_live_card() -> pd.DataFrame:
    ensure_data_dirs()
    target_event = _load_target_event()
    event_id = str(target_event.iloc[0]["event_id"]).strip()
    if not event_id:
        raise ValueError("Selected target event has blank event_id.")

    # build_live_card writes both the live card and selected-event metadata. Preserve
    # the richer target-event artifact, including commence_time fields, after the
    # live card is rebuilt for the selected event.
    live_card = build_live_card(event_id)
    target_event.to_parquet(SELECTED_LIVE_CARD_EVENT_PATH, index=False)

    print("========== TARGET LIVE CARD BUILT ==========")
    print("Event ID:", event_id)
    if "event_name" in target_event.columns:
        print("Event:", target_event.iloc[0].get("event_name"))
    print("Live card rows:", len(live_card))
    print("Saved live card:", LIVE_CARD_PATH)
    print("Preserved target event:", SELECTED_LIVE_CARD_EVENT_PATH)
    return live_card


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Build live card from selected target event artifact.").parse_args()


def main() -> None:
    parse_args()
    run_build_target_live_card()


if __name__ == "__main__":
    main()
