# ============================================================
# pipeline/market/run_fanduel_card_filter.py
# ============================================================

"""Match FanDuel UFC event-index rows to the current UFC live card.

Offline-only. Reads FanDuel event index + live card and writes matched FanDuel
provider_event_id rows for safe/relevant market discovery.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import (
    FANDUEL_EVENT_CARD_MATCH_PATH,
    FANDUEL_EVENT_INDEX_PATH,
    LIVE_CARD_PATH,
    ensure_data_dirs,
)
from ufc_odds_utils import composite_name_score


EVENT_CARD_MATCH_COLUMNS = [
    "match_run_id",
    "match_timestamp",
    "provider_event_id",
    "provider_event_name",
    "provider_start_time",
    "provider_competition_id",
    "provider_competition_name",
    "fight_id",
    "ufcstats_event_id",
    "ufcstats_event_name",
    "ufcstats_event_date",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "same_order_score",
    "same_order_min_score",
    "reversed_order_score",
    "reversed_order_min_score",
    "match_type",
    "match_score",
    "min_single_score",
    "is_matched",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter FanDuel UFC event IDs to fights on the current live card.")
    parser.add_argument("--event-index-path", default=str(FANDUEL_EVENT_INDEX_PATH))
    parser.add_argument("--live-card-path", default=str(LIVE_CARD_PATH))
    parser.add_argument("--output-path", default=str(FANDUEL_EVENT_CARD_MATCH_PATH))
    parser.add_argument("--min-match-score", type=float, default=80.0)
    parser.add_argument("--min-single-score", type=float, default=70.0)
    return parser.parse_args()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _safe_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _score_order(provider_home: Any, provider_away: Any, red_fighter: Any, blue_fighter: Any) -> tuple[float, float]:
    home_red_score = composite_name_score(_safe_str(provider_home), _safe_str(red_fighter))
    away_blue_score = composite_name_score(_safe_str(provider_away), _safe_str(blue_fighter))
    pair_score = (home_red_score + away_blue_score) / 2
    return float(pair_score), float(min(home_red_score, away_blue_score))


def _best_match_for_event(
    event_row: pd.Series,
    live_card_df: pd.DataFrame,
    *,
    min_match_score: float,
    min_single_score: float,
    match_run_id: str,
    match_timestamp: str,
) -> dict[str, Any]:
    provider_home = event_row.get("participant_home")
    provider_away = event_row.get("participant_away")

    candidates = []
    for _, live_row in live_card_df.iterrows():
        same_score, same_min = _score_order(provider_home, provider_away, live_row.get("red_fighter"), live_row.get("blue_fighter"))
        reversed_score, reversed_min = _score_order(provider_home, provider_away, live_row.get("blue_fighter"), live_row.get("red_fighter"))

        if same_score >= reversed_score:
            match_type = "same_order"
            match_score = same_score
            best_min = same_min
        else:
            match_type = "reversed_order"
            match_score = reversed_score
            best_min = reversed_min

        is_matched = bool(match_score >= min_match_score and best_min >= min_single_score)

        candidates.append({
            "match_run_id": match_run_id,
            "match_timestamp": match_timestamp,
            "provider_event_id": event_row.get("provider_event_id"),
            "provider_event_name": event_row.get("provider_event_name"),
            "provider_start_time": event_row.get("provider_start_time"),
            "provider_competition_id": event_row.get("provider_competition_id"),
            "provider_competition_name": event_row.get("provider_competition_name"),
            "fight_id": live_row.get("fight_id"),
            "ufcstats_event_id": live_row.get("event_id"),
            "ufcstats_event_name": live_row.get("event_name"),
            "ufcstats_event_date": live_row.get("event_date"),
            "red_fighter": live_row.get("red_fighter"),
            "blue_fighter": live_row.get("blue_fighter"),
            "red_fighter_id": live_row.get("red_fighter_id"),
            "blue_fighter_id": live_row.get("blue_fighter_id"),
            "same_order_score": same_score,
            "same_order_min_score": same_min,
            "reversed_order_score": reversed_score,
            "reversed_order_min_score": reversed_min,
            "match_type": match_type,
            "match_score": float(match_score),
            "min_single_score": float(best_min),
            "is_matched": is_matched,
        })

    if not candidates:
        return {
            "match_run_id": match_run_id,
            "match_timestamp": match_timestamp,
            "provider_event_id": event_row.get("provider_event_id"),
            "provider_event_name": event_row.get("provider_event_name"),
            "provider_start_time": event_row.get("provider_start_time"),
            "provider_competition_id": event_row.get("provider_competition_id"),
            "provider_competition_name": event_row.get("provider_competition_name"),
            "is_matched": False,
        }

    return sorted(candidates, key=lambda r: (r["is_matched"], r["match_score"], r["min_single_score"]), reverse=True)[0]


def ensure_event_card_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in EVENT_CARD_MATCH_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[EVENT_CARD_MATCH_COLUMNS]


def run_card_filter(
    *,
    event_index_path: Path = FANDUEL_EVENT_INDEX_PATH,
    live_card_path: Path = LIVE_CARD_PATH,
    output_path: Path = FANDUEL_EVENT_CARD_MATCH_PATH,
    min_match_score: float = 80.0,
    min_single_score: float = 70.0,
) -> pd.DataFrame:
    event_index_df = _load_required_parquet(event_index_path, "FanDuel event index")
    live_card_df = _load_required_parquet(live_card_path, "Live card")

    match_run_id = datetime.now(timezone.utc).strftime("fanduel_card_filter_%Y%m%d_%H%M%S")
    match_timestamp = datetime.now(timezone.utc).isoformat()

    rows = [
        _best_match_for_event(
            row,
            live_card_df,
            min_match_score=min_match_score,
            min_single_score=min_single_score,
            match_run_id=match_run_id,
            match_timestamp=match_timestamp,
        )
        for _, row in event_index_df.iterrows()
    ]

    match_df = ensure_event_card_match_columns(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_df.to_parquet(output_path, index=False)
    return match_df


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    print("=" * 80)
    print("FANDUEL EVENT CARD FILTER")
    print("=" * 80)
    print("Mode: offline event-index/live-card matching only")
    print("FanDuel requests: 0")
    print("Event index path:", args.event_index_path)
    print("Live card path:", args.live_card_path)
    print("Output path:", args.output_path)

    match_df = run_card_filter(
        event_index_path=Path(args.event_index_path),
        live_card_path=Path(args.live_card_path),
        output_path=Path(args.output_path),
        min_match_score=float(args.min_match_score),
        min_single_score=float(args.min_single_score),
    )

    matched_df = match_df[match_df["is_matched"].fillna(False)]

    print()
    print("========== FANDUEL EVENT CARD FILTER SUMMARY ==========")
    print("Indexed FanDuel events:", len(match_df))
    print("Matched live-card fights:", len(matched_df))
    print("Unmatched FanDuel events:", len(match_df) - len(matched_df))
    print("Output path:", args.output_path)

    if not matched_df.empty:
        print()
        print(matched_df[[
            "provider_event_id",
            "provider_event_name",
            "fight_id",
            "red_fighter",
            "blue_fighter",
            "match_score",
            "min_single_score",
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
