# ============================================================
# pipeline/market/run_draftkings_event_index.py
# ============================================================

"""Standalone DraftKings UFC event index runner.

This runner is intentionally limited to one low-volume league navigation
request. It discovers DraftKings fight-level event IDs for UFC and writes an
independent reusable index artifact.

It does not run DraftKings market discovery, canonical normalization, market
matching, betting outcomes, CLV calculations, dashboard updates, proxy rotation,
CAPTCHA handling, or ban-evasion logic.

Usage:
    python -m pipeline.market.run_draftkings_event_index

Optional:
    python -m pipeline.market.run_draftkings_event_index \
        --url "<draftkings-ufc-league-navigation-url>" \
        --output-path data/market/draftkings_event_index.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.common.paths import DRAFTKINGS_EVENT_INDEX_PATH, ensure_data_dirs
from pipeline.market.providers.draftkings_public import fetch_public_json, utc_snapshot

DEFAULT_DRAFTKINGS_UFC_LEAGUE_NAV_URL = (
    "https://sportsbook-nash.draftkings.com/sites/US-KS-SB/"
    "api/sportscontent/navigation/dkusks/v2/nav/leagues/9034"
)

EVENT_INDEX_COLUMNS = [
    "provider_event_id",
    "provider_event_name",
    "provider_event_slug",
    "provider_start_time",
    "provider_sport_id",
    "provider_sport_name",
    "provider_league_id",
    "provider_league_name",
    "participant_home",
    "participant_away",
    "participant_home_sdid",
    "participant_away_sdid",
    "request_url",
    "snapshot_timestamp",
    "snapshot_run_id",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover DraftKings UFC fight-level event IDs from the league index."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("DRAFTKINGS_UFC_LEAGUE_NAV_URL", DEFAULT_DRAFTKINGS_UFC_LEAGUE_NAV_URL),
        help="DraftKings UFC league navigation endpoint. Defaults to leagueId 9034.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DRAFTKINGS_EVENT_INDEX_PATH),
        help="Output parquet path for the DraftKings UFC event index.",
    )
    return parser.parse_args()


def _as_id(value: Any) -> str | None:
    """Return a provider ID as a stable string, preserving missing values as None."""

    if value in (None, ""):
        return None
    return str(value)


def _lookup_by_id(items: Any) -> dict[str, dict[str, Any]]:
    """Build a string-ID lookup for a provider collection."""

    if not isinstance(items, list):
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = _as_id(item.get("id"))
        if item_id:
            lookup[item_id] = item
    return lookup


def _participant_by_role(participants: Any, role: str) -> dict[str, Any]:
    """Return the first participant matching a DraftKings venue role."""

    if not isinstance(participants, list):
        return {}

    for participant in participants:
        if isinstance(participant, dict) and participant.get("venueRole") == role:
            return participant
    return {}


def _participant_sdid(participant: dict[str, Any]) -> Any:
    """Extract DraftKings participant sdid from common participant shapes."""

    metadata = participant.get("metadata")
    if isinstance(metadata, dict) and metadata.get("sdid") not in (None, ""):
        return metadata.get("sdid")
    return participant.get("sdid")


def _event_slug(event: dict[str, Any]) -> Any:
    """Extract the most useful event slug field available in the navigation payload."""

    for key in ["seoIdentifier", "seoName", "slug", "eventGroupSeoIdentifier"]:
        value = event.get(key)
        if value not in (None, ""):
            return value
    return None


def build_event_index(payload: dict[str, Any], *, request_url: str) -> pd.DataFrame:
    """Transform a DraftKings league navigation payload into the event index schema."""

    snapshot_run_id, snapshot_timestamp = utc_snapshot()
    sports_by_id = _lookup_by_id(payload.get("sports"))
    leagues_by_id = _lookup_by_id(payload.get("leagues"))

    rows: list[dict[str, Any]] = []
    events = payload.get("events")
    if not isinstance(events, list):
        events = []

    for event in events:
        if not isinstance(event, dict):
            continue

        provider_event_id = _as_id(event.get("id"))
        if not provider_event_id:
            continue

        league_id = _as_id(event.get("leagueId")) or "9034"
        league = leagues_by_id.get(league_id, {})
        sport_id = _as_id(event.get("sportId")) or _as_id(league.get("sportId")) or "43"
        sport = sports_by_id.get(sport_id, {})

        home = _participant_by_role(event.get("participants"), "Home")
        away = _participant_by_role(event.get("participants"), "Away")

        rows.append(
            {
                "provider_event_id": provider_event_id,
                "provider_event_name": event.get("name"),
                "provider_event_slug": _event_slug(event),
                "provider_start_time": event.get("startEventDate"),
                "provider_sport_id": sport_id,
                "provider_sport_name": sport.get("name"),
                "provider_league_id": league_id,
                "provider_league_name": league.get("name"),
                "participant_home": home.get("name"),
                "participant_away": away.get("name"),
                "participant_home_sdid": _participant_sdid(home),
                "participant_away_sdid": _participant_sdid(away),
                "request_url": request_url,
                "snapshot_timestamp": snapshot_timestamp,
                "snapshot_run_id": snapshot_run_id,
            }
        )

    return ensure_event_index_columns(pd.DataFrame(rows))


def ensure_event_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a stable event index schema even when no events are returned."""

    out = df.copy()
    for column in EVENT_INDEX_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[EVENT_INDEX_COLUMNS]


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    print("=" * 80)
    print("DRAFTKINGS UFC EVENT INDEX")
    print("=" * 80)
    print("Mode: one league navigation request + independent event index artifact")
    print("Market discovery, normalization, matching, outcomes: untouched")
    print("Request URL:", args.url)

    payload = fetch_public_json(args.url)
    event_index_df = build_event_index(payload, request_url=args.url)

    if event_index_df.empty:
        raise RuntimeError(
            "DraftKings UFC event index returned zero events. "
            "Existing event index artifact was not overwritten."
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_index_df.to_parquet(output_path, index=False)

    print()
    print("========== DRAFTKINGS UFC EVENT INDEX SUMMARY ==========")
    print("Snapshot run ID:", event_index_df["snapshot_run_id"].iloc[0])
    print("Snapshot timestamp:", event_index_df["snapshot_timestamp"].iloc[0])
    print("Events discovered:", len(event_index_df))
    print("Output path:", output_path)

    print()
    print("Discovered fights:")
    print(
        event_index_df[
            ["provider_event_id", "provider_event_name", "provider_start_time"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
