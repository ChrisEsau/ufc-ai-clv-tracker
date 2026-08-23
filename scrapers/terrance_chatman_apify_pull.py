"""Low-cost one-fighter regional MMA pull for Terrance Chatman.

Research-only. This resolves Chatman's SofaScore event IDs with one Apify search run,
then fetches detailed statistics for the discovered events in one exact-event run.
The token is read from APIFY_TOKEN and is never persisted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from pipeline.common.paths import AUDITS_DIR

FIGHTER = "Terrance Chatman"
FIGHTER_SURNAME = "chatman"
EXPECTED_PRE_UFC_OPPONENTS = {
    "prieto",
    "gurrola",
    "maters",
    "el-sahlah",
    "dennis",
    "torres",
}

DISCOVERY_ACTOR_ID = "gio21~sofascore-scraper"
STATS_ACTOR_ID = "automation-lab~sofascore-live-events-statistics-scraper"
DISCOVERY_URL = f"https://api.apify.com/v2/actors/{DISCOVERY_ACTOR_ID}/run-sync-get-dataset-items"
STATS_URL = f"https://api.apify.com/v2/actors/{STATS_ACTOR_ID}/run-sync-get-dataset-items"

OUTPUT_DIR = AUDITS_DIR / "regional_mma" / "sofascore" / "terrance_chatman"
DISCOVERY_MAX_ITEMS = 10
DISCOVERY_MAX_CHARGE_USD = 0.04
STATS_MAX_CHARGE_USD = 0.03


def _post_actor(url: str, token: str, payload: dict[str, Any], *, max_items: int, max_charge_usd: float) -> list[dict[str, Any]]:
    response = requests.post(
        url,
        params={"maxItems": max_items, "maxTotalChargeUsd": max_charge_usd},
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:2000].replace("\n", " ")
        raise RuntimeError(f"Apify request failed HTTP {response.status_code}: {body}") from exc
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Expected Apify dataset list, got {type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def discover_history(token: str) -> list[dict[str, Any]]:
    return _post_actor(
        DISCOVERY_URL,
        token,
        {
            "searchTerm": FIGHTER,
            "includeMatches": True,
            "maxItems": DISCOVERY_MAX_ITEMS,
        },
        max_items=DISCOVERY_MAX_ITEMS,
        max_charge_usd=DISCOVERY_MAX_CHARGE_USD,
    )


def _match_involves_chatman(item: dict[str, Any]) -> bool:
    if str(item.get("type", "")).casefold() != "match":
        return False
    home = str(item.get("homeTeam", "")).casefold()
    away = str(item.get("awayTeam", "")).casefold()
    return FIGHTER_SURNAME in home or FIGHTER_SURNAME in away


def _looks_pre_ufc(item: dict[str, Any]) -> bool:
    home = str(item.get("homeTeam", "")).casefold()
    away = str(item.get("awayTeam", "")).casefold()
    opponent = away if FIGHTER_SURNAME in home else home
    return any(last in opponent for last in EXPECTED_PRE_UFC_OPPONENTS)


def discovered_event_ids(items: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in items:
        if not _match_involves_chatman(item) or not _looks_pre_ufc(item):
            continue
        value = item.get("sofascoreId")
        if value is not None:
            ids.append(str(value))
    return list(dict.fromkeys(ids))


def fetch_event_stats(token: str, event_ids: list[str]) -> list[dict[str, Any]]:
    if not event_ids:
        return []
    return _post_actor(
        STATS_URL,
        token,
        {
            "eventIds": event_ids,
            "eventUrls": [],
            "includeStatistics": True,
            "status": "all",
            "team": "",
            "tournament": "",
            "maxItems": len(event_ids),
        },
        max_items=len(event_ids),
        max_charge_usd=STATS_MAX_CHARGE_USD,
    )


def _team_name(item: dict[str, Any], side: str) -> str:
    value = item.get(f"{side}Team")
    return str(value.get("name", "")) if isinstance(value, dict) else ""


def _fighter_side(item: dict[str, Any]) -> str | None:
    if FIGHTER_SURNAME in _team_name(item, "home").casefold():
        return "home"
    if FIGHTER_SURNAME in _team_name(item, "away").casefold():
        return "away"
    return None


def summarize_event(item: dict[str, Any]) -> dict[str, Any]:
    stats = item.get("statistics") if isinstance(item.get("statistics"), list) else []
    side = _fighter_side(item)
    home = _team_name(item, "home")
    away = _team_name(item, "away")
    opponent = away if side == "home" else home if side == "away" else None
    tournament = item.get("tournament") if isinstance(item.get("tournament"), dict) else {}

    all_values: dict[str, Any] = {}
    for row in stats:
        if not isinstance(row, dict) or row.get("period") != "ALL":
            continue
        name = row.get("name")
        if not name:
            continue
        all_values[str(name)] = row.get(side) if side in {"home", "away"} else None

    return {
        "event_id": item.get("eventId"),
        "start_time": item.get("startTime"),
        "tournament": tournament.get("name"),
        "home": home,
        "away": away,
        "opponent": opponent,
        "fighter_side": side,
        "status": item.get("status"),
        "has_statistics": bool(item.get("hasStatistics")),
        "statistics_rows": len(stats),
        "periods": sorted({str(r.get("period")) for r in stats if isinstance(r, dict) and r.get("period")}),
        "stat_names": sorted({str(r.get("name")) for r in stats if isinstance(r, dict) and r.get("name")}),
        "fighter_all_values": all_values,
    }


def main() -> None:
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    discovery = discover_history(token)
    event_ids = discovered_event_ids(discovery)
    events = fetch_event_stats(token, event_ids)

    summary = {
        "fighter": FIGHTER,
        "discovery_records": len(discovery),
        "discovered_pre_ufc_event_ids": event_ids,
        "detailed_events_returned": len(events),
        "events_with_statistics": sum(bool(item.get("hasStatistics")) for item in events),
        "hard_charge_cap_usd": round(DISCOVERY_MAX_CHARGE_USD + STATS_MAX_CHARGE_USD, 2),
        "events": [summarize_event(item) for item in events],
    }

    (OUTPUT_DIR / "discovery.json").write_text(json.dumps(discovery, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "raw_events.json").write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
