"""Low-cost one-fighter regional MMA pull for Terrance Chatman.

Research-only. Chatman's SofaScore fighter ID was resolved in the first one-fighter
probe. This follow-up performs one batched match search for his known pre-UFC bouts,
then one exact-event statistics pull. The Apify token is never persisted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

from pipeline.common.paths import AUDITS_DIR

FIGHTER = "Terrance Chatman"
FIGHTER_ID = 1159222
FIGHTER_SURNAME = "chatman"

# Maters was already resolved by the first one-fighter search.
KNOWN_EVENT_IDS = {"Dwight Maters": "15229560"}
MISSING_BOUT_QUERIES = {
    "Erick Prieto": "Terrance Chatman Erick Prieto",
    "Steven Gurrola": "Terrance Chatman Steven Gurrola",
    "Omar El-Sahlah": "Terrance Chatman Omar El-Sahlah",
    "Myron Dennis": "Terrance Chatman Myron Dennis",
    "Juan Torres": "Terrance Chatman Juan Torres",
}

MATCH_SEARCH_ACTOR_ID = "abotapi~sofascore-scraper"
STATS_ACTOR_ID = "automation-lab~sofascore-live-events-statistics-scraper"
MATCH_SEARCH_URL = f"https://api.apify.com/v2/actors/{MATCH_SEARCH_ACTOR_ID}/run-sync-get-dataset-items"
STATS_URL = f"https://api.apify.com/v2/actors/{STATS_ACTOR_ID}/run-sync-get-dataset-items"

OUTPUT_DIR = AUDITS_DIR / "regional_mma" / "sofascore" / "terrance_chatman"
MATCH_SEARCH_MAX_CHARGE_USD = 0.025
STATS_MAX_CHARGE_USD = 0.025


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


def search_missing_bouts(token: str) -> list[dict[str, Any]]:
    return _post_actor(
        MATCH_SEARCH_URL,
        token,
        {
            "mode": "search",
            "searchQueries": list(MISSING_BOUT_QUERIES.values()),
            "searchType": "match",
            "includeStatistics": False,
            "includeLineups": False,
            "includeIncidents": False,
            "includeOdds": False,
            "includeVotes": False,
            "includeStandings": False,
            "includeSquad": False,
            "maxItems": 10,
        },
        max_items=10,
        max_charge_usd=MATCH_SEARCH_MAX_CHARGE_USD,
    )


def _flat_team_name(item: dict[str, Any], side: str) -> str:
    value = item.get(f"{side}Team")
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value or "")


def _contains_name(text: str, name: str) -> bool:
    return name.casefold() in text.casefold()


def _valid_chatman_bout(item: dict[str, Any], opponent: str) -> bool:
    home = _flat_team_name(item, "home")
    away = _flat_team_name(item, "away")
    names = f"{home} {away} {item.get('name', '')}"
    return FIGHTER_SURNAME in names.casefold() and _contains_name(names, opponent)


def resolve_event_ids(search_items: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    resolved = dict(KNOWN_EVENT_IDS)
    evidence: list[dict[str, Any]] = []

    for opponent in MISSING_BOUT_QUERIES:
        candidates = [item for item in search_items if _valid_chatman_bout(item, opponent)]
        chosen = candidates[0] if candidates else None
        event_id = None
        if chosen is not None:
            event_id = chosen.get("id") or chosen.get("sofascoreId")
            if event_id is not None:
                resolved[opponent] = str(event_id)
        evidence.append(
            {
                "opponent": opponent,
                "matched": event_id is not None,
                "event_id": str(event_id) if event_id is not None else None,
                "candidate_count": len(candidates),
                "candidate_names": [str(item.get("name", "")) for item in candidates[:5]],
            }
        )

    return resolved, evidence


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


def _fighter_side(item: dict[str, Any]) -> str | None:
    if FIGHTER_SURNAME in _flat_team_name(item, "home").casefold():
        return "home"
    if FIGHTER_SURNAME in _flat_team_name(item, "away").casefold():
        return "away"
    return None


def summarize_event(item: dict[str, Any]) -> dict[str, Any]:
    stats = item.get("statistics") if isinstance(item.get("statistics"), list) else []
    side = _fighter_side(item)
    home = _flat_team_name(item, "home")
    away = _flat_team_name(item, "away")
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
        "fighter_all_values": all_values,
    }


def main() -> None:
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    search_items = search_missing_bouts(token)
    resolved, evidence = resolve_event_ids(search_items)
    event_ids = list(dict.fromkeys(resolved.values()))
    events = fetch_event_stats(token, event_ids)

    summary = {
        "fighter": FIGHTER,
        "fighter_sofascore_id": FIGHTER_ID,
        "target_pre_ufc_bouts": 6,
        "batched_search_records": len(search_items),
        "resolved_opponent_event_ids": resolved,
        "resolved_bout_count": len(resolved),
        "detailed_events_returned": len(events),
        "events_with_statistics": sum(bool(item.get("hasStatistics")) for item in events),
        "hard_charge_cap_usd": round(MATCH_SEARCH_MAX_CHARGE_USD + STATS_MAX_CHARGE_USD, 3),
        "search_evidence": evidence,
        "events": [summarize_event(item) for item in events],
    }

    (OUTPUT_DIR / "match_search.json").write_text(json.dumps(search_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "raw_events.json").write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
