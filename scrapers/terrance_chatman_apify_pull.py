"""Low-cost one-fighter regional MMA pull for Terrance Chatman.

Research-only. Chatman's SofaScore fighter ID was resolved in the first one-fighter
probe. This follow-up uses the same working search Actor for five precise bout-name
queries, then one exact-event statistics pull. The Apify token is never persisted.
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

KNOWN_EVENT_IDS = {"Dwight Maters": "15229560"}
MISSING_BOUT_QUERIES = {
    "Erick Prieto": "Terrance Chatman Erick Prieto",
    "Steven Gurrola": "Terrance Chatman Steven Gurrola",
    "Omar El-Sahlah": "Terrance Chatman Omar El-Sahlah",
    "Myron Dennis": "Terrance Chatman Myron Dennis",
    "Juan Torres": "Terrance Chatman Juan Torres",
}

SEARCH_ACTOR_ID = "gio21~sofascore-scraper"
STATS_ACTOR_ID = "automation-lab~sofascore-live-events-statistics-scraper"
SEARCH_URL = f"https://api.apify.com/v2/actors/{SEARCH_ACTOR_ID}/run-sync-get-dataset-items"
STATS_URL = f"https://api.apify.com/v2/actors/{STATS_ACTOR_ID}/run-sync-get-dataset-items"

OUTPUT_DIR = AUDITS_DIR / "regional_mma" / "sofascore" / "terrance_chatman"
SEARCH_MAX_ITEMS = 2
SEARCH_MAX_CHARGE_PER_BOUT_USD = 0.01
STATS_MAX_CHARGE_USD = 0.02


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


def _flat_team_name(item: dict[str, Any], side: str) -> str:
    value = item.get(f"{side}Team")
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return str(value or "")


def _candidate_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("name", "")),
            _flat_team_name(item, "home"),
            _flat_team_name(item, "away"),
            str(item.get("slug", "")),
        ]
    ).casefold()


def search_one_bout(token: str, opponent: str, query: str) -> tuple[str | None, list[dict[str, Any]]]:
    items = _post_actor(
        SEARCH_URL,
        token,
        {
            "searchTerm": query,
            "includeMatches": False,
            "maxItems": SEARCH_MAX_ITEMS,
        },
        max_items=SEARCH_MAX_ITEMS,
        max_charge_usd=SEARCH_MAX_CHARGE_PER_BOUT_USD,
    )

    opponent_last = opponent.split()[-1].casefold()
    for item in items:
        text = _candidate_text(item)
        if FIGHTER_SURNAME not in text or opponent_last not in text:
            continue
        event_id = item.get("sofascoreId") or item.get("id")
        if event_id is not None:
            return str(event_id), items
    return None, items


def resolve_event_ids(token: str) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved = dict(KNOWN_EVENT_IDS)
    evidence: list[dict[str, Any]] = []
    all_search_items: list[dict[str, Any]] = []

    for opponent, query in MISSING_BOUT_QUERIES.items():
        try:
            event_id, items = search_one_bout(token, opponent, query)
            all_search_items.extend(items)
            evidence.append(
                {
                    "opponent": opponent,
                    "query": query,
                    "returned": len(items),
                    "event_id": event_id,
                    "candidate_names": [str(item.get("name", "")) for item in items],
                }
            )
            if event_id is not None:
                resolved[opponent] = event_id
        except Exception as exc:
            evidence.append({"opponent": opponent, "query": query, "error": str(exc)})

    return resolved, evidence, all_search_items


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

    resolved, evidence, search_items = resolve_event_ids(token)
    event_ids = list(dict.fromkeys(resolved.values()))
    events = fetch_event_stats(token, event_ids)

    summary = {
        "fighter": FIGHTER,
        "fighter_sofascore_id": FIGHTER_ID,
        "target_pre_ufc_bouts": 6,
        "resolved_opponent_event_ids": resolved,
        "resolved_bout_count": len(resolved),
        "detailed_events_returned": len(events),
        "events_with_statistics": sum(bool(item.get("hasStatistics")) for item in events),
        "hard_charge_cap_usd": round(len(MISSING_BOUT_QUERIES) * SEARCH_MAX_CHARGE_PER_BOUT_USD + STATS_MAX_CHARGE_USD, 2),
        "search_evidence": evidence,
        "events": [summarize_event(item) for item in events],
    }

    (OUTPUT_DIR / "match_search.json").write_text(json.dumps(search_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "raw_events.json").write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
