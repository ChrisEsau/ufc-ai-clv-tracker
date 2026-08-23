"""Small Apify-backed SofaScore MMA coverage probe.

Retrieval only. The Apify token is read from the environment and is never persisted.
Outputs are written under data/audits/regional_mma/sofascore/ for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from pipeline.common.paths import AUDITS_DIR

ACTOR_ID = "automation-lab~sofascore-live-events-statistics-scraper"
API_URL = f"https://api.apify.com/v2/actors/{ACTOR_ID}/run-sync-get-dataset-items"
DEFAULT_OUTPUT = AUDITS_DIR / "regional_mma" / "sofascore" / "apify_lfa_probe.json"


def run_probe(
    *,
    token: str,
    date: str,
    tournament: str,
    max_items: int,
    max_charge_usd: float,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    payload = {
        "mode": "date",
        "sport": "mma",
        "date": date,
        "eventIds": [],
        "eventUrls": [],
        "includeStatistics": True,
        "status": "finished",
        "team": "",
        "tournament": tournament,
        "maxItems": max_items,
    }
    response = requests.post(
        API_URL,
        params={
            "maxItems": max_items,
            "maxTotalChargeUsd": max_charge_usd,
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1000].replace("\n", " ")
        raise RuntimeError(
            f"Apify Actor request failed with HTTP {response.status_code}: {body}"
        ) from exc

    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Expected Apify dataset list, got {type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for item in items:
        stats = item.get("statistics")
        stat_rows = stats if isinstance(stats, list) else []
        stat_names = sorted(
            {
                str(row.get("name"))
                for row in stat_rows
                if isinstance(row, dict) and row.get("name")
            }
        )
        periods = sorted(
            {
                str(row.get("period"))
                for row in stat_rows
                if isinstance(row, dict) and row.get("period")
            }
        )
        tournament = item.get("tournament") if isinstance(item.get("tournament"), dict) else {}
        home = item.get("homeTeam") if isinstance(item.get("homeTeam"), dict) else {}
        away = item.get("awayTeam") if isinstance(item.get("awayTeam"), dict) else {}
        events.append(
            {
                "event_id": item.get("eventId"),
                "start_time": item.get("startTime"),
                "tournament": tournament.get("name"),
                "home": home.get("name"),
                "away": away.get("name"),
                "status": item.get("status"),
                "has_statistics": item.get("hasStatistics"),
                "statistics_rows": len(stat_rows),
                "periods": periods,
                "stat_names": stat_names,
            }
        )
    return {"event_count": len(items), "events": events}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--tournament", default="LFA")
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--max-charge-usd", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set")
    if args.max_items < 1:
        raise ValueError("--max-items must be >= 1")

    items = run_probe(
        token=token,
        date=args.date,
        tournament=args.tournament,
        max_items=args.max_items,
        max_charge_usd=args.max_charge_usd,
        timeout_seconds=args.timeout_seconds,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = summarize(items)
    summary_path = args.output.with_name("apify_lfa_probe_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
