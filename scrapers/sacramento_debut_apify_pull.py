"""Pull pre-UFC regional MMA records for Sacramento 2026 debut fighters via Apify.

Research-only. This writes audit artifacts and does not touch FSR or UFC round stats.
The Apify token is read from the environment and is never persisted.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

from pipeline.common.paths import AUDITS_DIR

ACTOR_ID = "automation-lab~sofascore-live-events-statistics-scraper"
API_URL = f"https://api.apify.com/v2/actors/{ACTOR_ID}/run-sync-get-dataset-items"
OUTPUT_DIR = AUDITS_DIR / "regional_mma" / "sofascore" / "sacramento_debuts"
MAX_CHARGE_PER_LOOKUP_USD = 0.01

# Standard professional MMA only; Ryan Kuse's Karate Combat and Gamebred bare-knuckle
# bouts are intentionally excluded from this first cold-start pull.
TARGETS: dict[str, list[tuple[str, str]]] = {
    "Anthony Wint": [
        ("2023-04-14", "Luis Alfonso Garcia Diaz"),
        ("2024-06-14", "Jawaski Bethly"),
        ("2025-03-23", "Omar El-Sahlah"),
        ("2025-06-25", "Emmanuel Verdier"),
        ("2025-11-21", "Miles Banks"),
        ("2026-03-15", "Jamahl Tatum"),
        ("2026-08-11", "Matthew Adams"),
    ],
    "Terrance Chatman": [
        ("2025-02-21", "Erick Prieto"),
        ("2025-06-22", "Steven Gurrola"),
        ("2025-09-07", "Dwight Maters"),
        ("2026-01-18", "Omar El-Sahlah"),
        ("2026-04-12", "Myron Dennis"),
        ("2026-07-19", "Juan Torres"),
    ],
    "Ryan Kuse": [
        ("2020-08-28", "Earnest Walls"),
        ("2020-11-21", "Micaias Urena"),
        ("2021-03-26", "Damian Attie"),
        ("2021-08-06", "Amir Kazemi"),
        ("2021-09-17", "Rob Fuller"),
        ("2022-03-04", "Jamal Johnson"),
        ("2022-11-18", "Leonardo Morales"),
        ("2024-02-17", "Thiago Belo"),
        ("2024-04-21", "Cameron Smotherman"),
        ("2025-10-25", "Kevin Natividad"),
        ("2026-04-05", "Leandro Camargo"),
    ],
    "Stanley Dorsainvil": [
        ("2023-01-20", "Hernandez Banks"),
        ("2023-06-02", "Joao Guerra"),
        ("2024-07-12", "Cedric Katambwa"),
        ("2024-11-16", "Johnny Smith"),
        ("2026-05-22", "Daniel Holt"),
    ],
}


def run_lookup(token: str, fighter: str, date: str) -> list[dict[str, Any]]:
    payload = {
        "mode": "date",
        "sport": "mma",
        "date": date,
        "eventIds": [],
        "eventUrls": [],
        "includeStatistics": True,
        "status": "finished",
        "team": fighter,
        "tournament": "",
        "maxItems": 1,
    }
    response = requests.post(
        API_URL,
        params={"maxItems": 1, "maxTotalChargeUsd": MAX_CHARGE_PER_LOOKUP_USD},
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list from Apify for {fighter} {date}")
    return [item for item in data if isinstance(item, dict)]


def side_for_fighter(item: dict[str, Any], fighter: str) -> str | None:
    home = item.get("homeTeam") if isinstance(item.get("homeTeam"), dict) else {}
    away = item.get("awayTeam") if isinstance(item.get("awayTeam"), dict) else {}
    f = fighter.casefold()
    if f in str(home.get("name", "")).casefold():
        return "home"
    if f in str(away.get("name", "")).casefold():
        return "away"
    return None


def summarize_event(item: dict[str, Any], fighter: str, expected_opponent: str, target_date: str) -> dict[str, Any]:
    stats = item.get("statistics") if isinstance(item.get("statistics"), list) else []
    home = item.get("homeTeam") if isinstance(item.get("homeTeam"), dict) else {}
    away = item.get("awayTeam") if isinstance(item.get("awayTeam"), dict) else {}
    tournament = item.get("tournament") if isinstance(item.get("tournament"), dict) else {}
    side = side_for_fighter(item, fighter)
    actual_opponent = away.get("name") if side == "home" else home.get("name") if side == "away" else None
    names = sorted({str(r.get("name")) for r in stats if isinstance(r, dict) and r.get("name")})
    periods = sorted({str(r.get("period")) for r in stats if isinstance(r, dict) and r.get("period")})

    all_rows: dict[str, Any] = {}
    for row in stats:
        if not isinstance(row, dict) or row.get("period") not in {"ALL", "ALL15MIN"}:
            continue
        name = row.get("name")
        if not name:
            continue
        value = row.get(side) if side in {"home", "away"} else None
        all_rows[str(name)] = value

    return {
        "fighter": fighter,
        "target_date": target_date,
        "expected_opponent": expected_opponent,
        "event_id": item.get("eventId"),
        "start_time": item.get("startTime"),
        "tournament": tournament.get("name"),
        "home": home.get("name"),
        "away": away.get("name"),
        "actual_opponent": actual_opponent,
        "fighter_side": side,
        "status": item.get("status"),
        "has_statistics": bool(item.get("hasStatistics")),
        "statistics_rows": len(stats),
        "periods": periods,
        "stat_names": names,
        "fighter_all_period_values": all_rows,
    }


def main() -> None:
    token = os.environ.get("APIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("APIFY_TOKEN is not set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    total = sum(len(v) for v in TARGETS.values())
    index = 0
    for fighter, fights in TARGETS.items():
        for date, opponent in fights:
            index += 1
            print(f"[{index}/{total}] {fighter} vs {opponent} ({date})")
            try:
                items = run_lookup(token, fighter, date)
            except Exception as exc:
                errors.append({"fighter": fighter, "date": date, "opponent": opponent, "error": str(exc)})
                continue
            if not items:
                summaries.append({
                    "fighter": fighter,
                    "target_date": date,
                    "expected_opponent": opponent,
                    "matched": False,
                    "has_statistics": False,
                })
                continue
            item = items[0]
            raw_records.append({"fighter": fighter, "target_date": date, "expected_opponent": opponent, "record": item})
            summary = summarize_event(item, fighter, opponent, date)
            summary["matched"] = True
            summaries.append(summary)

    by_fighter: dict[str, dict[str, Any]] = defaultdict(lambda: {"targeted": 0, "matched": 0, "with_statistics": 0})
    for row in summaries:
        bucket = by_fighter[row["fighter"]]
        bucket["targeted"] += 1
        bucket["matched"] += int(bool(row.get("matched")))
        bucket["with_statistics"] += int(bool(row.get("has_statistics")))
    for err in errors:
        by_fighter[err["fighter"]]["targeted"] += 1

    report = {
        "scope": "Sacramento 2026 UFC debutants, pre-UFC standard professional MMA",
        "lookup_count": total,
        "max_charge_per_lookup_usd": MAX_CHARGE_PER_LOOKUP_USD,
        "theoretical_max_charge_usd": round(total * MAX_CHARGE_PER_LOOKUP_USD, 2),
        "by_fighter": dict(by_fighter),
        "fights": summaries,
        "errors": errors,
    }

    (OUTPUT_DIR / "raw.json").write_text(json.dumps(raw_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
