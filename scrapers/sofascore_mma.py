"""Minimal SofaScore MMA JSON coverage probe.

This module is intentionally isolated from UFCStats ingestion and FSR. It discovers MMA
fights for a date, optionally filters by promotion, fetches event/statistics JSON, saves
raw payloads, and writes a flattened audit CSV for inspection.

It does not modify UFC master data or data/fight_details/ufc_round_stats.parquet.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from pipeline.common.paths import AUDITS_DIR


DEFAULT_BASE_URL = "https://www.sofascore.com/api/v1"
DEFAULT_OUTPUT_DIR = AUDITS_DIR / "regional_mma" / "sofascore"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class SofaScoreError(RuntimeError):
    """Raised when a SofaScore request cannot be completed safely."""


@dataclass(frozen=True)
class FetchResult:
    url: str
    payload: dict[str, Any] | None
    status_code: int


class SofaScoreMMAClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json,text/plain,*/*",
                "User-Agent": user_agent,
                "Referer": "https://www.sofascore.com/",
            }
        )

    def _get_json(self, path: str, *, allow_404: bool = False) -> FetchResult:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise SofaScoreError(f"Request failed for {url}: {exc}") from exc

        if response.status_code == 404 and allow_404:
            return FetchResult(url=url, payload=None, status_code=404)

        if response.status_code in {401, 403, 429}:
            raise SofaScoreError(
                f"SofaScore returned HTTP {response.status_code} for {url}. "
                "Stop rather than attempting to bypass access controls."
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:500].replace("\n", " ")
            raise SofaScoreError(
                f"HTTP {response.status_code} for {url}: {body}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            content_type = response.headers.get("content-type")
            body = response.text[:500].replace("\n", " ")
            raise SofaScoreError(
                f"Expected JSON from {url}, got content-type={content_type!r}: {body}"
            ) from exc

        if not isinstance(payload, dict):
            raise SofaScoreError(
                f"Expected JSON object from {url}, got {type(payload).__name__}."
            )
        return FetchResult(url=url, payload=payload, status_code=response.status_code)

    def scheduled_events(self, date_text: str) -> FetchResult:
        _validate_date(date_text)
        return self._get_json(f"sport/mma/scheduled-events/{date_text}")

    def event(self, event_id: int) -> FetchResult:
        return self._get_json(f"event/{int(event_id)}")

    def statistics(self, event_id: int) -> FetchResult:
        return self._get_json(f"event/{int(event_id)}/statistics", allow_404=True)


def _validate_date(value: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM-DD date, received {value!r}."
        ) from exc


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "unknown"


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def event_promotion_labels(event: dict[str, Any]) -> list[str]:
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") or {}
    labels = [
        tournament.get("name"),
        tournament.get("slug"),
        unique.get("name"),
        unique.get("slug"),
        _nested(tournament, "category", "name"),
    ]
    return [str(value) for value in labels if value not in (None, "")]


def matches_promotion(event: dict[str, Any], promotion: str | None) -> bool:
    if not promotion:
        return True
    needle = promotion.casefold().strip()
    return any(needle in label.casefold() for label in event_promotion_labels(event))


def is_finished(event: dict[str, Any]) -> bool:
    status_type = str(_nested(event, "status", "type") or "").casefold()
    return status_type in {"finished", "ended"}


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    tournament = event.get("tournament") or {}
    unique = tournament.get("uniqueTournament") or {}
    timestamp = event.get("startTimestamp")
    start_utc = None
    if isinstance(timestamp, (int, float)):
        start_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    return {
        "event_id": event.get("id"),
        "start_utc": start_utc,
        "status": _nested(event, "status", "type"),
        "tournament": tournament.get("name"),
        "promotion": _first_nonempty(unique.get("name"), tournament.get("name")),
        "promotion_slug": _first_nonempty(unique.get("slug"), tournament.get("slug")),
        "home_fighter": home.get("name"),
        "home_fighter_id": home.get("id"),
        "away_fighter": away.get("name"),
        "away_fighter_id": away.get("id"),
        "winner_code": event.get("winnerCode"),
    }


def iter_statistics_items(
    payload: dict[str, Any],
) -> Iterable[tuple[str | None, str | None, dict[str, Any]]]:
    periods = payload.get("statistics") or []
    if not isinstance(periods, list):
        return
    for period_block in periods:
        if not isinstance(period_block, dict):
            continue
        period = period_block.get("period")
        groups = period_block.get("groups") or []
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_name = group.get("groupName") or group.get("name")
            items = group.get("statisticsItems") or group.get("statistics") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    yield period, group_name, item


def flatten_statistics(
    event: dict[str, Any], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    base = event_summary(event)
    rows: list[dict[str, Any]] = []
    for period, group, item in iter_statistics_items(payload):
        rows.append(
            {
                **base,
                "period": period,
                "group": group,
                "stat_name": item.get("name"),
                "stat_key": item.get("key"),
                "home_display": item.get("home"),
                "away_display": item.get("away"),
                "home_value": item.get("homeValue"),
                "away_value": item.get("awayValue"),
                "compare_code": item.get("compareCode"),
                "statistics_type": item.get("statisticsType"),
            }
        )
    return rows


def stat_keys(payload: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    for _, _, item in iter_statistics_items(payload):
        key = item.get("key") or item.get("name")
        if key:
            keys.add(str(key))
    return sorted(keys)


def stat_periods(payload: dict[str, Any]) -> list[str]:
    periods: set[str] = set()
    for period, _, _ in iter_statistics_items(payload):
        if period:
            periods.add(str(period))
    return sorted(periods)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _event_dir(root: Path, event: dict[str, Any]) -> Path:
    summary = event_summary(event)
    event_id = summary.get("event_id") or "unknown"
    match = (
        f"{summary.get('home_fighter') or 'home'}-vs-"
        f"{summary.get('away_fighter') or 'away'}"
    )
    return root / f"{event_id}_{_slug(match)}"


def probe_event(
    client: SofaScoreMMAClient,
    event_id: int,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_result = client.event(event_id)
    event_payload = event_result.payload or {}
    event = (
        event_payload.get("event")
        if isinstance(event_payload.get("event"), dict)
        else event_payload
    )
    if not isinstance(event, dict):
        raise SofaScoreError(f"No event object returned for event {event_id}.")

    event_dir = _event_dir(output_root, event)
    _write_json(event_dir / "event.json", event_result.payload)

    stats_result = client.statistics(event_id)
    stats_payload = stats_result.payload
    has_statistics = bool(stats_payload and (stats_payload.get("statistics") or []))
    flat_rows: list[dict[str, Any]] = []

    if stats_payload is not None:
        _write_json(event_dir / "statistics.json", stats_payload)
        flat_rows = flatten_statistics(event, stats_payload)

    summary = event_summary(event)
    summary.update(
        {
            "event_url": event_result.url,
            "statistics_url": stats_result.url,
            "statistics_http_status": stats_result.status_code,
            "has_statistics": has_statistics,
            "statistics_rows": len(flat_rows),
            "periods": stat_periods(stats_payload or {}),
            "stat_keys": stat_keys(stats_payload or {}),
        }
    )
    _write_json(event_dir / "summary.json", summary)
    return summary, flat_rows


def probe_date(
    client: SofaScoreMMAClient,
    date_text: str,
    output_root: Path,
    *,
    promotion: str | None,
    finished_only: bool,
    max_events: int,
    sleep_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = client.scheduled_events(date_text)
    payload = result.payload or {}
    day_dir = output_root / date_text
    _write_json(day_dir / "scheduled_events.json", payload)

    events = payload.get("events") or []
    if not isinstance(events, list):
        raise SofaScoreError("scheduled-events payload did not contain an events list.")

    selected: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if not matches_promotion(event, promotion):
            continue
        if finished_only and not is_finished(event):
            continue
        selected.append(event)

    if max_events > 0:
        selected = selected[:max_events]

    summaries: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for index, event in enumerate(selected, start=1):
        event_id = event.get("id")
        if not isinstance(event_id, int):
            continue
        label = event_summary(event)
        print(
            f"[{index}/{len(selected)}] {event_id}: "
            f"{label.get('home_fighter')} vs {label.get('away_fighter')} "
            f"({label.get('promotion')})"
        )
        try:
            summary, event_rows = probe_event(client, event_id, day_dir)
        except SofaScoreError as exc:
            summary = {
                **label,
                "event_id": event_id,
                "error": str(exc),
                "has_statistics": False,
            }
            event_rows = []
            _write_json(day_dir / f"{event_id}_error.json", summary)
        summaries.append(summary)
        flat_rows.extend(event_rows)
        if sleep_seconds > 0 and index < len(selected):
            time.sleep(sleep_seconds)

    _write_json(day_dir / "coverage_summary.json", summaries)
    _write_csv(day_dir / "statistics_flat.csv", flat_rows)
    return summaries, flat_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe SofaScore MMA event/statistics JSON without touching UFC/FSR data."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--date", type=str, help="MMA event date in YYYY-MM-DD format.")
    target.add_argument("--event-id", type=int, help="Single SofaScore event ID.")
    parser.add_argument(
        "--promotion",
        type=str,
        default=None,
        help="Case-insensitive substring filter, e.g. LFA, Fury, Cage Warriors.",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Include scheduled/live fights when probing a date.",
    )
    parser.add_argument("--max-events", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--sleep-seconds", type=float, default=0.75)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.date:
        _validate_date(args.date)
    if args.max_events < 0:
        parser.error("--max-events must be >= 0")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds must be >= 0")

    client = SofaScoreMMAClient(
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    output_root: Path = args.output_dir

    if args.event_id is not None:
        summary, rows = probe_event(client, args.event_id, output_root)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"flattened statistic rows: {len(rows)}")
        return

    summaries, rows = probe_date(
        client,
        args.date,
        output_root,
        promotion=args.promotion,
        finished_only=not args.include_unfinished,
        max_events=args.max_events,
        sleep_seconds=args.sleep_seconds,
    )
    with_stats = sum(bool(row.get("has_statistics")) for row in summaries)
    print(
        f"selected fights: {len(summaries)} | "
        f"with statistics: {with_stats} | flattened statistic rows: {len(rows)}"
    )
    print(f"audit output: {output_root / args.date}")


if __name__ == "__main__":
    main()
