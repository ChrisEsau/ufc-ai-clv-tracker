from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ufc_pipeline_utils import american_to_decimal, american_to_implied_prob

BOOKMAKER = "FanDuel"
SOURCE = "fanduel_public"
DEFAULT_REGION = os.getenv("FANDUEL_REGION", "KS")
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_URL = (
    "https://api.sportsbook.fanduel.com/sbapi/content-managed-page"
    "?page=SPORT&eventTypeId=26420387&_ak=FhMFpcPWXMeyZxOx&timezone=America%2FChicago"
)

RAW_MARKET_COLUMNS = [
    "snapshot_run_id", "snapshot_timestamp", "source", "bookmaker",
    "provider_event_id", "event_name", "event_start_timestamp",
    "provider_competition_id", "provider_competition_name",
    "provider_market_id", "raw_market_name", "provider_market_type_name",
    "provider_selection_id", "raw_selection_name", "selection_outcome_type",
    "selection_participant_name", "selection_participant_venue_role",
    "price_american", "price_decimal", "provider_decimal_odds", "true_odds",
    "implied_probability", "line", "is_parlay", "is_boost", "is_promo",
    "is_supported_market", "supported_market_family", "request_url",
    "raw_payload_path",
]

EVENT_INDEX_COLUMNS = [
    "provider_event_id", "provider_event_name", "provider_start_time",
    "provider_competition_id", "provider_competition_name",
    "participant_home", "participant_away", "request_url",
    "snapshot_timestamp", "snapshot_run_id",
]


@dataclass(frozen=True)
class FanDuelSnapshot:
    snapshot_run_id: str
    snapshot_timestamp: str
    raw_payload_path: Path


def utc_snapshot() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("fanduel_%Y%m%d_%H%M%S"), now.isoformat()


def fetch_public_json(url: str = DEFAULT_URL, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    # Region header is required. Do not add cookies, login headers, or x-px-context.
    headers = {
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "origin": "https://sportsbook.fanduel.com",
        "referer": "https://sportsbook.fanduel.com/",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "x-sportsbook-region": DEFAULT_REGION,
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def save_raw_snapshot(payload: dict[str, Any], *, raw_root: Path, request_label: str = "mma") -> FanDuelSnapshot:
    snapshot_run_id, snapshot_timestamp = utc_snapshot()
    raw_dir = raw_root / snapshot_timestamp[:10]
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_payload_path = raw_dir / f"snapshot_{snapshot_run_id}_{request_label}.json"
    raw_payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return FanDuelSnapshot(snapshot_run_id, snapshot_timestamp, raw_payload_path)


def _attachments(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("attachments", {}) if isinstance(payload.get("attachments"), dict) else {}


def _market_family(market_type: Any, market_name: Any) -> str | None:
    text = f"{market_type or ''} {market_name or ''}".lower()

    if "match_betting" in text or "moneyline" in text:
        return "moneyline"
    if "method_of_victory" in text:
        return "fighter_method_props"
    if "will_the_fight_go_the_distance" in text or "go the distance" in text:
        return "goes_distance"
    if "total_rounds" in text:
        return "total_rounds"
    if "round_betting" in text or "what_round" in text:
        return "round_betting"
    if "how_fight_will_end" in text:
        return "exact_method"
    if "method_&_round_combo" in text:
        return "method_round_combo"
    if "ko/tko_round_combos" in text:
        return "ko_tko_round_combo"
    if "submission_round_combos" in text:
        return "submission_round_combo"
    if "decision_no_bet" in text:
        return "decision_no_bet"
    if "double_chance" in text:
        return "double_chance"
    if "gone_in_60_seconds" in text:
        return "gone_in_60_seconds"
    if "winning_minute" in text:
        return "winning_minute"
    if "winning_round_&_minute" in text:
        return "winning_round_minute"

    return None


def _american_odds(runner: dict[str, Any]) -> Any:
    odds = runner.get("winRunnerOdds", {})
    display = odds.get("americanDisplayOdds", {}) if isinstance(odds, dict) else {}
    return display.get("americanOddsInt", display.get("americanOdds"))


def _decimal_odds(runner: dict[str, Any]) -> Any:
    odds = runner.get("winRunnerOdds", {})
    true_odds = odds.get("trueOdds", {}) if isinstance(odds, dict) else {}
    decimal = true_odds.get("decimalOdds", {}) if isinstance(true_odds, dict) else {}
    return decimal.get("decimalOdds")


def build_event_index(payload: dict[str, Any], *, request_url: str, snapshot: FanDuelSnapshot) -> pd.DataFrame:
    attachments = _attachments(payload)
    events = attachments.get("events", {}) or {}
    competitions = attachments.get("competitions", {}) or {}

    rows = []
    for event_id, event in events.items():
        if not isinstance(event, dict):
            continue
        competition_id = str(event.get("competitionId", ""))
        competition = competitions.get(competition_id, {}) or competitions.get(int(competition_id), {}) if competition_id.isdigit() else {}
        name = event.get("name") or ""
        parts = [x.strip() for x in name.split(" v ", 1)]
        rows.append({
            "provider_event_id": str(event.get("eventId") or event_id),
            "provider_event_name": name,
            "provider_start_time": event.get("openDate"),
            "provider_competition_id": event.get("competitionId"),
            "provider_competition_name": competition.get("name"),
            "participant_home": parts[0] if len(parts) == 2 else None,
            "participant_away": parts[1] if len(parts) == 2 else None,
            "request_url": request_url,
            "snapshot_timestamp": snapshot.snapshot_timestamp,
            "snapshot_run_id": snapshot.snapshot_run_id,
        })

    return pd.DataFrame(rows).reindex(columns=EVENT_INDEX_COLUMNS)


def flatten_market_diagnostics(payload: dict[str, Any], *, request_url: str, snapshot: FanDuelSnapshot) -> pd.DataFrame:
    attachments = _attachments(payload)
    events = attachments.get("events", {}) or {}
    markets = attachments.get("markets", {}) or {}
    competitions = attachments.get("competitions", {}) or {}

    rows = []
    for market_id, market in markets.items():
        if not isinstance(market, dict):
            continue

        event_id = str(market.get("eventId", ""))
        event = events.get(event_id, {}) or events.get(int(event_id), {}) if event_id.isdigit() else {}
        competition_id = str(market.get("competitionId", ""))
        competition = competitions.get(competition_id, {}) or competitions.get(int(competition_id), {}) if competition_id.isdigit() else {}

        market_type = market.get("marketType")
        market_name = market.get("marketName")
        family = _market_family(market_type, market_name)

        for runner in market.get("runners", []) or []:
            if not isinstance(runner, dict):
                continue

            price_american = _american_odds(runner)
            provider_decimal = _decimal_odds(runner)
            result = runner.get("result", {}) if isinstance(runner.get("result"), dict) else {}

            rows.append({
                "snapshot_run_id": snapshot.snapshot_run_id,
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "source": SOURCE,
                "bookmaker": BOOKMAKER,
                "provider_event_id": event_id,
                "event_name": event.get("name"),
                "event_start_timestamp": event.get("openDate") or market.get("marketTime"),
                "provider_competition_id": market.get("competitionId"),
                "provider_competition_name": competition.get("name"),
                "provider_market_id": market.get("marketId") or market_id,
                "raw_market_name": market_name,
                "provider_market_type_name": market_type,
                "provider_selection_id": runner.get("selectionId"),
                "raw_selection_name": runner.get("runnerName"),
                "selection_outcome_type": result.get("type"),
                "selection_participant_name": runner.get("runnerName"),
                "selection_participant_venue_role": result.get("type"),
                "price_american": price_american,
                "price_decimal": american_to_decimal(price_american),
                "provider_decimal_odds": provider_decimal,
                "true_odds": provider_decimal,
                "implied_probability": american_to_implied_prob(price_american),
                "line": runner.get("handicap"),
                "is_parlay": False,
                "is_boost": False,
                "is_promo": False,
                "is_supported_market": family is not None,
                "supported_market_family": family,
                "request_url": request_url,
                "raw_payload_path": str(snapshot.raw_payload_path),
            })

    return pd.DataFrame(rows).reindex(columns=RAW_MARKET_COLUMNS)
