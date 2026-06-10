# ============================================================
# pipeline/market/providers/draftkings_public.py
# ============================================================

"""DraftKings public market discovery adapter.

This module is intentionally limited to raw, read-only market discovery.
It does not perform UFCStats matching, canonical normalization, EV math,
CLV calculations, or betting decisions.

Approved scope:
- build known public DraftKings event/subcategory JSON URLs,
- fetch public JSON responses,
- save raw responses under data/market/raw/draftkings/,
- flatten discovered events/markets/selections into a diagnostic dataframe,
- attach provider registry metadata when available,
- flag parlays, boosts, promos, and currently recognized market families.

No bypassing, login automation, proxy rotation, CAPTCHA handling, or ban evasion
logic belongs in this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from ufc_pipeline_utils import american_to_decimal, american_to_implied_prob


BOOKMAKER = "DraftKings"
SOURCE = "draftkings_public"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = "ufc-ai-clv-tracker/market-discovery"
DEFAULT_BASE_URL = "https://sportsbook-nash.draftkings.com/sites/US-KS-SB/api/sportscontent"
EVENT_ENDPOINT = "/pagedata/event/v1/events"
EVENT_SUBCATEGORY_MARKETS_ENDPOINT = "/controldata/event/eventSubcategory/v1/markets"

RAW_MARKET_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "event_name",
    "event_start_timestamp",
    "provider_sport_id",
    "provider_league_id",
    "provider_subcategory_id",
    "provider_subcategory_name",
    "registry_family",
    "registry_outcome_type",
    "provider_market_id",
    "raw_market_name",
    "provider_market_type_id",
    "provider_market_type_name",
    "provider_market_tags",
    "provider_selection_id",
    "raw_selection_name",
    "selection_outcome_type",
    "selection_participant_name",
    "selection_participant_sdid",
    "selection_participant_venue_role",
    "price_american",
    "price_decimal",
    "provider_decimal_odds",
    "true_odds",
    "implied_probability",
    "line",
    "bet_percent",
    "handle_percent",
    "is_parlay",
    "is_boost",
    "is_promo",
    "is_supported_market",
    "supported_market_family",
    "request_url",
    "raw_payload_path",
]


@dataclass(frozen=True)
class DraftKingsSnapshot:
    """Metadata describing one saved DraftKings raw response."""

    snapshot_run_id: str
    snapshot_timestamp: str
    raw_payload_path: Path


def utc_snapshot() -> tuple[str, str]:
    """Return a stable run id and ISO timestamp for one market snapshot."""

    now = datetime.now(timezone.utc)
    return now.strftime("draftkings_%Y%m%d_%H%M%S"), now.isoformat()


def build_event_url(event_id: str, *, base_url: str = DEFAULT_BASE_URL) -> str:
    """Build the DraftKings public event metadata URL for one event id."""

    query = urlencode({"eventIds": str(event_id)})
    return f"{base_url}{EVENT_ENDPOINT}?{query}"


def build_event_subcategory_markets_url(
    event_id: str,
    subcategory_id: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Build the DraftKings public market URL for one event/subcategory pair."""

    event_id = str(event_id)
    subcategory_id = str(subcategory_id)
    markets_query = (
        f"$filter=eventId eq '{event_id}' "
        f"AND clientMetadata/subCategoryId eq '{subcategory_id}' "
        "AND tags/all(t: t ne 'SportcastBetBuilder') "
        "and tags/any(t: t eq 'OSB')"
    )
    query = urlencode(
        {
            "templateVars": f"{event_id},{subcategory_id}",
            "marketsQuery": markets_query,
            "entity": "markets",
        }
    )
    return f"{base_url}{EVENT_SUBCATEGORY_MARKETS_ENDPOINT}?{query}"


def fetch_public_json(url: str, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch a public DraftKings JSON endpoint.

    This function intentionally makes one normal request and fails on HTTP
    errors rather than retrying aggressively.
    """

    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def save_raw_snapshot(
    payload: dict[str, Any],
    *,
    raw_root: Path = Path("data/market/raw/draftkings"),
    snapshot_run_id: str | None = None,
    snapshot_timestamp: str | None = None,
    event_id: str | None = None,
    subcategory_id: str | None = None,
) -> DraftKingsSnapshot:
    """Persist one raw DraftKings payload and return its snapshot metadata."""

    if snapshot_run_id is None or snapshot_timestamp is None:
        generated_run_id, generated_timestamp = utc_snapshot()
        snapshot_run_id = snapshot_run_id or generated_run_id
        snapshot_timestamp = snapshot_timestamp or generated_timestamp

    date_part = snapshot_timestamp[:10]
    raw_dir = raw_root / date_part
    if event_id:
        raw_dir = raw_dir / f"event_{event_id}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    suffix_parts = [snapshot_run_id]
    if subcategory_id:
        suffix_parts.append(f"subcategory_{subcategory_id}")
    raw_payload_path = raw_dir / f"snapshot_{'_'.join(suffix_parts)}.json"
    raw_payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return DraftKingsSnapshot(
        snapshot_run_id=snapshot_run_id,
        snapshot_timestamp=snapshot_timestamp,
        raw_payload_path=raw_payload_path,
    )


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    """Return the first non-empty value from a provider dictionary."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _to_american_odds(value: Any) -> Any:
    """Extract an American odds value from common provider fields."""

    if isinstance(value, dict):
        return _first_present(value, ["american", "americanOdds", "oddsAmerican", "displayOdds"])
    return value


def _clean_american_odds(value: Any) -> Any:
    """Normalize DraftKings unicode minus odds strings for downstream math."""

    if value is None:
        return None
    if isinstance(value, str):
        return value.replace("−", "-").strip()
    return value


def _market_flags(raw_market_name: Any, raw_selection_name: Any) -> dict[str, Any]:
    """Classify a raw market/selection for downstream discovery."""

    text = f"{raw_market_name or ''} {raw_selection_name or ''}".lower()

    is_parlay = bool(re.search(r"\b(parlay|same game|sgp|builder)\b", text))
    is_boost = "boost" in text
    is_promo = any(token in text for token in ["promo", "special", "featured"])

    supported_market_family = None
    if "significant strike" in text:
        supported_market_family = "fighter_sig_strikes_total"
    elif "finish only moneyline" in text:
        supported_market_family = "finish_only_moneyline"
    elif "decision only moneyline" in text:
        supported_market_family = "decision_only_moneyline"
    elif "submission only moneyline" in text:
        supported_market_family = "submission_only_moneyline"
    elif "ko/tko/dq only moneyline" in text:
        supported_market_family = "ko_tko_only_moneyline"
    elif "round 1 only moneyline" in text:
        supported_market_family = "round_1_only_moneyline"
    elif "round and method" in text:
        supported_market_family = "round_method"
    elif "exact method" in text:
        supported_market_family = "exact_method"
    elif "alternate point spread" in text:
        supported_market_family = "alternate_point_spread"
    elif "point spread" in text:
        supported_market_family = "point_spread"
    elif any(token in text for token in ["moneyline", "fight winner", "winner"]):
        supported_market_family = "moneyline"
    elif "go" in text and "distance" in text:
        supported_market_family = "goes_distance"
    elif "total" in text and "round" in text:
        supported_market_family = "over_under_rounds"
    elif any(token in text for token in ["ko/tko", "ko", "tko", "knockout"]):
        supported_market_family = "ko_tko"
    elif "submission" in text:
        supported_market_family = "submission"
    elif "decision" in text:
        supported_market_family = "decision"
    elif "exact round" in text:
        supported_market_family = "exact_round"
    elif "round" in text and "win" in text:
        supported_market_family = "fighter_round_win"

    return {
        "is_parlay": is_parlay,
        "is_boost": is_boost,
        "is_promo": is_promo,
        "is_supported_market": supported_market_family is not None,
        "supported_market_family": supported_market_family,
    }


def _first_participant(selection: dict[str, Any]) -> dict[str, Any]:
    participants = selection.get("participants")
    if isinstance(participants, list) and participants and isinstance(participants[0], dict):
        return participants[0]
    return {}


def flatten_market_diagnostics(
    payload: dict[str, Any],
    *,
    snapshot: DraftKingsSnapshot,
    request_url: str | None = None,
    registry_entry: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Flatten DraftKings event/market/selection payloads for diagnostics.

    Real DraftKings UFC payloads expose top-level arrays:

        events[]
        markets[]
        selections[]

    The diagnostic grain is one provider market selection row.
    """

    events = {
        str(event.get("id")): event
        for event in payload.get("events", [])
        if isinstance(event, dict) and event.get("id") is not None
    }
    markets = {
        str(market.get("id")): market
        for market in payload.get("markets", [])
        if isinstance(market, dict) and market.get("id") is not None
    }
    selections = [
        selection
        for selection in payload.get("selections", [])
        if isinstance(selection, dict)
    ]

    rows: list[dict[str, Any]] = []
    registry_entry = registry_entry or {}

    for selection in selections:
        market_id = str(selection.get("marketId"))
        market = markets.get(market_id, {})
        event_id = str(market.get("eventId") or selection.get("eventId") or "")
        event = events.get(event_id, {})

        raw_market_name = market.get("name")
        market_type = market.get("marketType") if isinstance(market.get("marketType"), dict) else {}
        raw_selection_name = selection.get("label")
        display_odds = selection.get("displayOdds") if isinstance(selection.get("displayOdds"), dict) else {}
        price_american = _clean_american_odds(_to_american_odds(display_odds))
        provider_decimal_odds = display_odds.get("decimal")
        participant = _first_participant(selection)
        participant_metadata = participant.get("metadata") if isinstance(participant.get("metadata"), dict) else {}
        selection_metadata = selection.get("metadata") if isinstance(selection.get("metadata"), dict) else {}
        flags = _market_flags(raw_market_name, raw_selection_name)

        rows.append(
            {
                "snapshot_run_id": snapshot.snapshot_run_id,
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "source": SOURCE,
                "bookmaker": BOOKMAKER,
                "provider_event_id": event_id or pd.NA,
                "event_name": event.get("name"),
                "event_start_timestamp": event.get("startEventDate"),
                "provider_sport_id": market.get("sportId") or event.get("sportId"),
                "provider_league_id": market.get("leagueId") or event.get("leagueId"),
                "provider_subcategory_id": market.get("subcategoryId") or registry_entry.get("subcategory_id"),
                "provider_subcategory_name": registry_entry.get("name"),
                "registry_family": registry_entry.get("family"),
                "registry_outcome_type": registry_entry.get("outcome_type"),
                "provider_market_id": market.get("id"),
                "raw_market_name": raw_market_name,
                "provider_market_type_id": market_type.get("id"),
                "provider_market_type_name": market_type.get("name"),
                "provider_market_tags": market.get("tags"),
                "provider_selection_id": selection.get("id"),
                "raw_selection_name": raw_selection_name,
                "selection_outcome_type": selection.get("outcomeType"),
                "selection_participant_name": participant.get("name"),
                "selection_participant_sdid": participant_metadata.get("sdid"),
                "selection_participant_venue_role": participant.get("venueRole"),
                "price_american": price_american,
                "price_decimal": american_to_decimal(price_american),
                "provider_decimal_odds": provider_decimal_odds,
                "true_odds": selection.get("trueOdds"),
                "implied_probability": american_to_implied_prob(price_american),
                "line": _first_present(selection, ["line", "points", "handicap", "total"]),
                "bet_percent": selection_metadata.get("betPercent"),
                "handle_percent": selection_metadata.get("handlePercent"),
                "request_url": request_url,
                "raw_payload_path": str(snapshot.raw_payload_path),
                **flags,
            }
        )

    return ensure_raw_market_columns(pd.DataFrame(rows))


def ensure_raw_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stable diagnostic schema even when no markets are found."""

    out = df.copy()
    for column in RAW_MARKET_COLUMNS:
        if column not in out.columns:
            out[column] = pd.Series(dtype="object")
    return out[RAW_MARKET_COLUMNS]
