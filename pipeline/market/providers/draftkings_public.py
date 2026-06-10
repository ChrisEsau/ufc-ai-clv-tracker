# ============================================================
# pipeline/market/providers/draftkings_public.py
# ============================================================

"""DraftKings public market discovery adapter.

This module is intentionally limited to raw, read-only market discovery.
It does not perform UFCStats matching, canonical normalization, EV math,
CLV calculations, or betting decisions.

Approved scope:
- fetch a configured DraftKings public JSON URL,
- save the raw response under data/market/raw/draftkings/,
- flatten discovered markets/selections into a diagnostic dataframe,
- flag parlays, boosts, promos, and currently supported market families.

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

import pandas as pd
import requests

from ufc_pipeline_utils import american_to_decimal, american_to_implied_prob


BOOKMAKER = "DraftKings"
SOURCE = "draftkings_public"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = "ufc-ai-clv-tracker/market-discovery"

RAW_MARKET_COLUMNS = [
    "snapshot_run_id",
    "snapshot_timestamp",
    "source",
    "bookmaker",
    "provider_event_id",
    "event_name",
    "provider_market_id",
    "raw_market_name",
    "provider_selection_id",
    "raw_selection_name",
    "price_american",
    "price_decimal",
    "implied_probability",
    "line",
    "is_parlay",
    "is_boost",
    "is_promo",
    "is_supported_market",
    "supported_market_family",
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


def fetch_public_json(url: str, *, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Fetch a public DraftKings JSON endpoint.

    The caller must provide a URL discovered manually from a public DraftKings
    page. This function intentionally makes one normal request and fails on HTTP
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
) -> DraftKingsSnapshot:
    """Persist one raw DraftKings payload and return its snapshot metadata."""

    if snapshot_run_id is None or snapshot_timestamp is None:
        generated_run_id, generated_timestamp = utc_snapshot()
        snapshot_run_id = snapshot_run_id or generated_run_id
        snapshot_timestamp = snapshot_timestamp or generated_timestamp

    date_part = snapshot_timestamp[:10]
    raw_dir = raw_root / date_part
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_payload_path = raw_dir / f"snapshot_{snapshot_run_id}.json"
    raw_payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return DraftKingsSnapshot(
        snapshot_run_id=snapshot_run_id,
        snapshot_timestamp=snapshot_timestamp,
        raw_payload_path=raw_payload_path,
    )


def _walk_json(value: Any):
    """Yield every dictionary contained in a nested JSON-like object."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    """Return the first non-empty value from a provider dictionary."""

    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _looks_like_market(row: dict[str, Any]) -> bool:
    """Heuristic check for a DraftKings market-like dictionary."""

    has_market_name = _first_present(row, ["marketName", "name", "label", "title"]) is not None
    has_outcomes = any(isinstance(row.get(key), list) for key in ["outcomes", "selections", "participants"])
    return bool(has_market_name and has_outcomes)


def _selection_rows(market: dict[str, Any]) -> list[dict[str, Any]]:
    """Return provider selection dictionaries from common DraftKings fields."""

    for key in ["outcomes", "selections", "participants"]:
        value = market.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _to_american_odds(value: Any) -> Any:
    """Extract an American odds value from common provider fields."""

    if isinstance(value, dict):
        return _first_present(value, ["american", "americanOdds", "oddsAmerican", "displayOdds"])
    return value


def _market_flags(raw_market_name: Any, raw_selection_name: Any) -> dict[str, Any]:
    """Classify a raw market/selection for downstream discovery."""

    text = f"{raw_market_name or ''} {raw_selection_name or ''}".lower()

    is_parlay = bool(re.search(r"\b(parlay|same game|sgp|builder)\b", text))
    is_boost = "boost" in text
    is_promo = any(token in text for token in ["promo", "special", "featured"])

    supported_market_family = None
    if any(token in text for token in ["moneyline", "fight winner", "winner"]):
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


def flatten_market_diagnostics(
    payload: dict[str, Any],
    *,
    snapshot: DraftKingsSnapshot,
) -> pd.DataFrame:
    """Flatten all discovered DraftKings markets/selections for diagnostics.

    This is deliberately permissive because DraftKings response shapes can vary.
    Unknown markets are still preserved in the diagnostic table; canonical
    normalizers can be added later after real payloads are inspected.
    """

    rows: list[dict[str, Any]] = []

    for market in _walk_json(payload):
        if not _looks_like_market(market):
            continue

        raw_market_name = _first_present(market, ["marketName", "name", "label", "title"])
        provider_market_id = _first_present(market, ["marketId", "id", "dkMarketId"])
        provider_event_id = _first_present(market, ["eventId", "eventGroupId", "providerEventId"])
        event_name = _first_present(market, ["eventName", "name", "eventDescription"])

        for selection in _selection_rows(market):
            raw_selection_name = _first_present(selection, ["label", "name", "outcomeName", "participant", "title"])
            provider_selection_id = _first_present(selection, ["selectionId", "outcomeId", "id", "dkOutcomeId"])
            raw_odds = _first_present(selection, ["oddsAmerican", "americanOdds", "displayOdds", "odds"])
            price_american = _to_american_odds(raw_odds)
            line = _first_present(selection, ["line", "points", "handicap", "total"])
            flags = _market_flags(raw_market_name, raw_selection_name)

            rows.append(
                {
                    "snapshot_run_id": snapshot.snapshot_run_id,
                    "snapshot_timestamp": snapshot.snapshot_timestamp,
                    "source": SOURCE,
                    "bookmaker": BOOKMAKER,
                    "provider_event_id": provider_event_id,
                    "event_name": event_name,
                    "provider_market_id": provider_market_id,
                    "raw_market_name": raw_market_name,
                    "provider_selection_id": provider_selection_id,
                    "raw_selection_name": raw_selection_name,
                    "price_american": price_american,
                    "price_decimal": american_to_decimal(price_american),
                    "implied_probability": american_to_implied_prob(price_american),
                    "line": line,
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
