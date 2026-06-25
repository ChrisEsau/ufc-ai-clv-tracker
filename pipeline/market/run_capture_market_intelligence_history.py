from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import (
    MARKET_INTELLIGENCE_HISTORY_AUDIT_PATH,
    MARKET_INTELLIGENCE_HISTORY_PATH,
    MARKET_OUTCOMES_PATH,
)
from pipeline.market.signals.history_schema import (
    ensure_market_intelligence_history_audit_columns,
    ensure_market_intelligence_history_columns,
)


def _now() -> tuple[str, str]:
    dt = datetime.now(timezone.utc)
    return dt.strftime("market_intel_%Y%m%d_%H%M%S"), dt.isoformat()


def _first_existing(row: pd.Series, names: list[str]):
    for name in names:
        value = row.get(name)
        if pd.notna(value) and str(value).strip():
            return value
    return None


def _fight_display(row: pd.Series) -> str:
    red = row.get("red_fighter")
    blue = row.get("blue_fighter")
    if pd.notna(red) and pd.notna(blue):
        return f"{red} vs {blue}"
    return str(_first_existing(row, ["fight_display", "event_name"]) or "Unknown fight")


def _market_display(row: pd.Series) -> str:
    value = _first_existing(row, ["market_display", "market_key"])
    return str(value or "Unknown market").replace("_", " ").title()


def _outcome_display(row: pd.Series) -> str:
    value = _first_existing(row, ["fighter_name", "provider_selection_name", "outcome_display", "side", "outcome_key"])
    return str(value or "Unknown outcome")


def build_history_rows(market_outcomes: pd.DataFrame, refresh_id: str, refresh_timestamp: str) -> pd.DataFrame:
    rows = []

    for _, row in market_outcomes.iterrows():
        rows.append(
            {
                "refresh_id": refresh_id,
                "refresh_timestamp": refresh_timestamp,
                "source_run_id": _first_existing(row, ["match_run_id", "snapshot_run_id", "market_run_id"]),
                "bookmaker": row.get("bookmaker"),
                "fight_id": row.get("fight_id"),
                "event_name": row.get("event_name"),
                "fight_display": _fight_display(row),
                "market_key": row.get("market_key"),
                "market_display": _market_display(row),
                "outcome_key": row.get("outcome_key"),
                "outcome_display": _outcome_display(row),
                "side": row.get("side"),
                "fighter_name": row.get("fighter_name"),
                "american_odds": row.get("american_odds"),
                "implied_probability": row.get("implied_probability"),
                "decimal_odds": row.get("decimal_odds"),
                "line": row.get("line"),
                "provider_event_id": row.get("provider_event_id"),
                "provider_market_id": row.get("provider_market_id"),
                "provider_selection_id": row.get("provider_selection_id"),
                "provider_market_type_name": row.get("provider_market_type_name"),
                "snapshot_source_path": str(MARKET_OUTCOMES_PATH),
            }
        )

    return ensure_market_intelligence_history_columns(pd.DataFrame(rows))


def main() -> None:
    print("=" * 80)
    print("CAPTURE MARKET INTELLIGENCE HISTORY")
    print("=" * 80)

    if not MARKET_OUTCOMES_PATH.exists():
        raise FileNotFoundError(f"Missing market outcomes: {MARKET_OUTCOMES_PATH}")

    refresh_id, refresh_timestamp = _now()
    market_outcomes = pd.read_parquet(MARKET_OUTCOMES_PATH)
    current = build_history_rows(market_outcomes, refresh_id, refresh_timestamp)

    MARKET_INTELLIGENCE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKET_INTELLIGENCE_HISTORY_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MARKET_INTELLIGENCE_HISTORY_PATH.exists():
        existing = pd.read_parquet(MARKET_INTELLIGENCE_HISTORY_PATH)
        combined = pd.concat(
            [
                ensure_market_intelligence_history_columns(existing),
                current,
            ],
            ignore_index=True,
        )
    else:
        combined = current

    combined = ensure_market_intelligence_history_columns(combined)
    combined.to_parquet(MARKET_INTELLIGENCE_HISTORY_PATH, index=False)

    audit = ensure_market_intelligence_history_audit_columns(
        pd.DataFrame(
            [
                {
                    "refresh_id": refresh_id,
                    "refresh_timestamp": refresh_timestamp,
                    "source_market_rows": len(market_outcomes),
                    "history_rows_appended": len(current),
                    "total_history_rows": len(combined),
                    "passes_validation": True,
                    "notes": "Append-only market intelligence history capture.",
                }
            ]
        )
    )
    audit.to_parquet(MARKET_INTELLIGENCE_HISTORY_AUDIT_PATH, index=False)

    print("Refresh ID:", refresh_id)
    print("Market rows:", len(market_outcomes))
    print("Rows appended:", len(current))
    print("Total history rows:", len(combined))
    print("Output:", MARKET_INTELLIGENCE_HISTORY_PATH)
    print("Audit:", MARKET_INTELLIGENCE_HISTORY_AUDIT_PATH)


if __name__ == "__main__":
    main()
