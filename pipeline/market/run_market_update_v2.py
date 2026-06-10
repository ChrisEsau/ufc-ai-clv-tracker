# ============================================================
# pipeline/market/run_market_update_v2.py
# ============================================================

"""Market Pipeline V2 runner.

Phase 10A scope:
- Moneyline only operationally.
- Outcome-based artifacts.
- ID-based join contract using fight_id + market_key + outcome_fighter_id.
- Legacy market artifacts remain untouched.

Diagnostic extension:
- Raw provider market/outcome rows are saved before normalization so newly
  configured markets can be inspected before they are promoted into canonical
  market_outcomes.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from pipeline.common.paths import (
    LIVE_CARD_PATH,
    MARKET_MATCH_AUDIT_V2_PATH,
    MARKET_OUTCOME_AUDIT_PATH,
    MARKET_OUTCOME_SNAPSHOTS_PATH,
    MARKET_OUTCOMES_PATH,
    MARKET_REGISTRY_PATH,
    ensure_data_dirs,
)
from pipeline.market.market_validator import validate_market_outcomes
from pipeline.market.normalizers.moneyline import ensure_market_outcome_columns, normalize_moneyline_provider_row
from pipeline.market.outcome_matcher import build_match_audit_row, ensure_match_audit_columns, match_provider_row_to_live_card
from pipeline.market.providers.the_odds_api import (
    fetch_odds,
    flatten_moneyline_odds,
    flatten_provider_market_diagnostics,
)


DEFAULT_PROVIDER_MARKET_DIAGNOSTIC_PATH = Path("data/market/provider_market_diagnostic.parquet")


def _load_project_odds_key() -> str | None:
    """Load the configured odds provider key without hardcoding secrets."""

    env_value = os.getenv("ODDS_API_KEY")
    if env_value:
        return env_value

    try:
        from pipeline_config import ODDS_API_KEY as configured_value
    except Exception:
        return None

    return configured_value


def _utc_snapshot() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("market_%Y%m%d_%H%M%S"), now.isoformat()


def _load_market_config(path: Path = MARKET_REGISTRY_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Market registry not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _provider_market_diagnostic_path(config: dict) -> Path:
    outputs = config.get("outputs", {}) or {}
    return Path(outputs.get("provider_market_diagnostic_path") or DEFAULT_PROVIDER_MARKET_DIAGNOSTIC_PATH)


def _load_live_card(path: Path = LIVE_CARD_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Live card not found: {path}")

    live_card_df = pd.read_parquet(path)
    required_columns = [
        "fight_id",
        "event_name",
        "red_fighter",
        "blue_fighter",
        "red_fighter_id",
        "blue_fighter_id",
    ]
    missing_columns = [column for column in required_columns if column not in live_card_df.columns]
    if missing_columns:
        raise ValueError(f"Live card missing required Market V2 columns: {missing_columns}")

    return live_card_df.dropna(subset=["fight_id", "red_fighter", "blue_fighter"]).copy()


def _append_snapshot_history(latest_df: pd.DataFrame, snapshot_path: Path = MARKET_OUTCOME_SNAPSHOTS_PATH) -> pd.DataFrame:
    if snapshot_path.exists():
        existing_df = pd.read_parquet(snapshot_path)
        return pd.concat([existing_df, latest_df], ignore_index=True)

    return latest_df.copy()


def _build_market_outcomes(
    *,
    provider_df: pd.DataFrame,
    live_card_df: pd.DataFrame,
    snapshot_run_id: str,
    snapshot_timestamp: str,
    min_single_score: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_rows = []
    match_audit_rows = []

    for _, provider_row in provider_df.iterrows():
        match = match_provider_row_to_live_card(
            provider_row=provider_row,
            live_card_df=live_card_df,
            min_single_score=min_single_score,
        )

        match_audit_rows.append(
            build_match_audit_row(
                provider_row=provider_row,
                match=match,
                snapshot_run_id=snapshot_run_id,
                snapshot_timestamp=snapshot_timestamp,
            )
        )

        if match is None:
            continue

        market_rows.extend(
            normalize_moneyline_provider_row(
                provider_row=provider_row,
                match=match,
                snapshot_run_id=snapshot_run_id,
                snapshot_timestamp=snapshot_timestamp,
            )
        )

    return (
        ensure_market_outcome_columns(pd.DataFrame(market_rows)),
        ensure_match_audit_columns(pd.DataFrame(match_audit_rows)),
    )


def main() -> None:
    print("=" * 80)
    print("UFC MARKET PIPELINE V2")
    print("=" * 80)

    ensure_data_dirs()

    provider_key = _load_project_odds_key()
    if not provider_key:
        raise RuntimeError("Odds provider key is not configured.")

    snapshot_run_id, snapshot_timestamp = _utc_snapshot()
    config = _load_market_config()
    live_card_df = _load_live_card()

    bookmakers = config.get("bookmakers", []) or []
    min_single_score = float((config.get("matching", {}) or {}).get("min_single_score", 90))
    provider_diagnostic_path = _provider_market_diagnostic_path(config)

    print("Snapshot run ID:", snapshot_run_id)
    print("Live card rows:", len(live_card_df))
    print("Configured bookmakers:", bookmakers)
    print("Configured canonical markets:", config.get("markets", []))
    print("Minimum single-fighter match score:", min_single_score)

    odds_json = fetch_odds(api_key=provider_key, config=config)
    provider_diagnostic_df = flatten_provider_market_diagnostics(odds_json=odds_json, bookmakers=bookmakers)
    provider_df = flatten_moneyline_odds(odds_json=odds_json, bookmakers=bookmakers)

    provider_diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    provider_diagnostic_df.to_parquet(provider_diagnostic_path, index=False)

    print("Provider events fetched:", len(odds_json))
    print("Provider diagnostic rows:", len(provider_diagnostic_df))
    if "provider_market_key" in provider_diagnostic_df.columns:
        print("Provider market keys:", provider_diagnostic_df["provider_market_key"].value_counts(dropna=False).to_dict())
    print("Provider moneyline rows:", len(provider_df))

    market_df, match_audit_df = _build_market_outcomes(
        provider_df=provider_df,
        live_card_df=live_card_df,
        snapshot_run_id=snapshot_run_id,
        snapshot_timestamp=snapshot_timestamp,
        min_single_score=min_single_score,
    )

    validation_audit_df = validate_market_outcomes(
        market_df,
        snapshot_run_id=snapshot_run_id,
        snapshot_timestamp=snapshot_timestamp,
        artifact_name="market_outcomes",
    )

    snapshot_history_df = _append_snapshot_history(market_df)

    market_df.to_parquet(MARKET_OUTCOMES_PATH, index=False)
    snapshot_history_df.to_parquet(MARKET_OUTCOME_SNAPSHOTS_PATH, index=False)
    validation_audit_df.to_parquet(MARKET_OUTCOME_AUDIT_PATH, index=False)
    match_audit_df.to_parquet(MARKET_MATCH_AUDIT_V2_PATH, index=False)

    matched_provider_rows = int(match_audit_df["is_matched"].sum()) if "is_matched" in match_audit_df else 0
    unmatched_provider_rows = int(len(match_audit_df) - matched_provider_rows)

    print()
    print("========== MARKET V2 SUMMARY ==========")
    print("Provider rows:", len(provider_df))
    print("Provider diagnostic rows:", len(provider_diagnostic_df))
    print("Matched provider rows:", matched_provider_rows)
    print("Unmatched provider rows:", unmatched_provider_rows)
    print("Market outcome rows:", len(market_df))
    print("Snapshot history rows:", len(snapshot_history_df))
    print("Validation passes:", bool(validation_audit_df["passes_validation"].iloc[0]))
    print()
    print("Files saved:")
    print(MARKET_OUTCOMES_PATH)
    print(MARKET_OUTCOME_SNAPSHOTS_PATH)
    print(MARKET_OUTCOME_AUDIT_PATH)
    print(MARKET_MATCH_AUDIT_V2_PATH)
    print(provider_diagnostic_path)


if __name__ == "__main__":
    main()
