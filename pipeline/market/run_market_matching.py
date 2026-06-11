# ============================================================
# pipeline/market/run_market_matching.py
# ============================================================

"""Match canonical sportsbook markets to the UFC live card.

Input:
    data/market/canonical_market_catalog.parquet
    data/predictions/ufc_live_card.parquet

Output:
    data/market/market_outcomes.parquet
    data/audits/ufc_market_match_audit_v2.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from pipeline.common.paths import (
    CANONICAL_MARKET_CATALOG_PATH,
    LIVE_CARD_PATH,
    MARKET_MATCH_AUDIT_V2_PATH,
    MARKET_OUTCOMES_PATH,
    ensure_data_dirs,
)

DRAFTKINGS_REGISTRY_PATH = Path("configs/market/providers/draftkings_ufc_registry.yaml")
from pipeline.market.market_matcher import (
    build_market_match_audit_row,
    build_market_outcome_row,
    ensure_market_match_audit_columns,
    ensure_market_outcome_columns,
    match_canonical_market_row_to_live_card,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match canonical market catalog rows to UFC live card fights."
    )
    parser.add_argument(
        "--catalog-path",
        default=str(CANONICAL_MARKET_CATALOG_PATH),
        help="Canonical market catalog parquet path.",
    )
    parser.add_argument(
        "--live-card-path",
        default=str(LIVE_CARD_PATH),
        help="UFC live card parquet path.",
    )
    parser.add_argument(
        "--output-path",
        default=str(MARKET_OUTCOMES_PATH),
        help="Matched market outcomes output path.",
    )
    parser.add_argument(
        "--audit-path",
        default=str(MARKET_MATCH_AUDIT_V2_PATH),
        help="Market matching audit output path.",
    )
    parser.add_argument(
        "--registry-path",
        default=str(DRAFTKINGS_REGISTRY_PATH),
        help="Provider registry YAML used to resolve matching strategies.",
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=80.0,
        help="Minimum matching score required to emit a market outcome.",
    )
    return parser.parse_args()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)



def _load_matching_strategy_map(registry_path: Path) -> tuple[str, dict[str, str]]:
    """Load default and per-market matching strategies from provider registry."""

    if not registry_path.exists():
        return "fighter_name", {}

    config = yaml.safe_load(registry_path.read_text()) or {}
    matching = config.get("matching", {}) or {}

    default_strategy = str(matching.get("default_strategy", "fighter_name"))
    overrides = matching.get("market_overrides", {}) or {}

    strategy_map = {}
    for market_key, payload in overrides.items():
        if isinstance(payload, dict):
            strategy_map[str(market_key)] = str(payload.get("strategy", default_strategy))
        else:
            strategy_map[str(market_key)] = str(payload)

    return default_strategy, strategy_map


def _apply_matching_strategies(
    catalog_df: pd.DataFrame,
    *,
    default_strategy: str,
    strategy_map: dict[str, str],
) -> pd.DataFrame:
    """Attach registry-driven matching strategy to canonical market rows."""

    df = catalog_df.copy()
    if "matching_strategy" not in df.columns:
        df["matching_strategy"] = df["market_key"].astype(str).map(strategy_map).fillna(default_strategy)
    else:
        df["matching_strategy"] = (
            df["matching_strategy"]
            .fillna(df["market_key"].astype(str).map(strategy_map))
            .fillna(default_strategy)
        )
    return df

def run_market_matching(
    *,
    catalog_path: Path = CANONICAL_MARKET_CATALOG_PATH,
    live_card_path: Path = LIVE_CARD_PATH,
    output_path: Path = MARKET_OUTCOMES_PATH,
    audit_path: Path = MARKET_MATCH_AUDIT_V2_PATH,
    registry_path: Path = DRAFTKINGS_REGISTRY_PATH,
    min_match_score: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match canonical market rows to live-card fights."""

    catalog_df = _load_required_parquet(catalog_path, "Canonical market catalog")
    default_strategy, strategy_map = _load_matching_strategy_map(registry_path)
    catalog_df = _apply_matching_strategies(
        catalog_df,
        default_strategy=default_strategy,
        strategy_map=strategy_map,
    )
    live_card_df = _load_required_parquet(live_card_path, "Live card")

    outcome_rows = []
    audit_rows = []

    for _, catalog_row in catalog_df.iterrows():
        match = match_canonical_market_row_to_live_card(
            catalog_row,
            live_card_df,
            min_match_score=min_match_score,
        )
        audit_rows.append(build_market_match_audit_row(catalog_row, match))

        if match is not None:
            outcome_rows.append(build_market_outcome_row(catalog_row, match))

    market_df = ensure_market_outcome_columns(pd.DataFrame(outcome_rows))
    audit_df = ensure_market_match_audit_columns(pd.DataFrame(audit_rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    market_df.to_parquet(output_path, index=False)
    audit_df.to_parquet(audit_path, index=False)

    return market_df, audit_df


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    run_id = datetime.now(timezone.utc).strftime("match_%Y%m%d_%H%M%S")

    print("=" * 80)
    print("UFC MARKET MATCHING")
    print("=" * 80)
    print("Run ID:", run_id)
    print("Catalog path:", args.catalog_path)
    print("Live card path:", args.live_card_path)
    print("Output path:", args.output_path)
    print("Audit path:", args.audit_path)
    print("Registry path:", args.registry_path)
    print("Minimum match score:", args.min_match_score)

    market_df, audit_df = run_market_matching(
        catalog_path=Path(args.catalog_path),
        live_card_path=Path(args.live_card_path),
        output_path=Path(args.output_path),
        audit_path=Path(args.audit_path),
        registry_path=Path(args.registry_path),
        min_match_score=float(args.min_match_score),
    )

    matched_rows = int(audit_df["is_matched"].fillna(False).sum()) if "is_matched" in audit_df else 0

    print()
    print("========== MARKET MATCHING SUMMARY ==========")
    print("Catalog rows:", len(audit_df))
    print("Matched rows:", matched_rows)
    print("Market outcome rows:", len(market_df))
    if not market_df.empty and "market_key" in market_df.columns:
        print("Market keys:")
        print(market_df["market_key"].value_counts(dropna=False).to_string())
    print()
    print("Files saved:")
    print(args.output_path)
    print(args.audit_path)


if __name__ == "__main__":
    main()
