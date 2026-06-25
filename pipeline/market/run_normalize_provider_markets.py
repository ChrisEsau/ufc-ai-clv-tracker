# ============================================================
# pipeline/market/run_normalize_provider_markets.py
# ============================================================

"""Generic sportsbook provider market normalization runner.

This runner converts provider-specific diagnostic artifacts into the shared
canonical market catalog. Provider-specific quirks stay inside provider
normalizers; the output contract is sportsbook agnostic.

Usage:
    python -m pipeline.market.run_normalize_provider_markets --provider draftkings
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from pipeline.common.paths import (
    CANONICAL_MARKET_AUDIT_PATH,
    CANONICAL_MARKET_CATALOG_PATH,
    DRAFTKINGS_MARKET_CATALOG_PATH,
    DRAFTKINGS_MARKET_DIAGNOSTIC_PATH,
    FANDUEL_MARKET_CATALOG_PATH,
    FANDUEL_MARKET_DIAGNOSTIC_PATH,
    ensure_data_dirs,
)
from pipeline.market.normalizers.canonical_market_schema import (
    ensure_canonical_market_audit_columns,
)
from pipeline.market.normalizers.draftkings import normalize_draftkings_diagnostic_rows
from pipeline.market.normalizers.fanduel import normalize_fanduel_diagnostic_rows


NormalizerFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ProviderNormalizerConfig:
    """Runtime config for one provider normalizer."""

    provider: str
    source: str
    bookmaker: str
    input_path: Path
    output_path: Path
    normalizer: NormalizerFn


NORMALIZER_REGISTRY: dict[str, ProviderNormalizerConfig] = {
    "draftkings": ProviderNormalizerConfig(
        provider="draftkings",
        source="draftkings_public",
        bookmaker="DraftKings",
        input_path=DRAFTKINGS_MARKET_DIAGNOSTIC_PATH,
        output_path=DRAFTKINGS_MARKET_CATALOG_PATH,
        normalizer=normalize_draftkings_diagnostic_rows,
    ),
    "fanduel": ProviderNormalizerConfig(
        provider="fanduel",
        source="fanduel_public",
        bookmaker="FanDuel",
        input_path=FANDUEL_MARKET_DIAGNOSTIC_PATH,
        output_path=FANDUEL_MARKET_CATALOG_PATH,
        normalizer=normalize_fanduel_diagnostic_rows,
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize provider-specific market diagnostics into the canonical market catalog."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=sorted(NORMALIZER_REGISTRY.keys()),
        help="Provider key to normalize.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Optional provider diagnostic input override.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Provider market catalog output path. Defaults to provider-specific catalog path.",
    )
    parser.add_argument(
        "--audit-path",
        default=str(CANONICAL_MARKET_AUDIT_PATH),
        help="Canonical market catalog audit output path.",
    )
    parser.add_argument(
        "--allow-provider-event-drop",
        action="store_true",
        help=(
            "Allow normalization output to contain fewer provider_event_id values "
            "than the diagnostic input. Default is to fail fast."
        ),
    )
    return parser.parse_args()


def _safe_json_counts(series: pd.Series) -> str:
    """Return compact JSON value counts for audit rows."""

    if series.empty:
        return "{}"
    counts = series.fillna("<NA>").astype(str).value_counts(dropna=False).to_dict()
    return json.dumps(counts, sort_keys=True)


def _provider_event_ids(df: pd.DataFrame) -> set[str]:
    """Return non-empty provider_event_id values for coverage validation."""

    if "provider_event_id" not in df.columns or df.empty:
        return set()
    return {
        str(value).strip()
        for value in df["provider_event_id"].dropna().astype(str).tolist()
        if str(value).strip() not in {"", "nan", "None", "<NA>"}
    }


def _validate_provider_event_coverage(
    *,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    allow_provider_event_drop: bool,
) -> None:
    """Fail fast when normalization silently drops provider events.

    Market refresh is intended to preserve all provider_event_id values that have
    diagnostic rows. A reduction from many events to one event usually means a
    stale artifact, accidental single-event refresh, or unintended filter.
    """

    input_events = _provider_event_ids(input_df)
    output_events = _provider_event_ids(output_df)
    missing_events = sorted(input_events - output_events)

    print("Provider events in diagnostic input:", len(input_events))
    print("Provider events in canonical output:", len(output_events))
    if missing_events:
        print("Provider events missing after normalization:", missing_events)

    if missing_events and not allow_provider_event_drop:
        raise RuntimeError(
            "Provider event coverage dropped during market normalization. "
            f"Missing provider_event_id values: {missing_events}. "
            "Use --allow-provider-event-drop only for intentional one-off debugging."
        )


def _build_audit(
    *,
    provider_config: ProviderNormalizerConfig,
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a one-row audit for the canonical market catalog normalization."""

    unmapped_mask = output_df["market_key"].isna() | output_df["outcome_key"].isna()
    unmapped_rows = int(unmapped_mask.sum())
    output_rows = int(len(output_df))
    mapped_rate = 0.0 if output_rows == 0 else round((output_rows - unmapped_rows) / output_rows, 6)

    unmapped_market_names = []
    if "provider_market_name" in output_df.columns and unmapped_rows:
        unmapped_market_names = sorted(
            output_df.loc[unmapped_mask, "provider_market_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )[:50]

    snapshot_run_id = None
    snapshot_timestamp = None
    if "snapshot_run_id" in output_df.columns and not output_df.empty:
        snapshot_run_id = output_df["snapshot_run_id"].dropna().astype(str).iloc[0]
    elif "snapshot_run_id" in input_df.columns and not input_df.empty:
        snapshot_run_id = input_df["snapshot_run_id"].dropna().astype(str).iloc[0]

    if "snapshot_timestamp" in output_df.columns and not output_df.empty:
        snapshot_timestamp = output_df["snapshot_timestamp"].dropna().astype(str).iloc[0]
    elif "snapshot_timestamp" in input_df.columns and not input_df.empty:
        snapshot_timestamp = input_df["snapshot_timestamp"].dropna().astype(str).iloc[0]

    audit_df = pd.DataFrame(
        [
            {
                "snapshot_run_id": snapshot_run_id,
                "snapshot_timestamp": snapshot_timestamp,
                "source": provider_config.source,
                "bookmaker": provider_config.bookmaker,
                "input_rows": int(len(input_df)),
                "output_rows": output_rows,
                "unmapped_rows": unmapped_rows,
                "mapped_rate": mapped_rate,
                "market_family_counts": _safe_json_counts(output_df.get("market_family", pd.Series(dtype="object"))),
                "unmapped_market_names": json.dumps(unmapped_market_names),
                "passes_validation": bool(output_rows > 0 and mapped_rate >= 0.95),
            }
        ]
    )
    return ensure_canonical_market_audit_columns(audit_df)


def normalize_provider_markets(
    *,
    provider: str,
    input_path: Path | None = None,
    output_path: Path | None = None,
    audit_path: Path = CANONICAL_MARKET_AUDIT_PATH,
    allow_provider_event_drop: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize one provider diagnostic artifact into canonical market rows."""

    if provider not in NORMALIZER_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider}. Supported: {sorted(NORMALIZER_REGISTRY)}")

    provider_config = NORMALIZER_REGISTRY[provider]
    resolved_input_path = Path(input_path or provider_config.input_path)
    resolved_output_path = Path(output_path or provider_config.output_path)

    if not resolved_input_path.exists():
        raise FileNotFoundError(f"Provider diagnostic input not found: {resolved_input_path}")

    input_df = pd.read_parquet(resolved_input_path)
    output_df = provider_config.normalizer(input_df)
    _validate_provider_event_coverage(
        input_df=input_df,
        output_df=output_df,
        allow_provider_event_drop=allow_provider_event_drop,
    )
    audit_df = _build_audit(
        provider_config=provider_config,
        input_df=input_df,
        output_df=output_df,
    )

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(resolved_output_path, index=False)
    audit_df.to_parquet(audit_path, index=False)

    return output_df, audit_df


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()

    output_df, audit_df = normalize_provider_markets(
        provider=args.provider,
        input_path=Path(args.input_path) if args.input_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
        audit_path=Path(args.audit_path),
        allow_provider_event_drop=bool(args.allow_provider_event_drop),
    )

    print("=" * 80)
    print("NORMALIZE PROVIDER MARKETS")
    print("=" * 80)
    print("Provider:", args.provider)
    provider_config = NORMALIZER_REGISTRY[args.provider]
    print("Output path:", args.output_path or str(provider_config.output_path))
    print("Audit path:", args.audit_path)
    print("Canonical rows:", len(output_df))

    if not output_df.empty:
        print("Market keys:")
        print(output_df["market_key"].value_counts(dropna=False).head(30).to_string())

    print("Validation passes:", bool(audit_df["passes_validation"].iloc[0]))


if __name__ == "__main__":
    main()
