# ============================================================
# pipeline/betting/run_betting_join_key_diagnostic.py
# ============================================================

"""Diagnose Betting Outcomes V2 model/market join-key coverage.

This runner is read-only with respect to model and market inputs. It compares the
canonical Betting Outcomes V2 join keys:

    fight_id + market_key + outcome_join_key

between model outcome artifacts and market_outcomes.parquet, then writes a row-
level diagnostic artifact that explains which keys are shared, model-only, or
market-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.betting.betting_joiner import prepare_market_outcomes, prepare_model_predictions
from pipeline.betting.betting_schema import JOIN_KEYS
from pipeline.betting.run_betting_outcomes_v2 import (
    DEFAULT_REGISTRY_PATH,
    _load_model_outcomes,
)
from pipeline.common.paths import (
    BETTING_JOIN_KEY_DIAGNOSTIC_PATH,
    MARKET_OUTCOMES_PATH,
    ensure_data_dirs,
)

DIAGNOSTIC_COLUMNS = [
    "diagnostic_run_id",
    "diagnostic_timestamp",
    "key_status",
    "fight_id",
    "market_key",
    "outcome_join_key",
    "model_rows",
    "market_rows",
    "model_ids",
    "model_outcome_labels",
    "market_outcome_labels",
    "market_bookmakers",
    "model_fighter_ids",
    "market_fighter_ids",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Betting Outcomes V2 join-key diagnostic artifact."
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Model registry YAML used to load model-scoped predictions.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["production", "all", "single"],
        default="production",
        help="Same model mode used by Betting Outcomes V2.",
    )
    parser.add_argument(
        "--market-outcomes-path",
        default=str(MARKET_OUTCOMES_PATH),
        help="Market outcomes parquet path.",
    )
    parser.add_argument(
        "--output-path",
        default=str(BETTING_JOIN_KEY_DIAGNOSTIC_PATH),
        help="Output parquet path for the join-key diagnostic.",
    )
    return parser.parse_args()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("betting_join_key_diag_%Y%m%d_%H%M%S"), now.isoformat()


def _unique_joined(values: pd.Series) -> str:
    cleaned = [
        str(value)
        for value in values.dropna().astype(str).unique().tolist()
        if str(value).strip() not in {"", "nan", "None", "<NA>"}
    ]
    return ", ".join(sorted(cleaned))


def _aggregate_model_keys(model_df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_model_predictions(model_df)
    for optional_column in ["model_id", "outcome_label", "outcome_fighter_id"]:
        if optional_column not in out.columns:
            out[optional_column] = pd.NA

    return (
        out.groupby(JOIN_KEYS, dropna=False)
        .agg(
            model_rows=("fight_id", "size"),
            model_ids=("model_id", _unique_joined),
            model_outcome_labels=("outcome_label", _unique_joined),
            model_fighter_ids=("outcome_fighter_id", _unique_joined),
        )
        .reset_index()
    )


def _aggregate_market_keys(market_df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_market_outcomes(market_df)
    for optional_column in ["bookmaker", "outcome_label", "outcome_fighter_id"]:
        if optional_column not in out.columns:
            out[optional_column] = pd.NA

    return (
        out.groupby(JOIN_KEYS, dropna=False)
        .agg(
            market_rows=("fight_id", "size"),
            market_bookmakers=("bookmaker", _unique_joined),
            market_outcome_labels=("outcome_label", _unique_joined),
            market_fighter_ids=("outcome_fighter_id", _unique_joined),
        )
        .reset_index()
    )


def build_join_key_diagnostic(
    *,
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    diagnostic_run_id: str,
    diagnostic_timestamp: str,
) -> pd.DataFrame:
    """Return row-level model/market join-key diagnostic coverage."""

    model_keys = _aggregate_model_keys(model_df)
    market_keys = _aggregate_market_keys(market_df)

    diagnostic = model_keys.merge(
        market_keys,
        on=JOIN_KEYS,
        how="outer",
        indicator=True,
    )

    diagnostic["key_status"] = diagnostic["_merge"].map(
        {
            "both": "joined",
            "left_only": "model_only",
            "right_only": "market_only",
        }
    )
    diagnostic = diagnostic.drop(columns=["_merge"])
    diagnostic.insert(0, "diagnostic_timestamp", diagnostic_timestamp)
    diagnostic.insert(0, "diagnostic_run_id", diagnostic_run_id)

    for count_column in ["model_rows", "market_rows"]:
        if count_column not in diagnostic.columns:
            diagnostic[count_column] = 0
        diagnostic[count_column] = diagnostic[count_column].fillna(0).astype(int)

    for column in DIAGNOSTIC_COLUMNS:
        if column not in diagnostic.columns:
            diagnostic[column] = pd.NA

    return diagnostic[DIAGNOSTIC_COLUMNS]


def main() -> None:
    args = _parse_args()
    ensure_data_dirs()
    diagnostic_run_id, diagnostic_timestamp = _utc_run()

    print("=" * 80)
    print("BETTING JOIN KEY DIAGNOSTIC")
    print("=" * 80)
    print("Diagnostic run ID:", diagnostic_run_id)
    print("Model mode:", args.model_mode)
    print("Registry path:", args.registry_path)
    print("Market outcomes path:", args.market_outcomes_path)
    print("Output path:", args.output_path)

    model_df, selected_models = _load_model_outcomes(
        registry_path=Path(args.registry_path),
        model_mode=args.model_mode,
    )
    market_df = _load_required_parquet(Path(args.market_outcomes_path), "Market outcomes")

    diagnostic_df = build_join_key_diagnostic(
        model_df=model_df,
        market_df=market_df,
        diagnostic_run_id=diagnostic_run_id,
        diagnostic_timestamp=diagnostic_timestamp,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_df.to_parquet(output_path, index=False)

    print()
    print("========== JOIN KEY DIAGNOSTIC SUMMARY ==========")
    print("Selected models:", ", ".join(str(row.get("model_id")) for row in selected_models))
    print("Model rows:", len(model_df))
    print("Market rows:", len(market_df))
    print("Unique diagnostic keys:", len(diagnostic_df))
    print("Status counts:")
    print(diagnostic_df["key_status"].value_counts(dropna=False).to_string())
    print("Markets by status:")
    print(diagnostic_df.groupby(["key_status", "market_key"]).size().to_string())
    print()
    print("File saved:", output_path)


if __name__ == "__main__":
    main()
