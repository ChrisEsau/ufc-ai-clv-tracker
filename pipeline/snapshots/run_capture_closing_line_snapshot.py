from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    BETTING_OUTCOMES_PATH,
    CLOSING_LINE_SNAPSHOT_AUDIT_PATH,
    CLOSING_LINE_SNAPSHOTS_PATH,
    SELECTED_LIVE_CARD_EVENT_PATH,
    ensure_data_dirs,
)


SNAPSHOT_COLUMNS = [
    "closing_snapshot_run_id",
    "closing_snapshot_timestamp",
    "is_official_closing_snapshot",
    "event_id",
    "event_name",
    "commence_time_utc",
    "commence_time_cdt",
    "commence_time_source",
    "commence_time_updated_at",
    "betting_run_id",
    "betting_timestamp",
    "prediction_run_id",
    "prediction_timestamp",
    "snapshot_run_id",
    "snapshot_timestamp",
    "model_id",
    "model_family",
    "model_registry_status",
    "model_outcomes_path",
    "algorithm",
    "prediction_type",
    "fight_id",
    "fight_display",
    "red_fighter",
    "blue_fighter",
    "red_fighter_id",
    "blue_fighter_id",
    "market_key",
    "market_display",
    "bookmaker",
    "source",
    "outcome_label",
    "outcome_display",
    "outcome_fighter_id",
    "outcome_join_key",
    "outcome_side",
    "model_probability",
    "model_pick_probability",
    "is_model_pick",
    "model_pick",
    "model_confidence",
    "confidence_score",
    "confidence_pct",
    "confidence_tier",
    "american_odds",
    "decimal_odds",
    "implied_probability",
    "edge",
    "edge_pct",
    "ev",
    "ev_pct",
    "ev_dollars_at_100",
    "full_kelly_fraction",
    "fractional_kelly_fraction",
    "recommended_stake",
    "max_stake",
    "passes_edge_filter",
    "passes_confidence_filter",
    "passes_odds_filter",
    "passes_market_data_filter",
    "is_bet_candidate",
    "bet_status",
]

AUDIT_COLUMNS = [
    "closing_snapshot_run_id",
    "closing_snapshot_timestamp",
    "event_id",
    "event_name",
    "commence_time_utc",
    "betting_rows",
    "closing_rows",
    "bet_candidates",
    "unique_fights",
    "unique_markets",
    "unique_bookmakers",
    "passes_validation",
    "error",
]


def _utc_capture() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("closing_%Y%m%d_%H%M%S"), now.isoformat()


def _read_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _first_value(df: pd.DataFrame, column: str):
    if column not in df.columns or df.empty:
        return pd.NA
    values = df[column].dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[columns]


def _append_parquet(path: Path, new_rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = _read_optional_parquet(path)
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False) if not existing.empty else new_rows.copy()
    return _ensure_columns(combined, columns)


def _prepare_closing_rows(
    *,
    betting_df: pd.DataFrame,
    target_event_df: pd.DataFrame,
    closing_snapshot_run_id: str,
    closing_snapshot_timestamp: str,
    official: bool,
) -> pd.DataFrame:
    rows = betting_df.copy()
    rows["closing_snapshot_run_id"] = closing_snapshot_run_id
    rows["closing_snapshot_timestamp"] = closing_snapshot_timestamp
    rows["is_official_closing_snapshot"] = bool(official)

    for column in [
        "event_id",
        "event_name",
        "commence_time_utc",
        "commence_time_cdt",
        "commence_time_source",
        "commence_time_updated_at",
    ]:
        if column in target_event_df.columns:
            rows[column] = _first_value(target_event_df, column)

    return _ensure_columns(rows, SNAPSHOT_COLUMNS)


def _audit_row(
    *,
    closing_snapshot_run_id: str,
    closing_snapshot_timestamp: str,
    target_event_df: pd.DataFrame,
    betting_df: pd.DataFrame,
    closing_rows: pd.DataFrame,
    error: str | None,
) -> pd.DataFrame:
    return _ensure_columns(
        pd.DataFrame(
            [
                {
                    "closing_snapshot_run_id": closing_snapshot_run_id,
                    "closing_snapshot_timestamp": closing_snapshot_timestamp,
                    "event_id": _first_value(target_event_df, "event_id"),
                    "event_name": _first_value(target_event_df, "event_name"),
                    "commence_time_utc": _first_value(target_event_df, "commence_time_utc"),
                    "betting_rows": int(len(betting_df)),
                    "closing_rows": int(len(closing_rows)),
                    "bet_candidates": int(closing_rows.get("is_bet_candidate", pd.Series(dtype=bool)).fillna(False).sum()) if not closing_rows.empty else 0,
                    "unique_fights": int(closing_rows.get("fight_id", pd.Series(dtype=str)).nunique(dropna=True)) if not closing_rows.empty else 0,
                    "unique_markets": int(closing_rows.get("market_key", pd.Series(dtype=str)).nunique(dropna=True)) if not closing_rows.empty else 0,
                    "unique_bookmakers": int(closing_rows.get("bookmaker", pd.Series(dtype=str)).nunique(dropna=True)) if not closing_rows.empty else 0,
                    "passes_validation": bool(error is None and not closing_rows.empty),
                    "error": error,
                }
            ]
        ),
        AUDIT_COLUMNS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture official closing-line snapshot from the latest betting outcomes artifact.")
    parser.add_argument(
        "--official-closing-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flag rows as the official pre-fight closing snapshot. Defaults to true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    closing_snapshot_run_id, closing_snapshot_timestamp = _utc_capture()

    print("=" * 80)
    print("UFC CLOSING LINE SNAPSHOT CAPTURE")
    print("=" * 80)
    print("Closing snapshot run ID:", closing_snapshot_run_id)
    print("Official closing snapshot:", bool(args.official_closing_snapshot))

    target_event_df = betting_df = closing_rows = pd.DataFrame()
    error: str | None = None
    total_rows = 0

    try:
        target_event_df = _read_required_parquet(SELECTED_LIVE_CARD_EVENT_PATH, "Selected target event")
        betting_df = _read_required_parquet(BETTING_OUTCOMES_PATH, "Betting outcomes")
        closing_rows = _prepare_closing_rows(
            betting_df=betting_df,
            target_event_df=target_event_df,
            closing_snapshot_run_id=closing_snapshot_run_id,
            closing_snapshot_timestamp=closing_snapshot_timestamp,
            official=bool(args.official_closing_snapshot),
        )
        if closing_rows.empty:
            raise ValueError("Closing-line snapshot produced zero rows.")

        combined = _append_parquet(CLOSING_LINE_SNAPSHOTS_PATH, closing_rows, SNAPSHOT_COLUMNS)
        CLOSING_LINE_SNAPSHOTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(CLOSING_LINE_SNAPSHOTS_PATH, index=False)
        total_rows = len(combined)
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        audit = _audit_row(
            closing_snapshot_run_id=closing_snapshot_run_id,
            closing_snapshot_timestamp=closing_snapshot_timestamp,
            target_event_df=target_event_df,
            betting_df=betting_df,
            closing_rows=closing_rows,
            error=error,
        )
        combined_audit = _append_parquet(CLOSING_LINE_SNAPSHOT_AUDIT_PATH, audit, AUDIT_COLUMNS)
        CLOSING_LINE_SNAPSHOT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined_audit.to_parquet(CLOSING_LINE_SNAPSHOT_AUDIT_PATH, index=False)

    print()
    print("========== CLOSING LINE SNAPSHOT SUMMARY ==========")
    print("Event:", _first_value(target_event_df, "event_name"))
    print("Commence time UTC:", _first_value(target_event_df, "commence_time_utc"))
    print("Betting rows captured:", len(betting_df))
    print("Closing rows appended:", len(closing_rows))
    print("Closing snapshot history rows:", total_rows)
    print("Files saved:")
    print(CLOSING_LINE_SNAPSHOTS_PATH)
    print(CLOSING_LINE_SNAPSHOT_AUDIT_PATH)


if __name__ == "__main__":
    main()
