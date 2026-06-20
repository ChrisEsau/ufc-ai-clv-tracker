"""Run the canonical CLV artifact pipeline.

This runner intentionally consumes existing model-market snapshots and bankroll
ledger artifacts. It does not pull fresh odds; market ingestion remains owned by
the Market Refresh pipeline so the Betting Board contract stays stable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.clv.closing_lines import CLOSING_LINE_COLUMNS, build_closing_lines
from pipeline.clv.clv_results import CLV_RESULT_COLUMNS, build_clv_results
from pipeline.clv.line_movement import LINE_MOVEMENT_COLUMNS, build_line_movement
from pipeline.clv.market_normalization import (
    MARKET_NORMALIZATION_AUDIT_COLUMNS,
    NORMALIZED_MARKET_COLUMNS,
    build_market_normalization_audit,
    normalize_market_snapshots,
)
from pipeline.common.paths import (
    BET_LEDGER_PATH,
    CLOSING_LINES_PATH,
    CLV_MARKET_NORMALIZATION_AUDIT_PATH,
    CLV_RESULTS_PATH,
    LINE_MOVEMENT_PATH,
    MARKET_SNAPSHOTS_PATH,
    MODEL_MARKET_SNAPSHOTS_PATH,
    NORMALIZED_MARKET_SNAPSHOTS_PATH,
    ensure_data_dirs,
)
from utils.bankroll_artifacts import load_bet_ledger, save_bet_ledger


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[columns]
    output.to_parquet(path, index=False)


def _load_market_snapshot_source() -> tuple[pd.DataFrame, Path]:
    """Prefer Market Refresh V2 model-market snapshots, with legacy fallback."""

    model_market_snapshots = _read_parquet(MODEL_MARKET_SNAPSHOTS_PATH)
    if not model_market_snapshots.empty:
        return model_market_snapshots, MODEL_MARKET_SNAPSHOTS_PATH

    legacy_market_snapshots = _read_parquet(MARKET_SNAPSHOTS_PATH)
    return legacy_market_snapshots, MARKET_SNAPSHOTS_PATH


def _update_ledger_with_clv(ledger: pd.DataFrame, clv_results: pd.DataFrame) -> int:
    """Write calculated CLV and closing odds back to the bankroll ledger."""

    if ledger.empty or clv_results.empty or "bet_id" not in ledger.columns or "bet_id" not in clv_results.columns:
        return 0

    updates = clv_results.dropna(subset=["bet_id"]).copy()
    if updates.empty:
        return 0

    updates = updates.sort_values("closing_timestamp", na_position="first").drop_duplicates("bet_id", keep="last")
    updates_by_id = updates.set_index(updates["bet_id"].astype(str))
    ledger_out = ledger.copy()
    ledger_ids = ledger_out["bet_id"].astype(str)
    matched = ledger_ids.isin(updates_by_id.index)
    if not matched.any():
        return 0

    for column in ["clv", "closing_odds"]:
        if column not in ledger_out.columns:
            ledger_out[column] = pd.NA

    matched_ids = ledger_ids[matched]
    ledger_out.loc[matched, "clv"] = matched_ids.map(updates_by_id["clv_pct"])
    ledger_out.loc[matched, "closing_odds"] = matched_ids.map(updates_by_id["closing_odds"])
    save_bet_ledger(ledger_out)
    return int(matched.sum())


def main() -> None:
    ensure_data_dirs()

    market_snapshots, market_snapshot_source = _load_market_snapshot_source()
    ledger = load_bet_ledger() if BET_LEDGER_PATH.exists() else pd.DataFrame()

    normalized_snapshots = normalize_market_snapshots(market_snapshots)
    normalization_audit = build_market_normalization_audit(market_snapshots, normalized_snapshots)
    closing_lines = build_closing_lines(normalized_snapshots)
    line_movement = build_line_movement(normalized_snapshots)
    clv_results = build_clv_results(ledger, closing_lines)
    ledger_rows_updated = _update_ledger_with_clv(ledger, clv_results)

    _write_parquet(normalized_snapshots, NORMALIZED_MARKET_SNAPSHOTS_PATH, NORMALIZED_MARKET_COLUMNS)
    _write_parquet(normalization_audit, CLV_MARKET_NORMALIZATION_AUDIT_PATH, MARKET_NORMALIZATION_AUDIT_COLUMNS)
    _write_parquet(closing_lines, CLOSING_LINES_PATH, CLOSING_LINE_COLUMNS)
    _write_parquet(line_movement, LINE_MOVEMENT_PATH, LINE_MOVEMENT_COLUMNS)
    _write_parquet(clv_results, CLV_RESULTS_PATH, CLV_RESULT_COLUMNS)

    print("========== UFC CLV PIPELINE ==========")
    print(f"Market snapshot source: {market_snapshot_source}")
    print(f"Market snapshots loaded: {len(market_snapshots)}")
    print(f"Normalized market rows written: {len(normalized_snapshots)} -> {NORMALIZED_MARKET_SNAPSHOTS_PATH}")
    print(f"Market normalization audit written: {CLV_MARKET_NORMALIZATION_AUDIT_PATH}")
    print(f"Ledger bets loaded: {len(ledger)}")
    print(f"Ledger rows updated with CLV: {ledger_rows_updated}")
    print(f"Closing lines written: {len(closing_lines)} -> {CLOSING_LINES_PATH}")
    print(f"Line movement rows written: {len(line_movement)} -> {LINE_MOVEMENT_PATH}")
    print(f"CLV result rows written: {len(clv_results)} -> {CLV_RESULTS_PATH}")


if __name__ == "__main__":
    main()
