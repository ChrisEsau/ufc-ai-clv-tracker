from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.betting.betting_joiner import build_betting_outcomes
from pipeline.common.paths import (
    MARKET_OUTCOMES_PATH,
    MODEL_MARKET_SNAPSHOT_AUDIT_PATH,
    MODEL_MARKET_SNAPSHOTS_PATH,
    PREDICTIONS_DIR,
    ensure_data_dirs,
)
from pipeline.common.risk_settings import load_risk_settings


DEFAULT_REGISTRY_PATH = Path("configs/models/model_registry.yaml")
DEFAULT_MODEL_OUTPUT_TEMPLATE = "data/predictions/by_model/{model_id}/model_outcomes.parquet"
SNAPSHOT_SOURCE = "market_refresh_orchestrator"


SNAPSHOT_COLUMNS = [
    "capture_run_id",
    "capture_timestamp",
    "snapshot_source",
    "snapshot_model_mode",
    "market_snapshot_run_id",
    "market_snapshot_timestamp",
    "betting_run_id",
    "betting_timestamp",
    "prediction_run_id",
    "prediction_timestamp",
    "model_id",
    "model_family",
    "model_registry_status",
    "model_stage",
    "model_outcomes_path",
    "algorithm",
    "prediction_type",
    "event_id",
    "event_name",
    "commence_time",
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
    "capture_run_id",
    "capture_timestamp",
    "snapshot_model_mode",
    "registry_path",
    "selected_models",
    "loaded_models",
    "missing_model_artifacts",
    "model_rows",
    "market_rows",
    "snapshot_rows",
    "existing_snapshot_rows",
    "total_snapshot_rows",
    "unique_models",
    "unique_markets",
    "unique_bookmakers",
    "bet_candidates",
    "passes_validation",
    "error",
]


def _utc_capture() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("model_market_%Y%m%d_%H%M%S"), now.isoformat()


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_columns(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    path.parent.mkdir(parents=True, exist_ok=True)
    out[columns].to_parquet(path, index=False)


def _append_parquet(existing_path: Path, new_rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = _read_parquet(existing_path)
    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False) if not existing.empty else new_rows.copy()
    for column in columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined[columns]


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file) or {}
    if not isinstance(registry, dict):
        raise ValueError(f"Model registry must be a mapping: {path}")
    return registry


def _model_output_path(*, model_id: str, entry: dict[str, Any]) -> Path:
    explicit_path = entry.get("model_outcomes_path")
    if explicit_path:
        return Path(str(explicit_path))
    return Path(DEFAULT_MODEL_OUTPUT_TEMPLATE.format(model_id=model_id))


def _select_models(registry: dict[str, Any], *, model_mode: str) -> list[dict[str, Any]]:
    if model_mode == "single":
        return [
            {
                "model_id": "canonical_model_outcomes",
                "status": "single",
                "model_family": "single",
                "market_key": "single",
                "algorithm": pd.NA,
                "path": PREDICTIONS_DIR / "model_outcomes.parquet",
            }
        ]

    models = registry.get("models", {}) or {}
    if not isinstance(models, dict):
        raise ValueError("Model registry 'models' section must be a mapping.")

    selected: list[dict[str, Any]] = []
    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status == "archived":
            continue
        if model_mode == "production" and status != "production":
            continue

        model_family = str(entry.get("model_family") or "").strip().lower()
        market_key = str(entry.get("market_key") or "").strip().lower()
        if not market_key and model_family == "moneyline":
            market_key = "moneyline"

        selected.append(
            {
                "model_id": str(model_id),
                "status": status or "unknown",
                "model_family": model_family,
                "market_key": market_key,
                "algorithm": entry.get("algorithm"),
                "path": _model_output_path(model_id=str(model_id), entry=entry),
            }
        )

    return selected


def _load_model_outcomes(*, registry_path: Path, model_mode: str) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    registry = _load_registry(registry_path)
    selected = _select_models(registry, model_mode=model_mode)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for row in selected:
        path = Path(row["path"])
        model_id = str(row["model_id"])
        if not path.exists():
            missing.append(f"{model_id}: {path}")
            continue
        frame = pd.read_parquet(path).copy()
        if frame.empty:
            continue
        frame["model_id"] = frame.get("model_id", model_id)
        frame["model_id"] = frame["model_id"].fillna(model_id)
        frame["model_family"] = frame.get("model_family", row["model_family"])
        frame["model_family"] = frame["model_family"].fillna(row["model_family"])
        frame["market_key"] = frame.get("market_key", row["market_key"])
        frame["market_key"] = frame["market_key"].fillna(row["market_key"])
        frame["algorithm"] = frame.get("algorithm", row.get("algorithm"))
        frame["algorithm"] = frame["algorithm"].fillna(row.get("algorithm"))
        frame["model_registry_status"] = row["status"]
        frame["model_outcomes_path"] = str(path)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No model outcome artifacts were available for snapshot capture.")

    return pd.concat(frames, ignore_index=True, sort=False), selected, missing


def _prepare_snapshot_rows(*, joined: pd.DataFrame, capture_run_id: str, capture_timestamp: str, model_mode: str) -> pd.DataFrame:
    rows = joined.copy()
    rows["capture_run_id"] = capture_run_id
    rows["capture_timestamp"] = capture_timestamp
    rows["snapshot_source"] = SNAPSHOT_SOURCE
    rows["snapshot_model_mode"] = model_mode
    rows["market_snapshot_run_id"] = rows.get("snapshot_run_id", pd.NA)
    rows["market_snapshot_timestamp"] = rows.get("snapshot_timestamp", pd.NA)
    rows["model_stage"] = rows.get("model_registry_status", pd.NA).fillna("unknown")
    return rows


def _audit_row(
    *,
    capture_run_id: str,
    capture_timestamp: str,
    model_mode: str,
    registry_path: Path,
    selected_models: list[dict[str, Any]],
    missing: list[str],
    model_df: pd.DataFrame,
    market_df: pd.DataFrame,
    snapshot_rows: pd.DataFrame,
    existing_rows: int,
    total_rows: int,
    error: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "capture_run_id": capture_run_id,
                "capture_timestamp": capture_timestamp,
                "snapshot_model_mode": model_mode,
                "registry_path": str(registry_path),
                "selected_models": ", ".join(str(row["model_id"]) for row in selected_models),
                "loaded_models": ", ".join(sorted(snapshot_rows.get("model_id", pd.Series(dtype=str)).dropna().astype(str).unique())),
                "missing_model_artifacts": "; ".join(missing),
                "model_rows": len(model_df),
                "market_rows": len(market_df),
                "snapshot_rows": len(snapshot_rows),
                "existing_snapshot_rows": existing_rows,
                "total_snapshot_rows": total_rows,
                "unique_models": snapshot_rows.get("model_id", pd.Series(dtype=str)).nunique(dropna=True),
                "unique_markets": snapshot_rows.get("market_key", pd.Series(dtype=str)).nunique(dropna=True),
                "unique_bookmakers": snapshot_rows.get("bookmaker", pd.Series(dtype=str)).nunique(dropna=True),
                "bet_candidates": int(snapshot_rows.get("is_bet_candidate", pd.Series(dtype=bool)).fillna(False).sum()) if not snapshot_rows.empty else 0,
                "passes_validation": bool(error is None and not snapshot_rows.empty),
                "error": error,
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture append-only model-market snapshots for future CLV analysis.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument(
        "--model-mode",
        choices=["production", "all", "single"],
        default="all",
        help="Which model outcome artifacts to snapshot. 'all' includes production and draft non-archived models when artifacts exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_path)
    capture_run_id, capture_timestamp = _utc_capture()

    print("=" * 80)
    print("UFC MODEL-MARKET SNAPSHOT CAPTURE")
    print("=" * 80)
    print("Capture run ID:", capture_run_id)
    print("Model mode:", args.model_mode)

    ensure_data_dirs()
    existing_rows = len(_read_parquet(MODEL_MARKET_SNAPSHOTS_PATH))

    model_df = market_df = snapshot_rows = pd.DataFrame()
    selected_models: list[dict[str, Any]] = []
    missing: list[str] = []
    error: str | None = None

    try:
        model_df, selected_models, missing = _load_model_outcomes(registry_path=registry_path, model_mode=args.model_mode)
        market_df = _read_parquet(MARKET_OUTCOMES_PATH)
        if market_df.empty:
            raise FileNotFoundError(f"Market outcomes not found or empty: {MARKET_OUTCOMES_PATH}")

        joined = build_betting_outcomes(
            model_df=model_df,
            market_df=market_df,
            settings=load_risk_settings(),
            betting_run_id=capture_run_id,
            betting_timestamp=capture_timestamp,
        )
        snapshot_rows = _prepare_snapshot_rows(
            joined=joined,
            capture_run_id=capture_run_id,
            capture_timestamp=capture_timestamp,
            model_mode=args.model_mode,
        )
        if snapshot_rows.empty:
            raise ValueError("Snapshot join produced zero rows.")

        combined = _append_parquet(MODEL_MARKET_SNAPSHOTS_PATH, snapshot_rows, SNAPSHOT_COLUMNS)
        _write_columns(combined, MODEL_MARKET_SNAPSHOTS_PATH, SNAPSHOT_COLUMNS)
        total_rows = len(combined)
    except Exception as exc:
        error = str(exc)
        total_rows = existing_rows
        raise
    finally:
        audit = _audit_row(
            capture_run_id=capture_run_id,
            capture_timestamp=capture_timestamp,
            model_mode=args.model_mode,
            registry_path=registry_path,
            selected_models=selected_models,
            missing=missing,
            model_df=model_df,
            market_df=market_df,
            snapshot_rows=snapshot_rows,
            existing_rows=existing_rows,
            total_rows=total_rows,
            error=error,
        )
        combined_audit = _append_parquet(MODEL_MARKET_SNAPSHOT_AUDIT_PATH, audit, AUDIT_COLUMNS)
        _write_columns(combined_audit, MODEL_MARKET_SNAPSHOT_AUDIT_PATH, AUDIT_COLUMNS)

    print()
    print("========== SNAPSHOT SUMMARY ==========")
    print("Selected models:", ", ".join(str(row["model_id"]) for row in selected_models))
    if missing:
        print("Missing model artifacts skipped:", "; ".join(missing))
    print("Model rows:", len(model_df))
    print("Market rows:", len(market_df))
    print("Snapshot rows appended:", len(snapshot_rows))
    print("Snapshot history rows:", total_rows)
    print("Files saved:")
    print(MODEL_MARKET_SNAPSHOTS_PATH)
    print(MODEL_MARKET_SNAPSHOT_AUDIT_PATH)


if __name__ == "__main__":
    main()
