# ============================================================
# pipeline/betting/run_betting_outcomes_v2.py
# ============================================================

"""Build generic outcome-level betting opportunities.

Betting Outcomes V2 joins prediction outcomes to market outcomes using the
canonical ID-based key:

    fight_id + market_key + outcome_join_key

This runner consumes existing market_outcomes.parquet and registry-selected
model-scoped prediction artifacts. It does not scrape markets or generate model
predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline.betting.betting_audit import build_betting_audit
from pipeline.betting.betting_joiner import build_betting_outcomes
from pipeline.common.paths import (
    BETTING_OUTCOMES_AUDIT_PATH,
    BETTING_OUTCOMES_PATH,
    MARKET_OUTCOMES_PATH,
    PREDICTIONS_DIR,
    ensure_data_dirs,
)
from pipeline.common.risk_settings import load_risk_settings


CANONICAL_MODEL_OUTCOMES_PATH = PREDICTIONS_DIR / "model_outcomes.parquet"
DEFAULT_REGISTRY_PATH = Path("configs/models/model_registry.yaml")
DEFAULT_MODEL_OUTPUT_TEMPLATE = "data/predictions/by_model/{model_id}/model_outcomes.parquet"


class BettingOutcomeRunnerError(RuntimeError):
    """Raised when Betting Outcomes V2 cannot build the board artifact."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UFC Betting Outcomes V2.")
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Model registry YAML used for production/all model aggregation.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["production", "all", "single"],
        default="production",
        help=(
            "production loads status: production models from the registry; "
            "all loads non-archived registry models; single loads the canonical "
            "data/predictions/model_outcomes.parquet artifact."
        ),
    )
    return parser.parse_args()


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("betting_%Y%m%d_%H%M%S"), now.isoformat()


def _load_required_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_parquet(path)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BettingOutcomeRunnerError(f"Model registry not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file) or {}
    if not isinstance(registry, dict):
        raise BettingOutcomeRunnerError(f"Model registry must be a mapping: {path}")
    return registry


def _select_registry_models(registry: dict[str, Any], *, model_mode: str) -> list[dict[str, Any]]:
    models = registry.get("models", {}) or {}
    if not isinstance(models, dict):
        raise BettingOutcomeRunnerError("Model registry 'models' section must be a mapping.")

    selected: list[dict[str, Any]] = []
    production_by_market: dict[str, str] = {}

    for model_id, entry in models.items():
        if not isinstance(entry, dict):
            continue

        status = str(entry.get("status") or "").strip().lower()
        if model_mode == "production" and status != "production":
            continue
        if model_mode == "all" and status == "archived":
            continue

        model_family = str(entry.get("model_family") or "").strip().lower()
        market_key = str(entry.get("market_key") or "").strip().lower()
        if not market_key and model_family == "moneyline":
            market_key = "moneyline"

        if status == "production":
            existing = production_by_market.get(market_key)
            if existing:
                raise BettingOutcomeRunnerError(
                    "Only one production model is allowed per market. "
                    f"Market '{market_key}' has both '{existing}' and '{model_id}'."
                )
            production_by_market[market_key] = str(model_id)

        selected.append(
            {
                "model_id": str(model_id),
                "status": status,
                "model_family": model_family,
                "market_key": market_key,
                "path": _model_output_path(model_id=str(model_id), entry=entry),
            }
        )

    if not selected:
        raise BettingOutcomeRunnerError(f"No models selected for model_mode='{model_mode}'.")

    return selected


def _model_output_path(*, model_id: str, entry: dict[str, Any]) -> Path:
    explicit_path = entry.get("model_outcomes_path")
    if explicit_path:
        return Path(str(explicit_path))
    return Path(DEFAULT_MODEL_OUTPUT_TEMPLATE.format(model_id=model_id))


def _load_registry_model_outcomes(*, registry_path: Path, model_mode: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    registry = _load_registry(registry_path)
    selected = _select_registry_models(registry, model_mode=model_mode)

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

        if "model_id" not in frame.columns:
            frame["model_id"] = model_id
        else:
            frame["model_id"] = frame["model_id"].fillna(model_id)

        if "model_family" not in frame.columns:
            frame["model_family"] = row["model_family"]
        else:
            frame["model_family"] = frame["model_family"].fillna(row["model_family"])

        if "market_key" not in frame.columns:
            frame["market_key"] = row["market_key"]
        else:
            frame["market_key"] = frame["market_key"].fillna(row["market_key"])

        frame["model_registry_status"] = row["status"]
        frame["model_outcomes_path"] = str(path)
        frames.append(frame)

    if missing:
        raise BettingOutcomeRunnerError(
            "Missing model-scoped prediction artifacts. Run prediction for these models first: "
            + "; ".join(missing)
        )

    if not frames:
        raise BettingOutcomeRunnerError("Selected model outcome artifacts were empty.")

    return pd.concat(frames, ignore_index=True, sort=False), selected


def _load_model_outcomes(*, registry_path: Path, model_mode: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if model_mode == "single":
        return _load_required_parquet(CANONICAL_MODEL_OUTCOMES_PATH, "Model outcomes"), [
            {
                "model_id": "canonical_model_outcomes",
                "status": "single",
                "model_family": "single",
                "market_key": "single",
                "path": CANONICAL_MODEL_OUTCOMES_PATH,
            }
        ]

    return _load_registry_model_outcomes(registry_path=registry_path, model_mode=model_mode)


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_path)

    print("=" * 80)
    print("UFC BETTING OUTCOMES V2")
    print("=" * 80)

    ensure_data_dirs()
    betting_run_id, betting_timestamp = _utc_run()

    model_df, selected_models = _load_model_outcomes(
        registry_path=registry_path,
        model_mode=args.model_mode,
    )
    market_df = _load_required_parquet(MARKET_OUTCOMES_PATH, "Market outcomes")
    settings = load_risk_settings()

    print("Betting run ID:", betting_run_id)
    print("Model mode:", args.model_mode)
    print("Registry path:", registry_path)
    print("Selected models:", ", ".join(str(row["model_id"]) for row in selected_models))
    print("Market outcomes path:", MARKET_OUTCOMES_PATH)
    print("Model rows:", len(model_df))
    print("Market rows:", len(market_df))
    print("Risk settings:", asdict(settings))

    betting_df = build_betting_outcomes(
        model_df=model_df,
        market_df=market_df,
        settings=settings,
        betting_run_id=betting_run_id,
        betting_timestamp=betting_timestamp,
    )
    audit_df = build_betting_audit(
        model_df=model_df,
        market_df=market_df,
        betting_df=betting_df,
        betting_run_id=betting_run_id,
        betting_timestamp=betting_timestamp,
    )

    betting_df.to_parquet(BETTING_OUTCOMES_PATH, index=False)
    audit_df.to_parquet(BETTING_OUTCOMES_AUDIT_PATH, index=False)

    print()
    print("========== BETTING OUTCOMES V2 SUMMARY ==========")
    print("Joined rows:", len(betting_df))
    print("Bet candidates:", int(betting_df["is_bet_candidate"].fillna(False).sum()) if not betting_df.empty else 0)
    print("Validation passes:", bool(audit_df["passes_validation"].iloc[0]))
    print()
    print("Files saved:")
    print(BETTING_OUTCOMES_PATH)
    print(BETTING_OUTCOMES_AUDIT_PATH)


if __name__ == "__main__":
    main()
