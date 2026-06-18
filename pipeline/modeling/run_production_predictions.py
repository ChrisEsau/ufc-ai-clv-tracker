from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import yaml

from pipeline.common.paths import PREDICTIONS_DIR, ensure_data_dirs

DEFAULT_REGISTRY_PATH = Path("configs/models/model_registry.yaml")
DEFAULT_MODEL_OUTPUT_TEMPLATE = "data/predictions/by_model/{model_id}/model_outcomes.parquet"
PRODUCTION_PREDICTION_AUDIT_PATH = Path("data/audits/production_prediction_audit.parquet")

AUDIT_COLUMNS = [
    "production_prediction_run_id",
    "production_prediction_timestamp",
    "registry_path",
    "model_id",
    "model_family",
    "market_key",
    "status",
    "config_path",
    "model_outcomes_path",
    "return_code",
    "rows_written",
    "passes_validation",
    "error",
]


class ProductionPredictionRunnerError(RuntimeError):
    """Raised when one or more production model predictions fail."""


def _utc_run() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("production_pred_%Y%m%d_%H%M%S"), now.isoformat()


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file) or {}
    if not isinstance(registry, dict):
        raise ValueError(f"Model registry must be a mapping: {path}")
    if not isinstance(registry.get("models"), dict):
        raise ValueError("Model registry missing required mapping: models")
    return registry


def _model_output_path(*, model_id: str, entry: dict[str, Any]) -> Path:
    explicit_path = entry.get("model_outcomes_path")
    if explicit_path:
        return Path(str(explicit_path))
    return Path(DEFAULT_MODEL_OUTPUT_TEMPLATE.format(model_id=model_id))


def _select_production_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    production_by_market: dict[str, str] = {}

    for model_id, entry in (registry.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status != "production":
            continue

        model_family = str(entry.get("model_family") or "").strip().lower()
        market_key = str(entry.get("market_key") or "").strip().lower()
        if not market_key and model_family == "moneyline":
            market_key = "moneyline"

        if not model_family:
            raise ProductionPredictionRunnerError(f"Production model '{model_id}' is missing model_family.")
        if not entry.get("config_path"):
            raise ProductionPredictionRunnerError(f"Production model '{model_id}' is missing config_path.")
        if not market_key:
            raise ProductionPredictionRunnerError(f"Production model '{model_id}' is missing market_key.")

        existing = production_by_market.get(market_key)
        if existing:
            raise ProductionPredictionRunnerError(
                "Only one production model is allowed per market. "
                f"Market '{market_key}' has both '{existing}' and '{model_id}'."
            )
        production_by_market[market_key] = str(model_id)

        selected.append(
            {
                "model_id": str(model_id),
                "model_family": model_family,
                "market_key": market_key,
                "status": status,
                "config_path": str(entry.get("config_path")),
                "model_outcomes_path": _model_output_path(model_id=str(model_id), entry=entry),
            }
        )

    if not selected:
        raise ProductionPredictionRunnerError("No status=production models found in registry.")

    return selected


def _run_prediction_for_model(model: dict[str, Any], *, registry_path: Path, prefer_raw_model: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "pipeline.modeling.run_prediction",
        "--registry-path",
        str(registry_path),
        "--model-family",
        str(model["model_family"]),
        "--model-id",
        str(model["model_id"]),
    ]
    if prefer_raw_model:
        command.append("--prefer-raw-model")

    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _count_output_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Model-scoped prediction artifact not found after run: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Model-scoped prediction artifact is empty after run: {path}")
    return int(len(df))


def _audit_row(
    *,
    production_prediction_run_id: str,
    production_prediction_timestamp: str,
    registry_path: Path,
    model: dict[str, Any],
    return_code: int | None,
    rows_written: int | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "production_prediction_run_id": production_prediction_run_id,
        "production_prediction_timestamp": production_prediction_timestamp,
        "registry_path": str(registry_path),
        "model_id": model.get("model_id"),
        "model_family": model.get("model_family"),
        "market_key": model.get("market_key"),
        "status": model.get("status"),
        "config_path": model.get("config_path"),
        "model_outcomes_path": str(model.get("model_outcomes_path")),
        "return_code": return_code,
        "rows_written": rows_written,
        "passes_validation": bool(error is None and rows_written and rows_written > 0),
        "error": error,
    }


def _write_audit(rows: list[dict[str, Any]]) -> None:
    audit_df = pd.DataFrame(rows)
    for column in AUDIT_COLUMNS:
        if column not in audit_df.columns:
            audit_df[column] = pd.NA
    PRODUCTION_PREDICTION_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df[AUDIT_COLUMNS].to_parquet(PRODUCTION_PREDICTION_AUDIT_PATH, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions for every status=production model in the registry.")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--prefer-raw-model", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_data_dirs()
    registry_path = Path(args.registry_path)
    production_prediction_run_id, production_prediction_timestamp = _utc_run()

    print("=" * 80)
    print("UFC PRODUCTION MODEL PREDICTIONS")
    print("=" * 80)
    print("Production prediction run ID:", production_prediction_run_id)
    print("Registry path:", registry_path)

    registry = _load_registry(registry_path)
    models = _select_production_models(registry)
    print("Production models:", ", ".join(model["model_id"] for model in models))

    audit_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for model in models:
        rows_written: int | None = None
        error: str | None = None
        return_code: int | None = 0
        try:
            print()
            print("---------- RUN PRODUCTION MODEL ----------")
            print("Model ID:", model["model_id"])
            print("Family:", model["model_family"])
            print("Market:", model["market_key"])
            _run_prediction_for_model(model, registry_path=registry_path, prefer_raw_model=bool(args.prefer_raw_model))
            rows_written = _count_output_rows(Path(model["model_outcomes_path"]))
            print("Rows written:", rows_written)
        except subprocess.CalledProcessError as exc:
            return_code = int(exc.returncode)
            error = str(exc)
            failures.append(f"{model['model_id']}: {error}")
        except Exception as exc:
            return_code = 1
            error = str(exc)
            failures.append(f"{model['model_id']}: {error}")
        finally:
            audit_rows.append(
                _audit_row(
                    production_prediction_run_id=production_prediction_run_id,
                    production_prediction_timestamp=production_prediction_timestamp,
                    registry_path=registry_path,
                    model=model,
                    return_code=return_code,
                    rows_written=rows_written,
                    error=error,
                )
            )

    _write_audit(audit_rows)

    print()
    print("========== PRODUCTION PREDICTION SUMMARY ==========")
    print("Models attempted:", len(models))
    print("Models failed:", len(failures))
    print("Audit saved:", PRODUCTION_PREDICTION_AUDIT_PATH)

    if failures:
        raise ProductionPredictionRunnerError("Production prediction failures: " + "; ".join(failures))


if __name__ == "__main__":
    main()
