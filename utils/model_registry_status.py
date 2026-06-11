from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


MODEL_REGISTRY_PATH = Path("configs/models/model_registry.yaml")
PRODUCTION_MODE = "Production only"
ALL_MODE = "All models"
MODEL_MODE_OPTIONS = [PRODUCTION_MODE, ALL_MODE]


@pd.api.extensions.register_dataframe_accessor("_noop_model_registry_status")
class _NoopAccessor:
    """Private no-op accessor to keep pandas import from appearing unused in linters."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj



def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    """Load the model registry used by dashboard model filtering."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return payload if isinstance(payload, dict) else {}


def registered_model_status_map(path: Path = MODEL_REGISTRY_PATH) -> dict[str, str]:
    """Return model_id -> status from the registry."""

    registry = load_model_registry(path)
    models = registry.get("models", {}) or {}
    status_map: dict[str, str] = {}
    for model_id, entry in models.items():
        if isinstance(entry, dict):
            status_map[str(model_id)] = str(entry.get("status") or "").strip().lower()
    return status_map


def production_model_ids(path: Path = MODEL_REGISTRY_PATH) -> set[str]:
    """Return model IDs marked status: production."""

    return {
        model_id
        for model_id, status in registered_model_status_map(path).items()
        if status == "production"
    }


def filter_betting_outcomes_by_model_mode(
    outcomes: pd.DataFrame,
    *,
    model_mode: str | None,
    registry_path: Path = MODEL_REGISTRY_PATH,
) -> pd.DataFrame:
    """Filter Betting Board rows by registry status.

    Production mode intentionally uses only `status: production` from the model
    registry. All-model mode leaves rows untouched so draft/research models can be
    reviewed without becoming the production board default.
    """

    if outcomes is None or outcomes.empty:
        return pd.DataFrame()

    if model_mode == ALL_MODE:
        return outcomes.copy()

    if "model_id" not in outcomes.columns:
        return outcomes.copy()

    production_ids = production_model_ids(registry_path)
    if not production_ids:
        return outcomes.iloc[0:0].copy()

    return outcomes[outcomes["model_id"].astype(str).isin(production_ids)].copy()
