from __future__ import annotations

from copy import deepcopy
from typing import Any

from utils.model_lab_setup.config_io import build_config_path, load_model_config, normalize_model_config
from utils.model_lab_setup.registry_io import get_model_registry_entry
from utils.model_lab_setup.versioning import artifact_dir_for_market


EDITABLE_STATUSES = {"draft"}


def _status_flags(status: str, is_new_model: bool = False) -> dict[str, bool]:
    normalized_status = str(status or "draft").lower()
    is_editable = bool(is_new_model) or normalized_status in EDITABLE_STATUSES
    return {"is_editable": is_editable, "is_read_only": not is_editable}


def resolve_existing_model_context(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Resolve a registered model into a full Model Setup context."""

    entry = get_model_registry_entry(registry, model_id)
    config_path = str(entry.get("config_path") or build_config_path(model_id))
    config = normalize_model_config(load_model_config(config_path))

    model_family = str(entry.get("model_family") or config.get("model_family") or "moneyline").strip().lower()
    market_key = str(
        entry.get("market_key")
        or config.get("market_key")
        or (config.get("prediction") or {}).get("market_key")
        or "moneyline"
    ).strip().lower()
    status = str(entry.get("status") or config.get("status") or "draft").lower()
    artifact_dir = str(
        entry.get("artifact_dir")
        or (config.get("artifacts") or {}).get("output_dir")
        or artifact_dir_for_market(model_id, market_key)
    )

    return {
        "mode": "existing",
        "model_id": model_id,
        "display_name": str(entry.get("display_name") or config.get("display_name") or model_id),
        "description": str(entry.get("description") or config.get("description") or ""),
        "status": status,
        "model_family": model_family,
        "market_key": market_key,
        "algorithm": str(entry.get("algorithm") or config.get("algorithm") or "xgboost"),
        "config_path": config_path,
        "artifact_dir": artifact_dir,
        "dashboard_selectable": bool(entry.get("dashboard_selectable", False)),
        "config": config,
        "registry_entry": deepcopy(entry),
        "is_new_model": False,
        **_status_flags(status, is_new_model=False),
    }


def summarize_context_for_ui(context: dict[str, Any]) -> dict[str, str]:
    """Return a compact display-safe context summary."""

    return {
        "model_id": str(context.get("model_id") or ""),
        "status": str(context.get("status") or ""),
        "family": str(context.get("model_family") or ""),
        "market": str(context.get("market_key") or ""),
        "config_path": str(context.get("config_path") or ""),
        "artifact_dir": str(context.get("artifact_dir") or ""),
        "editable_label": "Editable" if context.get("is_editable") else "Read-only",
    }
