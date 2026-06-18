from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MODEL_REGISTRY_PATH = Path("configs/models/model_registry.yaml")


def _empty_registry() -> dict[str, Any]:
    return {"models": {}, "active_models": {}}


def load_model_registry(path: str | Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    """Load the model registry YAML, returning a valid empty registry if missing."""

    registry_path = Path(path)
    if not registry_path.exists():
        return _empty_registry()

    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Model registry must be a mapping: {registry_path}")

    registry = _empty_registry()
    registry.update(payload)
    registry["models"] = registry.get("models") or {}
    registry["active_models"] = registry.get("active_models") or {}
    return registry


def save_model_registry(
    registry: dict[str, Any],
    path: str | Path = MODEL_REGISTRY_PATH,
) -> dict[str, Any]:
    """Save the model registry to a local YAML file."""

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {"ok": True, "message": f"Saved model registry to {registry_path}", "path": str(registry_path)}


def get_registered_model_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized model rows for selectors/tables."""

    rows: list[dict[str, Any]] = []
    for model_id, entry in (registry.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "model_id": str(model_id),
                "display_name": str(entry.get("display_name") or model_id),
                "description": str(entry.get("description") or ""),
                "model_family": str(entry.get("model_family") or ""),
                "market_key": str(entry.get("market_key") or "moneyline"),
                "algorithm": str(entry.get("algorithm") or ""),
                "status": str(entry.get("status") or "draft"),
                "config_path": str(entry.get("config_path") or ""),
                "artifact_dir": str(entry.get("artifact_dir") or ""),
                "dashboard_selectable": bool(entry.get("dashboard_selectable", False)),
            }
        )
    return rows


def get_model_registry_entry(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Return a model registry entry or raise KeyError."""

    models = registry.get("models") or {}
    if model_id not in models:
        raise KeyError(f"Model not found in registry: {model_id}")
    entry = models[model_id]
    if not isinstance(entry, dict):
        raise ValueError(f"Model registry entry must be a mapping: {model_id}")
    return entry


def get_active_primary_model_id(registry: dict[str, Any], model_family: str) -> str:
    """Return the active primary model id for a model family."""

    return str(((registry.get("active_models") or {}).get(model_family) or {}).get("primary") or "")


def is_active_primary(registry: dict[str, Any], model_id: str, model_family: str) -> bool:
    """Return whether model_id is the active primary for model_family."""

    return get_active_primary_model_id(registry, model_family) == model_id


def upsert_model_registry_entry(
    registry: dict[str, Any],
    model_id: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Return a registry copy with one model entry inserted/replaced."""

    updated = deepcopy(registry)
    updated.setdefault("models", {})[model_id] = deepcopy(entry)
    updated.setdefault("active_models", registry.get("active_models") or {})
    return updated


def remove_model_registry_entry(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Return a registry copy with one model entry removed."""

    updated = deepcopy(registry)
    updated.setdefault("models", {}).pop(model_id, None)
    return updated
