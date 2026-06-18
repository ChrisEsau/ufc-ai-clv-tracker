from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from utils.model_lab_setup.config_io import build_config_payload_from_form, dump_model_config
from utils.model_lab_setup.registry_io import (
    remove_model_registry_entry,
    save_model_registry,
    upsert_model_registry_entry,
)
from utils.model_lab_setup.validators import (
    combine_validation_results,
    validate_delete_allowed,
    validate_model_id_available,
    validate_save_allowed,
)


def _write_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _build_registry_entry(context: dict[str, Any], payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    identity = payload.get("identity") or {}
    return {
        "display_name": str(identity.get("display_name") or context.get("display_name") or config.get("model_id")),
        "description": str(identity.get("description") or context.get("description") or ""),
        "model_family": str(config.get("model_family") or context.get("model_family") or "moneyline"),
        "market_key": str(config.get("market_key") or context.get("market_key") or "moneyline"),
        "algorithm": str(identity.get("algorithm") or context.get("algorithm") or config.get("algorithm") or "xgboost"),
        "config_path": str(context.get("config_path") or ""),
        "artifact_dir": str((config.get("artifacts") or {}).get("output_dir") or context.get("artifact_dir") or ""),
        "status": str(config.get("status") or context.get("status") or "draft"),
        "dashboard_selectable": bool(identity.get("dashboard_selectable", context.get("dashboard_selectable", False))),
        "outcome_architecture": True,
    }


def save_model_setup(context: dict[str, Any], registry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update a model config and registry entry.

    Save creates when context model_id is not present in the registry.
    Save updates when context model_id already exists and is editable.
    """

    model_id = str(context.get("model_id") or "")
    model_exists = model_id in (registry.get("models") or {})

    validation = validate_save_allowed(context)
    if context.get("is_new_model") or not model_exists:
        validation = combine_validation_results(validation, validate_model_id_available(registry, model_id))
    if not validation["ok"]:
        return {"ok": False, "message": "; ".join(validation["errors"]), "model_id": model_id, "config_path": str(context.get("config_path") or "")}

    config = build_config_payload_from_form(context, payload)
    config_path = str(context.get("config_path") or "")
    if not config_path:
        return {"ok": False, "message": "Config path is missing.", "model_id": model_id, "config_path": config_path}

    registry_entry = _build_registry_entry(context, payload, config)
    updated_registry = upsert_model_registry_entry(registry, model_id, registry_entry)

    _write_text(config_path, dump_model_config(config))
    registry_result = save_model_registry(updated_registry)
    if not registry_result.get("ok"):
        return {"ok": False, "message": str(registry_result.get("message") or "Registry save failed."), "model_id": model_id, "config_path": config_path}

    action = "Created" if not model_exists else "Updated"
    return {"ok": True, "message": f"{action} model {model_id}.", "model_id": model_id, "config_path": config_path}


def delete_model_setup(context: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """Delete a model config and registry entry. Artifacts are not deleted."""

    model_id = str(context.get("model_id") or "")
    validation = validate_delete_allowed(context, registry)
    if not validation["ok"]:
        return {"ok": False, "message": "; ".join(validation["errors"]), "model_id": model_id}

    config_path = str(context.get("config_path") or "")
    if config_path:
        target = Path(config_path)
        if target.exists():
            target.unlink()

    updated_registry = remove_model_registry_entry(deepcopy(registry), model_id)
    registry_result = save_model_registry(updated_registry)
    if not registry_result.get("ok"):
        return {"ok": False, "message": str(registry_result.get("message") or "Registry save failed."), "model_id": model_id}

    return {"ok": True, "message": f"Deleted model {model_id}. Artifacts were not deleted.", "model_id": model_id}
