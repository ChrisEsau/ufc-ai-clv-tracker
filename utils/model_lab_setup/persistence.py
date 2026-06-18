from __future__ import annotations

from copy import deepcopy
from typing import Any

import requests

import utils.model_lab_workflows as mlw
from utils.model_lab_setup.config_io import build_config_payload_from_form, dump_model_config
from utils.model_lab_setup.registry_io import remove_model_registry_entry, upsert_model_registry_entry
from utils.model_lab_setup.validators import (
    combine_validation_results,
    validate_delete_allowed,
    validate_model_id_available,
    validate_model_setup_form,
    validate_save_allowed,
)


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


def _github_delete_file(path: str, message: str) -> tuple[bool, str]:
    owner, repo, token, branch = mlw.get_github_config()
    if not owner or not repo or not token:
        return False, "Missing GitHub Streamlit secrets."

    ok, _, sha = mlw._github_read_file(path)
    if not ok:
        return False, "Could not inspect existing GitHub file before delete."
    if not sha:
        return True, f"No config file found at {path}; registry entry can still be removed."

    response = requests.delete(
        f"{mlw.GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}",
        headers=mlw.github_headers(token),
        json={"message": message, "sha": sha, "branch": branch},
        timeout=20,
    )
    if response.status_code in {200, 201}:
        return True, f"Deleted {path}."
    return False, f"GitHub API error {response.status_code}: {response.text}"


def save_model_setup(context: dict[str, Any], registry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Create or update a model config and registry entry on GitHub."""

    model_id = str(context.get("model_id") or "")
    model_exists = model_id in (registry.get("models") or {})

    validation = combine_validation_results(
        validate_save_allowed(context),
        validate_model_setup_form(context, registry, payload),
    )
    if context.get("is_new_model") or not model_exists:
        validation = combine_validation_results(validation, validate_model_id_available(registry, model_id))
    if not validation["ok"]:
        return {
            "ok": False,
            "message": "; ".join(validation["errors"]),
            "model_id": model_id,
            "config_path": str(context.get("config_path") or ""),
        }

    config = build_config_payload_from_form(context, payload)
    config_path = str(context.get("config_path") or "")
    if not config_path:
        return {"ok": False, "message": "Config path is missing.", "model_id": model_id, "config_path": config_path}

    registry_entry = _build_registry_entry(context, payload, config)
    updated_registry = upsert_model_registry_entry(registry, model_id, registry_entry)

    ok, msg = mlw._github_write_file(
        config_path,
        dump_model_config(config),
        f"Save draft model config {model_id}",
    )
    if not ok:
        return {"ok": False, "message": msg, "model_id": model_id, "config_path": config_path}

    ok, msg = mlw._save_registry(updated_registry)
    if not ok:
        return {"ok": False, "message": msg, "model_id": model_id, "config_path": config_path}

    action = "Created" if not model_exists else "Updated"
    return {"ok": True, "message": f"{action} model {model_id} on GitHub.", "model_id": model_id, "config_path": config_path}


def delete_model_setup(context: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """Delete a model config and registry entry on GitHub. Artifacts are not deleted."""

    model_id = str(context.get("model_id") or "")
    validation = validate_delete_allowed(context, registry)
    if not validation["ok"]:
        return {"ok": False, "message": "; ".join(validation["errors"]), "model_id": model_id}

    config_path = str(context.get("config_path") or "")
    if config_path:
        ok, msg = _github_delete_file(config_path, f"Delete model config {model_id}")
        if not ok:
            return {"ok": False, "message": msg, "model_id": model_id}

    updated_registry = remove_model_registry_entry(deepcopy(registry), model_id)
    ok, msg = mlw._save_registry(updated_registry)
    if not ok:
        return {"ok": False, "message": msg, "model_id": model_id}

    return {"ok": True, "message": f"Deleted model {model_id} on GitHub. Artifacts were not deleted.", "model_id": model_id}
