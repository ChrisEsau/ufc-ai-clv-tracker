from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path("configs/models/model_registry.yaml")
ENV_MODEL_ID = "UFC_MODEL_ID"


class ModelRegistryError(RuntimeError):
    """Raised when model registry configuration is invalid."""



def load_model_registry(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load the UFC model registry YAML.

    Parameters
    ----------
    registry_path:
        Path to model_registry.yaml.

    Returns
    -------
    dict
        Parsed registry dictionary.
    """

    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise ModelRegistryError(
            f"Model registry not found: {registry_path}"
        )

    with registry_path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    if not isinstance(registry, dict):
        raise ModelRegistryError(
            "Model registry must deserialize into a dictionary."
        )

    if "models" not in registry:
        raise ModelRegistryError(
            "Model registry missing required 'models' section."
        )

    return registry



def get_model_entry(
    model_id: str,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Return a model entry from the registry."""

    models = registry.get("models", {})

    if model_id not in models:
        available = sorted(models.keys())

        raise ModelRegistryError(
            f"Unknown model_id '{model_id}'. "
            f"Available models: {available}"
        )

    return models[model_id]



def get_dashboard_selectable_models(
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return models that can be selected in the dashboard."""

    selectable = []

    for model_id, entry in registry.get("models", {}).items():
        if entry.get("dashboard_selectable", False):
            selectable.append(
                {
                    "model_id": model_id,
                    **entry,
                }
            )

    return selectable



def get_active_model_id(
    model_family: str,
    registry: dict[str, Any],
) -> str:
    """Return the active model id for a model family."""

    active_models = registry.get("active_models", {})

    if model_family not in active_models:
        raise ModelRegistryError(
            f"No active model configured for family '{model_family}'."
        )

    primary_model = active_models[model_family].get("primary")

    if not primary_model:
        raise ModelRegistryError(
            f"Model family '{model_family}' has no primary model configured."
        )

    return str(primary_model)



def resolve_selected_model_id(
    model_family: str,
    registry: dict[str, Any],
    model_id: str | None = None,
) -> str:
    """Resolve model selection using approved precedence.

    Precedence:
        1. Explicit function argument.
        2. UFC_MODEL_ID environment variable.
        3. Active model from registry.
    """

    if model_id:
        return model_id

    env_model_id = os.getenv(ENV_MODEL_ID)

    if env_model_id:
        return env_model_id

    return get_active_model_id(
        model_family=model_family,
        registry=registry,
    )
