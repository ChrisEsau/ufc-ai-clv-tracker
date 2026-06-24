from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path("configs/models/model_registry.yaml")
ENV_MODEL_ID = "UFC_MODEL_ID"
ENV_MARKET_KEY = "UFC_MARKET_KEY"


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



def _normalize_market_key(market_key: str | None) -> str:
    return str(market_key or "").strip().lower()



def _family_active_config(model_family: str, registry: dict[str, Any]) -> dict[str, Any]:
    active_models = registry.get("active_models", {}) or {}

    if model_family not in active_models:
        raise ModelRegistryError(
            f"No active model configured for family '{model_family}'."
        )

    family_config = active_models[model_family] or {}
    if not isinstance(family_config, dict):
        raise ModelRegistryError(
            f"Active model config for family '{model_family}' must be a mapping."
        )

    return family_config



def get_active_model_id(
    model_family: str,
    registry: dict[str, Any],
    market_key: str | None = None,
) -> str:
    """Return the active model id for a model family and optional market.

    Backward compatibility:
    - Existing callers that only pass model_family still receive the family primary.
    - Market-aware callers can pass market_key so each prop market can have its own
      active production model without replacing other prop markets.
    """

    family = str(model_family or "").strip().lower()
    active_config = _family_active_config(family, registry)
    normalized_market_key = _normalize_market_key(market_key)

    if normalized_market_key:
        market_model = active_config.get(normalized_market_key)
        if market_model:
            return str(market_model)
        if family == "moneyline" and normalized_market_key == "moneyline" and active_config.get("primary"):
            return str(active_config["primary"])
        raise ModelRegistryError(
            f"No active model configured for family '{family}' and market '{normalized_market_key}'."
        )

    primary_model = active_config.get("primary")

    if not primary_model:
        raise ModelRegistryError(
            f"Model family '{family}' has no primary model configured."
        )

    return str(primary_model)



def get_active_model_ids_by_market(
    model_family: str,
    registry: dict[str, Any],
) -> dict[str, str]:
    """Return market_key -> active model id for one model family."""

    family = str(model_family or "").strip().lower()
    active_config = _family_active_config(family, registry)
    by_market: dict[str, str] = {}
    for key, value in active_config.items():
        normalized_key = _normalize_market_key(str(key))
        if not normalized_key or normalized_key == "primary" or not value:
            continue
        by_market[normalized_key] = str(value)
    return by_market



def resolve_selected_model_id(
    model_family: str,
    registry: dict[str, Any],
    model_id: str | None = None,
    market_key: str | None = None,
) -> str:
    """Resolve model selection using approved precedence.

    Precedence:
        1. Explicit function argument.
        2. UFC_MODEL_ID environment variable.
        3. Active model from registry, optionally scoped by market_key.
    """

    if model_id:
        return model_id

    env_model_id = os.getenv(ENV_MODEL_ID)

    if env_model_id:
        return env_model_id

    resolved_market_key = market_key or os.getenv(ENV_MARKET_KEY)

    return get_active_model_id(
        model_family=model_family,
        registry=registry,
        market_key=resolved_market_key,
    )
