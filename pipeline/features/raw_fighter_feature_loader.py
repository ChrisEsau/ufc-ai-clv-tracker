"""Loader for raw fighter feature plugins.

This module reads ``configs/features/raw_fighter_feature_registry.yaml`` and
validates/imports active plugin entries. It is intentionally separate from the
current fighter-state builder so plugins can be introduced and validated before
changing production feature generation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml


DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH = "configs/features/raw_fighter_feature_registry.yaml"


@dataclass(frozen=True)
class RawFighterFeaturePlugin:
    """Loaded raw fighter feature plugin metadata."""

    feature_group: str
    output_columns: list[str]
    calculate: Callable[[pd.DataFrame, pd.Series, dict | None], dict]


@lru_cache(maxsize=8)
def load_raw_fighter_feature_registry(
    registry_path: str = DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH,
) -> dict:
    """Load the raw fighter feature registry YAML."""

    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw fighter feature registry not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    if not isinstance(registry, dict):
        raise ValueError(f"Raw fighter feature registry must be a dictionary: {path}")

    return registry


def list_registered_feature_groups(
    registry_path: str = DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH,
) -> list[str]:
    """Return all registered feature group names."""

    registry = load_raw_fighter_feature_registry(registry_path)
    return list(registry.get("feature_groups", {}).keys())


def list_registered_outputs(
    registry_path: str = DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH,
) -> dict[str, list[str]]:
    """Return registered output columns by feature group."""

    registry = load_raw_fighter_feature_registry(registry_path)
    groups = registry.get("feature_groups", {})
    return {
        group_name: [str(column) for column in group.get("output_columns", [])]
        for group_name, group in groups.items()
    }


def load_active_raw_fighter_feature_plugins(
    registry_path: str = DEFAULT_RAW_FIGHTER_FEATURE_REGISTRY_PATH,
) -> list[RawFighterFeaturePlugin]:
    """Load active raw fighter feature plugins from the registry.

    Existing legacy groups are intentionally skipped until they are migrated to
    plugin implementations and marked ``status: active``.
    """

    registry = load_raw_fighter_feature_registry(registry_path)
    groups = registry.get("feature_groups", {})
    plugins: list[RawFighterFeaturePlugin] = []

    for group_name, group in groups.items():
        status = str(group.get("status", "")).lower()
        if status != "active":
            continue

        plugin_module = group.get("plugin")
        function_name = group.get("function")
        output_columns = [str(column) for column in group.get("output_columns", [])]

        if not plugin_module or not function_name:
            raise ValueError(f"Active feature group missing plugin/function: {group_name}")
        if not output_columns:
            raise ValueError(f"Active feature group missing output columns: {group_name}")

        module = importlib.import_module(str(plugin_module))
        calculate_func = getattr(module, str(function_name), None)
        if calculate_func is None or not callable(calculate_func):
            raise ValueError(
                f"Raw fighter feature function is not callable: {plugin_module}.{function_name}"
            )

        plugins.append(
            RawFighterFeaturePlugin(
                feature_group=str(group_name),
                output_columns=output_columns,
                calculate=calculate_func,
            )
        )

    return plugins
