"""Generic transform engine for UFC feature views.

This module applies reusable red/blue transform plugins to fighter feature
columns. Transform implementations are resolved from
``configs/features/transform_registry.yaml`` so new transforms can be added by
registering a plugin instead of editing this engine.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import yaml


DEFAULT_TRANSFORM_REGISTRY_PATH = "configs/features/transform_registry.yaml"


@dataclass(frozen=True)
class TransformResult:
    """Summary of transform engine output."""

    dataframe: pd.DataFrame
    generated_columns: list[str]
    missing_source_pairs: list[str]


@dataclass(frozen=True)
class TransformPlugin:
    """Loaded transform plugin metadata."""

    transform_id: str
    output_suffix: str
    apply: Callable[[pd.Series, pd.Series, dict | None], pd.Series]


def apply_red_blue_transforms(
    df: pd.DataFrame,
    base_columns: Iterable[str],
    transforms: Iterable[str],
    *,
    red_prefix: str = "r_pre_",
    blue_prefix: str = "b_pre_",
    transform_registry_path: str | Path = DEFAULT_TRANSFORM_REGISTRY_PATH,
    context: dict | None = None,
) -> TransformResult:
    """Apply registered red/blue transform plugins to a dataframe.

    Parameters
    ----------
    df:
        Input dataframe containing red and blue source columns.
    base_columns:
        Unprefixed base columns such as ``elo`` or ``splm``.
    transforms:
        Transform identifiers such as ``red_minus_blue``, ``absolute_gap``,
        and ``ratio``.
    red_prefix / blue_prefix:
        Source column prefixes. Defaults target prefight state columns like
        ``r_pre_elo`` and ``b_pre_elo``.
    transform_registry_path:
        YAML registry describing transform plugin module/function mappings.
    context:
        Optional metadata passed through to plugin ``apply`` functions.

    Returns
    -------
    TransformResult
        Copy of the input dataframe with generated columns appended or
        overwritten, plus generated-column and missing-source summaries.
    """

    out = df.copy()
    generated_columns: list[str] = []
    missing_source_pairs: list[str] = []
    new_columns: dict[str, pd.Series] = {}

    requested_transforms = [str(transform) for transform in transforms]
    plugins = load_transform_plugins(tuple(requested_transforms), str(transform_registry_path))

    for base_column in [str(column) for column in base_columns]:
        red_col = f"{red_prefix}{base_column}"
        blue_col = f"{blue_prefix}{base_column}"

        if red_col not in out.columns or blue_col not in out.columns:
            missing_source_pairs.append(f"{red_col}|{blue_col}")
            continue

        red_values = pd.to_numeric(out[red_col], errors="coerce")
        blue_values = pd.to_numeric(out[blue_col], errors="coerce")

        for plugin in plugins:
            output_col = f"{base_column}_{plugin.output_suffix}"
            new_columns[output_col] = plugin.apply(red_values, blue_values, context)
            generated_columns.append(output_col)

    if new_columns:
        out = out.drop(columns=[column for column in new_columns if column in out.columns])
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return TransformResult(
        dataframe=out,
        generated_columns=dedupe_preserve_order(generated_columns),
        missing_source_pairs=dedupe_preserve_order(missing_source_pairs),
    )


@lru_cache(maxsize=32)
def load_transform_plugins(
    transform_ids: tuple[str, ...],
    registry_path: str = DEFAULT_TRANSFORM_REGISTRY_PATH,
) -> list[TransformPlugin]:
    """Load transform plugins from the transform registry."""

    registry = load_transform_registry(registry_path)
    transform_registry = registry.get("transforms", {})
    plugins: list[TransformPlugin] = []

    for transform_id in dedupe_preserve_order(list(transform_ids)):
        if transform_id not in transform_registry:
            raise ValueError(f"Transform not found in registry: {transform_id}")

        entry = transform_registry[transform_id]
        status = str(entry.get("status", "")).lower()
        plugin_module = entry.get("plugin")
        function_name = entry.get("function")
        output_suffix = entry.get("output_suffix")

        if status != "active":
            raise ValueError(f"Transform is not active: {transform_id} (status={status})")
        if not plugin_module or not function_name:
            raise ValueError(f"Transform plugin/function missing for: {transform_id}")
        if not output_suffix:
            raise ValueError(f"Transform output_suffix missing for: {transform_id}")

        module = importlib.import_module(str(plugin_module))
        apply_func = getattr(module, str(function_name), None)
        if apply_func is None or not callable(apply_func):
            raise ValueError(
                f"Transform function is not callable: {plugin_module}.{function_name}"
            )

        plugins.append(
            TransformPlugin(
                transform_id=transform_id,
                output_suffix=str(output_suffix),
                apply=apply_func,
            )
        )

    return plugins


@lru_cache(maxsize=8)
def load_transform_registry(registry_path: str = DEFAULT_TRANSFORM_REGISTRY_PATH) -> dict:
    """Load the transform registry YAML."""

    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Transform registry not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    if not isinstance(registry, dict):
        raise ValueError(f"Transform registry must be a dictionary: {path}")

    return registry


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    """De-duplicate values while preserving first-seen order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
