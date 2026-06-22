"""Materialize feature_registry.yaml feature definitions.

This module is the bridge between the Model Lab feature registry and actual
feature-view/training dataframes. It currently supports the first production
slice: two-input red/blue transform features and formula features whose inputs
already exist in the dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pipeline.features.formula_engine import compute_formula_feature
from pipeline.features.transform_engine import load_transform_plugins
from utils.feature_registry import load_feature_registry

DEFAULT_FEATURE_REGISTRY_PATH = "configs/features/feature_registry.yaml"
DEFAULT_TRANSFORM_REGISTRY_PATH = "configs/features/transform_registry.yaml"


@dataclass(frozen=True)
class RegistryFeatureBuildResult:
    """Summary returned by the registry feature materializer."""

    dataframe: pd.DataFrame
    generated_columns: list[str]
    missing_inputs: dict[str, list[str]]
    skipped_features: dict[str, str]


def apply_registry_feature_definitions(
    df: pd.DataFrame,
    *,
    registry_path: str | Path = DEFAULT_FEATURE_REGISTRY_PATH,
    transform_registry_path: str | Path = DEFAULT_TRANSFORM_REGISTRY_PATH,
    selected_features: Iterable[str] | None = None,
    allowed_statuses: Iterable[str] | None = None,
    overwrite_existing: bool = True,
) -> RegistryFeatureBuildResult:
    registry = load_feature_registry(registry_path)
    definitions = registry.get("feature_definitions", {}) or {}
    requested = _resolve_requested_features_with_dependencies(
        definitions=definitions,
        selected_features=selected_features,
    )
    statuses = {str(item) for item in (allowed_statuses or {"active", "draft"})}

    out = df.copy()
    generated: list[str] = []
    missing_inputs: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}

    for feature_id, definition in definitions.items():
        feature_id = str(feature_id)
        if not isinstance(definition, dict):
            skipped[feature_id] = "definition is not a mapping"
            continue

        output_column = str(definition.get("output_column") or feature_id)
        if requested and feature_id not in requested and output_column not in requested:
            continue
        if str(definition.get("status", "")).lower() not in statuses:
            skipped[feature_id] = f"status not allowed: {definition.get('status')}"
            continue
        if output_column in out.columns and not overwrite_existing:
            skipped[feature_id] = f"output already exists: {output_column}"
            continue

        feature_type = str(definition.get("type", "")).lower()
        if feature_type == "transform":
            built = _apply_transform_feature(
                out=out,
                feature_id=feature_id,
                output_column=output_column,
                definition=definition,
                transform_registry_path=transform_registry_path,
            )
        elif feature_type == "formula":
            built = _apply_formula_feature(
                out=out,
                feature_id=feature_id,
                output_column=output_column,
                definition=definition,
            )
        else:
            skipped[feature_id] = f"unsupported type: {feature_type}"
            continue

        if built["status"] == "generated":
            out[output_column] = built["series"]
            generated.append(output_column)
        elif built["status"] == "missing_inputs":
            missing_inputs[feature_id] = built["missing"]
        else:
            skipped[feature_id] = built.get("reason", "not generated")

    return RegistryFeatureBuildResult(
        dataframe=out,
        generated_columns=dedupe_preserve_order(generated),
        missing_inputs=missing_inputs,
        skipped_features=skipped,
    )


def _resolve_requested_features_with_dependencies(
    *,
    definitions: dict[str, Any],
    selected_features: Iterable[str] | None,
) -> set[str]:
    """Return selected registry features plus registry-defined dependencies.

    Live prediction passes the exact model feature contract as ``selected_features``.
    Some formula features depend on intermediate registry formulas that are not
    model inputs, for example style ``*_net`` features depend on matching
    ``*_reverse`` features. Without dependency expansion, the materializer skips
    the intermediate feature and the requested feature fails missing-input
    validation.
    """

    requested = {str(item) for item in selected_features or [] if str(item).strip()}
    if not requested:
        return set()

    output_to_feature_id = {
        str(definition.get("output_column") or feature_id): str(feature_id)
        for feature_id, definition in definitions.items()
        if isinstance(definition, dict)
    }

    expanded = set(requested)
    changed = True
    while changed:
        changed = False
        for feature_id, definition in definitions.items():
            feature_id = str(feature_id)
            if not isinstance(definition, dict):
                continue

            output_column = str(definition.get("output_column") or feature_id)
            if feature_id not in expanded and output_column not in expanded:
                continue

            for input_name in definition.get("inputs", []) or []:
                input_name = str(input_name)
                dependency_feature_id = output_to_feature_id.get(input_name)
                if dependency_feature_id and dependency_feature_id not in expanded:
                    expanded.add(dependency_feature_id)
                    expanded.add(input_name)
                    changed = True

    return expanded


def _apply_transform_feature(
    *,
    out: pd.DataFrame,
    feature_id: str,
    output_column: str,
    definition: dict[str, Any],
    transform_registry_path: str | Path,
) -> dict[str, Any]:
    inputs = [str(item) for item in definition.get("inputs", []) or [] if str(item).strip()]
    if len(inputs) < 2:
        return {"status": "skipped", "reason": "transform feature requires at least two inputs"}

    red_col, blue_col = inputs[0], inputs[1]
    missing = [column for column in [red_col, blue_col] if column not in out.columns]
    if missing:
        return {"status": "missing_inputs", "missing": missing}

    transform_id = str(definition.get("transform") or "").strip()
    if not transform_id:
        return {"status": "skipped", "reason": "missing transform id"}

    plugin = load_transform_plugins((transform_id,), str(transform_registry_path))[0]
    series = plugin.apply(
        pd.to_numeric(out[red_col], errors="coerce"),
        pd.to_numeric(out[blue_col], errors="coerce"),
        {"feature_id": feature_id, "output_column": output_column, "definition": definition},
    )
    return {"status": "generated", "series": series}


def _apply_formula_feature(
    *,
    out: pd.DataFrame,
    feature_id: str,
    output_column: str,
    definition: dict[str, Any],
) -> dict[str, Any]:
    formula = str(definition.get("formula") or "").strip()
    if not formula:
        return {"status": "skipped", "reason": "missing formula"}

    inputs = [str(item) for item in definition.get("inputs", []) or [] if str(item).strip()]
    missing = [column for column in inputs if column not in out.columns]
    if missing:
        return {"status": "missing_inputs", "missing": missing}

    try:
        series = compute_formula_feature(out, formula)
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"formula failed: {exc}"}
    return {"status": "generated", "series": series}


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        value = str(value)
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
