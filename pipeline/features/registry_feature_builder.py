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
import yaml

from pipeline.features.formula_engine import compute_formula_feature
from pipeline.features.transform_engine import load_transform_plugins

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
    """Apply feature definitions from ``feature_registry.yaml`` to a dataframe.

    Parameters
    ----------
    df:
        Input feature dataframe.
    registry_path:
        Path to the canonical feature registry.
    transform_registry_path:
        Path to the transform plugin registry.
    selected_features:
        Optional set of feature IDs/output columns to build. When omitted, all
        buildable registry features with an allowed status are considered.
    allowed_statuses:
        Feature statuses allowed to materialize. Defaults to active and draft so
        Model Lab experiments can train before promotion.
    overwrite_existing:
        If true, registry definitions may overwrite an existing column of the
        same output name. This keeps definitions authoritative during migration.
    """

    registry = load_feature_registry(registry_path)
    definitions = registry.get("feature_definitions", {}) or {}
    requested = {str(item) for item in selected_features or [] if str(item).strip()}
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
        {
            "feature_id": feature_id,
            "output_column": output_column,
            "definition": definition,
        },
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
    except Exception as exc:  # noqa: BLE001 - report definition-level failure clearly
        return {"status": "skipped", "reason": f"formula failed: {exc}"}
    return {"status": "generated", "series": series}


def load_feature_registry(path: str | Path = DEFAULT_FEATURE_REGISTRY_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature registry not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Feature registry must be a dictionary: {path}")
    return payload


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
