"""Generic feature builder driven by the feature graph and transform engine.

This module is a shadow implementation. It generates candidate feature columns
from an already assembled moneyline-style dataframe, but it does not replace the
validated moneyline feature-view builder yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from pipeline.features.feature_graph import FeatureGraphPlan
from pipeline.features.transform_engine import apply_red_blue_transforms


@dataclass(frozen=True)
class GenericFeatureBuildResult:
    """Result returned by the generic feature builder."""

    dataframe: pd.DataFrame
    generated_columns: list[str]
    passthrough_columns: list[str]
    missing_source_pairs: list[str]


@dataclass(frozen=True)
class FeatureTransformSpec:
    """Source and output naming details for one transformed feature."""

    source_base_column: str
    output_base_column: str
    transform: str


def build_generic_features_from_plan(
    df: pd.DataFrame,
    plan: FeatureGraphPlan,
    *,
    state_prefix: str = "pre_",
) -> GenericFeatureBuildResult:
    """Generate graph-planned feature candidates from an input dataframe.

    Parameters
    ----------
    df:
        Assembled fight-level dataframe containing red/blue prefight source
        columns such as ``r_pre_elo`` and ``b_pre_elo``.
    plan:
        Resolved feature graph plan.
    state_prefix:
        Fighter-state prefix between side and base feature name. Defaults to
        ``pre_`` for current moneyline feature-view compatibility.
    """

    out = df.copy()
    generated_columns: list[str] = []
    missing_source_pairs: list[str] = []

    specs = infer_feature_transform_specs(plan.generated_feature_columns)
    grouped_specs = group_specs_by_source_prefix(specs)

    for source_prefix, group_specs in grouped_specs.items():
        source_base_columns = [spec.source_base_column for spec in group_specs]
        transforms = [spec.transform for spec in group_specs]
        red_prefix = f"r_{state_prefix}{source_prefix}"
        blue_prefix = f"b_{state_prefix}{source_prefix}"

        temp_source_columns = resolve_temp_source_columns(
            df=out,
            base_columns=source_base_columns,
            red_prefix=red_prefix,
            blue_prefix=blue_prefix,
        )
        temp_df = out[temp_source_columns].copy() if temp_source_columns else pd.DataFrame(index=out.index)

        transform_result = apply_red_blue_transforms(
            df=temp_df,
            base_columns=source_base_columns,
            transforms=transforms,
            red_prefix=red_prefix,
            blue_prefix=blue_prefix,
        )

        missing_source_pairs.extend(transform_result.missing_source_pairs)

        rename_map = build_generated_column_rename_map(group_specs)
        for source_column, target_column in rename_map.items():
            if source_column not in transform_result.dataframe.columns:
                continue
            out[target_column] = transform_result.dataframe[source_column]
            generated_columns.append(target_column)

    available_passthrough = [
        column for column in plan.passthrough_feature_columns if column in out.columns
    ]

    return GenericFeatureBuildResult(
        dataframe=out,
        generated_columns=dedupe_preserve_order(generated_columns),
        passthrough_columns=available_passthrough,
        missing_source_pairs=dedupe_preserve_order(missing_source_pairs),
    )


def resolve_temp_source_columns(
    df: pd.DataFrame,
    base_columns: Iterable[str],
    red_prefix: str,
    blue_prefix: str,
) -> list[str]:
    """Return source columns needed for an isolated transform-engine call."""

    columns: list[str] = []
    for base_column in base_columns:
        red_col = f"{red_prefix}{base_column}"
        blue_col = f"{blue_prefix}{base_column}"
        if red_col in df.columns:
            columns.append(red_col)
        if blue_col in df.columns:
            columns.append(blue_col)
    return dedupe_preserve_order(columns)


def infer_feature_transform_specs(generated_columns: Iterable[str]) -> list[FeatureTransformSpec]:
    """Infer transform specs from generated feature names."""

    specs: list[FeatureTransformSpec] = []
    for column in generated_columns:
        column = str(column)
        transform = infer_transform_from_column(column)
        output_base_column = strip_transform_suffix(column)
        if transform is None or output_base_column is None:
            continue

        _, source_base_column = split_output_base_column(output_base_column)
        specs.append(
            FeatureTransformSpec(
                source_base_column=source_base_column,
                output_base_column=output_base_column,
                transform=transform,
            )
        )
    return specs


def group_specs_by_source_prefix(
    specs: Iterable[FeatureTransformSpec],
) -> dict[str, list[FeatureTransformSpec]]:
    """Group transform specs by source prefix required in red/blue source columns."""

    grouped: dict[str, list[FeatureTransformSpec]] = {}
    for spec in specs:
        source_prefix, _ = split_output_base_column(spec.output_base_column)
        grouped.setdefault(source_prefix, []).append(spec)
    return grouped


def split_output_base_column(output_base_column: str) -> tuple[str, str]:
    """Split output base column into source prefix and source base column.

    Examples
    --------
    ``elo`` -> ("", "elo")
    ``ewm_elo`` -> ("ewm_", "elo")
    ``recent_form_elo`` -> ("recent_form_", "elo")
    """

    if output_base_column.startswith("recent_form_"):
        return "recent_form_", output_base_column.removeprefix("recent_form_")
    if output_base_column.startswith("ewm_"):
        return "ewm_", output_base_column.removeprefix("ewm_")
    return "", output_base_column


def build_generated_column_rename_map(specs: Iterable[FeatureTransformSpec]) -> dict[str, str]:
    """Build rename map from transform-engine output names to graph output names."""

    rename_map: dict[str, str] = {}
    for spec in specs:
        source_column = transform_output_column(spec.source_base_column, spec.transform)
        target_column = spec_to_output_column(spec)
        rename_map[source_column] = target_column
    return rename_map


def spec_to_output_column(spec: FeatureTransformSpec) -> str:
    """Return final output column name for a transform spec."""

    return transform_output_column(spec.output_base_column, spec.transform)


def infer_transform_from_column(column: str) -> str | None:
    """Infer transform ID from generated feature column name."""

    if column.endswith("_reverse_diff"):
        return "blue_minus_red"
    if column.endswith("_abs_gap"):
        return "absolute_gap"
    if column.endswith("_ratio"):
        return "ratio"
    if column.endswith("_diff"):
        return "red_minus_blue"
    return None


def transform_output_column(base_column: str, transform: str) -> str:
    """Build output column name for a base column and transform."""

    if transform == "red_minus_blue":
        return f"{base_column}_diff"
    if transform == "blue_minus_red":
        return f"{base_column}_reverse_diff"
    if transform == "absolute_gap":
        return f"{base_column}_abs_gap"
    if transform == "ratio":
        return f"{base_column}_ratio"
    return f"{base_column}_{transform}"


def infer_transformable_base_columns(generated_columns: Iterable[str]) -> list[str]:
    """Infer unprefixed base columns from generated feature names.

    Kept for backward-compatible imports from early migration scripts.
    """

    base_columns: list[str] = []
    for column in generated_columns:
        base_column = strip_transform_suffix(str(column))
        if base_column is not None:
            base_columns.append(base_column)
    return dedupe_preserve_order(base_columns)


def infer_transforms_from_generated_columns(generated_columns: Iterable[str]) -> list[str]:
    """Infer transform IDs from generated feature names.

    Kept for backward-compatible imports from early migration scripts.
    """

    transforms: list[str] = []
    for column in generated_columns:
        transform = infer_transform_from_column(str(column))
        if transform is not None:
            transforms.append(transform)
    return dedupe_preserve_order(transforms)


def strip_transform_suffix(column: str) -> str | None:
    """Strip known transform suffixes from a generated feature column name."""

    suffixes = [
        "_reverse_diff",
        "_abs_gap",
        "_ratio",
        "_diff",
    ]
    for suffix in suffixes:
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return None


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
