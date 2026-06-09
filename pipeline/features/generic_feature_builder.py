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

    transformable_columns = infer_transformable_base_columns(plan.generated_feature_columns)
    transforms = infer_transforms_from_generated_columns(plan.generated_feature_columns)

    transform_result = apply_red_blue_transforms(
        df=df,
        base_columns=transformable_columns,
        transforms=transforms,
        red_prefix=f"r_{state_prefix}",
        blue_prefix=f"b_{state_prefix}",
    )

    available_passthrough = [
        column for column in plan.passthrough_feature_columns if column in transform_result.dataframe.columns
    ]

    return GenericFeatureBuildResult(
        dataframe=transform_result.dataframe,
        generated_columns=transform_result.generated_columns,
        passthrough_columns=available_passthrough,
        missing_source_pairs=transform_result.missing_source_pairs,
    )


def infer_transformable_base_columns(generated_columns: Iterable[str]) -> list[str]:
    """Infer unprefixed base columns from generated feature names."""

    base_columns: list[str] = []
    for column in generated_columns:
        base_column = strip_transform_suffix(str(column))
        if base_column is not None:
            base_columns.append(base_column)
    return dedupe_preserve_order(base_columns)


def infer_transforms_from_generated_columns(generated_columns: Iterable[str]) -> list[str]:
    """Infer transform IDs from generated feature names."""

    transforms: list[str] = []
    for column in generated_columns:
        column = str(column)
        if column.endswith("_diff"):
            transforms.append("red_minus_blue")
        elif column.endswith("_reverse_diff"):
            transforms.append("blue_minus_red")
        elif column.endswith("_abs_gap"):
            transforms.append("absolute_gap")
        elif column.endswith("_ratio"):
            transforms.append("ratio")
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
