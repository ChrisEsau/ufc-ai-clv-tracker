"""Generic transform engine for UFC feature views.

This module applies reusable transforms to red/blue fighter feature columns.
It is intentionally standalone so it can be validated before replacing any
existing moneyline feature-view logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransformResult:
    """Summary of transform engine output."""

    dataframe: pd.DataFrame
    generated_columns: list[str]
    missing_source_pairs: list[str]


SAFE_DENOMINATOR_EPSILON = 1e-9


def apply_red_blue_transforms(
    df: pd.DataFrame,
    base_columns: Iterable[str],
    transforms: Iterable[str],
    *,
    red_prefix: str = "r_pre_",
    blue_prefix: str = "b_pre_",
) -> TransformResult:
    """Apply red/blue transforms to a dataframe.

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

    Returns
    -------
    TransformResult
        Copy of the input dataframe with generated columns appended, plus
        generated-column and missing-source summaries.
    """

    out = df.copy()
    generated_columns: list[str] = []
    missing_source_pairs: list[str] = []
    new_columns: dict[str, pd.Series] = {}

    requested_transforms = [str(transform) for transform in transforms]

    for base_column in [str(column) for column in base_columns]:
        red_col = f"{red_prefix}{base_column}"
        blue_col = f"{blue_prefix}{base_column}"

        if red_col not in out.columns or blue_col not in out.columns:
            missing_source_pairs.append(f"{red_col}|{blue_col}")
            continue

        red_values = pd.to_numeric(out[red_col], errors="coerce")
        blue_values = pd.to_numeric(out[blue_col], errors="coerce")

        for transform in requested_transforms:
            if transform == "red_minus_blue":
                output_col = f"{base_column}_diff"
                new_columns[output_col] = red_values - blue_values
            elif transform == "blue_minus_red":
                output_col = f"{base_column}_reverse_diff"
                new_columns[output_col] = blue_values - red_values
            elif transform == "absolute_gap":
                output_col = f"{base_column}_abs_gap"
                new_columns[output_col] = (red_values - blue_values).abs()
            elif transform == "ratio":
                output_col = f"{base_column}_ratio"
                new_columns[output_col] = safe_ratio(red_values, blue_values)
            else:
                continue

            generated_columns.append(output_col)

    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=out.index)], axis=1)

    return TransformResult(
        dataframe=out,
        generated_columns=dedupe_preserve_order(generated_columns),
        missing_source_pairs=dedupe_preserve_order(missing_source_pairs),
    )


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return numerator / denominator with safe handling for zero denominators."""

    denominator = pd.to_numeric(denominator, errors="coerce")
    numerator = pd.to_numeric(numerator, errors="coerce")

    safe_denominator = denominator.where(denominator.abs() > SAFE_DENOMINATOR_EPSILON, np.nan)
    return numerator / safe_denominator


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
