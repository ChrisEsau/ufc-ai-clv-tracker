from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.features.registry_feature_builder import (
    DEFAULT_FEATURE_REGISTRY_PATH,
    DEFAULT_TRANSFORM_REGISTRY_PATH,
    RegistryFeatureBuildResult,
    apply_registry_feature_definitions,
)


@dataclass(frozen=True)
class SharedFeatureMaterializationResult:
    dataframe: pd.DataFrame
    registry_result: RegistryFeatureBuildResult


def materialize_model_lab_registry_features(
    df: pd.DataFrame,
    *,
    selected_features: Iterable[str] | None = None,
    allowed_statuses: Iterable[str] | None = None,
    registry_path: str | Path = DEFAULT_FEATURE_REGISTRY_PATH,
    transform_registry_path: str | Path = DEFAULT_TRANSFORM_REGISTRY_PATH,
    overwrite_existing: bool = True,
) -> SharedFeatureMaterializationResult:
    """Apply Model Lab registry transforms and formulas to a feature dataframe."""

    registry_result = apply_registry_feature_definitions(
        df,
        registry_path=registry_path,
        transform_registry_path=transform_registry_path,
        selected_features=selected_features,
        allowed_statuses=allowed_statuses or {"active", "draft"},
        overwrite_existing=overwrite_existing,
    )
    return SharedFeatureMaterializationResult(
        dataframe=registry_result.dataframe,
        registry_result=registry_result,
    )


def print_registry_materialization_summary(result: SharedFeatureMaterializationResult) -> None:
    registry_result = result.registry_result
    if registry_result.generated_columns:
        print(
            "Registry features materialized: "
            f"{len(registry_result.generated_columns)} ({registry_result.generated_columns})"
        )
    if registry_result.missing_inputs:
        print(f"Registry features with missing inputs: {registry_result.missing_inputs}")
