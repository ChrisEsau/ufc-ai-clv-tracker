"""Feature selection and validation for UFC model training.

Final architecture:
- Feature engineering creates columns in feature warehouse files.
- Feature source registries describe where feature warehouses live.
- Model config YAML files explicitly list the exact feature columns used by each model.
- This module validates and returns those model-config feature columns.

No training module should infer a production model's feature list from naming
rules. Rule-based selectors remain only as backward-compatible notebook-parity
helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

UNSAFE_PREFIXES = ("r_pre_", "b_pre_", "R_", "B_", "r_", "b_")


class FeatureContractError(ValueError):
    """Raised when a model feature contract is invalid for the input dataframe."""


def load_model_config(config_path: str | Path) -> dict[str, Any]:
    """Load a model config YAML file."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise FeatureContractError(f"Invalid model config file: {config_path}")

    return config


def resolve_features_from_model_config(
    df: pd.DataFrame,
    model_config: dict[str, Any],
) -> list[str]:
    """Return validated feature columns from a model config dictionary.

    Expected model config shape:

    features:
      selection_mode: explicit
      expected_feature_count: 124
      allow_unsafe_features: false
      feature_columns:
        - elo_diff
        - win_pct_diff
    """
    feature_config = model_config.get("features")
    if not isinstance(feature_config, dict):
        raise FeatureContractError("Model config is missing required 'features' section")

    return resolve_feature_contract(df=df, contract=feature_config)


def resolve_features_from_model_config_path(
    df: pd.DataFrame,
    config_path: str | Path,
) -> list[str]:
    """Load a model config YAML and return validated feature columns."""
    model_config = load_model_config(config_path)
    return resolve_features_from_model_config(df=df, model_config=model_config)


def resolve_feature_contract(
    df: pd.DataFrame,
    contract: dict[str, Any],
) -> list[str]:
    """Resolve a model-config feature contract into concrete dataframe columns."""
    selection_mode = str(contract.get("selection_mode", "explicit")).strip().lower()
    allow_unsafe = bool(contract.get("allow_unsafe_features", False))
    expected_feature_count = contract.get("expected_feature_count")

    if selection_mode == "explicit":
        raw_features = contract.get("feature_columns", contract.get("features", []))
        feature_columns = _normalize_feature_list(raw_features)
    elif selection_mode == "diff_plus_engineered":
        # Notebook-parity fallback only. Production configs should use explicit.
        registered_features = _normalize_feature_list(contract.get("registered_features", []))
        feature_columns = select_diff_plus_engineered_features(
            df=df,
            registered_features=registered_features,
        )
    else:
        raise FeatureContractError(f"Unsupported selection_mode: {selection_mode}")

    validate_feature_columns(
        df=df,
        feature_columns=feature_columns,
        allow_unsafe_features=allow_unsafe,
        expected_feature_count=expected_feature_count,
    )

    return feature_columns


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    allow_unsafe_features: bool = False,
    expected_feature_count: int | None = None,
) -> None:
    """Validate that a model feature contract is usable for a dataframe."""
    if not feature_columns:
        raise FeatureContractError("Feature contract resolved to zero feature columns")

    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise FeatureContractError(f"Feature contract contains missing columns: {missing_cols}")

    if not allow_unsafe_features:
        unsafe_cols = [col for col in feature_columns if col.startswith(UNSAFE_PREFIXES)]
        if unsafe_cols:
            raise FeatureContractError(
                f"Unsafe raw red/blue fighter columns detected: {unsafe_cols}"
            )

    if expected_feature_count is not None and len(feature_columns) != int(expected_feature_count):
        raise FeatureContractError(
            f"Feature count mismatch: expected {expected_feature_count}, "
            f"observed {len(feature_columns)}"
        )


def select_diff_plus_engineered_features(
    df: pd.DataFrame,
    registered_features: list[str],
) -> list[str]:
    """Return notebook-compatible diff-plus-engineered feature columns.

    This helper is retained for validation and legacy notebook parity only.
    Production model configs should explicitly list all model input columns.
    """
    safe_cols: list[str] = []

    for col in df.columns:
        if col.endswith("_diff"):
            safe_cols.append(col)
        elif col in registered_features:
            safe_cols.append(col)

    return list(dict.fromkeys(safe_cols))


def select_moneyline_features(
    df: pd.DataFrame,
    registered_features: list[str],
) -> list[str]:
    """Backward-compatible alias for the old base moneyline selector."""
    feature_columns = select_diff_plus_engineered_features(df, registered_features)
    validate_feature_columns(df, feature_columns, allow_unsafe_features=False)
    return feature_columns


def load_registered_engineered_features(registry_path: str | Path) -> list[str]:
    """Legacy helper to load engineered feature names from CSV or YAML registries."""
    registry_path = Path(registry_path)

    if registry_path.suffix.lower() in {".yaml", ".yml"}:
        with registry_path.open("r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        if isinstance(registry, dict):
            for key in ("engineered_features", "features", "moneyline_features"):
                values = registry.get(key)
                if isinstance(values, list):
                    return _normalize_feature_list(values)

            values = []
            for item in registry.values():
                if isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, dict) and "feature" in entry:
                            values.append(str(entry["feature"]))
            if values:
                return values

        return []

    registry_df = pd.read_csv(registry_path)
    if "feature" not in registry_df.columns:
        raise FeatureContractError(
            f"Feature registry is missing required 'feature' column: {registry_path}"
        )

    return registry_df["feature"].dropna().astype(str).tolist()


def _normalize_feature_list(values: list[Any]) -> list[str]:
    """Normalize a feature list while preserving order and removing duplicates."""
    if not isinstance(values, list):
        raise FeatureContractError("Feature list must be a list")

    normalized = [str(value) for value in values if value is not None]
    return list(dict.fromkeys(normalized))
