"""Feature contract management for UFC model training.

Notebook source sections:
- SECTION 5 — LOAD FEATURE REGISTRY
- SECTION 6 — SYMMETRY-SAFE FEATURE FILTER

Responsibilities:
- Preserve the original moneyline ``diff_plus_engineered`` contract.
- Support future explicit model feature contracts.
- Validate missing and unsafe feature columns before training.
- Keep feature engineering separate from feature selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

UNSAFE_PREFIXES = ("r_pre_", "b_pre_", "R_", "B_", "r_", "b_")
DEFAULT_SELECTION_MODE = "diff_plus_engineered"


class FeatureContractError(ValueError):
    """Raised when a feature contract is invalid for the input dataframe."""


def load_registered_engineered_features(registry_path: str | Path) -> list[str]:
    """Load engineered feature names from CSV or YAML registry files."""
    registry_path = Path(registry_path)

    if registry_path.suffix.lower() in {".yaml", ".yml"}:
        with registry_path.open("r", encoding="utf-8") as f:
            registry = yaml.safe_load(f) or {}

        if isinstance(registry, dict):
            for key in ("engineered_features", "features", "moneyline_features"):
                values = registry.get(key)
                if isinstance(values, list):
                    return _normalize_feature_list(values)

            # Fallback for richer feature registries that store records.
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


def load_feature_contracts(contract_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load model feature contracts from a YAML file.

    Expected shape:

    feature_sets:
      moneyline_xgb_base:
        selection_mode: diff_plus_engineered
        expected_feature_count: 124
      moneyline_xgb_market:
        selection_mode: explicit
        features:
          - elo_diff
          - market_implied_prob_diff
    """
    contract_path = Path(contract_path)
    with contract_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    if "feature_sets" in raw_config:
        contracts = raw_config["feature_sets"]
    else:
        contracts = raw_config

    if not isinstance(contracts, dict):
        raise FeatureContractError(f"Invalid feature contract file: {contract_path}")

    return contracts


def resolve_feature_contract(
    df: pd.DataFrame,
    contract: dict[str, Any],
    registered_features: list[str] | None = None,
) -> list[str]:
    """Resolve a feature contract into concrete dataframe columns."""
    selection_mode = contract.get("selection_mode", DEFAULT_SELECTION_MODE)
    allow_unsafe = bool(contract.get("allow_unsafe_features", False))
    expected_feature_count = contract.get("expected_feature_count")

    if selection_mode == "diff_plus_engineered":
        feature_columns = select_diff_plus_engineered_features(
            df=df,
            registered_features=registered_features or [],
        )
    elif selection_mode == "explicit":
        feature_columns = _normalize_feature_list(contract.get("features", []))
    else:
        raise FeatureContractError(f"Unsupported selection_mode: {selection_mode}")

    validate_feature_columns(
        df=df,
        feature_columns=feature_columns,
        allow_unsafe_features=allow_unsafe,
        expected_feature_count=expected_feature_count,
    )

    return feature_columns


def resolve_feature_contract_by_name(
    df: pd.DataFrame,
    contract_path: str | Path,
    contract_name: str,
    registered_features: list[str] | None = None,
) -> list[str]:
    """Load a contract file and return feature columns for one named contract."""
    contracts = load_feature_contracts(contract_path)

    if contract_name not in contracts:
        available = sorted(contracts.keys())
        raise FeatureContractError(
            f"Feature contract '{contract_name}' not found. Available contracts: {available}"
        )

    return resolve_feature_contract(
        df=df,
        contract=contracts[contract_name],
        registered_features=registered_features,
    )


def select_diff_plus_engineered_features(
    df: pd.DataFrame,
    registered_features: list[str],
) -> list[str]:
    """Return the original notebook-compatible moneyline feature set.

    Rule:
    - include all columns ending in ``_diff``
    - include explicitly registered engineered features
    - reject unsafe raw red/blue fighter-specific columns later in validation
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
    """Backward-compatible alias for the base moneyline feature selector."""
    feature_columns = select_diff_plus_engineered_features(df, registered_features)
    validate_feature_columns(df, feature_columns, allow_unsafe_features=False)
    return feature_columns


def validate_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    allow_unsafe_features: bool = False,
    expected_feature_count: int | None = None,
) -> None:
    """Validate that a model feature contract is usable for a dataframe."""
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


def _normalize_feature_list(values: list[Any]) -> list[str]:
    """Normalize a feature list while preserving order and removing duplicates."""
    normalized = [str(value) for value in values if value is not None]
    return list(dict.fromkeys(normalized))
