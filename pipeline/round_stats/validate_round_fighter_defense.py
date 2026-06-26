"""Validate Round Fighter State P1.4 defensive degradation artifacts.

Run from repo root:

    python -m pipeline.round_stats.validate_round_fighter_defense
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH,
    ROUND_FIGHTER_DEFENSE_P1_4_VALIDATION_PATH,
    ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH,
    ensure_data_dirs,
)
from pipeline.round_stats.build_round_fighter_defense import (
    DEFENSE_FIGHT_OBSERVATION_COLUMNS,
)


REQUIRED_METADATA_COLUMNS = [
    "event_id",
    "fight_id",
    "fighter_id",
    "opponent_id",
    "fighter_name",
    "opponent_name",
    "event_name",
    "date",
    "corner",
]

REQUIRED_STATE_COLUMNS = [
    "rfs_def_prior_fight_count",
    "rfs_def_prior_valid_defense_count",
    "rfs_def_has_state",
]


@dataclass(frozen=True)
class ValidationCheck:
    check_name: str
    status: str
    severity: str
    observed_value: Any
    threshold: Any
    message: str


def _audit_value_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _check(
    checks: list[ValidationCheck],
    *,
    name: str,
    passed: bool,
    severity: str,
    observed: Any,
    threshold: Any,
    message: str,
) -> None:
    checks.append(
        ValidationCheck(
            check_name=name,
            status="PASS" if passed else "FAIL",
            severity=severity,
            observed_value=observed,
            threshold=threshold,
            message=message,
        )
    )


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def validate_round_fighter_defense() -> pd.DataFrame:
    checks: list[ValidationCheck] = []

    history_df = _read_if_exists(ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH)
    latest_df = _read_if_exists(ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH)

    _check(
        checks,
        name="history_artifact_exists",
        passed=history_df is not None,
        severity="fatal",
        observed=str(ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH),
        threshold="file exists",
        message="P1.4 defensive degradation history artifact must exist.",
    )
    _check(
        checks,
        name="latest_artifact_exists",
        passed=latest_df is not None,
        severity="fatal",
        observed=str(ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH),
        threshold="file exists",
        message="P1.4 latest defensive degradation artifact must exist.",
    )

    if history_df is None or latest_df is None:
        return pd.DataFrame(asdict(check) for check in checks)

    required_history_cols = [
        *REQUIRED_METADATA_COLUMNS,
        *DEFENSE_FIGHT_OBSERVATION_COLUMNS,
        *REQUIRED_STATE_COLUMNS,
    ]
    missing_history = [column for column in required_history_cols if column not in history_df.columns]

    _check(
        checks,
        name="history_required_columns_present",
        passed=not missing_history,
        severity="fatal",
        observed=missing_history,
        threshold=[],
        message="History artifact must include required P1.4 defensive degradation columns.",
    )

    state_cols = [
        column
        for column in history_df.columns
        if column.startswith(("rfs_def_exp_", "rfs_def_last3_", "rfs_def_ewm_"))
    ]

    latest_fight_obs_cols = [
        column
        for column in latest_df.columns
        if column.startswith("rfs_def_fight_")
    ]

    _check(
        checks,
        name="history_unique_fight_fighter_grain",
        passed=not history_df.duplicated(["fight_id", "fighter_id"]).any(),
        severity="fatal",
        observed=int(history_df.duplicated(["fight_id", "fighter_id"]).sum()),
        threshold=0,
        message="History must have one row per fight_id/fighter_id.",
    )

    _check(
        checks,
        name="latest_unique_fighter_grain",
        passed=not latest_df.duplicated(["fighter_id"]).any(),
        severity="fatal",
        observed=int(latest_df.duplicated(["fighter_id"]).sum()),
        threshold=0,
        message="Latest artifact must have one row per fighter_id.",
    )

    _check(
        checks,
        name="fight_observation_columns_present",
        passed=all(column in history_df.columns for column in DEFENSE_FIGHT_OBSERVATION_COLUMNS),
        severity="fatal",
        observed=sum(column in history_df.columns for column in DEFENSE_FIGHT_OBSERVATION_COLUMNS),
        threshold=len(DEFENSE_FIGHT_OBSERVATION_COLUMNS),
        message="History must contain registry-approved P1.4 fight-observation columns.",
    )

    _check(
        checks,
        name="state_columns_present",
        passed=len(state_cols) == len(DEFENSE_FIGHT_OBSERVATION_COLUMNS) * 3,
        severity="fatal",
        observed=len(state_cols),
        threshold=len(DEFENSE_FIGHT_OBSERVATION_COLUMNS) * 3,
        message="History must contain exp/last3/ewm state summaries for each P1.4 observation.",
    )

    _check(
        checks,
        name="latest_has_no_current_fight_observation_columns",
        passed=len(latest_fight_obs_cols) == 0,
        severity="fatal",
        observed=latest_fight_obs_cols,
        threshold=[],
        message="Latest defense artifact must not include current-fight observation columns.",
    )

    prior_count = pd.to_numeric(history_df["rfs_def_prior_fight_count"], errors="coerce").fillna(0)
    valid_count = pd.to_numeric(
        history_df["rfs_def_prior_valid_defense_count"],
        errors="coerce",
    ).fillna(0)

    _check(
        checks,
        name="valid_count_lte_prior_count",
        passed=bool((valid_count <= prior_count).all()),
        severity="fatal",
        observed=int((valid_count > prior_count).sum()),
        threshold=0,
        message="Valid defensive prior count cannot exceed prior fight count.",
    )

    any_state_rows = int(history_df[state_cols].notna().any(axis=1).sum()) if state_cols else 0
    has_state_rows = int(
        pd.to_numeric(history_df["rfs_def_has_state"], errors="coerce").fillna(0).sum()
    )

    _check(
        checks,
        name="has_state_matches_non_null_state",
        passed=any_state_rows == has_state_rows,
        severity="fatal",
        observed=has_state_rows,
        threshold=f"must equal rows with any state {any_state_rows}",
        message="rfs_def_has_state must match presence of non-null state columns.",
    )

    numeric_cols = [
        column
        for column in [*DEFENSE_FIGHT_OBSERVATION_COLUMNS, *state_cols]
        if column in history_df.columns
    ]
    finite_ok = bool(
        np.isfinite(
            history_df[numeric_cols]
            .select_dtypes(include=[np.number])
            .fillna(0)
        )
        .all()
        .all()
    )

    _check(
        checks,
        name="numeric_values_are_finite",
        passed=finite_ok,
        severity="fatal",
        observed="finite" if finite_ok else "inf_or_negative_inf_found",
        threshold="finite or NaN",
        message="P1.4 numeric values must not contain infinities.",
    )

    fight_obs_rows = int(history_df[DEFENSE_FIGHT_OBSERVATION_COLUMNS].notna().any(axis=1).sum())

    _check(
        checks,
        name="coverage_has_any_fight_observation",
        passed=fight_obs_rows > 0,
        severity="warn",
        observed=fight_obs_rows,
        threshold="> 0",
        message="P1.4 should usually have current-fight observations when two-round+ defensive data exists.",
    )

    _check(
        checks,
        name="coverage_has_any_model_state",
        passed=has_state_rows > 0,
        severity="warn",
        observed=has_state_rows,
        threshold="> 0 after historical backfill",
        message="Expected to fail until enough repeated fighter history is backfilled.",
    )

    head_feature_rows = int(
        history_df["rfs_def_fight_head_absorbed_slope"].notna().sum()
        if "rfs_def_fight_head_absorbed_slope" in history_df.columns
        else 0
    )

    _check(
        checks,
        name="coverage_has_head_absorbed_signal",
        passed=head_feature_rows > 0,
        severity="warn",
        observed=head_feature_rows,
        threshold="> 0 if round feed includes head splits",
        message="Head absorbed features require head strike columns in round stats; warning only.",
    )

    return pd.DataFrame(asdict(check) for check in checks)


def main() -> None:
    ensure_data_dirs()

    audit_df = validate_round_fighter_defense()

    parquet_audit_df = audit_df.copy()
    for column in ["observed_value", "threshold", "message"]:
        if column in parquet_audit_df.columns:
            parquet_audit_df[column] = parquet_audit_df[column].map(_audit_value_to_string)

    parquet_audit_df.to_parquet(ROUND_FIGHTER_DEFENSE_P1_4_VALIDATION_PATH, index=False)

    fatal_failures = audit_df[
        audit_df["severity"].eq("fatal") & audit_df["status"].eq("FAIL")
    ]

    print("=" * 80)
    print("ROUND FIGHTER DEFENSE P1.4 VALIDATION")
    print("=" * 80)
    print(f"History path     : {ROUND_FIGHTER_DEFENSE_P1_4_HISTORY_PATH}")
    print(f"Latest path      : {ROUND_LATEST_FIGHTER_DEFENSE_P1_4_PATH}")
    print(f"Audit path       : {ROUND_FIGHTER_DEFENSE_P1_4_VALIDATION_PATH}")
    print(f"Checks           : {len(audit_df)}")
    print(f"Fatal failures   : {len(fatal_failures)}")
    print("=" * 80)
    print(audit_df.to_string(index=False))

    if not fatal_failures.empty:
        raise SystemExit("Round Fighter Defense P1.4 validation failed fatal checks.")


if __name__ == "__main__":
    main()
