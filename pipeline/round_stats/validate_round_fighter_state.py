"""Validate Round Fighter State P0.1 artifacts.

This validator checks the standalone Round Fighter State feature store only.
It does not touch production fighter-state artifacts, prediction artifacts, or model views.

Run from repo root:

    python -m pipeline.round_stats.validate_round_fighter_state
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import (
    ROUND_FIGHTER_STATE_HISTORY_PATH,
    ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH,
    ROUND_LATEST_FIGHTER_STATE_PATH,
    ensure_data_dirs,
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
    "rfs_traj_prior_fight_count",
    "rfs_traj_prior_valid_trajectory_count",
    "rfs_traj_has_state",
]

REQUIRED_FIGHT_OBSERVATION_COLUMNS = [
    "rfs_traj_fight_rounds_observed",
    "rfs_traj_fight_sig_attempt_slope",
    "rfs_traj_fight_total_attempt_slope",
    "rfs_traj_fight_sig_landed_slope",
    "rfs_traj_fight_total_landed_slope",
    "rfs_traj_fight_sig_attempt_late_ratio",
    "rfs_traj_fight_total_attempt_late_ratio",
    "rfs_traj_fight_sig_landed_late_ratio",
    "rfs_traj_fight_sig_accuracy_slope",
    "rfs_traj_fight_total_accuracy_slope",
    "rfs_traj_fight_sig_accuracy_late_diff",
    "rfs_traj_fight_total_accuracy_late_diff",
    "rfs_traj_fight_td_attempt_slope",
    "rfs_traj_fight_td_accuracy_slope",
    "rfs_traj_fight_td_attempt_late_ratio",
    "rfs_traj_fight_control_seconds_slope",
    "rfs_traj_fight_control_late_ratio",
    "rfs_traj_fight_opp_sig_accuracy_allowed_slope",
    "rfs_traj_fight_opp_total_accuracy_allowed_slope",
    "rfs_traj_fight_opp_sig_attempt_allowed_slope",
    "rfs_traj_fight_opp_control_allowed_slope",
]



RFS_FIGHT_OBSERVATION_PREFIXES = (
    "rfs_traj_fight_",
    "rfs_open_fight_",
    "rfs_phase_base_fight_",
    "rfs_phase_interact_fight_",
    "rfs_dynamic_response_fight_",
    "rfs_finish_state_fight_",
)

@dataclass(frozen=True)
class ValidationCheck:
    """One validation check result."""

    check_name: str
    status: str
    severity: str
    observed_value: str
    threshold: str
    message: str


class RoundFighterStateValidationError(RuntimeError):
    """Raised when fatal Round Fighter State validation checks fail."""


def _add_check(
    checks: list[ValidationCheck],
    check_name: str,
    passed: bool,
    severity: str,
    observed_value: object,
    threshold: object,
    message: str,
) -> None:
    """Append a validation check result."""
    checks.append(
        ValidationCheck(
            check_name=check_name,
            status="PASS" if passed else "FAIL",
            severity=severity,
            observed_value=str(observed_value),
            threshold=str(threshold),
            message=message,
        )
    )


def _write_audit(checks: list[ValidationCheck]) -> pd.DataFrame:
    """Write validation audit parquet."""
    ensure_data_dirs()
    ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    audit_df = pd.DataFrame([asdict(check) for check in checks])
    audit_df.to_parquet(ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH, index=False)

    return audit_df


def _read_artifact(path: Path, artifact_name: str) -> pd.DataFrame:
    """Read a parquet artifact with a clear error."""
    if not path.exists():
        raise RoundFighterStateValidationError(
            f"{artifact_name} artifact not found: {path}. "
            "Run python -m pipeline.round_stats.build_round_fighter_state first."
        )

    return pd.read_parquet(path)


def validate_round_fighter_state(
    history_path: Path = ROUND_FIGHTER_STATE_HISTORY_PATH,
    latest_path: Path = ROUND_LATEST_FIGHTER_STATE_PATH,
) -> pd.DataFrame:
    """Validate Round Fighter State history/latest artifacts and write audit."""
    checks: list[ValidationCheck] = []

    history_exists = Path(history_path).exists()
    latest_exists = Path(latest_path).exists()

    _add_check(
        checks,
        "history_artifact_exists",
        history_exists,
        "fatal",
        history_path,
        "file exists",
        "Round Fighter State history artifact must exist.",
    )
    _add_check(
        checks,
        "latest_artifact_exists",
        latest_exists,
        "fatal",
        latest_path,
        "file exists",
        "Round latest fighter state artifact must exist.",
    )

    if not history_exists or not latest_exists:
        audit_df = _write_audit(checks)
        raise RoundFighterStateValidationError(
            "Missing required Round Fighter State artifact. "
            f"Audit written to {ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH}"
        )

    history = _read_artifact(Path(history_path), "history")
    latest = _read_artifact(Path(latest_path), "latest")

    required_columns = (
        REQUIRED_METADATA_COLUMNS
        + REQUIRED_STATE_COLUMNS
        + REQUIRED_FIGHT_OBSERVATION_COLUMNS
    )
    missing_history_columns = [
        column for column in required_columns if column not in history.columns
    ]
    latest_required_columns = REQUIRED_METADATA_COLUMNS + REQUIRED_STATE_COLUMNS
    missing_latest_columns = [
        column for column in latest_required_columns if column not in latest.columns
    ]

    _add_check(
        checks,
        "history_required_columns_present",
        not missing_history_columns,
        "fatal",
        missing_history_columns,
        "[]",
        "History artifact must include required P0.1 columns.",
    )
    _add_check(
        checks,
        "latest_required_columns_present",
        not missing_latest_columns,
        "fatal",
        missing_latest_columns,
        "[]",
        "Latest artifact must include metadata and current prior-state columns.",
    )

    history_duplicate_keys = (
        int(history.duplicated(subset=["fight_id", "fighter_id"]).sum())
        if not missing_history_columns
        else -1
    )
    latest_duplicate_fighters = (
        int(latest.duplicated(subset=["fighter_id"]).sum())
        if "fighter_id" in latest.columns
        else -1
    )

    _add_check(
        checks,
        "history_unique_fight_fighter_grain",
        history_duplicate_keys == 0,
        "fatal",
        history_duplicate_keys,
        "0",
        "History must have one row per fight_id/fighter_id.",
    )
    _add_check(
        checks,
        "latest_unique_fighter_grain",
        latest_duplicate_fighters == 0,
        "fatal",
        latest_duplicate_fighters,
        "0",
        "Latest artifact must have one row per fighter_id.",
    )

    latest_fight_observation_columns = [
        column
        for column in latest.columns
        if column.startswith(
            RFS_FIGHT_OBSERVATION_PREFIXES
        )
    ]
    _add_check(
        checks,
        "latest_has_no_current_fight_observation_columns",
        len(latest_fight_observation_columns) == 0,
        "fatal",
        latest_fight_observation_columns,
        "[]",
        "Latest artifact must not expose current-fight observation columns to live joins.",
    )


    rfs_cols = [
        column
        for column in history.columns
        if column.startswith("rfs_")
    ]

    fight_cols = [
        column
        for column in rfs_cols
        if column.startswith(
            RFS_FIGHT_OBSERVATION_PREFIXES
        )
    ]

    state_cols = [
        column
        for column in rfs_cols
        if not column.startswith(
            RFS_FIGHT_OBSERVATION_PREFIXES
        )
    ]

    _add_check(
        checks,
        "rfs_columns_present",
        len(rfs_cols) > 0,
        "fatal",
        len(rfs_cols),
        "> 0",
        "History must contain RFS feature columns.",
    )
    _add_check(
        checks,
        "fight_observation_columns_present",
        len(fight_cols) >= len(REQUIRED_FIGHT_OBSERVATION_COLUMNS),
        "fatal",
        len(fight_cols),
        f">= {len(REQUIRED_FIGHT_OBSERVATION_COLUMNS)}",
        "History must contain required fight-observation trajectory columns.",
    )
    _add_check(
        checks,
        "prior_state_columns_present",
        len(state_cols) > len(REQUIRED_STATE_COLUMNS),
        "warn",
        len(state_cols),
        f"> {len(REQUIRED_STATE_COLUMNS)}",
        "History should contain prior-state summary columns.",
    )

    if not missing_history_columns:
        history_sorted = history.copy()
        history_sorted["date"] = pd.to_datetime(history_sorted["date"], errors="coerce")
        history_sorted = history_sorted.sort_values(
            ["fighter_id", "date", "fight_id"]
        ).reset_index(drop=True)

        expected_prior_fight_count = history_sorted.groupby("fighter_id").cumcount()
        prior_fight_count_matches = (
            history_sorted["rfs_traj_prior_fight_count"]
            .fillna(-1)
            .astype(int)
            .eq(expected_prior_fight_count)
            .all()
        )

        _add_check(
            checks,
            "prior_fight_count_is_point_in_time",
            bool(prior_fight_count_matches),
            "fatal",
            "matches" if prior_fight_count_matches else "mismatch",
            "matches groupwise cumcount",
            "Prior fight count must equal fighter-level historical cumcount.",
        )

        prior_valid_le_prior_fights = (
            history_sorted["rfs_traj_prior_valid_trajectory_count"]
            <= history_sorted["rfs_traj_prior_fight_count"]
        ).all()

        _add_check(
            checks,
            "prior_valid_count_not_ahead_of_prior_fights",
            bool(prior_valid_le_prior_fights),
            "fatal",
            "valid" if prior_valid_le_prior_fights else "invalid",
            "prior_valid <= prior_fight_count",
            "Prior valid trajectory count cannot exceed prior fight count.",
        )

        expected_has_state = (
            history_sorted["rfs_traj_prior_valid_trajectory_count"].gt(0).astype(int)
        )
        has_state_matches = (
            history_sorted["rfs_traj_has_state"]
            .fillna(-1)
            .astype(int)
            .eq(expected_has_state)
            .all()
        )

        _add_check(
            checks,
            "has_state_matches_prior_valid_count",
            bool(has_state_matches),
            "fatal",
            "matches" if has_state_matches else "mismatch",
            "has_state == prior_valid_count > 0",
            "rfs_traj_has_state must be derived from prior valid trajectory count.",
        )

        first_fight_rows = history_sorted["rfs_traj_prior_fight_count"].eq(0)
        first_fight_has_no_state = (
            history_sorted.loc[first_fight_rows, "rfs_traj_has_state"].eq(0).all()
        )

        _add_check(
            checks,
            "first_fight_rows_have_no_prior_state",
            bool(first_fight_has_no_state),
            "fatal",
            "valid" if first_fight_has_no_state else "invalid",
            "0",
            "A fighter's first observed row must not have prior RFS state.",
        )

    numeric = history.select_dtypes(include=[np.number])
    if not numeric.empty:
        inf_count = int(np.isinf(numeric.to_numpy()).sum())
    else:
        inf_count = 0

    _add_check(
        checks,
        "numeric_values_are_finite_or_null",
        inf_count == 0,
        "fatal",
        inf_count,
        "0",
        "Numeric feature values may be null, but not positive/negative infinity.",
    )

    rounds_ok = True
    bad_round_count = 0
    if "rfs_traj_fight_rounds_observed" in history.columns:
        round_values = pd.to_numeric(
            history["rfs_traj_fight_rounds_observed"],
            errors="coerce",
        )
        bad_round_mask = round_values.lt(1) | round_values.gt(5) | round_values.isna()
        bad_round_count = int(bad_round_mask.sum())
        rounds_ok = bad_round_count == 0

    _add_check(
        checks,
        "rounds_observed_range",
        rounds_ok,
        "fatal",
        bad_round_count,
        "0 bad rows; expected 1..5",
        "Observed round count should be between 1 and 5.",
    )

    slope_columns = [column for column in history.columns if column.endswith("_slope")]
    attempt_slope_columns = [
        column for column in slope_columns if "attempt" in column
    ]
    accuracy_slope_columns = [
        column for column in slope_columns if "accuracy" in column
    ]
    late_ratio_columns = [
        column for column in history.columns if column.endswith("_late_ratio")
    ]
    accuracy_late_diff_columns = [
        column for column in history.columns if column.endswith("_accuracy_late_diff")
    ]

    bad_attempt_slope_count = 0
    if attempt_slope_columns:
        attempt_slopes = history[attempt_slope_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        bad_attempt_slope_count = int(attempt_slopes.abs().gt(200).sum().sum())

    _add_check(
        checks,
        "attempt_slope_outlier_bounds",
        bad_attempt_slope_count == 0,
        "warn",
        bad_attempt_slope_count,
        "abs(attempt slope) <= 200",
        "Attempt slopes above 200 attempts/round are suspicious.",
    )

    bad_accuracy_slope_count = 0
    if accuracy_slope_columns:
        accuracy_slopes = history[accuracy_slope_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        bad_accuracy_slope_count = int(accuracy_slopes.abs().gt(1).sum().sum())

    _add_check(
        checks,
        "accuracy_slope_bounds",
        bad_accuracy_slope_count == 0,
        "fatal",
        bad_accuracy_slope_count,
        "abs(accuracy slope) <= 1",
        "Accuracy slopes should be within [-1, 1].",
    )

    bad_late_ratio_count = 0
    if late_ratio_columns:
        late_ratios = history[late_ratio_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        bad_late_ratio_count = int(late_ratios.gt(10).sum().sum())

    _add_check(
        checks,
        "late_ratio_outlier_bounds",
        bad_late_ratio_count == 0,
        "warn",
        bad_late_ratio_count,
        "late ratio <= 10",
        "Late ratios above 10 are suspicious.",
    )

    bad_accuracy_late_diff_count = 0
    if accuracy_late_diff_columns:
        accuracy_diffs = history[accuracy_late_diff_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        bad_accuracy_late_diff_count = int(accuracy_diffs.abs().gt(1).sum().sum())

    _add_check(
        checks,
        "accuracy_late_diff_bounds",
        bad_accuracy_late_diff_count == 0,
        "fatal",
        bad_accuracy_late_diff_count,
        "abs(accuracy late diff) <= 1",
        "Accuracy late differences should be within [-1, 1].",
    )

    audit_df = _write_audit(checks)

    fatal_failures = audit_df[
        (audit_df["severity"] == "fatal")
        & (audit_df["status"] == "FAIL")
    ]

    print("=" * 80)
    print("ROUND FIGHTER STATE P0.1 VALIDATION")
    print("=" * 80)
    print(f"History path     : {history_path}")
    print(f"Latest path      : {latest_path}")
    print(f"Audit path       : {ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH}")
    print(f"History shape    : {history.shape}")
    print(f"Latest shape     : {latest.shape}")
    print(f"Checks           : {len(audit_df)}")
    print(f"Fatal failures   : {len(fatal_failures)}")
    print("=" * 80)
    print(audit_df.to_string(index=False))

    if not fatal_failures.empty:
        raise RoundFighterStateValidationError(
            "Round Fighter State validation failed fatal checks. "
            f"Audit written to {ROUND_FIGHTER_STATE_P0_1_VALIDATION_PATH}"
        )

    print("ROUND FIGHTER STATE VALIDATION PASSED")
    return audit_df


def main() -> None:
    """Run Round Fighter State P0.1 validation."""
    validate_round_fighter_state()


if __name__ == "__main__":
    main()
