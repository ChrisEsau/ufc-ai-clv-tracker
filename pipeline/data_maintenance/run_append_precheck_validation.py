from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
    APPEND_PRECHECK_PATH,
    APPEND_DUPLICATE_CHECK_PATH,
    APPEND_REQUIRED_FIELD_AUDIT_PATH,
)

STAGED_PATH = STAGED_MASTER_ROWS_PROFILED_PATH
PRECHECK_OUTPUT = APPEND_PRECHECK_PATH
DUPLICATE_OUTPUT = APPEND_DUPLICATE_CHECK_PATH
REQUIRED_FIELD_OUTPUT = APPEND_REQUIRED_FIELD_AUDIT_PATH


def clean_id_series(s):
    return (
        s.astype(str)
        .str.strip()
        .replace(["", "nan", "None", "NaN"], pd.NA)
    )


def missing_value_mask(df, column_name):
    if column_name not in df.columns:
        return pd.Series(True, index=df.index)

    series = df[column_name]

    return series.isna() | (
        series.astype(str)
        .str.strip()
        .isin(["", "nan", "None", "NaN", "NaT"])
    )


def missing_value_count(df, column_name):
    return int(missing_value_mask(df, column_name).sum())


def add_check(checks, check_name, status, failure_count, details, severity="block"):
    checks.append({
        "check_name": check_name,
        "severity": severity,
        "status": status,
        "failure_count": failure_count,
        "details": details,
    })


def run_append_precheck_validation():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    master = pd.read_parquet(MASTER_PATH)
    staged = pd.read_parquet(STAGED_PATH)

    print()
    print("========== PRECHECK INPUTS ==========")
    print("Master rows:", len(master))
    print("Staged rows:", len(staged))
    print("Master cols:", len(master.columns))
    print("Staged cols:", len(staged.columns))

    checks = []

    master_cols = list(master.columns)
    staged_cols = list(staged.columns)

    column_count_match = len(master_cols) == len(staged_cols)
    column_order_match = master_cols == staged_cols

    add_check(
        checks,
        check_name="column_count_match",
        status="pass" if column_count_match else "fail",
        failure_count=0 if column_count_match else 1,
        details=f"master={len(master_cols)}, staged={len(staged_cols)}",
    )

    add_check(
        checks,
        check_name="column_order_match",
        status="pass" if column_order_match else "fail",
        failure_count=0 if column_order_match else 1,
        details="staged column order matches master"
        if column_order_match
        else "staged column order does not match master",
    )

    staged_fight_ids = clean_id_series(staged["fight_id"])
    master_fight_ids = clean_id_series(master["fight_id"])

    valid_staged_fight_ids = staged_fight_ids.dropna()
    valid_master_fight_ids = master_fight_ids.dropna()

    duplicate_in_staged = staged[
        staged_fight_ids.notna()
        & staged_fight_ids.duplicated(keep=False)
    ].copy()

    already_in_master = staged[
        staged_fight_ids.notna()
        & staged_fight_ids.isin(set(valid_master_fight_ids))
    ].copy()

    duplicate_in_staged_count = len(duplicate_in_staged)
    already_in_master_count = len(already_in_master)

    add_check(
        checks,
        check_name="duplicate_fight_ids_in_staged",
        status="pass" if duplicate_in_staged_count == 0 else "fail",
        failure_count=duplicate_in_staged_count,
        details=f"{duplicate_in_staged_count} duplicate staged fight_id rows",
    )

    add_check(
        checks,
        check_name="fight_ids_already_in_master",
        status="pass" if already_in_master_count == 0 else "fail",
        failure_count=already_in_master_count,
        details=f"{already_in_master_count} staged fight_id rows already exist in master",
    )

    duplicate_check = pd.concat(
        [
            duplicate_in_staged.assign(duplicate_type="duplicate_in_staged"),
            already_in_master.assign(duplicate_type="already_in_master"),
        ],
        ignore_index=True,
    )

    duplicate_check.to_parquet(DUPLICATE_OUTPUT, index=False)

    required_fields = [
        "event_id",
        "event_name",
        "date",
        "location",
        "fight_id",
        "division",
        "title_fight",
        "total_rounds",
        "r_name",
        "b_name",
        "r_id",
        "b_id",
        "winner",
        "winner_id",
        "method",
        "finish_round",
        "match_time_sec",
        "r_sig_str_landed",
        "b_sig_str_landed",
        "r_total_str_landed",
        "b_total_str_landed",
    ]

    profile_warning_fields = [
        "r_height",
        "b_height",
        "r_reach",
        "b_reach",
        "r_stance",
        "b_stance",
        "r_dob",
        "b_dob",
    ]

    required_rows = []

    for col in required_fields:
        missing_count = missing_value_count(staged, col)

        required_rows.append({
            "column_name": col,
            "severity": "block",
            "missing_count": missing_count,
            "status": "pass" if missing_count == 0 else "fail",
        })

    for col in profile_warning_fields:
        missing_count = missing_value_count(staged, col)

        required_rows.append({
            "column_name": col,
            "severity": "warning",
            "missing_count": missing_count,
            "status": "pass" if missing_count == 0 else "fail",
        })

    required_audit = pd.DataFrame(required_rows)
    required_audit.to_parquet(REQUIRED_FIELD_OUTPUT, index=False)

    failed_required = required_audit[
        required_audit["status"] == "fail"
    ]

    print()
    print("========== FAILED REQUIRED FIELDS ==========")

    if failed_required.empty:
        print("None")
    else:
        print(
            failed_required[
                ["column_name", "missing_count"]
            ].to_string(index=False)
        )

    failed_required_blockers = required_audit[
        (required_audit["severity"] == "block")
        & (required_audit["status"] == "fail")
    ]

    required_failures = len(failed_required_blockers)

    add_check(
        checks,
        check_name="required_fields_populated",
        status="pass" if required_failures == 0 else "fail",
        failure_count=required_failures,
        details=f"{required_failures} required blocking fields have missing values",
    )

    fighter_identity_fields = ["r_id", "b_id", "winner_id"]
    missing_identity_mask = pd.concat(
        [missing_value_mask(staged, field) for field in fighter_identity_fields],
        axis=1,
    ).any(axis=1)
    missing_identity_count = int(missing_identity_mask.sum())

    add_check(
        checks,
        check_name="fighter_identity_complete",
        status="pass" if missing_identity_count == 0 else "fail",
        failure_count=missing_identity_count,
        details=(
            f"{missing_identity_count} staged rows missing "
            "r_id, b_id, or winner_id"
        ),
    )

    failed_profile_warnings = required_audit[
        (required_audit["severity"] == "warning")
        & (required_audit["status"] == "fail")
    ]

    profile_warning_count = len(failed_profile_warnings)

    add_check(
        checks,
        check_name="profile_completeness_warning",
        severity="warning",
        status="pass" if profile_warning_count == 0 else "fail",
        failure_count=profile_warning_count,
        details=(
            f"{profile_warning_count} optional profile fields have "
            "missing values"
        ),
    )

    stat_cols = [
        c for c in staged.columns
        if any(
            token in c
            for token in [
                "_kd",
                "_landed",
                "_atmpted",
                "_acc",
                "_td_",
                "_sub_att",
                "_ctrl",
                "_per",
                "match_time_sec",
            ]
        )
    ]

    negative_count = 0

    for col in stat_cols:
        vals = pd.to_numeric(staged[col], errors="coerce")
        negative_count += int((vals < 0).sum())

    add_check(
        checks,
        check_name="negative_stat_check",
        status="pass" if negative_count == 0 else "fail",
        failure_count=negative_count,
        details=f"{negative_count} negative numeric stat values found",
    )

    precheck = pd.DataFrame(checks)

    blocking_checks = precheck[precheck["severity"] == "block"]
    append_ready = bool((blocking_checks["status"] == "pass").all())

    precheck["run_id"] = run_id
    precheck["run_timestamp"] = run_timestamp
    precheck["staged_rows"] = len(staged)
    precheck["master_rows"] = len(master)
    precheck["append_ready"] = append_ready

    print()
    print("========== APPEND GATE ==========")
    print("Append ready:", append_ready)

    blocking_failures = precheck[
        (precheck["severity"] == "block")
        & (precheck["status"] == "fail")
    ]

    warning_failures = precheck[
        (precheck["severity"] == "warning")
        & (precheck["status"] == "fail")
    ]

    if not blocking_failures.empty:
        print()
        print("Blocking failed checks:")
        print(
            blocking_failures[
                ["check_name", "failure_count"]
            ].to_string(index=False)
        )

    if not warning_failures.empty:
        print()
        print("Warning checks:")
        print(
            warning_failures[
                ["check_name", "failure_count"]
            ].to_string(index=False)
        )

    precheck.to_parquet(PRECHECK_OUTPUT, index=False)

    print("========== APPEND PRECHECK VALIDATION ==========")
    print("Master rows:", len(master))
    print("Staged rows:", len(staged))
    print("Append ready:", append_ready)
    print()
    print(precheck[["check_name", "severity", "status", "failure_count", "details"]])
    print()
    print("Saved:", PRECHECK_OUTPUT)
    print("Saved:", DUPLICATE_OUTPUT)
    print("Saved:", REQUIRED_FIELD_OUTPUT)
    print()
    print("========== FIGHT ID DEBUG ==========")
    print("Staged fight_id nulls:", int(staged_fight_ids.isna().sum()))
    print("Staged unique valid fight_ids:", int(valid_staged_fight_ids.nunique()))
    print("Master unique valid fight_ids:", int(valid_master_fight_ids.nunique()))
    print(
        "Overlap count:",
        int(staged_fight_ids.isin(set(valid_master_fight_ids)).sum()),
    )

    print()
    print("========== OVERLAPPING FIGHTS ==========")

    overlap_cols = [
        "event_name",
        "date",
        "location",
        "fight_id",
        "division",
        "title_fight",
        "total_rounds",
        "r_name",
        "b_name",
    ]

    overlap_cols = [
        c for c in overlap_cols
        if c in already_in_master.columns
    ]

    if already_in_master.empty:
        print("None")
    else:
        print(
            already_in_master[overlap_cols]
            .head(25)
            .to_string(index=False)
        )

    return precheck, append_ready


if __name__ == "__main__":
    run_append_precheck_validation()