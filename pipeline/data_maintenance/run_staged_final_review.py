from datetime import datetime, timezone

import pandas as pd

from pipeline.common.paths import (
    MASTER_PATH,
    STAGED_MASTER_ROWS_PROFILED_PATH,
    STAGED_FINAL_REVIEW_PATH,
)

STAGED_PATH = STAGED_MASTER_ROWS_PROFILED_PATH
FINAL_REVIEW_OUTPUT = STAGED_FINAL_REVIEW_PATH

MISSING_STRINGS = {"", "nan", "None", "NaN", "NaT", "<NA>"}
VALID_STANCES = {
    "Open Stance",
    "Orthodox",
    "Sideways",
    "Southpaw",
    "Switch",
}


def clean_string(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in MISSING_STRINGS:
        return None

    return value


def normalize_name(value):
    value = clean_string(value)

    if value is None:
        return None

    return " ".join(value.lower().split())


def missing_mask(df, column_name):
    if column_name not in df.columns:
        return pd.Series(True, index=df.index)

    series = df[column_name]

    return series.isna() | (
        series.astype(str)
        .str.strip()
        .isin(MISSING_STRINGS)
    )


def to_numeric_series(df, column_name):
    if column_name not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")

    return pd.to_numeric(df[column_name], errors="coerce")


def add_check(checks, check_name, severity, failure_mask, details):
    failure_count = int(failure_mask.sum())

    checks.append(
        {
            "check_name": check_name,
            "severity": severity,
            "status": "pass" if failure_count == 0 else "fail",
            "failure_count": failure_count,
            "details": details,
        }
    )


def landed_attempted_pairs(columns):
    pairs = []

    for landed_col in columns:
        if not landed_col.endswith("_landed"):
            continue

        attempted_col = landed_col.replace("_landed", "_atmpted")

        if attempted_col in columns:
            pairs.append((landed_col, attempted_col))

    return pairs


def percentage_columns(columns):
    return [
        col for col in columns
        if col.endswith("_acc") or col.endswith("_per") or col.endswith("_pct")
    ]


def run_staged_final_review():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_timestamp = datetime.now(timezone.utc).isoformat()

    master = pd.read_parquet(MASTER_PATH)
    staged = pd.read_parquet(STAGED_PATH)

    checks = []

    print()
    print("========== STAGED FINAL REVIEW INPUTS ==========")
    print("Master rows:", len(master))
    print("Staged rows:", len(staged))

    add_check(
        checks,
        check_name="staged_rows_present",
        severity="block",
        failure_mask=pd.Series([len(staged) == 0]),
        details=f"staged_rows={len(staged)}",
    )

    identity_fields = [
        "event_id",
        "fight_id",
        "r_name",
        "b_name",
        "r_id",
        "b_id",
        "winner",
        "winner_id",
    ]
    missing_identity = pd.Series(False, index=staged.index)

    for field in identity_fields:
        missing_identity = missing_identity | missing_mask(staged, field)

    add_check(
        checks,
        check_name="identity_fields_present",
        severity="block",
        failure_mask=missing_identity,
        details=(
            "requires event_id, fight_id, r_name, b_name, r_id, b_id, "
            "winner, and winner_id"
        ),
    )

    if {"r_id", "b_id"}.issubset(staged.columns):
        same_fighter = (
            staged["r_id"].astype(str).str.strip()
            == staged["b_id"].astype(str).str.strip()
        ) & ~missing_mask(staged, "r_id") & ~missing_mask(staged, "b_id")
    else:
        same_fighter = pd.Series(True, index=staged.index)

    add_check(
        checks,
        check_name="red_blue_fighters_distinct",
        severity="block",
        failure_mask=same_fighter,
        details="r_id and b_id must identify different fighters",
    )

    if {"winner_id", "r_id", "b_id"}.issubset(staged.columns):
        winner_id_matches = staged.apply(
            lambda row: clean_string(row.get("winner_id"))
            in {clean_string(row.get("r_id")), clean_string(row.get("b_id"))},
            axis=1,
        )
        winner_id_mismatch = ~winner_id_matches
    else:
        winner_id_mismatch = pd.Series(True, index=staged.index)

    add_check(
        checks,
        check_name="winner_id_matches_fighter_side",
        severity="block",
        failure_mask=winner_id_mismatch,
        details="winner_id must equal either r_id or b_id",
    )

    if {"winner", "r_name", "b_name"}.issubset(staged.columns):
        winner_name_matches = staged.apply(
            lambda row: normalize_name(row.get("winner"))
            in {normalize_name(row.get("r_name")), normalize_name(row.get("b_name"))},
            axis=1,
        )
        winner_name_mismatch = ~winner_name_matches
    else:
        winner_name_mismatch = pd.Series(True, index=staged.index)

    add_check(
        checks,
        check_name="winner_name_matches_fighter_side",
        severity="block",
        failure_mask=winner_name_mismatch,
        details="winner must equal either r_name or b_name",
    )

    if "fight_id" in staged.columns and "fight_id" in master.columns:
        master_ids = set(master["fight_id"].dropna().astype(str).str.strip())
        staged_ids = staged["fight_id"].astype(str).str.strip()
        duplicate_fight_ids = staged_ids.isin(master_ids)
    else:
        duplicate_fight_ids = pd.Series(True, index=staged.index)

    add_check(
        checks,
        check_name="fight_id_not_in_master",
        severity="block",
        failure_mask=duplicate_fight_ids,
        details="staged fight_id values must not already exist in master",
    )

    if "fight_id" in staged.columns:
        duplicate_staged_fight_ids = (
            staged["fight_id"].astype(str).str.strip().duplicated(keep=False)
            & ~missing_mask(staged, "fight_id")
        )
    else:
        duplicate_staged_fight_ids = pd.Series(True, index=staged.index)

    add_check(
        checks,
        check_name="fight_id_unique_in_staged",
        severity="block",
        failure_mask=duplicate_staged_fight_ids,
        details="staged fight_id values must be unique inside staged rows",
    )

    date_parse = (
        pd.to_datetime(staged["date"], errors="coerce")
        if "date" in staged.columns
        else pd.Series(pd.NaT, index=staged.index)
    )
    bad_dates = date_parse.isna()

    add_check(
        checks,
        check_name="event_date_parseable",
        severity="block",
        failure_mask=bad_dates,
        details="date must be parseable before append",
    )

    metadata_fields = ["location", "division", "title_fight", "total_rounds"]
    missing_metadata = pd.Series(False, index=staged.index)

    for field in metadata_fields:
        missing_metadata = missing_metadata | missing_mask(staged, field)

    add_check(
        checks,
        check_name="fight_metadata_present",
        severity="block",
        failure_mask=missing_metadata,
        details="requires location, division, title_fight, and total_rounds",
    )

    title_fight = to_numeric_series(staged, "title_fight")
    bad_title_fight = title_fight.isna() | ~title_fight.isin([0, 1])

    add_check(
        checks,
        check_name="title_fight_flag_valid",
        severity="block",
        failure_mask=bad_title_fight,
        details="title_fight must be 1 for yes or 0 for no",
    )

    total_rounds = to_numeric_series(staged, "total_rounds")
    bad_total_rounds = total_rounds.isna() | ~total_rounds.isin([3, 5])

    add_check(
        checks,
        check_name="total_rounds_plausible",
        severity="block",
        failure_mask=bad_total_rounds,
        details="total_rounds must be populated as 3 or 5",
    )

    event_conflict = pd.Series(False, index=staged.index)

    if {"event_id", "event_name", "date"}.issubset(staged.columns) and {
        "event_id",
        "event_name",
        "date",
    }.issubset(master.columns):
        master_events = master[["event_id", "event_name", "date"]].dropna(
            subset=["event_id"]
        )

        for idx, row in staged.iterrows():
            event_id = clean_string(row.get("event_id"))

            if event_id is None:
                continue

            matches = master_events[
                master_events["event_id"].astype(str).str.strip() == event_id
            ]

            if matches.empty:
                continue

            staged_name = normalize_name(row.get("event_name"))
            staged_date = pd.to_datetime(row.get("date"), errors="coerce")
            master_names = {normalize_name(value) for value in matches["event_name"]}
            master_dates = pd.to_datetime(matches["date"], errors="coerce")

            date_matches = False
            if not pd.isna(staged_date):
                date_matches = bool((master_dates.dt.date == staged_date.date()).any())

            if staged_name not in master_names or not date_matches:
                event_conflict.loc[idx] = True

    add_check(
        checks,
        check_name="event_identity_consistent_with_master",
        severity="warning",
        failure_mask=event_conflict,
        details="existing master event_id should have matching event_name and date",
    )

    finish_round = to_numeric_series(staged, "finish_round")
    total_rounds = to_numeric_series(staged, "total_rounds")
    bad_finish_round = (
        finish_round.isna()
        | (finish_round < 1)
        | (finish_round > 5)
        | (total_rounds.notna() & (finish_round > total_rounds))
    )

    add_check(
        checks,
        check_name="finish_round_plausible",
        severity="block",
        failure_mask=bad_finish_round,
        details="finish_round must be numeric, between 1 and 5, and not exceed total_rounds",
    )

    match_time_sec = to_numeric_series(staged, "match_time_sec")
    bad_match_time = (
        match_time_sec.isna()
        | (match_time_sec <= 0)
        | (match_time_sec > 1500)
    )

    add_check(
        checks,
        check_name="match_time_plausible",
        severity="block",
        failure_mask=bad_match_time,
        details="match_time_sec must be positive and no more than 25 minutes",
    )

    landed_attempted_failures = pd.Series(False, index=staged.index)

    for landed_col, attempted_col in landed_attempted_pairs(staged.columns):
        landed = pd.to_numeric(staged[landed_col], errors="coerce")
        attempted = pd.to_numeric(staged[attempted_col], errors="coerce")
        landed_attempted_failures = landed_attempted_failures | (
            landed.notna()
            & attempted.notna()
            & (landed > attempted)
        )

    add_check(
        checks,
        check_name="landed_not_greater_than_attempted",
        severity="block",
        failure_mask=landed_attempted_failures,
        details="all *_landed values must be less than or equal to matching *_atmpted values",
    )

    percent_failures = pd.Series(False, index=staged.index)

    for col in percentage_columns(staged.columns):
        values = pd.to_numeric(staged[col], errors="coerce")
        percent_failures = percent_failures | (
            values.notna()
            & ((values < 0) | (values > 100))
        )

    add_check(
        checks,
        check_name="percentage_values_in_range",
        severity="warning",
        failure_mask=percent_failures,
        details="accuracy and percentage values should be between 0 and 100 when populated",
    )

    profile_plausibility_failures = pd.Series(False, index=staged.index)

    for col in ["r_height", "b_height"]:
        values = to_numeric_series(staged, col)
        profile_plausibility_failures = profile_plausibility_failures | (
            values.notna()
            & ((values < 120) | (values > 230))
        )

    for col in ["r_reach", "b_reach"]:
        values = to_numeric_series(staged, col)
        profile_plausibility_failures = profile_plausibility_failures | (
            values.notna()
            & ((values < 120) | (values > 230))
        )

    for col in ["r_stance", "b_stance"]:
        if col in staged.columns:
            stances = staged[col].apply(clean_string)
            profile_plausibility_failures = profile_plausibility_failures | (
                stances.notna()
                & ~stances.isin(VALID_STANCES)
            )

    add_check(
        checks,
        check_name="profile_values_plausible",
        severity="warning",
        failure_mask=profile_plausibility_failures,
        details="height/reach should be plausible centimeters and stance should be recognized",
    )

    review = pd.DataFrame(checks)
    blocking_checks = review[review["severity"] == "block"]
    final_review_pass = bool((blocking_checks["status"] == "pass").all())

    review["run_id"] = run_id
    review["run_timestamp"] = run_timestamp
    review["staged_rows"] = len(staged)
    review["master_rows"] = len(master)
    review["final_review_pass"] = final_review_pass

    print()
    print("========== STAGED FINAL REVIEW ==========")
    print("Final review pass:", final_review_pass)

    failed = review[review["status"] == "fail"]

    if failed.empty:
        print("Failed checks: None")
    else:
        print("Failed checks:")
        print(
            failed[["check_name", "severity", "failure_count", "details"]]
            .to_string(index=False)
        )

    review.to_parquet(FINAL_REVIEW_OUTPUT, index=False)

    print()
    print(review[["check_name", "severity", "status", "failure_count", "details"]])
    print()
    print("Saved:", FINAL_REVIEW_OUTPUT)

    return review, final_review_pass


if __name__ == "__main__":
    run_staged_final_review()
