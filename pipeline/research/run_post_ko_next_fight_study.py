"""Study fighter performance in the fight after a KO/TKO loss.

This is a standalone research runner. It reads the canonical historical master
file and writes research-only artifacts under data/research/. It does not change
master, feature, model, prediction, market, or dashboard artifacts.

Run from repo root:

    python -m pipeline.research.run_post_ko_next_fight_study

Optional:

    python -m pipeline.research.run_post_ko_next_fight_study \
        --master-path data/master/ufc_master.parquet \
        --output-dir data/research \
        --modern-era-start 2005-01-01 \
        --recent-era-start 2015-01-01
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.common.paths import MASTER_PATH

DEFAULT_OUTPUT_DIR = Path("data/research")
DEFAULT_MODERN_ERA_START = "2005-01-01"
DEFAULT_RECENT_ERA_START = "2015-01-01"
DEFAULT_PRIME_ERA_START = "2020-01-01"

DETAIL_OUTPUT_NAME = "post_ko_next_fight_study.parquet"
DETAIL_CSV_NAME = "post_ko_next_fight_study.csv"
SUMMARY_OUTPUT_NAME = "post_ko_next_fight_summary.csv"
BUCKET_OUTPUT_NAME = "post_ko_next_fight_bucket_summary.csv"
ERA_SUMMARY_OUTPUT_NAME = "post_ko_next_fight_era_summary.csv"
MODERN_DETAIL_OUTPUT_NAME = "post_ko_next_fight_modern_study.parquet"
MODERN_DETAIL_CSV_NAME = "post_ko_next_fight_modern_study.csv"
MODERN_SUMMARY_OUTPUT_NAME = "post_ko_next_fight_modern_summary.csv"
MODERN_BUCKET_OUTPUT_NAME = "post_ko_next_fight_modern_bucket_summary.csv"
AGE_SUMMARY_OUTPUT_NAME = "post_ko_next_fight_age_summary.csv"
KO_COUNT_SUMMARY_OUTPUT_NAME = "post_ko_next_fight_ko_count_summary.csv"

KO_METHOD_PATTERNS = (
    "ko/tko",
    "tko",
    "ko",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how UFC fighters perform in their next fight after a KO/TKO loss."
    )
    parser.add_argument(
        "--master-path",
        default=str(MASTER_PATH),
        help="Path to ufc_master.parquet. Defaults to pipeline.common.paths.MASTER_PATH.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for research outputs. Defaults to data/research.",
    )
    parser.add_argument(
        "--modern-era-start",
        default=DEFAULT_MODERN_ERA_START,
        help="Modern-era start date for filtered outputs. Defaults to 2005-01-01.",
    )
    parser.add_argument(
        "--recent-era-start",
        default=DEFAULT_RECENT_ERA_START,
        help="Recent-era start date for era comparison. Defaults to 2015-01-01.",
    )
    parser.add_argument(
        "--prime-era-start",
        default=DEFAULT_PRIME_ERA_START,
        help="Prime-era start date for era comparison. Defaults to 2020-01-01.",
    )
    parser.add_argument(
        "--include-same-day-next-fights",
        action="store_true",
        help="Keep same-day next fights in modern filtered outputs. Default excludes them.",
    )
    parser.add_argument(
        "--include-doctor-stoppage",
        action="store_true",
        help=(
            "Include method strings containing doctor stoppage as KO/TKO losses. "
            "Default is to exclude them because injury/doctor stoppages may not reflect concussion damage."
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Master file is missing required columns: {missing}")


def normalize_method(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def is_ko_tko_method(method: object, include_doctor_stoppage: bool = False) -> bool:
    method_norm = normalize_method(method)
    if not method_norm:
        return False

    if not include_doctor_stoppage and "doctor" in method_norm:
        return False

    return any(pattern in method_norm for pattern in KO_METHOD_PATTERNS)


def parse_dates(date_series: pd.Series) -> pd.Series:
    # Master schema documents M/D/YYYY, but this fallback keeps the study usable if
    # imported files contain ISO or mixed date formats.
    parsed = pd.to_datetime(date_series, format="%m/%d/%Y", errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(date_series.loc[missing_mask], errors="coerce")
    return parsed


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_corner_age_series(df: pd.DataFrame, corner: str) -> pd.Series:
    """Return fighter age for a corner when available, otherwise NA.

    The master schema may evolve, so this supports common age column names while
    keeping the study runnable when age is not present.
    """

    direct_candidates = [
        f"{corner}_age",
        f"{corner}_fighter_age",
        f"{corner}_age_years",
    ]
    direct_col = first_existing_column(df, direct_candidates)
    if direct_col:
        return pd.to_numeric(df[direct_col], errors="coerce")

    dob_candidates = [
        f"{corner}_dob",
        f"{corner}_date_of_birth",
        f"{corner}_fighter_dob",
        f"{corner}_fighter_date_of_birth",
    ]
    dob_col = first_existing_column(df, dob_candidates)
    if dob_col:
        dob = pd.to_datetime(df[dob_col], errors="coerce")
        fight_date = df["fight_date"]
        return (fight_date - dob).dt.days / 365.25

    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def build_fighter_fight_history(master: pd.DataFrame, include_doctor_stoppage: bool = False) -> pd.DataFrame:
    required = [
        "event_id",
        "event_name",
        "date",
        "fight_id",
        "division",
        "method",
        "finish_round",
        "match_time_sec",
        "total_rounds",
        "r_name",
        "r_id",
        "b_name",
        "b_id",
        "winner_id",
    ]
    require_columns(master, required)

    df = master.copy()
    df["fight_date"] = parse_dates(df["date"])
    df["method_norm"] = df["method"].map(normalize_method)
    df["is_ko_tko_fight"] = df["method"].map(
        lambda method: is_ko_tko_method(method, include_doctor_stoppage=include_doctor_stoppage)
    )
    df["r_fighter_age"] = build_corner_age_series(df, "r")
    df["b_fighter_age"] = build_corner_age_series(df, "b")

    red = df.assign(
        fighter_id=df["r_id"],
        fighter_name=df["r_name"],
        opponent_id=df["b_id"],
        opponent_name=df["b_name"],
        fighter_age=df["r_fighter_age"],
        corner="red",
    )
    blue = df.assign(
        fighter_id=df["b_id"],
        fighter_name=df["b_name"],
        opponent_id=df["r_id"],
        opponent_name=df["r_name"],
        fighter_age=df["b_fighter_age"],
        corner="blue",
    )

    history = pd.concat([red, blue], ignore_index=True, sort=False)
    history["won_fight"] = history["fighter_id"].astype(str) == history["winner_id"].astype(str)
    history["lost_fight"] = ~history["won_fight"]
    history["lost_by_ko_tko"] = history["lost_fight"] & history["is_ko_tko_fight"]

    history = history.sort_values(
        ["fighter_id", "fight_date", "event_id", "fight_id", "corner"],
        na_position="last",
    ).reset_index(drop=True)

    history["fighter_fight_number"] = history.groupby("fighter_id").cumcount() + 1
    history["career_ko_loss_number"] = history.groupby("fighter_id")["lost_by_ko_tko"].cumsum()
    history["next_fight_date"] = history.groupby("fighter_id")["fight_date"].shift(-1)
    history["next_fight_id"] = history.groupby("fighter_id")["fight_id"].shift(-1)
    history["next_event_id"] = history.groupby("fighter_id")["event_id"].shift(-1)
    history["next_event_name"] = history.groupby("fighter_id")["event_name"].shift(-1)
    history["next_division"] = history.groupby("fighter_id")["division"].shift(-1)
    history["next_opponent_id"] = history.groupby("fighter_id")["opponent_id"].shift(-1)
    history["next_opponent_name"] = history.groupby("fighter_id")["opponent_name"].shift(-1)
    history["next_method"] = history.groupby("fighter_id")["method"].shift(-1)
    history["next_finish_round"] = history.groupby("fighter_id")["finish_round"].shift(-1)
    history["next_match_time_sec"] = history.groupby("fighter_id")["match_time_sec"].shift(-1)
    history["won_next_fight"] = history.groupby("fighter_id")["won_fight"].shift(-1)
    history["lost_next_fight"] = history.groupby("fighter_id")["lost_fight"].shift(-1)
    history["next_fight_was_ko_tko"] = history.groupby("fighter_id")["is_ko_tko_fight"].shift(-1)
    history["next_fight_lost_by_ko_tko"] = history.groupby("fighter_id")["lost_by_ko_tko"].shift(-1)
    history["days_to_next_fight"] = (history["next_fight_date"] - history["fight_date"]).dt.days

    return history


def add_research_buckets(study: pd.DataFrame) -> pd.DataFrame:
    study = study.copy()

    study["layoff_bucket"] = pd.cut(
        study["days_to_next_fight"],
        bins=[-1, 120, 180, 270, 365, 540, 10_000],
        labels=["0-120", "121-180", "181-270", "271-365", "366-540", "541+"],
    )

    study["age_bucket"] = pd.cut(
        pd.to_numeric(study["fighter_age"], errors="coerce"),
        bins=[0, 29.999, 34.999, 100],
        labels=["<30", "30-34", "35+"],
    )

    ko_loss_number = pd.to_numeric(study["career_ko_loss_number"], errors="coerce")
    study["career_ko_loss_bucket"] = pd.cut(
        ko_loss_number,
        bins=[0, 1, 2, 10_000],
        labels=["1", "2", "3+"],
    )

    return study


def build_post_ko_study(history: pd.DataFrame) -> pd.DataFrame:
    study = history.loc[history["lost_by_ko_tko"]].copy()
    study = study.loc[study["next_fight_id"].notna()].copy()

    output_columns = [
        "fighter_id",
        "fighter_name",
        "fighter_fight_number",
        "fighter_age",
        "career_ko_loss_number",
        "fight_date",
        "event_name",
        "fight_id",
        "division",
        "opponent_id",
        "opponent_name",
        "method",
        "finish_round",
        "match_time_sec",
        "days_to_next_fight",
        "next_fight_date",
        "next_event_name",
        "next_fight_id",
        "next_division",
        "next_opponent_id",
        "next_opponent_name",
        "next_method",
        "next_finish_round",
        "next_match_time_sec",
        "won_next_fight",
        "lost_next_fight",
        "next_fight_was_ko_tko",
        "next_fight_lost_by_ko_tko",
    ]
    study = study[output_columns].copy()

    bool_columns = [
        "won_next_fight",
        "lost_next_fight",
        "next_fight_was_ko_tko",
        "next_fight_lost_by_ko_tko",
    ]
    for col in bool_columns:
        study[col] = study[col].astype("boolean")

    study = add_research_buckets(study)

    return study.sort_values(["fight_date", "fighter_name"]).reset_index(drop=True)


def filter_study(
    study: pd.DataFrame,
    start_date: str,
    include_same_day_next_fights: bool = False,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    filtered = study.loc[study["fight_date"] >= start_ts].copy()
    if not include_same_day_next_fights:
        filtered = filtered.loc[filtered["days_to_next_fight"] > 0].copy()
    return filtered.reset_index(drop=True)


def rate(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mean())


def build_summary(study: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("ko_loss_with_next_fight_count", len(study)),
        ("unique_fighters", study["fighter_id"].nunique()),
        ("won_next_fight_count", int(study["won_next_fight"].fillna(False).sum())),
        ("won_next_fight_rate", rate(study["won_next_fight"].astype("float"))),
        ("lost_next_fight_count", int(study["lost_next_fight"].fillna(False).sum())),
        ("lost_next_fight_rate", rate(study["lost_next_fight"].astype("float"))),
        (
            "lost_next_by_ko_tko_count",
            int(study["next_fight_lost_by_ko_tko"].fillna(False).sum()),
        ),
        (
            "lost_next_by_ko_tko_rate",
            rate(study["next_fight_lost_by_ko_tko"].astype("float")),
        ),
        ("avg_days_to_next_fight", float(study["days_to_next_fight"].mean())),
        ("median_days_to_next_fight", float(study["days_to_next_fight"].median())),
        ("avg_fighter_age", float(pd.to_numeric(study["fighter_age"], errors="coerce").mean())),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_group_summary(study: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        study.groupby([group_col], observed=True)
        .agg(
            ko_loss_with_next_fight_count=("fighter_id", "size"),
            unique_fighters=("fighter_id", "nunique"),
            won_next_fight_rate=("won_next_fight", lambda s: rate(s.astype("float"))),
            lost_next_by_ko_tko_rate=(
                "next_fight_lost_by_ko_tko",
                lambda s: rate(s.astype("float")),
            ),
            avg_days_to_next_fight=("days_to_next_fight", "mean"),
            avg_fighter_age=("fighter_age", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )
    return grouped


def build_era_summary(
    study: pd.DataFrame,
    modern_start: str,
    recent_start: str,
    prime_start: str,
    include_same_day_next_fights: bool = False,
) -> pd.DataFrame:
    eras = [
        ("all_history", None),
        (f"modern_{modern_start}", modern_start),
        (f"recent_{recent_start}", recent_start),
        (f"prime_{prime_start}", prime_start),
    ]

    rows: list[dict[str, object]] = []
    for era_name, start_date in eras:
        era_study = study.copy()
        if start_date is not None:
            era_study = filter_study(
                era_study,
                start_date=start_date,
                include_same_day_next_fights=include_same_day_next_fights,
            )
        summary = build_summary(era_study)
        row = {"era": era_name, "start_date": start_date or ""}
        row.update(dict(zip(summary["metric"], summary["value"], strict=False)))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    master_path = Path(args.master_path)
    output_dir = Path(args.output_dir)

    if not master_path.exists():
        raise FileNotFoundError(f"Master file not found: {master_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(master_path)
    history = build_fighter_fight_history(
        master,
        include_doctor_stoppage=args.include_doctor_stoppage,
    )
    study = build_post_ko_study(history)
    modern_study = filter_study(
        study,
        start_date=args.modern_era_start,
        include_same_day_next_fights=args.include_same_day_next_fights,
    )

    summary = build_summary(study)
    bucket_summary = build_group_summary(study, "layoff_bucket")
    era_summary = build_era_summary(
        study,
        modern_start=args.modern_era_start,
        recent_start=args.recent_era_start,
        prime_start=args.prime_era_start,
        include_same_day_next_fights=args.include_same_day_next_fights,
    )
    modern_summary = build_summary(modern_study)
    modern_bucket_summary = build_group_summary(modern_study, "layoff_bucket")
    age_summary = build_group_summary(modern_study, "age_bucket")
    ko_count_summary = build_group_summary(modern_study, "career_ko_loss_bucket")

    detail_parquet_path = output_dir / DETAIL_OUTPUT_NAME
    detail_csv_path = output_dir / DETAIL_CSV_NAME
    summary_path = output_dir / SUMMARY_OUTPUT_NAME
    bucket_path = output_dir / BUCKET_OUTPUT_NAME
    era_summary_path = output_dir / ERA_SUMMARY_OUTPUT_NAME
    modern_detail_parquet_path = output_dir / MODERN_DETAIL_OUTPUT_NAME
    modern_detail_csv_path = output_dir / MODERN_DETAIL_CSV_NAME
    modern_summary_path = output_dir / MODERN_SUMMARY_OUTPUT_NAME
    modern_bucket_path = output_dir / MODERN_BUCKET_OUTPUT_NAME
    age_summary_path = output_dir / AGE_SUMMARY_OUTPUT_NAME
    ko_count_summary_path = output_dir / KO_COUNT_SUMMARY_OUTPUT_NAME

    study.to_parquet(detail_parquet_path, index=False)
    study.to_csv(detail_csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    bucket_summary.to_csv(bucket_path, index=False)
    era_summary.to_csv(era_summary_path, index=False)
    modern_study.to_parquet(modern_detail_parquet_path, index=False)
    modern_study.to_csv(modern_detail_csv_path, index=False)
    modern_summary.to_csv(modern_summary_path, index=False)
    modern_bucket_summary.to_csv(modern_bucket_path, index=False)
    age_summary.to_csv(age_summary_path, index=False)
    ko_count_summary.to_csv(ko_count_summary_path, index=False)

    print("Post-KO next-fight study complete")
    print(f"Input master rows: {len(master):,}")
    print(f"All-history KO/TKO losses with a next fight: {len(study):,}")
    print(f"Modern KO/TKO losses with a next fight: {len(modern_study):,}")
    print(f"Modern era start: {args.modern_era_start}")
    print(f"Same-day next fights included in modern outputs: {args.include_same_day_next_fights}")
    print(f"Detail parquet: {detail_parquet_path}")
    print(f"Modern detail parquet: {modern_detail_parquet_path}")
    print(f"Era summary CSV: {era_summary_path}")
    print(f"Modern summary CSV: {modern_summary_path}")
    print("\nModern summary:")
    print(modern_summary.to_string(index=False))
    print("\nEra comparison:")
    print(era_summary.to_string(index=False))


if __name__ == "__main__":
    main()
