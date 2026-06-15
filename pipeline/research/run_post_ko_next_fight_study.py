"""Study fighter performance in the fight after a KO/TKO loss.

This is a standalone research runner. It reads the canonical historical master
file and writes research-only artifacts under data/research/. It does not change
master, feature, model, prediction, market, or dashboard artifacts.

Run from repo root:

    python -m pipeline.research.run_post_ko_next_fight_study

Optional:

    python -m pipeline.research.run_post_ko_next_fight_study \
        --master-path data/master/ufc_master.parquet \
        --output-dir data/research
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from pipeline.common.paths import MASTER_PATH

DEFAULT_OUTPUT_DIR = Path("data/research")
DETAIL_OUTPUT_NAME = "post_ko_next_fight_study.parquet"
DETAIL_CSV_NAME = "post_ko_next_fight_study.csv"
SUMMARY_OUTPUT_NAME = "post_ko_next_fight_summary.csv"
BUCKET_OUTPUT_NAME = "post_ko_next_fight_bucket_summary.csv"

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

    red = df.assign(
        fighter_id=df["r_id"],
        fighter_name=df["r_name"],
        opponent_id=df["b_id"],
        opponent_name=df["b_name"],
        corner="red",
    )
    blue = df.assign(
        fighter_id=df["b_id"],
        fighter_name=df["b_name"],
        opponent_id=df["r_id"],
        opponent_name=df["r_name"],
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


def build_post_ko_study(history: pd.DataFrame) -> pd.DataFrame:
    study = history.loc[history["lost_by_ko_tko"]].copy()
    study = study.loc[study["next_fight_id"].notna()].copy()

    output_columns = [
        "fighter_id",
        "fighter_name",
        "fighter_fight_number",
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

    study["layoff_bucket"] = pd.cut(
        study["days_to_next_fight"],
        bins=[-1, 120, 180, 270, 365, 540, 10_000],
        labels=["0-120", "121-180", "181-270", "271-365", "366-540", "541+"],
    )

    return study.sort_values(["fight_date", "fighter_name"]).reset_index(drop=True)


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
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def build_bucket_summary(study: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["layoff_bucket"]
    bucket = (
        study.groupby(group_cols, observed=True)
        .agg(
            ko_loss_with_next_fight_count=("fighter_id", "size"),
            unique_fighters=("fighter_id", "nunique"),
            won_next_fight_rate=("won_next_fight", lambda s: rate(s.astype("float"))),
            lost_next_by_ko_tko_rate=(
                "next_fight_lost_by_ko_tko",
                lambda s: rate(s.astype("float")),
            ),
            avg_days_to_next_fight=("days_to_next_fight", "mean"),
        )
        .reset_index()
    )
    return bucket


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
    summary = build_summary(study)
    bucket_summary = build_bucket_summary(study)

    detail_parquet_path = output_dir / DETAIL_OUTPUT_NAME
    detail_csv_path = output_dir / DETAIL_CSV_NAME
    summary_path = output_dir / SUMMARY_OUTPUT_NAME
    bucket_path = output_dir / BUCKET_OUTPUT_NAME

    study.to_parquet(detail_parquet_path, index=False)
    study.to_csv(detail_csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    bucket_summary.to_csv(bucket_path, index=False)

    print("Post-KO next-fight study complete")
    print(f"Input master rows: {len(master):,}")
    print(f"KO/TKO losses with a next fight: {len(study):,}")
    print(f"Detail parquet: {detail_parquet_path}")
    print(f"Detail CSV: {detail_csv_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Bucket summary CSV: {bucket_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
