from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MASTER_PATH = Path("data/master/ufc_master.parquet")
ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
QUEUE_PATH = Path("data/status/ufc_round_stats_backfill_queue.parquet")
AUDIT_PATH = Path("data/audits/ufc_round_stats_backfill_queue_audit.parquet")

DEFAULT_START_DATE = "2026-03-26"


def fighter_url(fighter_id: str | None) -> str | None:
    if pd.isna(fighter_id) or not str(fighter_id).strip():
        return None
    return f"http://ufcstats.com/fighter-details/{fighter_id}"


def fight_url(fight_id: str | None) -> str | None:
    if pd.isna(fight_id) or not str(fight_id).strip():
        return None
    return f"http://ufcstats.com/fight-details/{fight_id}"


def event_url(event_id: str | None) -> str | None:
    if pd.isna(event_id) or not str(event_id).strip():
        return None
    return f"http://ufcstats.com/event-details/{event_id}"


def _parse_date(value: str | None, *, default: pd.Timestamp) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        return default
    parsed = pd.to_datetime(value, errors="raise")
    return pd.Timestamp(parsed).normalize()


def _load_existing_round_fight_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    round_stats = pd.read_parquet(path)
    if round_stats.empty or "fight_id" not in round_stats.columns:
        return set()

    return set(round_stats["fight_id"].dropna().astype(str).str.strip())


def _load_existing_queue(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    queue = pd.read_parquet(path)
    if queue.empty or "fight_id" not in queue.columns:
        return pd.DataFrame()

    queue = queue.copy()
    queue["fight_id"] = queue["fight_id"].astype(str).str.strip()
    return queue.drop_duplicates("fight_id", keep="last")


def _existing_queue_metadata(existing_queue: pd.DataFrame) -> pd.DataFrame:
    preserve_cols = [
        "fight_id",
        "status",
        "attempt_count",
        "last_attempt_at",
        "last_success_at",
        "last_error",
        "round_rows_scraped",
    ]

    if existing_queue.empty:
        return pd.DataFrame(columns=preserve_cols)

    cols = [c for c in preserve_cols if c in existing_queue.columns]
    return existing_queue[cols].copy()


def build_queue(
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    reset_status: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_ts = datetime.now(timezone.utc).isoformat()

    master = pd.read_parquet(MASTER_PATH).copy()

    required_cols = [
        "event_id",
        "event_name",
        "date",
        "location",
        "fight_id",
        "division",
        "title_fight",
        "finish_round",
        "match_time_sec",
        "total_rounds",
        "r_name",
        "r_id",
        "b_name",
        "b_id",
        "winner",
        "winner_id",
    ]

    missing_master_cols = [c for c in required_cols if c not in master.columns]
    if missing_master_cols:
        raise SystemExit(f"Master missing required columns: {missing_master_cols}")

    master = master[required_cols].dropna(subset=["fight_id"]).copy()
    master["fight_id"] = master["fight_id"].astype(str).str.strip()
    master["date"] = pd.to_datetime(master["date"], errors="coerce")

    master_window = master[
        master["date"].between(start_date, end_date, inclusive="both")
    ].drop_duplicates("fight_id").copy()

    existing_round_fight_ids = _load_existing_round_fight_ids(ROUND_STATS_PATH)

    missing_round_stats = master_window[
        ~master_window["fight_id"].isin(existing_round_fight_ids)
    ].copy()

    queue = missing_round_stats.copy()

    queue["red_fighter"] = queue["r_name"]
    queue["blue_fighter"] = queue["b_name"]
    queue["red_fighter_id"] = queue["r_id"]
    queue["blue_fighter_id"] = queue["b_id"]

    queue["event_url"] = queue["event_id"].map(event_url)
    queue["fight_url"] = queue["fight_id"].map(fight_url)
    queue["red_fighter_url"] = queue["red_fighter_id"].map(fighter_url)
    queue["blue_fighter_url"] = queue["blue_fighter_id"].map(fighter_url)

    existing_queue = _load_existing_queue(QUEUE_PATH)
    existing_meta = _existing_queue_metadata(existing_queue)

    if not reset_status and not existing_meta.empty:
        queue = queue.merge(
            existing_meta,
            on="fight_id",
            how="left",
            suffixes=("", "_existing"),
        )
    else:
        for col in [
            "status",
            "attempt_count",
            "last_attempt_at",
            "last_success_at",
            "last_error",
            "round_rows_scraped",
        ]:
            queue[col] = pd.NA

    queue["status"] = queue["status"].fillna("pending")
    queue["attempt_count"] = pd.to_numeric(queue["attempt_count"], errors="coerce").fillna(0).astype(int)
    queue["last_attempt_at"] = queue["last_attempt_at"].astype("object")
    queue["last_success_at"] = queue["last_success_at"].astype("object")
    queue["last_error"] = queue["last_error"].astype("object")
    queue["round_rows_scraped"] = pd.to_numeric(queue["round_rows_scraped"], errors="coerce").fillna(0).astype(int)

    if reset_status:
        queue["status"] = "pending"
        queue["attempt_count"] = 0
        queue["last_attempt_at"] = pd.NA
        queue["last_success_at"] = pd.NA
        queue["last_error"] = pd.NA
        queue["round_rows_scraped"] = 0

    queue["queue_created_at"] = run_ts
    queue["queue_source"] = str(MASTER_PATH)
    queue["round_stats_path"] = str(ROUND_STATS_PATH)
    queue["queue_start_date"] = start_date.date().isoformat()
    queue["queue_end_date"] = end_date.date().isoformat()

    output_cols = [
        "status",
        "attempt_count",
        "last_attempt_at",
        "last_success_at",
        "last_error",
        "round_rows_scraped",
        "queue_created_at",
        "queue_source",
        "round_stats_path",
        "queue_start_date",
        "queue_end_date",
        "date",
        "event_id",
        "event_name",
        "event_url",
        "location",
        "fight_id",
        "fight_url",
        "division",
        "title_fight",
        "finish_round",
        "match_time_sec",
        "total_rounds",
        "red_fighter",
        "red_fighter_id",
        "red_fighter_url",
        "blue_fighter",
        "blue_fighter_id",
        "blue_fighter_url",
        "winner",
        "winner_id",
    ]

    queue["_sort_date"] = pd.to_datetime(queue["date"], errors="coerce")
    queue = (
        queue[output_cols + ["_sort_date"]]
        .sort_values(["_sort_date", "event_name", "fight_id"], ascending=[True, True, True])
        .drop(columns=["_sort_date"])
        .reset_index(drop=True)
    )

    audit = pd.DataFrame(
        [
            {
                "run_timestamp": run_ts,
                "master_path": str(MASTER_PATH),
                "round_stats_path": str(ROUND_STATS_PATH),
                "queue_path": str(QUEUE_PATH),
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "reset_status": reset_status,
                "master_rows": len(master),
                "master_unique_fights": master["fight_id"].nunique(),
                "master_window_unique_fights": master_window["fight_id"].nunique(),
                "existing_round_stats_unique_fights": len(existing_round_fight_ids),
                "missing_round_stats_fights": missing_round_stats["fight_id"].nunique(),
                "queue_rows": len(queue),
                "earliest_queue_date": pd.to_datetime(queue["date"], errors="coerce").min() if not queue.empty else pd.NaT,
                "latest_queue_date": pd.to_datetime(queue["date"], errors="coerce").max() if not queue.empty else pd.NaT,
            }
        ]
    )

    return queue, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build UFCStats round-stats queue from canonical master fights missing round stats."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--reset-status", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    today = pd.Timestamp.today().normalize()
    start_date = _parse_date(args.start_date, default=pd.Timestamp(DEFAULT_START_DATE))
    end_date = _parse_date(args.end_date, default=today)

    if start_date > end_date:
        raise SystemExit(f"start_date must be <= end_date. Got {start_date.date()} > {end_date.date()}")

    queue, audit = build_queue(
        start_date=start_date,
        end_date=end_date,
        reset_status=bool(args.reset_status),
    )

    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    queue.to_parquet(QUEUE_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)

    print("=" * 80)
    print("ROUND STATS BACKFILL QUEUE")
    print("=" * 80)
    print("Source               :", MASTER_PATH)
    print("Round stats          :", ROUND_STATS_PATH)
    print("Start date           :", start_date.date())
    print("End date             :", end_date.date())
    print("Master window fights :", int(audit.loc[0, "master_window_unique_fights"]))
    print("Existing RS fights   :", int(audit.loc[0, "existing_round_stats_unique_fights"]))
    print("Missing RS fights    :", int(audit.loc[0, "missing_round_stats_fights"]))
    print("Queue rows           :", len(queue))
    print("Earliest queue date  :", audit.loc[0, "earliest_queue_date"])
    print("Latest queue date    :", audit.loc[0, "latest_queue_date"])
    print()
    print("Saved queue:", QUEUE_PATH)
    print("Saved audit:", AUDIT_PATH)

    if not queue.empty:
        print()
        preview_cols = ["date", "event_name", "fight_id", "red_fighter", "blue_fighter", "status"]
        print(queue[preview_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
