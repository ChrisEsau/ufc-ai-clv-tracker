from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ODDS_PATH = Path("data/market/historical_market_outcomes.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
QUEUE_PATH = Path("data/status/ufc_round_stats_backfill_queue.parquet")
AUDIT_PATH = Path("data/audits/ufc_round_stats_backfill_queue_audit.parquet")


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


def main() -> None:
    run_ts = datetime.now(timezone.utc).isoformat()

    odds = pd.read_parquet(ODDS_PATH)
    master = pd.read_parquet(MASTER_PATH)

    odds_fights = odds[["fight_id"]].dropna().drop_duplicates().copy()
    odds_fights["fight_id"] = odds_fights["fight_id"].astype(str)

    master = master.copy()
    master["fight_id"] = master["fight_id"].astype(str)

    master_cols = [
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

    missing_master_cols = [c for c in master_cols if c not in master.columns]
    if missing_master_cols:
        raise SystemExit(f"Master missing required columns: {missing_master_cols}")

    master_fights = master[master_cols].drop_duplicates("fight_id").copy()

    joined = odds_fights.merge(
        master_fights,
        on="fight_id",
        how="left",
        indicator=True,
    )

    matched = joined[joined["_merge"].eq("both")].copy()
    missing = joined[joined["_merge"].eq("left_only")].copy()

    queue = matched.drop(columns=["_merge"]).copy()

    # Use master IDs as canonical join keys.
    queue["red_fighter"] = queue["r_name"]
    queue["blue_fighter"] = queue["b_name"]
    queue["red_fighter_id"] = queue["r_id"]
    queue["blue_fighter_id"] = queue["b_id"]

    # Reconstruct UFCStats URLs from canonical IDs for scraper staging.
    queue["event_url"] = queue["event_id"].map(event_url)
    queue["fight_url"] = queue["fight_id"].map(fight_url)
    queue["red_fighter_url"] = queue["red_fighter_id"].map(fighter_url)
    queue["blue_fighter_url"] = queue["blue_fighter_id"].map(fighter_url)

    queue["status"] = "pending"
    queue["attempt_count"] = 0
    queue["last_attempt_at"] = pd.NA
    queue["last_success_at"] = pd.NA
    queue["last_error"] = pd.NA
    queue["round_rows_scraped"] = 0
    queue["queue_created_at"] = run_ts
    queue["queue_source"] = str(ODDS_PATH)

    queue["_sort_date"] = pd.to_datetime(queue["date"], errors="coerce")

    queue = queue[
        [
            "status",
            "attempt_count",
            "last_attempt_at",
            "last_success_at",
            "last_error",
            "round_rows_scraped",
            "queue_created_at",
            "queue_source",
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
    ].assign(_sort_date=queue["_sort_date"]).sort_values(["_sort_date", "event_name", "fight_id"]).drop(columns=["_sort_date"])

    audit = pd.DataFrame(
        [
            {
                "run_timestamp": run_ts,
                "odds_path": str(ODDS_PATH),
                "master_path": str(MASTER_PATH),
                "odds_rows": len(odds),
                "odds_unique_fights": odds_fights["fight_id"].nunique(),
                "master_rows": len(master),
                "master_unique_fights": master["fight_id"].nunique(),
                "matched_fights": len(matched),
                "missing_from_master": len(missing),
                "queue_rows": len(queue),
                "earliest_queue_date": pd.to_datetime(queue["date"], errors="coerce").min(),
                "latest_queue_date": pd.to_datetime(queue["date"], errors="coerce").max(),
            }
        ]
    )

    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    queue.to_parquet(QUEUE_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)

    print("=" * 80)
    print("ROUND STATS BACKFILL QUEUE")
    print("=" * 80)
    print("Odds unique fights :", odds_fights["fight_id"].nunique())
    print("Master unique fights:", master["fight_id"].nunique())
    print("Matched fights      :", len(matched))
    print("Missing from master :", len(missing))
    print("Queue rows          :", len(queue))
    print("Earliest queue date :", audit.loc[0, "earliest_queue_date"])
    print("Latest queue date   :", audit.loc[0, "latest_queue_date"])
    print()
    print("Saved queue:", QUEUE_PATH)
    print("Saved audit:", AUDIT_PATH)

    if len(missing):
        print()
        print("Sample missing fight_ids from odds not found in master:")
        print(missing[["fight_id"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
