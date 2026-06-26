from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pipeline.round_stats.ufcstats_round_stats import scrape_round_stats_for_queue_row


QUEUE_PATH = Path("data/status/ufc_round_stats_backfill_queue.parquet")
ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
AUDIT_PATH = Path("data/audits/ufc_round_stats_scrape_audit.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone UFCStats round-stats backfill batch.")
    p.add_argument("--queue-path", default=str(QUEUE_PATH))
    p.add_argument("--output-path", default=str(ROUND_STATS_PATH))
    p.add_argument("--audit-path", default=str(AUDIT_PATH))
    p.add_argument("--max-fights", type=int, default=5)
    p.add_argument("--sleep-seconds", type=float, default=8.0)
    p.add_argument("--jitter-seconds", type=float, default=4.0)
    p.add_argument("--max-failures", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_ts = datetime.now(timezone.utc).isoformat()

    queue_path = Path(args.queue_path)
    output_path = Path(args.output_path)
    audit_path = Path(args.audit_path)

    queue = pd.read_parquet(queue_path)

    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    existing_fights = set(existing["fight_id"].astype(str)) if not existing.empty and "fight_id" in existing.columns else set()

    pending = queue[
        queue["status"].astype(str).eq("pending")
        & ~queue["fight_id"].astype(str).isin(existing_fights)
    ].head(args.max_fights).copy()

    all_rows = []
    audit_rows = []

    print("=" * 80)
    print("ROUND STATS BACKFILL BATCH")
    print("=" * 80)
    print("Queue:", queue_path)
    print("Output:", output_path)
    print("Pending selected:", len(pending))
    print("Dry run:", args.dry_run)

    for n, (idx, row) in enumerate(pending.iterrows(), start=1):
        fight_id = str(row["fight_id"])
        print()
        print("=" * 80)
        print(f"Fight {n}/{len(pending)}:", row["date"], row["event_name"])
        print(row["red_fighter"], "vs", row["blue_fighter"])
        print(row["fight_url"])

        attempt_ts = datetime.now(timezone.utc).isoformat()
        queue.loc[idx, "status"] = "running"
        queue.loc[idx, "attempt_count"] = int(queue.loc[idx, "attempt_count"]) + 1
        queue.loc[idx, "last_attempt_at"] = attempt_ts
        if not args.dry_run:
            queue.to_parquet(queue_path, index=False)

        try:
            df = scrape_round_stats_for_queue_row(row)
            dupes = int(df.duplicated(["fight_id", "fighter_id", "round"]).sum())

            if df.empty:
                raise RuntimeError("Scrape returned 0 rows")
            if dupes:
                raise RuntimeError(f"Scrape returned duplicate round keys: {dupes}")

            all_rows.append(df)

            queue.loc[idx, "status"] = "success"
            queue.loc[idx, "last_success_at"] = run_ts
            queue.loc[idx, "last_error"] = pd.NA
            queue.loc[idx, "round_rows_scraped"] = len(df)

            audit_rows.append({
                "run_timestamp": run_ts,
                "fight_id": fight_id,
                "event_id": row.get("event_id"),
                "event_name": row.get("event_name"),
                "date": row.get("date"),
                "red_fighter": row.get("red_fighter"),
                "blue_fighter": row.get("blue_fighter"),
                "status": "success",
                "round_rows": len(df),
                "error": None,
            })

            print("SUCCESS rows:", len(df))

        except Exception as exc:
            queue.loc[idx, "status"] = "failed"
            queue.loc[idx, "last_error"] = str(exc)

            audit_rows.append({
                "run_timestamp": run_ts,
                "fight_id": fight_id,
                "event_id": row.get("event_id"),
                "event_name": row.get("event_name"),
                "date": row.get("date"),
                "red_fighter": row.get("red_fighter"),
                "blue_fighter": row.get("blue_fighter"),
                "status": "failed",
                "round_rows": 0,
                "error": str(exc),
            })

            print("FAILED:", exc)

            if sum(a["status"] == "failed" for a in audit_rows) >= args.max_failures:
                print("Stopping because max failures reached.")
                break

        if n < len(pending):
            wait = float(args.sleep_seconds) + random.uniform(0, float(args.jitter_seconds))
            print(f"Sleeping {wait:.1f} seconds...")
            time.sleep(wait)

    new_rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    audit = pd.DataFrame(audit_rows)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Successful fights:", sum(a["status"] == "success" for a in audit_rows))
    print("Failed fights:", sum(a["status"] == "failed" for a in audit_rows))
    print("New round rows:", len(new_rows))

    if args.dry_run:
        print("Dry run: no files written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    if not new_rows.empty:
        combined = pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
        combined = combined.drop_duplicates(["fight_id", "fighter_id", "round"], keep="last")
        combined.to_parquet(output_path, index=False)

    if audit_path.exists():
        old_audit = pd.read_parquet(audit_path)
        audit = pd.concat([old_audit, audit], ignore_index=True)
    audit.to_parquet(audit_path, index=False)

    queue.to_parquet(queue_path, index=False)

    print("Wrote:", output_path)
    print("Wrote:", audit_path)
    print("Updated:", queue_path)


if __name__ == "__main__":
    main()
