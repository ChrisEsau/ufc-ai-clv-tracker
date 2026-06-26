from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from pipeline.round_stats.ufcstats_round_stats import scrape_round_stats_for_queue_row


QUEUE_PATH = Path("data/status/ufc_round_stats_backfill_queue.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview UFCStats round-stat scraping from the standalone queue.")
    parser.add_argument("--queue-path", default=str(QUEUE_PATH))
    parser.add_argument("--max-fights", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=8.0)
    parser.add_argument("--jitter-seconds", type=float, default=3.0)
    parser.add_argument("--max-failures", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queue = pd.read_parquet(args.queue_path)
    pending = queue[queue["status"].astype(str).eq("pending")].head(args.max_fights).copy()

    all_rows = []
    failures = []

    print("=" * 80)
    print("ROUND STATS BATCH PREVIEW")
    print("=" * 80)
    print("Queue:", args.queue_path)
    print("Pending selected:", len(pending))
    print("Writes: disabled")
    print()

    for idx, (_, row) in enumerate(pending.iterrows(), start=1):
        label = f'{row["date"]} | {row["event_name"]} | {row["red_fighter"]} vs {row["blue_fighter"]}'
        print("=" * 80)
        print(f"Fight {idx}/{len(pending)}")
        print(label)
        print(row["fight_url"])

        try:
            scraped = scrape_round_stats_for_queue_row(row)
            duplicate_keys = scraped.duplicated(["fight_id", "fighter_id", "round"]).sum()

            print("Rows:", len(scraped))
            print("Duplicate keys:", duplicate_keys)

            print(scraped[[
                "fight_id",
                "round",
                "corner",
                "fighter_id",
                "fighter_name",
                "sig_str_attempted",
                "td_attempted",
                "ctrl_sec",
            ]].to_string(index=False))

            all_rows.append(scraped)

        except Exception as exc:
            failures.append({"fight_id": row.get("fight_id"), "error": str(exc)})
            print("ERROR:", exc)

            if len(failures) >= args.max_failures:
                print("Stopping because max failures reached.")
                break

        if idx < len(pending):
            wait = float(args.sleep_seconds) + random.uniform(0, float(args.jitter_seconds))
            print(f"Sleeping {wait:.1f} seconds...")
            time.sleep(wait)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Fights attempted:", len(all_rows) + len(failures))
    print("Successful fights:", len(all_rows))
    print("Failed fights:", len(failures))
    print("Round rows parsed:", len(combined))

    if not combined.empty:
        print("Duplicate key count:", combined.duplicated(["fight_id", "fighter_id", "round"]).sum())

    if failures:
        print()
        print("Failures:")
        print(pd.DataFrame(failures).to_string(index=False))


if __name__ == "__main__":
    main()
