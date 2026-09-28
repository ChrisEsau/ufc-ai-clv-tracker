"""Measurement-only inventory for FSR V3 cold-start inputs.

Scans repository parquet schemas without modifying data and reports fields that
could support leakage-safe pre-UFC / early-career priors. It also checks whether
UFCStats pathway events are already in the round feed and whether profile record
fields appear historically stable or event-time varying.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOTS = [Path("data/master"), Path("data/fight_details"), Path("data/features"), Path("data/research")]
KEYWORDS = (
    "fighter", "opponent", "date", "dob", "age", "height", "reach", "weight",
    "record", "win", "loss", "draw", "method", "promotion", "org", "event",
    "amateur", "pro", "wrest", "grap", "school", "college", "stance",
    "ko", "sub", "decision", "round", "time",
)
PATHWAY_PATTERNS = (
    "contender", "road to ufc", "the ultimate fighter", "tuf",
)


def relevant(columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        low = c.lower()
        if any(k in low for k in KEYWORDS):
            out.append(c)
    return out


def pathway_inventory() -> None:
    path = Path("data/fight_details/ufc_round_stats.parquet")
    if not path.exists():
        return
    x = pd.read_parquet(path, columns=["event_name", "event_date", "fight_id"])
    names = x[["event_name", "event_date", "fight_id"]].drop_duplicates()
    mask = names["event_name"].fillna("").str.lower().map(
        lambda s: any(p in s for p in PATHWAY_PATTERNS)
    )
    hit = names[mask].sort_values(["event_date", "event_name"])
    print("\nPATHWAY EVENTS ALREADY IN UFC ROUND FEED")
    if hit.empty:
        print("<none>")
        return
    by_event = hit.groupby(["event_name", "event_date"], as_index=False)["fight_id"].nunique()
    print(f"events={len(by_event)} fights={hit['fight_id'].nunique()}")
    print(by_event.tail(40).to_string(index=False))


def master_record_temporality() -> None:
    path = Path("data/master/ufc_master.parquet")
    if not path.exists():
        return
    cols = [
        "date", "fight_id",
        "r_id", "r_name", "r_wins", "r_losses", "r_draws",
        "b_id", "b_name", "b_wins", "b_losses", "b_draws",
    ]
    x = pd.read_parquet(path, columns=cols)
    pieces = []
    for side in ("r", "b"):
        z = x[["date", "fight_id", f"{side}_id", f"{side}_name", f"{side}_wins", f"{side}_losses", f"{side}_draws"]].copy()
        z.columns = ["date", "fight_id", "fighter_id", "fighter_name", "wins", "losses", "draws"]
        pieces.append(z)
    long = pd.concat(pieces, ignore_index=True).dropna(subset=["fighter_id"])
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["record"] = long.apply(
        lambda row: f"{row['wins']}-{row['losses']}-{row['draws']}",
        axis=1,
    )
    variation = long.groupby("fighter_id")["record"].nunique()
    multi = variation[variation > 1]
    print("\nUFC MASTER PROFILE-RECORD TEMPORALITY")
    print(f"fighters={variation.size} fighters_with_multiple_distinct_record_values={multi.size} ({multi.size / max(variation.size, 1):.1%})")
    print("If this fraction is low, r/b wins-losses-draws are likely current-profile fields and unsafe for historical cold-start validation.")
    samples = []
    for fid in multi.index[:10]:
        g = long[long["fighter_id"] == fid].sort_values("date")
        samples.append(g.iloc[0][["fighter_name", "date", "record"]].to_dict() | {"which": "earliest"})
        samples.append(g.iloc[-1][["fighter_name", "date", "record"]].to_dict() | {"which": "latest"})
    if samples:
        print(pd.DataFrame(samples).to_string(index=False))


def main() -> None:
    print("=" * 120)
    print("FSR V3 COLD-START DATA INVENTORY — READ ONLY")
    print("=" * 120)
    files = []
    for root in ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*.parquet")))
    for path in files:
        try:
            schema = pq.read_schema(path)
        except Exception as exc:
            print(f"SKIP {path}: {exc}")
            continue
        cols = list(schema.names)
        rel = relevant(cols)
        print(f"\n[{path}] rows={pq.ParquetFile(path).metadata.num_rows} cols={len(cols)}")
        print("relevant:", ", ".join(rel) if rel else "<none>")
        if path.as_posix().endswith("data/master/ufc_master.parquet"):
            print("ALL UFC MASTER COLUMNS:")
            print("\n".join(cols))
    pathway_inventory()
    master_record_temporality()


if __name__ == "__main__":
    main()
