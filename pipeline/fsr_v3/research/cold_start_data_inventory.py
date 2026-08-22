"""Measurement-only inventory for FSR V3 cold-start inputs.

Scans repository parquet schemas without modifying data and reports fields that
could support leakage-safe pre-UFC / early-career priors.
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

ROOTS = [Path("data/master"), Path("data/fight_details"), Path("data/features"), Path("data/research")]
KEYWORDS = (
    "fighter", "opponent", "date", "dob", "age", "height", "reach", "weight",
    "record", "win", "loss", "draw", "method", "promotion", "org", "event",
    "amateur", "pro", "wrest", "grap", "school", "college", "stance",
    "ko", "sub", "decision", "round", "time",
)


def relevant(columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        low = c.lower()
        if any(k in low for k in KEYWORDS):
            out.append(c)
    return out


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


if __name__ == "__main__":
    main()
