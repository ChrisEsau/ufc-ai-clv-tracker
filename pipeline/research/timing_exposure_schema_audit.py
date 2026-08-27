from __future__ import annotations

import json
import pandas as pd
import numpy as np

from pipeline.simulation.event_clock_mc_v2.diagnostics.stage8_structural_population import MASTER, ROUND_STATS, pick_col


def main():
    master = pd.read_parquet(MASTER).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    dc = pick_col(master, "date", "event_date")
    master["event_date"] = pd.to_datetime(master[dc], errors="coerce")

    rounds = pd.read_parquet(ROUND_STATS).copy()
    fc = pick_col(rounds, "fight_id", "bout_id")
    rounds[fc] = rounds[fc].astype(str)
    rc = pick_col(rounds, "round", "round_number", required=False)

    out = {
        "master_columns": list(master.columns),
        "round_columns": list(rounds.columns),
        "round_col": rc,
        "samples": [],
    }

    candidates = master.sort_values("event_date").copy()
    method_col = pick_col(candidates, "method", required=False)
    if method_col:
        candidates = candidates[candidates[method_col].astype(str).str.contains("Decision", case=False, na=False)]

    # Sample old, middle, and recent decisions with round-stat coverage.
    available = set(rounds[fc])
    candidates = candidates[candidates.fight_id.isin(available)]
    picks = []
    for lo, hi in [("2010-01-01","2016-01-01"),("2019-01-01","2024-01-01"),("2025-01-01","2027-01-01")]:
        s = candidates[(candidates.event_date >= lo) & (candidates.event_date < hi)].head(3)
        picks.extend(s.to_dict("records"))

    inspect_fields = [c for c in master.columns if any(k in c.lower() for k in ("round", "time", "duration", "elapsed"))]
    for row in picks:
        fid = str(row["fight_id"])
        fr = rounds[rounds[fc].eq(fid)].copy()
        item = {
            "fight_id": fid,
            "event_date": str(pd.Timestamp(row["event_date"]).date()),
            "r_name": row.get("r_name"),
            "b_name": row.get("b_name"),
            "method": row.get(method_col) if method_col else None,
            "master_timing_fields": {k: (None if pd.isna(row.get(k)) else row.get(k)) for k in inspect_fields},
            "round_rows": int(len(fr)),
            "round_values": [] if rc is None else sorted(pd.to_numeric(fr[rc], errors="coerce").dropna().astype(float).unique().tolist()),
        }
        out["samples"].append(item)

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
