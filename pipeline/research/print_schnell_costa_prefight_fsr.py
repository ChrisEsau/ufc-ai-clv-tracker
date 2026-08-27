from __future__ import annotations
import json
import pandas as pd
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
import pipeline.research.fsr_recency_cohort_shadow as recency

FIGHT_ID = "5d2eedd05081ed23"
EWM_DECAY = 0.50


def main():
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)
    recency.EWM_DECAY = EWM_DECAY
    recency.EWM_CANONICAL_BLEND = 0.0
    ewm = recency.build_variant(canonical, "ewm")
    rows = ewm.loc[ewm["fight_id"].eq(FIGHT_ID)].copy()
    if len(rows) != 2:
        raise RuntimeError(f"expected 2 rows for {FIGHT_ID}, got {len(rows)}")
    keep = [c for c in rows.columns if c not in {"event_date"}]
    out = rows[keep].to_dict("records")
    print("SCHNELL_COSTA_PREFIGHT_FSR_EWM05")
    print(json.dumps(out, indent=2, sort_keys=True, default=str))

if __name__ == "__main__":
    main()
