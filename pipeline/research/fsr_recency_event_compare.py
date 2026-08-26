from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.research.fsr_recency_cohort_shadow import build_variant, actual_red_win
from pipeline.simulation.event_clock_mc_v2.calibration.runner import run
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture

EVENT_NAME = os.environ.get("EVENT_NAME", "UFC Fight Night: Muhammad vs. Bonfim")
PATHS = int(os.environ.get("PATHS", "1000"))
OUTDIR = Path("data/diagnostics/fsr_recency_event")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    canonical["event_date"] = pd.to_datetime(canonical["event_date"]).dt.normalize()
    canonical["fight_id"] = canonical["fight_id"].astype(str)
    canonical["fighter_id"] = canonical["fighter_id"].astype(str)

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    selected = master[master["event_name"].astype(str).eq(EVENT_NAME)].copy()
    if selected.empty:
        names = sorted(master.loc[master["event_name"].astype(str).str.contains("Muhammad|Bonfim", case=False, na=False), "event_name"].astype(str).unique())
        raise ValueError(f"event not found: {EVENT_NAME!r}; nearby={names}")
    selected = selected[selected["total_rounds"].isin([3,5])].copy()
    selected["actual_red_win"] = selected.apply(actual_red_win, axis=1)
    selected = selected[selected["actual_red_win"].notna()].copy()

    variants = {
        "canonical": canonical,
        "last3_all_v3": build_variant(canonical, "last3"),
    }
    results = {}
    for name, snapshots in variants.items():
        snapshots.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
        out = OUTDIR / f"{name}.json"
        rec = run(
            split="calibration",
            paths_per_fight=PATHS,
            config_path=Path("configs/event_clock_v2/calibration/default.yaml"),
            output=out,
            ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
            event_name=EVENT_NAME,
        )
        probs = rec["simulator_metrics"]["fight_probabilities"]
        rows = []
        for r in selected.itertuples(index=False):
            p = probs[str(r.fight_id)]
            rows.append({
                "fight_id": str(r.fight_id),
                "red": str(r.r_name),
                "blue": str(r.b_name),
                "actual_red_win": int(r.actual_red_win),
                "red_moneyline": float(p["red_moneyline"]),
                "blue_moneyline": float(p["blue_moneyline"]),
                "red_ko": float(p["red_ko_tko"]),
                "red_sub": float(p["red_submission"]),
                "red_dec": float(p["red_decision"]),
                "blue_ko": float(p["blue_ko_tko"]),
                "blue_sub": float(p["blue_submission"]),
                "blue_dec": float(p["blue_decision"]),
            })
        results[name] = rows

    canonical.to_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH, index=False)
    summary = {
        "event_name": EVENT_NAME,
        "paths_per_fight": PATHS,
        "design": "canonical vs last-3 all V3-native fight-relevant traits; mechanics/seeds fixed",
        "results": results,
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
