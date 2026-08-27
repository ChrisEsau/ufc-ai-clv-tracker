"""Research-only audit of modeled landed-strike exposure vs UFCStats sig-strike landings.

Runs the frozen Event Clock V2 calibration engine on the canonical calibration
cohort and compares its eligible landed strike rate to the historical UFCStats
significant-strike landed rate over the same fights. This establishes whether a
KO hazard fitted per historical significant-strike landing can be transferred
onto one modeled landed strike without an exposure-scale correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import (
    EVENT_CLOCK_V2_COHORT_MANIFEST_PATH,
    MASTER_PATH,
    ROUND_STATS_PATH,
)
from pipeline.simulation.event_clock_mc_v2.calibration.runner import run
from pipeline.simulation.event_clock_mc_v2.mechanics.config import KOKDArchitecture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths-per-fight", type=int, default=10)
    p.add_argument("--calibration-config", type=Path, default=Path("configs/event_clock_v2/calibration/default.yaml"))
    p.add_argument("--ledger", type=Path, default=Path("data/research/event2_ko_hazard_v2/exposure-ledger.json"))
    p.add_argument("--output", type=Path, default=Path("data/research/event2_ko_hazard_v2/exposure-summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    record = run(
        split="calibration",
        paths_per_fight=args.paths_per_fight,
        config_path=args.calibration_config,
        output=args.ledger,
        ko_kd_architecture=KOKDArchitecture.EMPIRICAL_EVENT2,
    )
    metrics = record["simulator_metrics"]
    fight_count = int(record["fight_count"])
    total_paths = fight_count * int(record["paths_per_fight"])
    sim_seconds = float(metrics["mean_fight_duration_seconds"]) * total_paths
    breakdown = metrics["phase_ko_kd_breakdown"]
    sim_by_phase = {
        phase: int(values.get("landed_eligible_strikes", 0))
        for phase, values in breakdown.items()
    }
    sim_landed = int(sum(sim_by_phase.values()))
    sim_rate = float(sim_landed * 900.0 / (2.0 * sim_seconds))

    manifest = pd.read_csv(EVENT_CLOCK_V2_COHORT_MANIFEST_PATH, dtype={"bout_id": str})
    ids = set(manifest.loc[manifest.cohort_split.eq("calibration"), "bout_id"].astype(str))
    master = pd.read_parquet(MASTER_PATH).copy()
    master["fight_id"] = master.fight_id.astype(str)
    master = master[master.fight_id.isin(ids)].drop_duplicates("fight_id")
    rounds = pd.read_parquet(ROUND_STATS_PATH).copy()
    rounds["fight_id"] = rounds.fight_id.astype(str)
    rounds = rounds[rounds.fight_id.isin(ids)]

    hist_seconds = float(pd.to_numeric(master.match_time_sec, errors="coerce").fillna(0).sum())
    hist_landed = float(pd.to_numeric(rounds.sig_str_landed, errors="coerce").fillna(0).sum())
    hist_rate = float(hist_landed * 900.0 / (2.0 * hist_seconds))

    phase_cols = {
        "standing": None,
        "clinch": "clinch_landed",
        "ground": "ground_landed",
    }
    hist_by_phase = {}
    for phase, col in phase_cols.items():
        if phase == "standing":
            vals = (
                pd.to_numeric(rounds.sig_str_landed, errors="coerce").fillna(0)
                - pd.to_numeric(rounds.clinch_landed, errors="coerce").fillna(0)
                - pd.to_numeric(rounds.ground_landed, errors="coerce").fillna(0)
            )
            hist_by_phase[phase] = float(vals.clip(lower=0).sum())
        else:
            hist_by_phase[phase] = float(pd.to_numeric(rounds[col], errors="coerce").fillna(0).sum())

    ratio = float(sim_rate / hist_rate) if hist_rate else None
    # Exact independent-event correction preserving cumulative hazard if one
    # modeled landing represents 1/ratio historical significant-strike landings.
    summary = {
        "study": "event2_modeled_landed_strike_exposure_vs_ufcstats_sig_landed",
        "production_changed": False,
        "cohort": "calibration",
        "fight_count": fight_count,
        "paths_per_fight": int(record["paths_per_fight"]),
        "historical": {
            "fight_count": int(len(master)),
            "fighter_seconds": 2.0 * hist_seconds,
            "sig_str_landed": hist_landed,
            "sig_str_landed_per_fighter_15": hist_rate,
            "landed_counts_by_phase": hist_by_phase,
        },
        "simulator": {
            "total_paths": total_paths,
            "fighter_seconds": 2.0 * sim_seconds,
            "eligible_landed_strikes": sim_landed,
            "eligible_landed_per_fighter_15": sim_rate,
            "landed_counts_by_phase": sim_by_phase,
            "standing_attempts_per_fighter_15": metrics["standing_attempts_per_fighter_15"],
            "clinch_strikes_per_fighter_15": metrics["clinch_strikes_per_fighter_15"],
            "ground_strikes_per_fighter_15": metrics["ground_strikes_per_fighter_15"],
        },
        "sim_to_historical_landed_rate_ratio": ratio,
        "interpretation": (
            "ratio near 1 supports direct transfer of a per-UFCStats-significant-landing hazard; "
            "material deviation implies intercept/exposure correction before simulator promotion"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EVENT2_STRIKE_EXPOSURE_AUDIT")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
