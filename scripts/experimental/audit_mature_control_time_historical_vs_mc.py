"""Compare historical UFCStats CTRL time with current locked shadow MC control time.

Research-only audit. Uses the same aligned mature 2020+ FSR-32 cohort as the
full-cohort KO validation and the exact current locked shadow simulator config.

Historical CTRL is UFCStats control time and may include both clinch/cage and
ground control. Simulated control is the simulator's clinch_control_seconds +
ground_control_seconds. This audit intentionally compares the broad totals first
before decomposing the simulator by phase.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH
from pipeline.round_stats import build_round_fighter_wrestling as wrestle
from scripts.experimental import fsr_mature_2020plus_full_cohort_ko_validation_r3_d60_s0 as full

DEFAULT_PATHS = 10
DEFAULT_SEED = 20260811
ROUNDS = (1, 2, 3)


def _historical_control(cohort: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(ROUND_STATS_PATH)
    rounds = wrestle.standardize_round_stats(raw)
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    wanted = set(cohort["bout_id"].astype(str))
    rounds = rounds[rounds["fight_id"].isin(wanted)].copy()
    rounds = rounds[pd.to_numeric(rounds["round"], errors="coerce").isin(ROUNDS)].copy()
    rounds["round"] = pd.to_numeric(rounds["round"], errors="coerce").astype(int)
    rounds["control_seconds"] = pd.to_numeric(rounds["control_seconds"], errors="coerce").fillna(0.0)

    # Sum both fighters. UFCStats CTRL is fighter-specific; this gives the total
    # observed controlled time in the round, directly analogous to summing the
    # simulator's fighter control counters.
    return (
        rounds.groupby(["fight_id", "round"], as_index=False)["control_seconds"]
        .sum()
        .rename(columns={"fight_id": "bout_id", "control_seconds": "hist_control_seconds"})
    )


def _sim_round_control(sim) -> tuple[float, float, float]:
    total = float(sum(s.control_seconds for s in sim.stats))
    clinch = float(sum(s.clinch_control_seconds for s in sim.stats))
    ground = float(sum(s.ground_control_seconds for s in sim.stats))
    return total, clinch, ground


def main() -> None:
    p = argparse.ArgumentParser(description="Audit historical vs simulated mature-cohort control time")
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-bouts", type=int, default=None, help="optional quick-run limit")
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    full._configure_locked_candidate()
    cohort, pairs = full._build_full_cohort()
    if args.max_bouts is not None:
        cohort = cohort.head(args.max_bouts).reset_index(drop=True)

    hist = _historical_control(cohort)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | int | str]] = []

    for idx, bout in cohort.iterrows():
        bout_id = str(bout["bout_id"])
        pair = pairs[bout_id]
        red, blue = pair
        r_age = float(bout["r_age"]) if pd.notna(bout.get("r_age")) else None
        b_age = float(bout["b_age"]) if pd.notna(bout.get("b_age")) else None
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)

        totals = {r: {"reached": 0, "control": 0.0, "clinch": 0.0, "ground": 0.0} for r in ROUNDS}
        for seed in seeds:
            sims = {}
            paths = {}
            for r in ROUNDS:
                sim, path, _, _ = full._run_prefix(
                    red, blue, rounds=r, seed=int(seed), red_age=r_age, blue_age=b_age
                )
                sims[r] = sim
                paths[r] = path

            c1, cl1, gr1 = _sim_round_control(sims[1])
            totals[1]["reached"] += 1
            totals[1]["control"] += c1
            totals[1]["clinch"] += cl1
            totals[1]["ground"] += gr1

            if paths[1].finish is None:
                c2, cl2, gr2 = _sim_round_control(sims[2])
                totals[2]["reached"] += 1
                totals[2]["control"] += max(0.0, c2 - c1)
                totals[2]["clinch"] += max(0.0, cl2 - cl1)
                totals[2]["ground"] += max(0.0, gr2 - gr1)

            if paths[2].finish is None:
                c2, cl2, gr2 = _sim_round_control(sims[2])
                c3, cl3, gr3 = _sim_round_control(sims[3])
                totals[3]["reached"] += 1
                totals[3]["control"] += max(0.0, c3 - c2)
                totals[3]["clinch"] += max(0.0, cl3 - cl2)
                totals[3]["ground"] += max(0.0, gr3 - gr2)

        for r in ROUNDS:
            reached = totals[r]["reached"]
            if reached:
                rows.append({
                    "bout_id": bout_id,
                    "round": r,
                    "sim_reached_paths": reached,
                    "sim_control_seconds": totals[r]["control"] / reached,
                    "sim_clinch_control_seconds": totals[r]["clinch"] / reached,
                    "sim_ground_control_seconds": totals[r]["ground"] / reached,
                })

        if (idx + 1) % 100 == 0 or idx + 1 == len(cohort):
            print(f"bouts {idx + 1:,}/{len(cohort):,}", flush=True)

    sim = pd.DataFrame(rows)
    merged = hist.merge(sim, on=["bout_id", "round"], how="inner", validate="one_to_one")

    print("\n" + "=" * 112)
    print("MATURE 2020+ CONTROL-TIME AUDIT — HISTORICAL UFCSTATS CTRL VS CURRENT LOCKED SHADOW MC")
    print("=" * 112)
    print(f"cohort bouts requested: {len(cohort):,}")
    print(f"aligned historical/sim fight-rounds: {len(merged):,}")
    print(f"paths per bout: {args.paths:,}")
    print("Historical CTRL may include clinch/cage + ground control; simulator total control does too.")

    print("\nROUND MEANS — CONDITIONAL ON ROUND BEING OBSERVED/REACHED")
    print(" rnd   n_hist   hist_ctrl   sim_ctrl   error     sim_clinch   sim_ground   ground_share")
    for r in ROUNDS:
        g = merged[merged["round"] == r]
        if g.empty:
            continue
        h = float(g["hist_control_seconds"].mean())
        s = float(g["sim_control_seconds"].mean())
        cl = float(g["sim_clinch_control_seconds"].mean())
        gr = float(g["sim_ground_control_seconds"].mean())
        err = (s / h - 1.0) if h > 0 else np.nan
        share = gr / s if s > 0 else np.nan
        print(f" R{r:<1d} {len(g):8,d} {h:11.2f} {s:10.2f} {err:8.2%} {cl:13.2f} {gr:12.2f} {share:12.2%}")

    print("\nALL ALIGNED FIGHT-ROUNDS")
    h = float(merged["hist_control_seconds"].mean())
    s = float(merged["sim_control_seconds"].mean())
    cl = float(merged["sim_clinch_control_seconds"].mean())
    gr = float(merged["sim_ground_control_seconds"].mean())
    print(f"historical mean CTRL: {h:.2f}s")
    print(f"simulated mean CTRL:  {s:.2f}s ({(s / h - 1.0):+.2%} vs historical)" if h > 0 else f"simulated mean CTRL: {s:.2f}s")
    print(f"simulated clinch CTRL: {cl:.2f}s")
    print(f"simulated ground CTRL: {gr:.2f}s")
    print(f"simulated ground share of CTRL: {gr / s:.2%}" if s > 0 else "simulated ground share of CTRL: n/a")

    print("\nINTERPRETATION NOTE")
    print("A match in total CTRL does NOT prove phase realism. The simulator can match broad CTRL while allocating too much of it to GROUND and too little to CLINCH. That decomposition is the key diagnostic here.")
    print("Research-only audit; no simulator physics modified.")


if __name__ == "__main__":
    main()
