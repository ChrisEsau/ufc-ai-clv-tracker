"""Research-only 200-bout MC round exposure audit.

Purpose
-------
Measure simulated significant-strike exposure by round using the current
KD-base -8.70 / collapse scale 2.0 / curvature 20.0 configuration, with no
simulator constants changed. Compare the simulation directly with the exact
same 200 mature 2020+ historical bouts when round stats are available.

Outputs per round:
- fight-round exposures
- combined significant strikes landed
- mean significant strikes landed per fight-round
- surviving knockdowns
- mean knockdowns per fight-round
- KO/TKO finishes
- KO/TKO rate per fight-round
- KD per 100 landed significant strikes
- KO per 1,000 landed significant strikes

No production simulator or FSR artifact is modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as current

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = current.DEFAULT_SEED
ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
OUTPUT_PATH = Path("data/experimental/mc_round_exposure_200.csv")
HIST_OUTPUT_PATH = Path("data/experimental/historical_same200_round_exposure.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--round-stats-path", type=Path, default=ROUND_STATS_PATH)
    return p.parse_args()


def _combined_sig_landed(sim: current.AuditSim) -> int:
    return int(sim.stats[0].sig_landed) + int(sim.stats[1].sig_landed)


def _combined_kd(sim: current.AuditSim) -> int:
    return int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)


def _finish_round(path: Any) -> int:
    if path.finish is None:
        return 0
    return int(getattr(path.finish, "round", 0) or 0)


def _run_prefix(red, blue, *, rounds: int, seed: int, red_age, blue_age):
    sim = current.AuditSim(
        red,
        blue,
        rounds=rounds,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )
    path = sim.run()
    return sim, path, _combined_sig_landed(sim), _combined_kd(sim), _finish_round(path)


def _rate(numer: float, denom: float, scale: float = 1.0) -> float:
    return float(numer / denom * scale) if denom else float("nan")


def _summarize_sim(round_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rnd, g in round_rows.groupby("round", sort=True):
        sig = int(g["sig_str_landed"].sum())
        kd = int(g["knockdowns"].sum())
        ko = int(g["ko_tko"].sum())
        n = len(g)
        rows.append(
            {
                "source": "simulated",
                "round": int(rnd),
                "fight_rounds": n,
                "sig_str_landed": sig,
                "knockdowns": kd,
                "ko_tko_finishes": ko,
                "mean_sig_str_landed": sig / n,
                "mean_kd_per_round": kd / n,
                "p_ko_tko": ko / n,
                "kd_per_100_sig_landed": _rate(kd, sig, 100.0),
                "ko_per_1000_sig_landed": _rate(ko, sig, 1000.0),
            }
        )
    return pd.DataFrame(rows)


def _historical_same_bouts(cohort: pd.DataFrame, round_stats_path: Path) -> pd.DataFrame:
    if not round_stats_path.exists():
        raise FileNotFoundError(f"Round stats dataset not found: {round_stats_path}")

    rs = pd.read_parquet(round_stats_path).copy()
    required = {"fight_id", "round", "sig_str_landed", "kd"}
    missing = sorted(required - set(rs.columns))
    if missing:
        raise ValueError(f"Round stats missing required columns: {missing}")

    rs["fight_id"] = rs["fight_id"].astype(str)
    rs["round"] = pd.to_numeric(rs["round"], errors="coerce")
    rs["sig_str_landed"] = pd.to_numeric(rs["sig_str_landed"], errors="coerce").fillna(0)
    rs["kd"] = pd.to_numeric(rs["kd"], errors="coerce").fillna(0)

    selected = cohort.copy()
    selected["bout_id"] = selected["bout_id"].astype(str)
    wanted = set(selected["bout_id"])
    rs = rs[rs["fight_id"].isin(wanted) & rs["round"].isin([1, 2, 3])].copy()

    fight_round = (
        rs.groupby(["fight_id", "round"], as_index=False)
        .agg(sig_str_landed=("sig_str_landed", "sum"), knockdowns=("kd", "sum"))
    )

    outcomes = selected[["bout_id", "actual_ko_tko", "actual_finish_round"]].rename(
        columns={"bout_id": "fight_id"}
    )
    fight_round = fight_round.merge(outcomes, on="fight_id", how="left", validate="many_to_one")
    fight_round["ko_tko"] = (
        fight_round["actual_ko_tko"].eq(1)
        & pd.to_numeric(fight_round["actual_finish_round"], errors="coerce").eq(fight_round["round"])
    ).astype(int)

    rows: list[dict[str, Any]] = []
    for rnd, g in fight_round.groupby("round", sort=True):
        sig = int(g["sig_str_landed"].sum())
        kd = int(g["knockdowns"].sum())
        ko = int(g["ko_tko"].sum())
        n = len(g)
        rows.append(
            {
                "source": "historical_same_bouts",
                "round": int(rnd),
                "fight_rounds": n,
                "sig_str_landed": sig,
                "knockdowns": kd,
                "ko_tko_finishes": ko,
                "mean_sig_str_landed": sig / n,
                "mean_kd_per_round": kd / n,
                "p_ko_tko": ko / n,
                "kd_per_100_sig_landed": _rate(kd, sig, 100.0),
                "ko_per_1000_sig_landed": _rate(ko, sig, 1000.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bouts <= 0 or args.paths <= 0:
        raise ValueError("--bouts and --paths must be positive")

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    round_rows: list[dict[str, Any]] = []
    completed = 0

    print("\n" + "=" * 150)
    print("CURRENT MC ROUND EXPOSURE AUDIT — 200 BOUTS x 10 PATHS")
    print("=" * 150)
    print(f"contact sigma={current.CONTACT_SIGMA:.2f}; power scale={current.POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={current.KD_BASE_LOGIT:.2f}; shock={current.KD_SHOCK_COEFFICIENT:.0f}; depletion={current.KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={current.COLLAPSE_SCALE:.1f}; curvature={current.COLLAPSE_CURVATURE:.1f}")
    print("No constants changed; significant strikes are read from simulator FighterStats.sig_landed.")

    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for path_idx, seed in enumerate(seed_matrix[bout_idx]):
            sim1, path1, sig1, kd1, fin1 = _run_prefix(
                red, blue, rounds=1, seed=int(seed), red_age=r_age, blue_age=b_age
            )
            sim2, path2, sig2, kd2, fin2 = _run_prefix(
                red, blue, rounds=2, seed=int(seed), red_age=r_age, blue_age=b_age
            )
            sim3, path3, sig3, kd3, fin3 = _run_prefix(
                red, blue, rounds=3, seed=int(seed), red_age=r_age, blue_age=b_age
            )

            # Prefix differencing gives round-specific exposure while preserving the
            # established deterministic-seed comparison convention used by the
            # existing 200-bout calibration scripts.
            round_rows.append({
                "bout_id": str(bout["bout_id"]), "path": path_idx, "round": 1,
                "sig_str_landed": sig1, "knockdowns": kd1,
                "ko_tko": int(path3.finish is not None and fin3 == 1),
            })

            # Only include a later-round exposure if that prefix actually reached
            # the round. A round-1 finish produces no R2/R3 fight-round exposure.
            if path1.finish is None:
                round_rows.append({
                    "bout_id": str(bout["bout_id"]), "path": path_idx, "round": 2,
                    "sig_str_landed": max(0, sig2 - sig1),
                    "knockdowns": max(0, kd2 - kd1),
                    "ko_tko": int(path3.finish is not None and fin3 == 2),
                })
            if path2.finish is None:
                round_rows.append({
                    "bout_id": str(bout["bout_id"]), "path": path_idx, "round": 3,
                    "sig_str_landed": max(0, sig3 - sig2),
                    "knockdowns": max(0, kd3 - kd2),
                    "ko_tko": int(path3.finish is not None and fin3 == 3),
                })

        completed += args.paths
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}", flush=True)

    sim_rounds = pd.DataFrame(round_rows)
    sim_summary = _summarize_sim(sim_rounds)
    hist_summary = _historical_same_bouts(cohort, args.round_stats_path)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sim_summary.to_csv(OUTPUT_PATH, index=False)
    hist_summary.to_csv(HIST_OUTPUT_PATH, index=False)

    display_cols = [
        "round", "fight_rounds", "sig_str_landed", "mean_sig_str_landed",
        "knockdowns", "mean_kd_per_round", "ko_tko_finishes", "p_ko_tko",
        "kd_per_100_sig_landed", "ko_per_1000_sig_landed",
    ]

    print("\nSIMULATED — ROUND TOTALS")
    print(sim_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nHISTORICAL — EXACT SAME 200 BOUTS")
    print(hist_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    merged = sim_summary.merge(hist_summary, on="round", suffixes=("_sim", "_hist"))
    compare_rows = []
    for _, r in merged.iterrows():
        compare_rows.append({
            "round": int(r["round"]),
            "sim_mean_sig": r["mean_sig_str_landed_sim"],
            "hist_mean_sig": r["mean_sig_str_landed_hist"],
            "sig_ratio_sim_hist": _rate(r["mean_sig_str_landed_sim"], r["mean_sig_str_landed_hist"]),
            "sim_kd_per100": r["kd_per_100_sig_landed_sim"],
            "hist_kd_per100": r["kd_per_100_sig_landed_hist"],
            "sim_ko_rate": r["p_ko_tko_sim"],
            "hist_ko_rate": r["p_ko_tko_hist"],
        })
    compare = pd.DataFrame(compare_rows)
    print("\nDIRECT COMPARISON")
    print(compare.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nSaved simulated summary: {OUTPUT_PATH}")
    print(f"Saved historical summary: {HIST_OUTPUT_PATH}")
    print("Research only: no simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
