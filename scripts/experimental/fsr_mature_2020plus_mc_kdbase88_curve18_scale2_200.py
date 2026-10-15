"""Single research audit: KD base -8.80, collapse scale 2.0, curvature 18.0.

Everything matches the existing -8.80 / curve-20 comparison run except collapse curvature.
This script also freshly recalculates exact same-200 historical round totals from the
canonical round-stats parquet and prints fresh simulated round totals from 200 x 10 paths.
No production simulator or FSR artifact is modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as run87
from scripts.experimental import historical_sigstr_kd_ko_exposure_2020plus_mature as hist
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
KD_BASE_LOGIT = -8.80
COLLAPSE_SCALE = 2.0
COLLAPSE_CURVATURE = 18.0
OUTPUT_PATH = Path("data/experimental/kdbase88_curve18_scale2_200.csv")
SIM_ROUND_OUTPUT = Path("data/experimental/kdbase88_curve18_scale2_200_round_totals.csv")
HIST_ROUND_OUTPUT = Path("data/experimental/historical_same200_round_totals_recalc.csv")


def _sig_total(sim) -> int:
    return int(sim.stats[0].sig_landed) + int(sim.stats[1].sig_landed)


def _summarize_sim(rows: list[dict[str, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    out = []
    for rnd, g in frame.groupby("round", sort=True):
        sig = int(g["sig_str_landed"].sum())
        kd = int(g["knockdowns"].sum())
        ko = int(g["ko_tko"].sum())
        n = len(g)
        out.append({
            "round": int(rnd),
            "fight_rounds": n,
            "sig_str_landed": sig,
            "mean_sig_str_landed": sig / n if n else np.nan,
            "knockdowns": kd,
            "mean_kd_per_round": kd / n if n else np.nan,
            "ko_tko_finishes": ko,
            "p_ko_tko": ko / n if n else np.nan,
            "kd_per_100_sig_landed": 100.0 * kd / sig if sig else np.nan,
            "ko_per_1000_sig_landed": 1000.0 * ko / sig if sig else np.nan,
        })
    return pd.DataFrame(out)


def _recalc_historical_same200(cohort: pd.DataFrame) -> pd.DataFrame:
    round_stats = hist._load_round_stats(hist.ROUND_STATS_PATH)
    rounds = hist._build_fight_rounds(round_stats, cohort)
    rounds = rounds[rounds["round"].isin([1, 2, 3])].copy()
    rows = []
    for rnd, g in rounds.groupby("round", sort=True):
        sig = int(g["sig_str_landed"].sum())
        kd = int(g["kd"].sum())
        ko = int(g["ko_tko"].sum())
        n = len(g)
        rows.append({
            "round": int(rnd),
            "fight_rounds": n,
            "sig_str_landed": sig,
            "mean_sig_str_landed": sig / n if n else np.nan,
            "knockdowns": kd,
            "mean_kd_per_round": kd / n if n else np.nan,
            "ko_tko_finishes": ko,
            "p_ko_tko": ko / n if n else np.nan,
            "kd_per_100_sig_landed": 100.0 * kd / sig if sig else np.nan,
            "ko_per_1000_sig_landed": 1000.0 * ko / sig if sig else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    run87.KD_BASE_LOGIT = KD_BASE_LOGIT
    run87.COLLAPSE_CURVATURE = COLLAPSE_CURVATURE
    run87.COLLAPSE = collapse_mod.CollapseCandidate(
        f"scale{COLLAPSE_SCALE:.1f}_curve{COLLAPSE_CURVATURE:.1f}",
        COLLAPSE_SCALE,
        COLLAPSE_CURVATURE,
    )
    run87.OUTPUT_PATH = OUTPUT_PATH

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(DEFAULT_BOUTS).reset_index(drop=True)
    total_paths = len(cohort) * DEFAULT_PATHS

    historical = _recalc_historical_same200(cohort)

    rng = np.random.default_rng(DEFAULT_SEED)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), DEFAULT_PATHS), dtype=np.int64)

    sim_round_rows: list[dict[str, int]] = []
    terminal_ko = 0
    direct_ko = 0
    completed = 0

    print("\n" + "=" * 150)
    print(
        f"KD BASE {KD_BASE_LOGIT:.2f} + COLLAPSE CURVE {COLLAPSE_CURVATURE:g} — "
        f"SCALE {COLLAPSE_SCALE:.1f} — {DEFAULT_BOUTS} BOUTS x {DEFAULT_PATHS} PATHS"
    )
    print("=" * 150)
    print(f"contact sigma={run87.CONTACT_SIGMA:.2f}; power scale={run87.POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={KD_BASE_LOGIT:.2f}; shock={run87.KD_SHOCK_COEFFICIENT:.0f}; depletion={run87.KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print("terminal collapse = KO/TKO only; surviving knockdown = KD")
    print("Historical totals are freshly recalculated from the exact same first 200 aligned bouts.")

    for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
        red, blue = pairs[str(bout["bout_id"])]
        r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
        b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

        for seed in seed_matrix[bout_idx]:
            sim1, path1, kd1, fr1 = run87._run_prefix(
                red, blue, rounds=1, seed=int(seed), red_age=r_age, blue_age=b_age
            )
            sim2, path2, kd2, fr2 = run87._run_prefix(
                red, blue, rounds=2, seed=int(seed), red_age=r_age, blue_age=b_age
            )
            sim3, path3, kd3, fr3 = run87._run_prefix(
                red, blue, rounds=3, seed=int(seed), red_age=r_age, blue_age=b_age
            )

            sig1 = _sig_total(sim1)
            sig2 = _sig_total(sim2)
            sig3 = _sig_total(sim3)

            sim_round_rows.append({
                "round": 1,
                "sig_str_landed": sig1,
                "knockdowns": kd1,
                "ko_tko": int(path1.finish is not None and fr1 == 1),
            })
            if path1.finish is None:
                sim_round_rows.append({
                    "round": 2,
                    "sig_str_landed": max(0, sig2 - sig1),
                    "knockdowns": max(0, kd2 - kd1),
                    "ko_tko": int(path2.finish is not None and fr2 == 2),
                })
            if path2.finish is None:
                sim_round_rows.append({
                    "round": 3,
                    "sig_str_landed": max(0, sig3 - sig2),
                    "knockdowns": max(0, kd3 - kd2),
                    "ko_tko": int(path3.finish is not None and fr3 == 3),
                })

            if path3.finish is not None:
                terminal_ko += int(sim3.terminal_collapse_finishes > 0)
                direct_ko += int(sim3.direct_strike_finishes > 0)

        completed += DEFAULT_PATHS
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"paths {completed:,}/{total_paths:,}", flush=True)

    simulated = _summarize_sim(sim_round_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    simulated.to_csv(SIM_ROUND_OUTPUT, index=False)
    historical.to_csv(HIST_ROUND_OUTPUT, index=False)

    r1 = simulated.loc[simulated["round"].eq(1)].iloc[0]
    h1 = historical.loc[historical["round"].eq(1)].iloc[0]
    overall = pd.DataFrame([{
        "kd_base_logit": KD_BASE_LOGIT,
        "kd_shock": run87.KD_SHOCK_COEFFICIENT,
        "collapse_scale": COLLAPSE_SCALE,
        "curvature": COLLAPSE_CURVATURE,
        "r1_sig_mean": r1["mean_sig_str_landed"],
        "r1_kd_mean": r1["mean_kd_per_round"],
        "r1_ko": r1["p_ko_tko"],
        "historical_r1_sig_mean_exact200": h1["mean_sig_str_landed"],
        "historical_r1_kd_mean_exact200": h1["mean_kd_per_round"],
        "historical_r1_ko_exact200": h1["p_ko_tko"],
        "terminal_collapse_ko_paths": terminal_ko / total_paths,
        "direct_strike_ko_paths": direct_ko / total_paths,
    }])
    overall.to_csv(OUTPUT_PATH, index=False)

    print("\nSIMULATED — ROUND TOTALS")
    print(simulated.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nHISTORICAL — EXACT SAME 200 BOUTS — FRESH RECALCULATION")
    print(historical.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    comparison = simulated.merge(historical, on="round", suffixes=("_sim", "_hist"))
    display = pd.DataFrame({
        "round": comparison["round"],
        "sim_mean_sig": comparison["mean_sig_str_landed_sim"],
        "hist_mean_sig": comparison["mean_sig_str_landed_hist"],
        "sim_kd_mean": comparison["mean_kd_per_round_sim"],
        "hist_kd_mean": comparison["mean_kd_per_round_hist"],
        "sim_ko_rate": comparison["p_ko_tko_sim"],
        "hist_ko_rate": comparison["p_ko_tko_hist"],
    })
    print("\nDIRECT COMPARISON")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nterminal-collapse KO paths={terminal_ko / total_paths:.2%}; direct-strike KO paths={direct_ko / total_paths:.2%}")
    print(f"Saved overall: {OUTPUT_PATH}")
    print(f"Saved simulated round totals: {SIM_ROUND_OUTPUT}")
    print(f"Saved historical round totals: {HIST_ROUND_OUTPUT}")
    print("Research only: no simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
