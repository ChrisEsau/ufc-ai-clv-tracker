"""Research-only KD-collapse SCALE sweep with curvature fixed at zero.

Goal
----
Tune surviving knockdown frequency first, before using curvature to selectively
turn the highest-shock KDs into KO/TKO finishes.

Fixed architecture
------------------
- FSR-32 fresh striking_power active through rolling stamina.
- Contact quality sigma = 0.80.
- Power magnitude scale = 75.
- KD base logit = -9.15.
- KD shock coefficient = 100.
- KD depletion coefficient = 0.
- Collapse curvature = 0.0.
- Terminal collapse = KO/TKO only; surviving collapse = KD.

200 bouts x 10 paths. CSV is rewritten after every candidate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_collapse_curvature_sweep_200 as curv
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
CURVATURE = 0.0
SCALES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
OUTPUT_PATH = Path("data/experimental/collapse_scale_sweep_curve0_200.csv")

HIST_R1_KD_MEAN = 0.2281
HIST_TOTAL_KD_MEAN = 0.4364
HIST_ANY_KD = 0.3578
HIST_R1_KO = 0.1406
HIST_TOTAL_KO = 0.3144


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    print("\n" + "=" * 146)
    print("KD COLLAPSE SCALE SWEEP — CURVATURE FIXED AT 0.0 — 200 BOUTS x 10 PATHS")
    print("=" * 146)
    print(f"contact sigma={curv.CONTACT_SIGMA:.2f}; power scale={curv.POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={curv.KD_BASE_LOGIT:.2f}; shock={curv.KD_SHOCK_COEFFICIENT:.0f}; depletion={curv.KD_DEPLETION_COEFFICIENT:.2f}")
    print("terminal collapse = KO/TKO only; surviving knockdown = KD")
    print(f"CSV: {OUTPUT_PATH}")

    for scale in SCALES:
        name = f"scale{scale:.1f}_curve0.0"
        collapse = collapse_mod.CollapseCandidate(name, scale, CURVATURE)

        r1_kd = r2_kd = r3_kd = 0
        any_kd = 0
        r1_ko = r2_ko = r3_ko = 0
        ko_total = 0
        ko_round_sum = 0
        terminal_ko = 0
        direct_ko = 0
        completed = 0

        for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
            red, blue = pairs[str(bout["bout_id"])]
            r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
            b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

            for seed in seed_matrix[bout_idx]:
                _, _, kd1, _ = curv._run_prefix(
                    red, blue, rounds=1, seed=int(seed), collapse=collapse,
                    red_age=r_age, blue_age=b_age,
                )
                _, _, kd2, _ = curv._run_prefix(
                    red, blue, rounds=2, seed=int(seed), collapse=collapse,
                    red_age=r_age, blue_age=b_age,
                )
                sim3, path3, kd3, finish_round = curv._run_prefix(
                    red, blue, rounds=3, seed=int(seed), collapse=collapse,
                    red_age=r_age, blue_age=b_age,
                )

                r1_kd += kd1
                r2_kd += max(0, kd2 - kd1)
                r3_kd += max(0, kd3 - kd2)
                any_kd += int(kd3 > 0)

                if path3.finish is not None:
                    ko_total += 1
                    ko_round_sum += finish_round
                    r1_ko += int(finish_round == 1)
                    r2_ko += int(finish_round == 2)
                    r3_ko += int(finish_round == 3)
                    terminal_ko += int(sim3.terminal_collapse_finishes > 0)
                    direct_ko += int(sim3.direct_strike_finishes > 0)

            completed += args.paths
            if completed % 500 == 0 or bout_idx + 1 == len(cohort):
                print(f"[{name}] paths {completed:,}/{total_paths:,}", flush=True)

        total_kd = r1_kd + r2_kd + r3_kd
        row: dict[str, float | str] = {
            "candidate": name,
            "collapse_scale": scale,
            "curvature": CURVATURE,
            "r1_kd_mean": r1_kd / total_paths,
            "r2_kd_mean": r2_kd / total_paths,
            "r3_kd_mean": r3_kd / total_paths,
            "total_kd_mean": total_kd / total_paths,
            "any_kd": any_kd / total_paths,
            "r1_ko": r1_ko / total_paths,
            "r2_ko": r2_ko / total_paths,
            "r3_ko": r3_ko / total_paths,
            "total_ko": ko_total / total_paths,
            "mean_ko_round": ko_round_sum / ko_total if ko_total else float("nan"),
            "terminal_collapse_ko": terminal_ko / total_paths,
            "direct_strike_ko": direct_ko / total_paths,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
        print(
            f"  -> KD R1/R2/R3={row['r1_kd_mean']:.4f}/{row['r2_kd_mean']:.4f}/{row['r3_kd_mean']:.4f}; "
            f"KO R1/R2/R3={row['r1_ko']:.2%}/{row['r2_ko']:.2%}/{row['r3_ko']:.2%}; "
            f"total KD={row['total_kd_mean']:.4f}; any KD={row['any_kd']:.2%}; total KO={row['total_ko']:.2%}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    out["kd_abs_error"] = (out["total_kd_mean"] - HIST_TOTAL_KD_MEAN).abs()
    out["r1_kd_abs_error"] = (out["r1_kd_mean"] - HIST_R1_KD_MEAN).abs()
    out["kd_score"] = out["kd_abs_error"] + out["r1_kd_abs_error"]
    out = out.sort_values(["kd_score", "collapse_scale"]).reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nHISTORICAL REFERENCES")
    print(f"R1 KD mean={HIST_R1_KD_MEAN:.4f}; total KD mean={HIST_TOTAL_KD_MEAN:.4f}; any KD={HIST_ANY_KD:.2%}")
    print(f"R1 KO={HIST_R1_KO:.2%}; total KO={HIST_TOTAL_KO:.2%}")

    print("\nRANKED BY KD FIT")
    print(
        f"{'candidate':>20} {'R1KD':>8} {'R2KD':>8} {'R3KD':>8} {'KDtot':>8} {'KDany':>8} "
        f"{'R1KO':>8} {'R2KO':>8} {'R3KO':>8} {'KOtot':>8} {'termKO':>8} {'direct':>8}"
    )
    for _, row in out.iterrows():
        print(
            f"{row['candidate']:>20} {row['r1_kd_mean']:8.4f} {row['r2_kd_mean']:8.4f} {row['r3_kd_mean']:8.4f} "
            f"{row['total_kd_mean']:8.4f} {row['any_kd']:8.2%} {row['r1_ko']:8.2%} {row['r2_ko']:8.2%} "
            f"{row['r3_ko']:8.2%} {row['total_ko']:8.2%} {row['terminal_collapse_ko']:8.2%} {row['direct_strike_ko']:8.2%}"
        )

    print(f"\nSaved CSV: {OUTPUT_PATH}")
    print("Research only: no production simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
