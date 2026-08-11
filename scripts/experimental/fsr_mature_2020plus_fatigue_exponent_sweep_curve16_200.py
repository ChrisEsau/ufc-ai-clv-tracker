"""Research-only fatigue exponent sweep for the locked curve-16 candidate.

Sweeps FATIGUE_CURVE_EXPONENT over 2.0, 1.5, and 1.0 while holding fixed:
- first 200 aligned mature 2020+ bouts x 10 paths
- contact sigma 0.80
- power magnitude scale 75
- base damage at power 50 = 1.18
- KD base -8.80
- KD shock 100
- KD depletion 0
- collapse scale 2.0
- collapse curvature 16.0
- max fatigue rating penalty 45
- resilience scale 80
- global stamina recovery 40% of missing
- all action/phase stamina costs

Exponent 1.0 is linear in missing stamina before the fighter-specific
performance-resilience multiplier is applied.

No production simulator or FSR artifact is modified.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_kdbase87_curve20_scale2_200 as run87
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as v31

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810
EXPONENTS = (2.0, 1.5, 1.0)

KD_BASE_LOGIT = -8.80
COLLAPSE_SCALE = 2.0
COLLAPSE_CURVATURE = 16.0

OUTPUT_PATH = Path("data/experimental/fatigue_exponent_sweep_curve16_200.csv")


def _configure_locked_candidate() -> None:
    run87.KD_BASE_LOGIT = KD_BASE_LOGIT
    run87.COLLAPSE_CURVATURE = COLLAPSE_CURVATURE
    run87.COLLAPSE = collapse_mod.CollapseCandidate(
        f"scale{COLLAPSE_SCALE:.1f}_curve{COLLAPSE_CURVATURE:.1f}",
        COLLAPSE_SCALE,
        COLLAPSE_CURVATURE,
    )


def _checkpoint_mean(sim: run87.AuditSim, round_no: int, metric: str) -> float:
    events = [
        e for e in sim.effective_fsr_events
        if int(e["round"]) == round_no and int(e["segment"]) == 1
    ]
    if not events:
        return float("nan")
    return float(np.mean([float(e[metric]) for e in events]))


def _run_exponent(
    exponent: float,
    cohort: pd.DataFrame,
    pairs: dict,
    seed_matrix: np.ndarray,
) -> dict[str, float]:
    # V3.1 fatigue_penalty reads this module-level constant at runtime.
    v31.FATIGUE_CURVE_EXPONENT = float(exponent)

    total_paths = len(cohort) * seed_matrix.shape[1]
    reached = {1: 0, 2: 0, 3: 0}
    ko = {1: 0, 2: 0, 3: 0}
    kd = {1: 0, 2: 0, 3: 0}
    sig = {1: 0, 2: 0, 3: 0}

    start_r2_stamina: list[float] = []
    start_r2_penalty: list[float] = []
    start_r2_power: list[float] = []
    start_r3_stamina: list[float] = []
    start_r3_penalty: list[float] = []
    start_r3_power: list[float] = []

    completed = 0
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

            sig1 = int(sim1.stats[0].sig_landed) + int(sim1.stats[1].sig_landed)
            sig2 = int(sim2.stats[0].sig_landed) + int(sim2.stats[1].sig_landed)
            sig3 = int(sim3.stats[0].sig_landed) + int(sim3.stats[1].sig_landed)

            reached[1] += 1
            sig[1] += sig1
            kd[1] += kd1
            ko[1] += int(path1.finish is not None and fr1 == 1)

            if path1.finish is None:
                reached[2] += 1
                sig[2] += max(0, sig2 - sig1)
                kd[2] += max(0, kd2 - kd1)
                ko[2] += int(path2.finish is not None and fr2 == 2)

            if path2.finish is None:
                reached[3] += 1
                sig[3] += max(0, sig3 - sig2)
                kd[3] += max(0, kd3 - kd2)
                ko[3] += int(path3.finish is not None and fr3 == 3)

            # Use the single 3-round path for checkpoint state. This does not
            # add RNG calls; it only reads events already recorded by the sim.
            for round_no, stamina_store, penalty_store, power_store in (
                (2, start_r2_stamina, start_r2_penalty, start_r2_power),
                (3, start_r3_stamina, start_r3_penalty, start_r3_power),
            ):
                events = [
                    e for e in sim3.effective_fsr_events
                    if int(e["round"]) == round_no and int(e["segment"]) == 1
                ]
                for e in events:
                    stamina_store.append(float(e["stamina_fraction"]))
                    penalty_store.append(float(e["fatigue_penalty"]))
                    power_store.append(float(e["effective_striking_power"]))

        completed += seed_matrix.shape[1]
        if completed % 500 == 0 or bout_idx + 1 == len(cohort):
            print(f"  exponent {exponent:.1f}: paths {completed:,}/{total_paths:,}", flush=True)

    row: dict[str, float] = {
        "fatigue_exponent": float(exponent),
        "max_fatigue_penalty": float(v31.MAX_FATIGUE_RATING_PENALTY),
        "r1_rounds": float(reached[1]),
        "r2_rounds": float(reached[2]),
        "r3_rounds": float(reached[3]),
        "r1_sig_mean": sig[1] / reached[1],
        "r2_sig_mean": sig[2] / reached[2] if reached[2] else float("nan"),
        "r3_sig_mean": sig[3] / reached[3] if reached[3] else float("nan"),
        "r1_kd_mean": kd[1] / reached[1],
        "r2_kd_mean": kd[2] / reached[2] if reached[2] else float("nan"),
        "r3_kd_mean": kd[3] / reached[3] if reached[3] else float("nan"),
        "r1_ko_rate": ko[1] / reached[1],
        "r2_ko_rate": ko[2] / reached[2] if reached[2] else float("nan"),
        "r3_ko_rate": ko[3] / reached[3] if reached[3] else float("nan"),
        "start_r2_stamina_mean": float(np.mean(start_r2_stamina)) if start_r2_stamina else float("nan"),
        "start_r2_penalty_mean": float(np.mean(start_r2_penalty)) if start_r2_penalty else float("nan"),
        "start_r2_effective_power_mean": float(np.mean(start_r2_power)) if start_r2_power else float("nan"),
        "start_r3_stamina_mean": float(np.mean(start_r3_stamina)) if start_r3_stamina else float("nan"),
        "start_r3_penalty_mean": float(np.mean(start_r3_penalty)) if start_r3_penalty else float("nan"),
        "start_r3_effective_power_mean": float(np.mean(start_r3_power)) if start_r3_power else float("nan"),
    }
    return row


def main() -> None:
    _configure_locked_candidate()

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(DEFAULT_BOUTS).reset_index(drop=True)

    rng = np.random.default_rng(DEFAULT_SEED)
    seed_matrix = rng.integers(
        0,
        2**31 - 1,
        size=(len(cohort), DEFAULT_PATHS),
        dtype=np.int64,
    )

    original_exponent = float(v31.FATIGUE_CURVE_EXPONENT)

    print("\n" + "=" * 150)
    print("FATIGUE EXPONENT SWEEP — LOCKED CURVE 16 — 200 BOUTS x 10 PATHS")
    print("=" * 150)
    print(f"exponents={EXPONENTS}")
    print(f"max fatigue penalty={v31.MAX_FATIGUE_RATING_PENALTY:.1f}; resilience scale={v31.FATIGUE_RESILIENCE_SCALE:.1f}")
    print("exponent 1.0 = linear in missing stamina before resilience scaling")
    print("global stamina recovery=40% of missing between rounds")
    print(f"KD base={run87.KD_BASE_LOGIT:.2f}; shock={run87.KD_SHOCK_COEFFICIENT:.0f}; depletion={run87.KD_DEPLETION_COEFFICIENT:.2f}")
    print(f"collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print("research only: all other simulator/FSR settings unchanged")

    rows: list[dict[str, float]] = []
    try:
        for exponent in EXPONENTS:
            rows.append(_run_exponent(exponent, cohort, pairs, seed_matrix))
    finally:
        v31.FATIGUE_CURVE_EXPONENT = original_exponent

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nSWEEP RESULTS")
    display_cols = [
        "fatigue_exponent",
        "r1_sig_mean", "r1_kd_mean", "r1_ko_rate",
        "r2_sig_mean", "r2_kd_mean", "r2_ko_rate",
        "r3_sig_mean", "r3_kd_mean", "r3_ko_rate",
        "start_r2_stamina_mean", "start_r2_penalty_mean", "start_r2_effective_power_mean",
        "start_r3_stamina_mean", "start_r3_penalty_mean", "start_r3_effective_power_mean",
    ]
    print(out[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nHISTORICAL — EXACT SAME 200 BOUTS")
    print("R1: sig=33.7450; KD=0.1750; KO=10.00%")
    print("R2: sig=38.8447; KD=0.2174; KO=15.53%")
    print("R3: sig=38.6107; KD=0.1221; KO=4.58%")

    print(f"\nSaved: {OUTPUT_PATH}")
    print("Research only: no production simulator constants or FSR artifacts modified.")


if __name__ == "__main__":
    main()
