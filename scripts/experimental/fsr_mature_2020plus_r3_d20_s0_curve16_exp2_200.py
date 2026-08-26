"""Standalone research runner: R3 entry damage recovery 20%, stamina recovery 0%.

R2 entry remains fixed after R1 at:
- damage recovery: 20% of missing
- stamina recovery: 40% of missing

After R2, entering R3:
- damage recovery: 20% of missing
- stamina recovery: 0% of missing

All other locked curve-16 / fatigue-exponent-2.0 simulator settings are unchanged.
Uses the exact same first 200 mature 2020+ bouts x 10 paths and seed matrix used by
other same-200 calibration experiments.
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

KD_BASE_LOGIT = -8.80
COLLAPSE_SCALE = 2.0
COLLAPSE_CURVATURE = 16.0
FATIGUE_EXPONENT = 2.0

R2_DAMAGE_RECOVERY = 0.20
R2_STAMINA_RECOVERY = 0.40
R3_DAMAGE_RECOVERY = 0.20
R3_STAMINA_RECOVERY = 0.00

OUTPUT_PATH = Path("data/experimental/r3_d20_s0_curve16_exp2_200.csv")


def _configure_locked_candidate() -> None:
    run87.KD_BASE_LOGIT = KD_BASE_LOGIT
    run87.COLLAPSE_CURVATURE = COLLAPSE_CURVATURE
    run87.COLLAPSE = collapse_mod.CollapseCandidate(
        f"scale{COLLAPSE_SCALE:.1f}_curve{COLLAPSE_CURVATURE:.1f}",
        COLLAPSE_SCALE,
        COLLAPSE_CURVATURE,
    )
    v31.FATIGUE_CURVE_EXPONENT = FATIGUE_EXPONENT


class RecoveryAuditSim(run87.AuditSim):
    """Locked simulator with round-specific global recovery only."""

    def _apply_between_round_recovery(self, completed_round: int) -> None:
        if completed_round == 1:
            damage_fraction = R2_DAMAGE_RECOVERY
            stamina_fraction = R2_STAMINA_RECOVERY
        else:
            damage_fraction = R3_DAMAGE_RECOVERY
            stamina_fraction = R3_STAMINA_RECOVERY

        for fighter_index, state in enumerate(self.damage_state):
            missing = max(0.0, state.reservoir_capacity - state.reservoir_current)
            before = float(state.reservoir_current)
            restored = min(missing * damage_fraction, missing)
            state.reservoir_current = min(
                state.reservoir_capacity,
                state.reservoir_current + restored,
            )
            actual_restored = float(state.reservoir_current - before)
            self.total_round_recovery[fighter_index] += actual_restored
            self.round_recovery_events.append({
                "after_round": int(completed_round),
                "fighter": int(fighter_index),
                "recovery_mode": "round_specific_global",
                "fraction_of_missing": float(damage_fraction),
                "reservoir_before": before,
                "reservoir_after": float(state.reservoir_current),
                "restored": actual_restored,
            })

        for fighter_index, state in enumerate(self.stamina_state):
            missing = max(0.0, state.capacity - state.current)
            before = float(state.current)
            restored = min(missing * stamina_fraction, missing)
            state.current = min(state.capacity, state.current + restored)
            actual_restored = float(state.current - before)
            self.total_stamina_recovered[fighter_index] += actual_restored
            self.stamina_round_events.append({
                "after_round": int(completed_round),
                "fighter": int(fighter_index),
                "recovery_mode": "round_specific_global",
                "fraction_of_missing": float(stamina_fraction),
                "stamina_before": before,
                "stamina_after": float(state.current),
                "restored": actual_restored,
            })


def _run_prefix(red, blue, *, rounds: int, seed: int, red_age, blue_age):
    sim = RecoveryAuditSim(
        red,
        blue,
        rounds=rounds,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
    )
    path = sim.run()
    kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
    finish_round = int(getattr(path.finish, "round", 0) or 0) if path.finish is not None else 0
    return sim, path, kd, finish_round


def _start_round_values(sim, round_no: int):
    events = [
        e for e in sim.effective_fsr_events
        if int(e["round"]) == round_no and int(e["segment"]) == 1
    ]
    stamina = [float(e["stamina_fraction"]) for e in events]
    penalty = [float(e["fatigue_penalty"]) for e in events]
    power = [float(e["effective_striking_power"]) for e in events]
    return stamina, penalty, power


def _start_r3_reservoir(sim):
    values = []
    for event in sim.round_recovery_events:
        if int(event["after_round"]) != 2:
            continue
        fighter = int(event["fighter"])
        capacity = float(sim.damage_state[fighter].reservoir_capacity)
        after = float(event["reservoir_after"])
        if capacity > 0:
            values.append(after / capacity)
    return values


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

    reached = {1: 0, 2: 0, 3: 0}
    sig = {1: 0, 2: 0, 3: 0}
    kd = {1: 0, 2: 0, 3: 0}
    ko = {1: 0, 2: 0, 3: 0}

    r3_stamina: list[float] = []
    r3_penalty: list[float] = []
    r3_power: list[float] = []
    r3_reservoir: list[float] = []
    terminal_r3 = 0
    direct_r3 = 0

    total_paths = len(cohort) * DEFAULT_PATHS
    completed = 0
    original_exponent = float(v31.FATIGUE_CURVE_EXPONENT)

    print("\n" + "=" * 160)
    print("STANDALONE R3 RECOVERY RUN — DAMAGE 20% / STAMINA 0% — CURVE 16 / EXPONENT 2.0")
    print("=" * 160)
    print("Exact same first 200 mature 2020+ bouts x 10 paths")
    print("R2 entry after R1: damage=20% of missing; stamina=40% of missing")
    print("R3 entry after R2: damage=20% of missing; stamina=0% of missing")
    print(f"KD base={KD_BASE_LOGIT:.2f}; collapse scale={COLLAPSE_SCALE:.1f}; curvature={COLLAPSE_CURVATURE:.1f}")
    print("all other simulator/FSR settings unchanged")

    try:
        for bout_idx, (_, bout) in enumerate(cohort.iterrows()):
            red, blue = pairs[str(bout["bout_id"])]
            r_age = float(bout["r_age"]) if not np.isnan(bout["r_age"]) else None
            b_age = float(bout["b_age"]) if not np.isnan(bout["b_age"]) else None

            for seed in seed_matrix[bout_idx]:
                sim1, path1, kd1, fr1 = _run_prefix(
                    red, blue, rounds=1, seed=int(seed), red_age=r_age, blue_age=b_age
                )
                sim2, path2, kd2, fr2 = _run_prefix(
                    red, blue, rounds=2, seed=int(seed), red_age=r_age, blue_age=b_age
                )
                sim3, path3, kd3, fr3 = _run_prefix(
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

                    s, p, pw = _start_round_values(sim3, 3)
                    r3_stamina.extend(s)
                    r3_penalty.extend(p)
                    r3_power.extend(pw)
                    r3_reservoir.extend(_start_r3_reservoir(sim3))

                    if path3.finish is not None and fr3 == 3:
                        terminal_r3 += int(sim3.terminal_collapse_finishes > 0)
                        direct_r3 += int(sim3.direct_strike_finishes > 0)

            completed += DEFAULT_PATHS
            if completed % 500 == 0 or bout_idx + 1 == len(cohort):
                print(f"paths {completed:,}/{total_paths:,}", flush=True)
    finally:
        v31.FATIGUE_CURVE_EXPONENT = original_exponent

    row = {
        "candidate": "r3_d20_s0",
        "r2_entry_damage_recovery": R2_DAMAGE_RECOVERY,
        "r2_entry_stamina_recovery": R2_STAMINA_RECOVERY,
        "r3_entry_damage_recovery": R3_DAMAGE_RECOVERY,
        "r3_entry_stamina_recovery": R3_STAMINA_RECOVERY,
        "r1_sig_mean": sig[1] / reached[1],
        "r1_kd_mean": kd[1] / reached[1],
        "r1_ko_rate": ko[1] / reached[1],
        "r2_sig_mean": sig[2] / reached[2],
        "r2_kd_mean": kd[2] / reached[2],
        "r2_ko_rate": ko[2] / reached[2],
        "r3_sig_mean": sig[3] / reached[3],
        "r3_kd_mean": kd[3] / reached[3],
        "r3_ko_rate": ko[3] / reached[3],
        "r3_rounds": reached[3],
        "start_r3_stamina_mean": float(np.mean(r3_stamina)),
        "start_r3_penalty_mean": float(np.mean(r3_penalty)),
        "start_r3_effective_power_mean": float(np.mean(r3_power)),
        "start_r3_reservoir_mean": float(np.mean(r3_reservoir)),
        "start_r3_reservoir_median": float(np.median(r3_reservoir)),
        "r3_terminal_collapse": terminal_r3,
        "r3_direct_strike": direct_r3,
    }

    out = pd.DataFrame([row])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nRESULT")
    display_cols = [
        "candidate",
        "r1_sig_mean", "r1_kd_mean", "r1_ko_rate",
        "r2_sig_mean", "r2_kd_mean", "r2_ko_rate",
        "r3_sig_mean", "r3_kd_mean", "r3_ko_rate",
        "start_r3_stamina_mean", "start_r3_penalty_mean", "start_r3_effective_power_mean",
        "start_r3_reservoir_mean", "start_r3_reservoir_median",
        "r3_terminal_collapse", "r3_direct_strike",
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
