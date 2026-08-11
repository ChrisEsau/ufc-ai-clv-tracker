"""Research-only KD-collapse curvature sweep with terminal-KO accounting.

Fixed working architecture
--------------------------
- FSR-32 fresh striking_power active through the rolling stamina engine.
- Contact quality: mean-1 lognormal, sigma=0.80.
- Power magnitude scale: 75.
- KD curve: base logit=-9.15, shock coefficient=100, depletion=0.
- Collapse scale is FIXED at 5.0.
- Only collapse shock curvature is swept.

Accounting contract
-------------------
A confirmed knockdown that survives collapse is recorded as a KD.
A confirmed knockdown whose collapse exhausts the reservoir is recorded as
KO/TKO only and is NOT added to KD counts.
A strike that directly exhausts the reservoir remains KO/TKO without requiring
or recording a KD.

This is a 200-bout x 10-path diagnostic and does not modify production code.
CSV output is rewritten after every completed candidate.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2 as ko
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse_mod
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33
from scripts.experimental import fsr_static_mc_v0 as base

DEFAULT_BOUTS = 200
DEFAULT_PATHS = 10
DEFAULT_SEED = 20260810

CONTACT_SIGMA = 0.80
POWER_MAGNITUDE_SCALE = 75.0
BASE_DAMAGE_AT_POWER_50 = 1.18
KD_BASE_LOGIT = -9.15
KD_SHOCK_COEFFICIENT = 100.0
KD_DEPLETION_COEFFICIENT = 0.0
COLLAPSE_SCALE = 5.0

CURVATURES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

OUTPUT_PATH = Path("data/experimental/collapse_curvature_sweep_200.csv")

HIST_R1_KD_MEAN = 0.2281
HIST_ANY_R1_KD = 0.2000
HIST_TOTAL_KD_MEAN = 0.4364
HIST_ANY_KD = 0.3578
HIST_R1_KO = 0.1406
HIST_TOTAL_KO = 0.3144
HIST_MEAN_KO_ROUND = 1.835


@dataclass(frozen=True)
class Candidate:
    curvature: float

    @property
    def name(self) -> str:
        return f"scale{COLLAPSE_SCALE:.1f}_curve{self.curvature:.1f}"

    @property
    def collapse(self) -> collapse_mod.CollapseCandidate:
        return collapse_mod.CollapseCandidate(self.name, COLLAPSE_SCALE, self.curvature)


CANDIDATES = tuple(Candidate(c) for c in CURVATURES)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + exp(-x))


def _power_multiplier(power: float) -> float:
    return float(exp((float(power) - 50.0) / POWER_MAGNITUDE_SCALE))


class CurvatureAuditSim(v33.StaticFSRMCKOTKOV33GlobalRecovery):
    """Current power/contact model with terminal-collapse-is-KO-only accounting."""

    def __init__(self, *args, collapse: collapse_mod.CollapseCandidate, **kwargs) -> None:
        self.collapse = collapse
        self.terminal_collapse_finishes = 0
        self.direct_strike_finishes = 0
        self.surviving_kds_by_round = {1: 0, 2: 0, 3: 0}
        super().__init__(*args, collapse=collapse, **kwargs)

    def _draw_contact_quality(self) -> float:
        sigma = CONTACT_SIGMA
        return float(self.rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma))

    def _draw_strike_damage(self, attacker: int) -> float:
        effective_power = base._value(self.fighters[attacker], "striking_power")
        q = self._draw_contact_quality()
        return max(0.0, BASE_DAMAGE_AT_POWER_50 * q * _power_multiplier(effective_power))

    def _knockdown_probability(self, defender: int, strike_damage: float) -> float:
        state = self.damage_state[defender]
        resistance = base._value(self.fighters[defender], "knockdown_resistance")
        shock_fraction = strike_damage / state.reservoir_capacity
        logit_p = (
            KD_BASE_LOGIT
            + KD_SHOCK_COEFFICIENT * shock_fraction
            + (50.0 - resistance) / damage.KD_RESISTANCE_SCALE
            + KD_DEPLETION_COEFFICIENT * (1.0 - state.reservoir_fraction)
            + (damage.KD_RECENT_KD_LOGIT_BONUS if state.recent_knockdown else 0.0)
        )
        return float(np.clip(_sigmoid(logit_p), 0.0, 0.95))

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        defender = self._other(attacker)
        total_damage = 0.0
        surviving_kds = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            reservoir_before = float(state.reservoir_current)

            raw_damage = self._draw_strike_damage(attacker)
            effective_damage = raw_damage
            if recent_kd_before:
                effective_damage *= ko.POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            state.reservoir_current = max(0.0, state.reservoir_current - effective_damage)

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(attacker_stats.max_single_strike_damage, effective_damage)
            defender_stats.max_single_strike_damage = max(defender_stats.max_single_strike_damage, effective_damage)
            total_damage += effective_damage

            # Direct reservoir exhaustion is an immediate KO/TKO. Do not run a
            # KD lottery after the fight is already over.
            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                self.direct_strike_finishes += 1
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=False,
                    recent_kd_before=bool(recent_kd_before),
                )
                break

            p_kd = self._knockdown_probability(defender, effective_damage)
            knockdown = self.rng.random() < p_kd
            if not knockdown:
                continue

            shock_fraction = effective_damage / state.reservoir_capacity
            collapse_fraction = self._kd_collapse_fraction(shock_fraction)
            collapse_damage = collapse_fraction * state.reservoir_capacity
            collapse_damage = min(collapse_damage, state.reservoir_current)
            state.reservoir_current = max(0.0, state.reservoir_current - collapse_damage)

            attacker_stats.damage_dealt += collapse_damage
            defender_stats.damage_absorbed += collapse_damage
            total_damage += collapse_damage

            if state.reservoir_current <= ko.RESERVOIR_FINISH_EPSILON:
                # Terminal collapse is classified as KO/TKO only.
                self.terminal_collapse_finishes += 1
                self.finish = ko.FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=True,
                    recent_kd_before=bool(recent_kd_before),
                )
                break

            # Only a surviving knockdown is recorded as a KD.
            attacker_stats.knockdowns_scored += 1
            defender_stats.knockdowns_absorbed += 1
            state.recent_knockdown_segments = max(
                state.recent_knockdown_segments,
                damage.RECENT_KD_SEGMENTS,
            )
            surviving_kds += 1

            round_no = int(getattr(self, "_audit_round", 0) or 0)
            if round_no in self.surviving_kds_by_round:
                self.surviving_kds_by_round[round_no] += 1

        return total_damage, surviving_kds

    def _generate_striking(self, phase: str) -> list[str]:
        return super()._generate_striking(phase)

    def run(self):
        # Reimplement only the tiny bit needed to expose current round to the KD
        # accounting hook, while preserving V3.3's run semantics.
        original_generate = self._generate_striking

        def wrapped_generate(phase: str):
            return original_generate(phase)

        self._generate_striking = wrapped_generate  # type: ignore[method-assign]
        # V3.3 sets finish.round before returning. For per-round KD counts, the
        # simulator is replayed as 1/2/3-round prefixes in the audit below, so we
        # do not depend on mutable round internals here.
        return super().run()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bouts", type=int, default=DEFAULT_BOUTS)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def _run_prefix(red, blue, *, rounds: int, seed: int, collapse, red_age, blue_age):
    sim = CurvatureAuditSim(
        red,
        blue,
        rounds=rounds,
        seed=seed,
        red_age=red_age,
        blue_age=blue_age,
        collapse=collapse,
    )
    path = sim.run()
    kd = int(sim.stats[0].knockdowns_scored) + int(sim.stats[1].knockdowns_scored)
    finish_round = int(getattr(path.finish, "round", 0) or 0) if path.finish is not None else 0
    return sim, path, kd, finish_round


def main() -> None:
    args = parse_args()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.head(args.bouts).reset_index(drop=True)
    total_paths = len(cohort) * args.paths

    rng = np.random.default_rng(args.seed)
    seed_matrix = rng.integers(0, 2**31 - 1, size=(len(cohort), args.paths), dtype=np.int64)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 144)
    print("KD COLLAPSE CURVATURE SWEEP — SCALE FIXED AT 5.0 — 200 BOUTS x 10 PATHS")
    print("=" * 144)
    print(f"contact sigma={CONTACT_SIGMA:.2f}; power scale={POWER_MAGNITUDE_SCALE:.0f}")
    print(f"KD base={KD_BASE_LOGIT:.2f}; shock={KD_SHOCK_COEFFICIENT:.0f}; depletion={KD_DEPLETION_COEFFICIENT:.2f}")
    print("terminal collapse = KO/TKO only; surviving knockdown = KD")
    print(f"CSV: {OUTPUT_PATH}")

    for candidate in CANDIDATES:
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
                # Prefix replays with identical seed give exact cumulative R1/R2/R3
                # counts under the same random path until a prior finish occurs.
                _, _, kd1, _ = _run_prefix(red, blue, rounds=1, seed=int(seed), collapse=candidate.collapse, red_age=r_age, blue_age=b_age)
                _, _, kd2, _ = _run_prefix(red, blue, rounds=2, seed=int(seed), collapse=candidate.collapse, red_age=r_age, blue_age=b_age)
                sim3, path3, kd3, finish_round = _run_prefix(red, blue, rounds=3, seed=int(seed), collapse=candidate.collapse, red_age=r_age, blue_age=b_age)

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
                print(f"[{candidate.name}] paths {completed:,}/{total_paths:,}", flush=True)

        total_kd = r1_kd + r2_kd + r3_kd
        row = {
            "candidate": candidate.name,
            "collapse_scale": COLLAPSE_SCALE,
            "curvature": candidate.curvature,
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
            f"total KD={row['total_kd_mean']:.4f}; total KO={row['total_ko']:.2%}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nHISTORICAL REFERENCES")
    print(f"R1 KD mean={HIST_R1_KD_MEAN:.4f}; any R1 KD={HIST_ANY_R1_KD:.2%}")
    print(f"total KD mean={HIST_TOTAL_KD_MEAN:.4f}; any KD={HIST_ANY_KD:.2%}")
    print(f"R1 KO={HIST_R1_KO:.2%}; total KO={HIST_TOTAL_KO:.2%}; mean KO round={HIST_MEAN_KO_ROUND:.3f}")

    print("\nRESULTS")
    cols = [
        "candidate", "r1_kd_mean", "r2_kd_mean", "r3_kd_mean", "total_kd_mean", "any_kd",
        "r1_ko", "r2_ko", "r3_ko", "total_ko", "mean_ko_round", "terminal_collapse_ko", "direct_strike_ko"
    ]
    print(out[cols].to_string(index=False))
    print(f"\nSaved CSV: {OUTPUT_PATH}")
    print("Research only: no production simulator or FSR artifact modified.")


if __name__ == "__main__":
    main()
