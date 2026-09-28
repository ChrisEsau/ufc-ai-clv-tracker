"""Shadow KO/TKO V2: reservoir-exhaustion finish architecture.

This replaces the earlier shock-curve experiment. There is NO independent
``P(KO | strike)`` lottery in this module.

Architecture
------------
landed significant strike
    -> Damage Reservoir V1 severity / striking-power draw
    -> reservoir depletion
    -> locked Damage V1 knockdown check

if a knockdown occurs:
    -> the existing short-lived ``recent_knockdown`` state becomes active
    -> subsequent landed strikes during that state receive a provisional
       follow-up damage multiplier

if reservoir_current reaches zero:
    -> deterministic KO/TKO stoppage

Age modifier contract
---------------------
Stored/pre-fight FSR profiles are immutable. Fight-night age modifiers are read
from ``config/fsr_age_modifiers.yaml`` through the generic evaluator in
``fsr_age_modifiers.py``. The simulator contains no trait-specific age
coefficients. Only YAML entries that are both enabled and calibrated are applied.

Important boundaries
--------------------
- The KD-causing strike receives no arbitrary bonus damage merely because it KD'd.
- No generic strike-level KO hazard exists.
- No hidden consciousness/hurt meter exists.
- Ground persistence uses the isolated 0.17 shadow candidate, leaving prior
  Damage V1 and frozen V0 baselines unchanged.
- The post-KD multiplier is intentionally provisional and must be calibrated only
  after the full finish/dynamic architecture is present enough for useful tests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.experimental import fsr_age_modifiers as age_modifiers
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_damage_v1_ground017 as ground017
from scripts.experimental import fsr_static_mc_v0 as base


FSR_PATH = damage.FSR_PATH

# Architecture-enabling shadow value only. Do not treat as a calibration lock.
POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER = 2.0
RESERVOIR_FINISH_EPSILON = 1e-9

# Compatibility alias for older diagnostics/tests. The authoritative list is
# loaded from YAML; no trait-specific age constants live in simulator code.
AGE_ADJUSTED_TRAITS = age_modifiers.enabled_calibrated_traits()


def age_adjustment_penalty(age: float | None) -> float:
    """Compatibility helper for the legacy KD-resistance rule.

    Returns the magnitude of the configured knockdown-resistance modifier.
    New code should call ``fsr_age_modifiers.trait_age_modifier`` directly.
    """
    return max(
        0.0,
        -age_modifiers.trait_age_modifier("knockdown_resistance", age),
    )


def age_adjusted_effective_trait(value: float, age: float | None) -> float:
    """Compatibility helper using the configured KD-resistance age equation."""
    modifier = age_modifiers.trait_age_modifier("knockdown_resistance", age)
    cfg = age_modifiers.load_age_modifier_config()
    bounds = cfg["rating_bounds"]
    return float(min(max(float(value) + modifier, float(bounds["min"])), float(bounds["max"])))


def apply_locked_age_adjustment(
    profile: pd.Series,
    age: float | None,
) -> pd.Series:
    """Compatibility wrapper around the generic YAML-driven age layer."""
    effective, _ = age_modifiers.apply_age_modifiers(profile, age)
    return effective


@dataclass
class FinishResult:
    winner: int
    loser: int
    method: str
    raw_strike_damage: float
    effective_strike_damage: float
    reservoir_before: float
    reservoir_after: float
    knockdown_on_strike: bool
    recent_kd_before: bool
    round: int | None = None
    segment: int | None = None
    clock_start: str | None = None


@dataclass
class KOPath(base.FightPath):
    finish: FinishResult | None = None


class StaticFSRMCKOTKOV2(ground017.StaticFSRMCDamageV1Ground017):
    """Damage V1 + ground017 + deterministic reservoir-exhaustion KO/TKO."""

    def __init__(
        self,
        red: pd.Series,
        blue: pd.Series,
        *,
        rounds: int = base.DEFAULT_ROUNDS,
        seed: int = 7,
        red_age: float | None = None,
        blue_age: float | None = None,
    ) -> None:
        # Preserve leakage-safe stored FSR exactly as supplied. Fight-night
        # effective profiles are separate copies produced from the external YAML.
        self.raw_fighters = [red.copy(deep=True), blue.copy(deep=True)]
        self.fighter_ages = [
            None if red_age is None or pd.isna(red_age) else float(red_age),
            None if blue_age is None or pd.isna(blue_age) else float(blue_age),
        ]
        red_effective, red_applied = age_modifiers.apply_age_modifiers(
            red,
            self.fighter_ages[0],
        )
        blue_effective, blue_applied = age_modifiers.apply_age_modifiers(
            blue,
            self.fighter_ages[1],
        )
        self.age_effective_fighters = [
            red_effective.copy(deep=True),
            blue_effective.copy(deep=True),
        ]
        self.age_modifier_values = [red_applied, blue_applied]

        super().__init__(red_effective, blue_effective, rounds=rounds, seed=seed)
        self.finish: FinishResult | None = None

    def _apply_landed_strikes(self, attacker: int, landed: int) -> tuple[float, int]:
        """Apply landed strikes sequentially and stop only at reservoir exhaustion."""
        defender = self._other(attacker)
        total_damage = 0.0
        knockdowns = 0

        for _ in range(int(landed)):
            if self.finish is not None:
                break

            state = self.damage_state[defender]
            recent_kd_before = state.recent_knockdown
            reservoir_before = float(state.reservoir_current)

            # Draw raw severity exactly as Damage Reservoir V1 does.
            raw_damage = self._draw_strike_damage(attacker)

            # Recent KD represents temporary defensive compromise. The
            # multiplier applies only to FOLLOW-UP strikes; the strike that first
            # creates the KD state does not receive a retroactive bonus.
            effective_damage = raw_damage
            if recent_kd_before:
                effective_damage *= POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER

            state.reservoir_current = max(
                0.0,
                state.reservoir_current - effective_damage,
            )

            # Keep the existing KD probability architecture and coefficients.
            p_kd = self._knockdown_probability(defender, effective_damage)
            knockdown = self.rng.random() < p_kd

            attacker_stats = self.stats[attacker]
            defender_stats = self.stats[defender]
            assert isinstance(attacker_stats, damage.DamageFighterStats)
            assert isinstance(defender_stats, damage.DamageFighterStats)

            attacker_stats.damage_dealt += effective_damage
            defender_stats.damage_absorbed += effective_damage
            attacker_stats.max_single_strike_damage = max(
                attacker_stats.max_single_strike_damage,
                effective_damage,
            )
            defender_stats.max_single_strike_damage = max(
                defender_stats.max_single_strike_damage,
                effective_damage,
            )
            total_damage += effective_damage

            if knockdown:
                attacker_stats.knockdowns_scored += 1
                defender_stats.knockdowns_absorbed += 1
                state.recent_knockdown_segments = max(
                    state.recent_knockdown_segments,
                    damage.RECENT_KD_SEGMENTS,
                )
                knockdowns += 1

            if state.reservoir_current <= RESERVOIR_FINISH_EPSILON:
                self.finish = FinishResult(
                    winner=attacker,
                    loser=defender,
                    method="KO/TKO",
                    raw_strike_damage=float(raw_damage),
                    effective_strike_damage=float(effective_damage),
                    reservoir_before=reservoir_before,
                    reservoir_after=float(state.reservoir_current),
                    knockdown_on_strike=bool(knockdown),
                    recent_kd_before=bool(recent_kd_before),
                )
                break

        return total_damage, knockdowns

    def _generate_striking(self, phase: str) -> list[str]:
        """Generate one segment with randomized fighter resolution order."""
        self._advance_damage_timers()
        notes: list[str] = []

        if phase == "GROUND" and self.ground_controller is not None:
            top = self.ground_controller
            bottom = self._other(top)
            actors = [
                (top, 1.0, "top"),
                (bottom, base.BOTTOM_GROUND_STRIKE_RATE_MULTIPLIER, "bottom"),
            ]
        else:
            actors = [(0, 1.0, None), (1, 1.0, None)]

        # V2 still works at 10-second segment resolution. Randomizing which
        # fighter's batch resolves first removes fixed red/blue stoppage bias.
        for idx in self.rng.permutation(len(actors)):
            fighter, multiplier, label = actors[int(idx)]
            note = self._generate_strikes_for_fighter(
                fighter,
                phase,
                rate_multiplier=multiplier,
            )
            if note:
                if label:
                    note = f"{note} ({label})"
                notes.append(note)
            if self.finish is not None:
                notes.append(
                    f"STOPPAGE: {self.names[self.finish.winner]} defeats "
                    f"{self.names[self.finish.loser]} by KO/TKO"
                )
                break

        return notes

    def run(self) -> KOPath:
        events: list[dict[str, Any]] = []

        for round_no in range(1, self.rounds + 1):
            self.phase = "DISTANCE"
            self.ground_controller = None
            self.clinch_controller = None
            self.clinch_initiator = None

            for segment_no in range(1, base.SEGMENTS_PER_ROUND + 1):
                phase_start = self.phase
                ground_controller_start = self.ground_controller
                clinch_controller_start = self.clinch_controller

                for stats in self.stats:
                    stats.phase_segments[phase_start] += 1

                strike_notes = self._generate_striking(phase_start)

                if self.finish is not None:
                    self.finish.round = round_no
                    self.finish.segment = segment_no
                    self.finish.clock_start = self._clock_start(segment_no)
                    transition_note = (
                        f"fight stopped: {self.names[self.finish.winner]} "
                        f"KO/TKO {self.names[self.finish.loser]}"
                    )
                elif phase_start == "DISTANCE":
                    transition_note = self._distance_transition()
                elif phase_start == "CLINCH":
                    transition_note = self._clinch_transition()
                else:
                    transition_note = self._ground_transition()

                events.append(
                    {
                        "round": round_no,
                        "segment": segment_no,
                        "clock_start": self._clock_start(segment_no),
                        "phase_start": phase_start,
                        "phase_end": self.phase,
                        "top_start": (
                            self.names[ground_controller_start]
                            if ground_controller_start is not None
                            else None
                        ),
                        "top_end": (
                            self.names[self.ground_controller]
                            if self.ground_controller is not None
                            else None
                        ),
                        "clinch_controller_start": (
                            self.names[clinch_controller_start]
                            if clinch_controller_start is not None
                            else None
                        ),
                        "clinch_controller_end": (
                            self.names[self.clinch_controller]
                            if self.clinch_controller is not None
                            else None
                        ),
                        "striking": "; ".join(strike_notes) if strike_notes else "no sig attempts",
                        "transition": transition_note,
                        "finish": self.finish is not None,
                    }
                )

                if self.finish is not None:
                    return KOPath(events=events, stats=self.stats, finish=self.finish)

        return KOPath(events=events, stats=self.stats, finish=None)


def print_ko_summary(sim: StaticFSRMCKOTKOV2, path: KOPath) -> None:
    damage.print_damage_summary(sim)
    print("\nCONFIGURED AGE MODIFIERS")
    print("-" * 100)
    for i, name in enumerate(sim.names):
        age = sim.fighter_ages[i]
        if age is None:
            print(f"{name}: age not supplied; no age adjustment applied")
            continue
        applied = sim.age_modifier_values[i]
        if not applied:
            print(f"{name}: age={age:.2f} | no enabled calibrated age modifiers")
            continue
        raw = sim.raw_fighters[i]
        effective = sim.age_effective_fighters[i]
        pieces = []
        for trait, modifier in applied.items():
            pieces.append(
                f"{trait} {base._value(raw, trait):.2f}->{base._value(effective, trait):.2f} "
                f"({modifier:+.2f})"
            )
        print(f"{name}: age={age:.2f} | " + " | ".join(pieces))
    print(
        f"Config: {age_modifiers.DEFAULT_CONFIG_PATH} | enabled+calibrated: "
        f"{', '.join(age_modifiers.enabled_calibrated_traits()) or 'none'}"
    )

    print("\nKO/TKO V2 FINISH SUMMARY")
    print("-" * 100)
    if path.finish is None:
        print("No reservoir-exhaustion KO/TKO stoppage occurred.")
    else:
        f = path.finish
        print(
            f"{sim.names[f.winner]} def. {sim.names[f.loser]} by {f.method} | "
            f"R{f.round} {f.clock_start} | raw_damage={f.raw_strike_damage:.2f} | "
            f"effective_damage={f.effective_strike_damage:.2f} | "
            f"reservoir {f.reservoir_before:.2f}->{f.reservoir_after:.2f} | "
            f"KD_on_strike={f.knockdown_on_strike} | "
            f"recent_KD_before={f.recent_kd_before}"
        )
    print(
        "\nV2 NOTE: reservoir exhaustion is the only KO/TKO trigger. "
        f"Post-KD follow-up multiplier={POST_KD_FOLLOWUP_DAMAGE_MULTIPLIER:.2f}x "
        "is provisional shadow architecture, not a calibrated lock."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run shadow reservoir-exhaustion Static FSR MC KO/TKO V2"
    )
    parser.add_argument("--red", help="fighter name or fighter_id")
    parser.add_argument("--blue", help="fighter name or fighter_id")
    parser.add_argument("--red-age", type=float, help="red fighter age on fight date")
    parser.add_argument("--blue-age", type=float, help="blue fighter age on fight date")
    parser.add_argument("--rounds", type=int, default=base.DEFAULT_ROUNDS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fsr-path", type=Path, default=FSR_PATH)
    args = parser.parse_args()

    print(f"[FSR MC KO/TKO V2] loading profiles from {args.fsr_path}", flush=True)
    profiles = damage.load_profiles(args.fsr_path)
    print(f"[FSR MC KO/TKO V2] latest fighter profiles: {len(profiles):,}", flush=True)

    if args.red and args.blue:
        red = base.find_profile(profiles, args.red)
        blue = base.find_profile(profiles, args.blue)
    elif args.red or args.blue:
        raise SystemExit("Provide both --red and --blue, or neither.")
    else:
        red, blue = base._default_matchup(profiles)
        print("[FSR MC KO/TKO V2] no matchup supplied; using first two profiles.", flush=True)

    sim = StaticFSRMCKOTKOV2(
        red,
        blue,
        rounds=args.rounds,
        seed=args.seed,
        red_age=args.red_age,
        blue_age=args.blue_age,
    )
    path = sim.run()
    base.print_path(path, sim.names)
    print_ko_summary(sim, path)


if __name__ == "__main__":
    main()
