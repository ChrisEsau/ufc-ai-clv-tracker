"""Trace the exact transition hazards offered to the current MC every segment.

This is a read-only diagnostic around the existing full-fight simulator. It does
not change any hazard, transition, finish, stamina, damage, or judging equation.
For each 10-second segment it records the deterministic candidate probabilities
that the current engine presents to its transition sampler, then records the
actual transition note produced by the unchanged simulator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_2026_baseline_age_power_same_decay as age_power
from scripts.experimental import run_single_historical_age_power_diagnostic as diag


DEFAULT_SEED = 20260811
OUT_DIR = Path("data/experimental/single_historical_td_hazard_trace")


class HazardTraceSim(full.StaticFSRMCFullFightV1):
    """Unchanged current simulator plus deterministic hazard observations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hazard_trace: list[dict[str, object]] = []

    def _profile_snapshot(self, fighter: int) -> dict[str, float]:
        d, c, w = base._style_preferences(self.fighters[fighter])
        return {
            "wrestling_entry": base._value(self.fighters[fighter], "wrestling_entry"),
            "control_imposition": base._value(self.fighters[fighter], "control_imposition"),
            "distance_pressure": base._value(self.fighters[fighter], "distance_striking_pressure"),
            "clinch_pressure": base._value(self.fighters[fighter], "clinch_striking_pressure"),
            "distance_pref": d,
            "clinch_pref": c,
            "wrestling_pref": w,
        }

    def _distance_transition(self) -> str:
        row: dict[str, object] = {
            "phase": "DISTANCE",
            "red_clinch_hazard": self._distance_clinch_hazard(0),
            "blue_clinch_hazard": self._distance_clinch_hazard(1),
            "red_td_hazard": self._td_attempt_hazard(0, "DISTANCE"),
            "blue_td_hazard": self._td_attempt_hazard(1, "DISTANCE"),
            "clinch_separate_hazard": None,
            "ground_exit_hazard": None,
            "reversal_given_exit": None,
        }
        note = super()._distance_transition()
        row["transition"] = note
        self.hazard_trace.append(row)
        return note

    def _clinch_transition(self) -> str:
        controller = self.clinch_controller
        assert controller is not None
        row: dict[str, object] = {
            "phase": "CLINCH",
            "red_clinch_hazard": None,
            "blue_clinch_hazard": None,
            "red_td_hazard": self._td_attempt_hazard(0, "CLINCH"),
            "blue_td_hazard": self._td_attempt_hazard(1, "CLINCH"),
            "clinch_separate_hazard": self._clinch_separate_hazard(controller),
            "ground_exit_hazard": None,
            "reversal_given_exit": None,
        }
        note = super()._clinch_transition()
        row["transition"] = note
        self.hazard_trace.append(row)
        return note

    def _ground_transition(self) -> str:
        controller = self.ground_controller
        assert controller is not None
        bottom = self._other(controller)
        row: dict[str, object] = {
            "phase": "GROUND",
            "red_clinch_hazard": None,
            "blue_clinch_hazard": None,
            "red_td_hazard": None,
            "blue_td_hazard": None,
            "clinch_separate_hazard": None,
            "ground_exit_hazard": self._ground_exit_hazard(controller),
            "reversal_given_exit": self._reversal_probability(bottom, controller),
        }
        note = super()._ground_transition()
        row["transition"] = note
        self.hazard_trace.append(row)
        return note


def _fmt_probability(value: object) -> str:
    if value is None or pd.isna(value):
        return "   -   "
    return f"{100.0 * float(value):6.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--red", required=True)
    parser.add_argument("--blue", required=True)
    parser.add_argument("--date")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    bout, red, blue = diag._select_bout(args.red, args.blue, args.date)
    bid = str(bout["bout_id"])
    red_name = base._display_name(red)
    blue_name = base._display_name(blue)
    red_age = diag._age(bout, "r_age")
    blue_age = diag._age(bout, "b_age")

    age_power._install_uniform_physical_age_layer()
    red_profile, _ = age_power._apply_same_physical_age_decay(red, red_age, enabled=True)
    blue_profile, _ = age_power._apply_same_physical_age_decay(blue, blue_age, enabled=True)

    sim = HazardTraceSim(
        red_profile,
        blue_profile,
        rounds=3,
        seed=args.seed,
        red_age=red_age,
        blue_age=blue_age,
    )

    red_style = sim._profile_snapshot(0)
    blue_style = sim._profile_snapshot(1)
    path = sim.run()

    if len(sim.hazard_trace) != len(path.events):
        raise RuntimeError(
            f"hazard/event trace length mismatch: hazards={len(sim.hazard_trace)} events={len(path.events)}"
        )

    rows: list[dict[str, object]] = []
    for event, hazard in zip(path.events, sim.hazard_trace):
        row = {
            "round": event["round"],
            "segment": event["segment"],
            "clock_start": event["clock_start"],
            "phase_start": event["phase_start"],
            "red_clinch_hazard": hazard["red_clinch_hazard"],
            "blue_clinch_hazard": hazard["blue_clinch_hazard"],
            "red_td_hazard": hazard["red_td_hazard"],
            "blue_td_hazard": hazard["blue_td_hazard"],
            "clinch_separate_hazard": hazard["clinch_separate_hazard"],
            "ground_exit_hazard": hazard["ground_exit_hazard"],
            "reversal_given_exit": hazard["reversal_given_exit"],
            "transition": hazard["transition"],
            "phase_end": event["phase_end"],
            "red_stamina_after": event["red_stamina_after"],
            "blue_stamina_after": event["blue_stamina_after"],
        }
        rows.append(row)

    frame = pd.DataFrame(rows)

    print("=" * 160)
    print("SINGLE-FIGHT MC TRANSITION HAZARD TRACE")
    print("=" * 160)
    print(f"bout_id: {bid}")
    print(f"fight: {red_name} vs {blue_name}")
    print(f"seed: {args.seed}")
    print("contract: current FSR-32 profiles + current age layer + unchanged MC mechanics")

    print("\nSTYLE INPUTS USED BY TD HAZARD")
    print(f"{'input':28s} {red_name:>18s} {blue_name:>18s}")
    print("-" * 68)
    for key in (
        "wrestling_entry",
        "control_imposition",
        "distance_pressure",
        "clinch_pressure",
        "distance_pref",
        "clinch_pref",
        "wrestling_pref",
    ):
        print(f"{key:28s} {red_style[key]:18.3f} {blue_style[key]:18.3f}")

    print("\nPER-SEGMENT CANDIDATE HAZARDS + ACTUAL SELECTED TRANSITION")
    print(
        f"{'R':>2s} {'SEG':>3s} {'CLOCK':>5s} {'PHASE':>8s} "
        f"{'R_CLINCH':>9s} {'B_CLINCH':>9s} {'R_TD':>9s} {'B_TD':>9s} "
        f"{'SEP':>9s} {'G_EXIT':>9s} {'REV|EXIT':>9s}  TRANSITION"
    )
    print("-" * 160)
    for row in rows:
        print(
            f"{int(row['round']):2d} {int(row['segment']):3d} {str(row['clock_start']):>5s} "
            f"{str(row['phase_start']):>8s} "
            f"{_fmt_probability(row['red_clinch_hazard']):>9s} "
            f"{_fmt_probability(row['blue_clinch_hazard']):>9s} "
            f"{_fmt_probability(row['red_td_hazard']):>9s} "
            f"{_fmt_probability(row['blue_td_hazard']):>9s} "
            f"{_fmt_probability(row['clinch_separate_hazard']):>9s} "
            f"{_fmt_probability(row['ground_exit_hazard']):>9s} "
            f"{_fmt_probability(row['reversal_given_exit']):>9s}  "
            f"{row['transition']}"
        )

    print("\nPATH RESULT")
    print(f"winner: {red_name if path.winner == 0 else blue_name}")
    print(f"method: {path.method}")
    print(f"TD attempts: {red_name}={sim.stats[0].td_att} | {blue_name}={sim.stats[1].td_att}")
    print(f"TD landed:   {red_name}={sim.stats[0].td_landed} | {blue_name}={sim.stats[1].td_landed}")
    print(f"control sec: {red_name}={sim.stats[0].control_seconds} | {blue_name}={sim.stats[1].control_seconds}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{bid}_seed_{args.seed}.csv"
    frame.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
