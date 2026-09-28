"""Print and persist 10-second rolling-FSR stamina diagnostics for one historical bout.

The stored FSR-32 profile remains immutable.  For every simulated segment this
script reports the pre-action stamina/effective FSR used by the segment and the
post-action stamina after queued action costs are applied.

Console output is intentionally compact.  A long-form CSV contains every
fatigue-sensitive effective FSR trait for both fighters at every simulated
segment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_damage_v1 as damage
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_1_rolling_fsr as rolling
from scripts.experimental import fsr_static_mc_v0 as base


DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)
DEFAULT_OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/segment_diagnostics"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bout-id", required=True)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def _effective_value(base_profile: pd.Series, trait: str, penalty: float) -> float:
    value = float(base_profile[trait])
    if trait not in rolling.FATIGUE_SENSITIVE_TRAITS:
        return value
    return max(rolling.MIN_EFFECTIVE_FSR_RATING, value - float(penalty))


def _tail_probability(power: float) -> float:
    return damage._sigmoid(
        damage._logit(damage.POWER_TAIL_BASE_PROBABILITY)
        + (float(power) - 50.0) / rolling.ROLLING_POWER_TAIL_RATING_SCALE
    )


def _phase_traits(phase: str) -> tuple[str, str, str]:
    return (
        base.PHASE_PRESSURE[phase],
        base.PHASE_PRECISION[phase],
        base.PHASE_DEFENSE[phase],
    )


def _build_rows(sim: rolling.StaticFSRMCKOTKOV31RollingFSR, path) -> pd.DataFrame:
    path_events = {
        (int(event["round"]), int(event["segment"])): event
        for event in path.events
    }
    effective_events = {
        (int(event["round"]), int(event["segment"]), int(event["fighter"])): event
        for event in sim.effective_fsr_events
    }

    rows: list[dict[str, object]] = []
    for (round_no, segment_no), path_event in path_events.items():
        phase = str(path_event["phase_start"])
        pressure_trait, precision_trait, defense_trait = _phase_traits(phase)

        for fighter in (0, 1):
            eff_event = effective_events[(round_no, segment_no, fighter)]
            base_profile = sim.base_fighters[fighter]
            penalty = float(eff_event["fatigue_penalty"])
            capacity = float(sim.stamina_state[fighter].capacity)
            stamina_before_fraction = float(eff_event["stamina_fraction"])
            after_key = "red_stamina_after" if fighter == 0 else "blue_stamina_after"
            stamina_after_fraction = float(path_event[after_key])
            spent = max(
                0.0,
                stamina_before_fraction * capacity - stamina_after_fraction * capacity,
            )

            row: dict[str, object] = {
                "round": round_no,
                "segment": segment_no,
                "clock_start": path_event["clock_start"],
                "phase": phase,
                "fighter": fighter,
                "fighter_name": sim.names[fighter],
                "stamina_capacity": capacity,
                "stamina_before": stamina_before_fraction * capacity,
                "stamina_before_fraction": stamina_before_fraction,
                "stamina_spent_segment": spent,
                "stamina_after": stamina_after_fraction * capacity,
                "stamina_after_fraction": stamina_after_fraction,
                "fatigue_penalty": penalty,
                "base_striking_power": float(base_profile["striking_power"]),
                "effective_striking_power": _effective_value(
                    base_profile, "striking_power", penalty
                ),
                "effective_power_tail_probability": _tail_probability(
                    _effective_value(base_profile, "striking_power", penalty)
                ),
                "phase_pressure_trait": pressure_trait,
                "effective_phase_pressure": _effective_value(
                    base_profile, pressure_trait, penalty
                ),
                "phase_precision_trait": precision_trait,
                "effective_phase_precision": _effective_value(
                    base_profile, precision_trait, penalty
                ),
                "phase_defense_trait": defense_trait,
                "effective_phase_defense": _effective_value(
                    base_profile, defense_trait, penalty
                ),
                "effective_wrestling_entry": _effective_value(
                    base_profile, "wrestling_entry", penalty
                ),
                "effective_wrestling_conversion": _effective_value(
                    base_profile, "wrestling_conversion", penalty
                ),
                "effective_td_defense": _effective_value(
                    base_profile, "td_defense", penalty
                ),
                "effective_control_imposition": _effective_value(
                    base_profile, "control_imposition", penalty
                ),
                "effective_control_resistance": _effective_value(
                    base_profile, "control_resistance", penalty
                ),
                "effective_submission_pressure": _effective_value(
                    base_profile, "submission_pressure", penalty
                ),
                "effective_reversal_ability": _effective_value(
                    base_profile, "reversal_ability", penalty
                ),
                "striking_event": path_event.get("striking", ""),
                "transition_event": path_event.get("transition", ""),
                "finish": bool(path_event.get("finish", False)),
            }

            for trait in sorted(rolling.FATIGUE_SENSITIVE_TRAITS):
                if trait in base_profile.index and pd.notna(base_profile[trait]):
                    row[f"base__{trait}"] = float(base_profile[trait])
                    row[f"effective__{trait}"] = _effective_value(
                        base_profile, trait, penalty
                    )

            rows.append(row)

    return pd.DataFrame(rows)


def _print_round_tables(frame: pd.DataFrame) -> None:
    compact_cols = [
        "segment",
        "clock_start",
        "phase",
        "fighter_name",
        "stamina_before",
        "stamina_spent_segment",
        "stamina_after",
        "fatigue_penalty",
        "base_striking_power",
        "effective_striking_power",
        "effective_power_tail_probability",
        "effective_phase_pressure",
        "effective_phase_precision",
        "effective_phase_defense",
        "effective_wrestling_entry",
        "effective_wrestling_conversion",
        "effective_control_imposition",
        "effective_control_resistance",
    ]

    for round_no, group in frame.groupby("round", sort=True):
        print("\n" + "=" * 180)
        print(f"ROUND {round_no} — 10-SECOND ROLLING FSR TRACE")
        print("=" * 180)
        print(
            group[compact_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

        summary = (
            group.groupby("fighter_name", sort=False)
            .agg(
                stamina_start=("stamina_before", "first"),
                stamina_end=("stamina_after", "last"),
                fatigue_penalty_end=("fatigue_penalty", "last"),
                effective_power_start=("effective_striking_power", "first"),
                effective_power_end=("effective_striking_power", "last"),
                mean_effective_power=("effective_striking_power", "mean"),
                mean_tail_probability=("effective_power_tail_probability", "mean"),
            )
            .reset_index()
        )
        print("\nROUND SUMMARY")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    args = _parse_args()
    cohort, pairs = cohort32.build_aligned_cohort()
    match = cohort[cohort["bout_id"].astype(str).eq(str(args.bout_id))]
    if len(match) != 1:
        raise ValueError(
            f"Expected one aligned cohort bout for {args.bout_id}; found {len(match)}"
        )

    bout = match.iloc[0]
    red, blue = pairs[str(args.bout_id)]
    r_age = float(bout["r_age"]) if pd.notna(bout["r_age"]) else None
    b_age = float(bout["b_age"]) if pd.notna(bout["b_age"]) else None

    sim = rolling.StaticFSRMCKOTKOV31RollingFSR(
        red,
        blue,
        collapse=STRONG_COLLAPSE,
        rounds=args.rounds,
        seed=args.seed,
        red_age=r_age,
        blue_age=b_age,
    )
    path = sim.run()
    frame = _build_rows(sim, path)

    print("\n" + "=" * 120)
    print("ROLLING-FSR STAMINA SEGMENT DIAGNOSTIC")
    print("=" * 120)
    print(f"bout_id: {args.bout_id}")
    print(f"fight: {sim.names[0]} vs {sim.names[1]}")
    print(f"event_date: {bout['event_date']}")
    print(f"seed: {args.seed}; requested horizon: {args.rounds} rounds")
    if path.finish is not None:
        print(
            f"simulated finish: R{path.finish.round} segment {path.finish.segment} "
            f"— {sim.names[path.finish.winner]} KO/TKO"
        )
    else:
        print("simulated finish: none")

    _print_round_tables(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.bout_id}_seed_{args.seed}_rolling_fsr_segments.csv"
    frame.to_csv(output, index=False)
    print(f"\nWrote full effective-FSR segment trace ({len(frame):,} fighter-segment rows) to {output}")
    print(
        "Timing contract: stamina_before/effective FSR -> segment actions -> "
        "stamina_spent_segment -> stamina_after."
    )


if __name__ == "__main__":
    main()
