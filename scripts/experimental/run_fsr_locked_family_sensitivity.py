"""Same-seed MC V2 sensitivity audit for locked FSR families.

Shadow/research only.

For one historical target fight:
1. build the leakage-safe locked-family PRE-fight cards;
2. run a baseline MC population;
3. perturb one RED fighter FSR skill by +/-5 rating points;
4. replay identical seeds;
5. report mechanistic simulator deltas.

This tests adapter pathway behavior, not predictive accuracy or final calibration.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1 as locked


DEFAULT_SIMULATIONS = 250
DEFAULT_SEED = 2026080700
DEFAULT_DELTA = 5.0

ALIAS_MAP = {
    "distance_precision": "distance_accuracy",
    "wrestling_entry": "td_initiative",
    "wrestling_conversion": "td_completion",
    "striking_power": "finishing_power",
    "damage_resistance": "damage_absorption",
}

PRIMARY_METRIC = {
    "distance_precision": ("red_distance_landed", +1),
    "distance_defense": ("blue_distance_landed", -1),
    "wrestling_entry": ("red_takedowns", +1),
    "wrestling_conversion": ("red_takedowns", +1),
    "td_defense": ("blue_takedowns", -1),
    "control_imposition": ("red_control_seconds", +1),
    "control_resistance": ("blue_control_seconds", -1),
    "submission_pressure": ("red_submission_attempts", +1),
    "submission_conversion": ("red_submission_pct", +1),
    "submission_resistance": ("blue_submission_pct", -1),
    "striking_power": ("red_knockdowns", +1),
    "chin_resistance": ("blue_ko_pct", -1),
    "damage_resistance": ("red_final_damage", -1),
}


def pct(count: float, n: int) -> float:
    return 100.0 * count / n


def perturb_card(
    card: dict[str, float],
    skill: str,
    delta: float,
) -> dict[str, float]:
    """Return a copied card with one locked FSR rating perturbed."""

    out = dict(card)
    out[skill] = float(out[skill]) + float(delta)

    alias = ALIAS_MAP.get(skill)
    if alias is not None:
        out[alias] = out[skill]

    return out


def build_inputs(
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
):
    """Translate two FSR cards into the existing MC V2 contracts."""

    return (
        locked.build_transition(red_card),
        locked.build_transition(blue_card),
        locked.build_phase(red_card, blue_card, baselines),
        locked.build_phase(blue_card, red_card, baselines),
        locked.build_dynamic(red_card),
        locked.build_dynamic(blue_card),
    )


def run_population(
    *,
    red_card: dict[str, float],
    blue_card: dict[str, float],
    baselines: dict[str, float],
    scheduled_rounds: int,
    simulations: int,
    seed_start: int,
) -> dict[str, float]:
    """Run identical-seed MC paths and retain pathway-level mechanics."""

    (
        red_transition,
        blue_transition,
        red_phase,
        blue_phase,
        red_dynamic,
        blue_dynamic,
    ) = build_inputs(
        red_card,
        blue_card,
        baselines,
    )

    candidate = base.Candidate(
        landed_ko_hazard=base.V1_LANDED_KO_HAZARD,
        knockdown_bonus_hazard=base.V1_KNOCKDOWN_BONUS_HAZARD,
    )

    dynamic_cal = base.state_calibration(candidate)
    phase_cal = base.phase_effect_calibration(candidate)
    transition_cal = base.zero_transition_effect_calibration()
    finish_cal = base.finish_calibration(candidate)

    totals = defaultdict(float)

    for i in range(simulations):
        seed = seed_start + i

        path = base.run_finish_enabled_dynamic_path(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            red_dynamic,
            blue_dynamic,
            dynamic_state_calibration=dynamic_cal,
            phase_effect_calibration=phase_cal,
            transition_effect_calibration=transition_cal,
            finish_probability_calibration=finish_cal,
            scheduled_rounds=scheduled_rounds,
            seed=seed,
            red_intrinsic_power_multiplier=(
                base.intrinsic_power_multiplier(
                    red_card["finishing_power"]
                )
            ),
            blue_intrinsic_power_multiplier=(
                base.intrinsic_power_multiplier(
                    blue_card["finishing_power"]
                )
            ),
            red_intrinsic_ko_vulnerability_multiplier=(
                base.intrinsic_ko_vulnerability_multiplier(
                    red_card["chin_resistance"]
                )
            ),
            blue_intrinsic_ko_vulnerability_multiplier=(
                base.intrinsic_ko_vulnerability_multiplier(
                    blue_card["chin_resistance"]
                )
            ),
        )

        for segment in path.segments:
            activity = segment.activity

            red_activity = activity.red
            blue_activity = activity.blue

            totals["red_distance_attempted"] += getattr(
                red_activity, "sig_str_attempted", 0
            )
            totals["blue_distance_attempted"] += getattr(
                blue_activity, "sig_str_attempted", 0
            )
            totals["red_distance_landed"] += getattr(
                red_activity, "sig_str_landed", 0
            )
            totals["blue_distance_landed"] += getattr(
                blue_activity, "sig_str_landed", 0
            )

            totals["red_control_seconds"] += getattr(
                red_activity, "control_seconds", 0
            )
            totals["blue_control_seconds"] += getattr(
                blue_activity, "control_seconds", 0
            )

            totals["red_submission_attempts"] += getattr(
                red_activity, "submission_attempts", 0
            )
            totals["blue_submission_attempts"] += getattr(
                blue_activity, "submission_attempts", 0
            )

            totals["red_knockdowns"] += getattr(
                red_activity, "knockdowns", 0
            )
            totals["blue_knockdowns"] += getattr(
                blue_activity, "knockdowns", 0
            )

            if segment.state.phase.value == "ground":
                totals["ground_seconds"] += 30.0

            transition = segment.transition
            if (
                transition is not None
                and transition.event.value == "takedown"
            ):
                if transition.actor is base.FighterSide.RED:
                    totals["red_takedowns"] += 1
                elif transition.actor is base.FighterSide.BLUE:
                    totals["blue_takedowns"] += 1

        final_state = path.segments[-1].dynamic_state_after_segment
        totals["red_final_damage"] += final_state.red.damage
        totals["blue_final_damage"] += final_state.blue.damage

        result = base.resolve_final_fight_result(path)

        if result.winner is base.FighterSide.RED:
            totals["red_wins"] += 1
        elif result.winner is base.FighterSide.BLUE:
            totals["blue_wins"] += 1
        else:
            totals["draws"] += 1

        if result.finish is None:
            totals["decisions"] += 1
        else:
            method = result.finish.method.value.lower()

            if "ko" in method:
                if result.winner is base.FighterSide.RED:
                    totals["red_ko"] += 1
                elif result.winner is base.FighterSide.BLUE:
                    totals["blue_ko"] += 1

            if "submission" in method:
                if result.winner is base.FighterSide.RED:
                    totals["red_submission"] += 1
                elif result.winner is base.FighterSide.BLUE:
                    totals["blue_submission"] += 1

    mean_keys = (
        "red_distance_attempted",
        "blue_distance_attempted",
        "red_distance_landed",
        "blue_distance_landed",
        "red_takedowns",
        "blue_takedowns",
        "red_control_seconds",
        "blue_control_seconds",
        "red_submission_attempts",
        "blue_submission_attempts",
        "red_knockdowns",
        "blue_knockdowns",
        "ground_seconds",
        "red_final_damage",
        "blue_final_damage",
    )

    out = {
        key: totals[key] / simulations
        for key in mean_keys
    }

    out.update(
        {
            "red_win_pct": pct(totals["red_wins"], simulations),
            "blue_win_pct": pct(totals["blue_wins"], simulations),
            "draw_pct": pct(totals["draws"], simulations),
            "red_ko_pct": pct(totals["red_ko"], simulations),
            "blue_ko_pct": pct(totals["blue_ko"], simulations),
            "red_submission_pct": pct(
                totals["red_submission"], simulations
            ),
            "blue_submission_pct": pct(
                totals["blue_submission"], simulations
            ),
            "decision_pct": pct(totals["decisions"], simulations),
        }
    )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fight_id")
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
    )
    args = parser.parse_args()

    (
        rounds,
        target_date,
        red_info,
        blue_info,
        scheduled_rounds,
    ) = base.load_target_fight(str(args.fight_id))

    locked.run_rating_builders(str(args.fight_id))

    red_card, _ = locked.build_full_card(
        str(args.fight_id),
        red_info["fighter_id"],
    )
    blue_card, _ = locked.build_full_card(
        str(args.fight_id),
        blue_info["fighter_id"],
    )

    baselines = base.population_baselines(
        rounds,
        target_date,
    )

    baseline = run_population(
        red_card=red_card,
        blue_card=blue_card,
        baselines=baselines,
        scheduled_rounds=scheduled_rounds,
        simulations=args.simulations,
        seed_start=args.seed,
    )

    rows = []

    for skill in locked.LOCKED_SKILLS:
        for direction, delta in (
            ("minus", -abs(args.delta)),
            ("plus", abs(args.delta)),
        ):
            variant_card = perturb_card(
                red_card,
                skill,
                delta,
            )

            metrics = run_population(
                red_card=variant_card,
                blue_card=blue_card,
                baselines=baselines,
                scheduled_rounds=scheduled_rounds,
                simulations=args.simulations,
                seed_start=args.seed,
            )

            row = {
                "skill": skill,
                "direction": direction,
                "rating_delta": delta,
            }

            for key, value in metrics.items():
                row[key] = value
                row[f"delta_{key}"] = (
                    value - baseline[key]
                )

            rows.append(row)

    out = pd.DataFrame(rows)

    output_path = (
        Path(base.OUTPUT_DIR)
        / (
            f"fsr_{args.fight_id}"
            "_locked_family_sensitivity.csv"
        )
    )
    out.to_csv(output_path, index=False)

    print()
    print("=" * 120)
    print("LOCKED FSR -> MC V2 SAME-SEED SENSITIVITY")
    print("=" * 120)
    print(
        f"Fight: {red_info['fighter_name']} vs "
        f"{blue_info['fighter_name']}"
    )
    print(f"Simulations per variant: {args.simulations}")
    print(f"Rating perturbation: +/-{abs(args.delta):.1f}")
    print()

    summary_rows = []

    for skill in locked.LOCKED_SKILLS:
        metric, expected_sign = PRIMARY_METRIC[skill]

        minus_row = out.loc[
            (out["skill"] == skill)
            & (out["direction"] == "minus")
        ].iloc[0]

        plus_row = out.loc[
            (out["skill"] == skill)
            & (out["direction"] == "plus")
        ].iloc[0]

        plus_delta = plus_row[f"delta_{metric}"]
        minus_delta = minus_row[f"delta_{metric}"]

        if expected_sign > 0:
            directional_pass = (
                plus_delta >= 0.0
                and minus_delta <= 0.0
            )
        else:
            directional_pass = (
                plus_delta <= 0.0
                and minus_delta >= 0.0
            )

        summary_rows.append(
            {
                "skill": skill,
                "primary_metric": metric,
                "minus_delta": minus_delta,
                "plus_delta": plus_delta,
                "directional_pass": directional_pass,
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Baseline mechanics:")
    for key in (
        "red_win_pct",
        "blue_win_pct",
        "red_ko_pct",
        "blue_ko_pct",
        "red_submission_pct",
        "blue_submission_pct",
        "red_takedowns",
        "blue_takedowns",
        "red_control_seconds",
        "blue_control_seconds",
        "red_submission_attempts",
        "blue_submission_attempts",
        "red_knockdowns",
        "blue_knockdowns",
        "red_final_damage",
        "blue_final_damage",
    ):
        print(f"  {key:<28} {baseline[key]:.4f}")

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
