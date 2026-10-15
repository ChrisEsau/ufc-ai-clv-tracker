"""Run one historical fight through the current provisional FSR/MC V1.4 stack.

Shadow/research only.

This launcher preserves the existing fight-id workflow from
``run_fsr_historical_fight_v1.py`` while installing the current V1.4
calibration checkpoint:

- locked population-centered FSR V1.1 fighter cards;
- V1.2 phase-conditioned activity conversion using V1.4 neutral exposure;
- RFS Phase Baseline style -> transition tendency;
- two-stage, multi-attempt takedown chains;
- takedown sequence-initiation scale = 1.75;
- V1.3 KO/TKO hazards;
- submission attempt generation = 1.00x;
- submission base hazard = 0.12;
- current cardio/dynamic-state/judging behavior.

In addition to the normal winner/method output, the diagnostic pass prints
simulated population-mean round totals beside the realized UFCStats round totals
for both fighters. Simulated round means are conditional on reaching that round,
which keeps finish-censored paths comparable to a realized fight that may end
mid-round.

Usage
-----
PYTHONPATH=. python \
    scripts/experimental/run_fsr_historical_fight_v1_4_compare.py \
    <fight_id> --simulations 500
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import FighterSide
from pipeline.simulation.rfs_mc_v2_shared_state.transition_contracts import (
    TAKEDOWN_EVENTS,
    TransitionEvent,
)

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2
from scripts.experimental import run_fsr_historical_fight_locked_v1_3 as v1_3
from scripts.experimental import run_fsr_v1_4_transition_style_grid as style
from scripts.experimental import run_fsr_v1_4_two_fight_full_validation as v1_4


SUBMISSION_HAZARD = 0.12
RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")

# Capture simulator entrypoints before installing wrapper overrides.
_ORIGINAL_RUN_MATCHUP_MONTE_CARLO = base.run_matchup_monte_carlo
_ORIGINAL_RUN_FINISH_PATH = base.run_finish_enabled_dynamic_path
_V1_3_BUILD_FULL_CARD = None
_STYLE_HISTORY: pd.DataFrame | None = None


def _load_style_history() -> pd.DataFrame:
    """Load only the leakage-safe style columns needed by the V1.4 adapter."""

    global _STYLE_HISTORY

    if _STYLE_HISTORY is None:
        history = pd.read_parquet(
            RFS_HISTORY_PATH,
            columns=[
                "fight_id",
                "fighter_id",
                *style.STYLE_COLUMNS.values(),
            ],
        )
        history["fight_id"] = history["fight_id"].astype(str)
        history["fighter_id"] = history["fighter_id"].astype(str)
        _STYLE_HISTORY = history

    return _STYLE_HISTORY


def build_full_card_v1_4(
    fight_id: str,
    fighter_id: str,
) -> tuple[dict[str, float], int]:
    """Build the installed V1.1 card and attach PRE-fight RFS style state."""

    if _V1_3_BUILD_FULL_CARD is None:
        raise RuntimeError("V1.4 wrapper was not installed before card build")

    card, fight_count = _V1_3_BUILD_FULL_CARD(
        fight_id,
        fighter_id,
    )

    card = style.attach_style(
        card,
        _load_style_history(),
        fight_id=fight_id,
        fighter_id=fighter_id,
    )

    return card, fight_count


def finish_calibration_v1_4(base_candidate):
    """Preserve V1.3 KO hazards and use the selected V1.4 submission hazard."""

    calibration = v1_3.finish_calibration(base_candidate)

    return replace(
        calibration,
        submission=replace(
            calibration.submission,
            base_probability_per_attempt=SUBMISSION_HAZARD,
        ),
    )


def run_matchup_monte_carlo_v1_4(*args, **kwargs):
    """Inject the selected V1.4 shared transition calibration."""

    kwargs["shared_path_calibration"] = v1_4.V1_4_CALIBRATION
    return _ORIGINAL_RUN_MATCHUP_MONTE_CARLO(*args, **kwargs)


def _first_existing_column(
    frame: pd.DataFrame,
    *candidates: str,
) -> str | None:
    """Return the first available schema alias."""

    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _numeric_sum(
    frame: pd.DataFrame,
    *candidates: str,
) -> float:
    """Sum one numeric metric across whichever supported alias exists."""

    column = _first_existing_column(frame, *candidates)
    if column is None or frame.empty:
        return float("nan")

    return float(
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(0.0)
        .sum()
    )


def _actual_round_rows(
    fight_id: str,
    *,
    scheduled_rounds: int,
) -> pd.DataFrame:
    """Build realized RED/BLUE round totals from UFCStats round data."""

    rounds = pd.read_parquet(base.ROUND_PATH)
    rounds["fight_id"] = rounds["fight_id"].astype(str)

    fight = rounds.loc[
        rounds["fight_id"] == str(fight_id)
    ].copy()

    if fight.empty:
        raise RuntimeError(f"No actual round rows found for fight {fight_id}")

    rows: list[dict[str, object]] = []

    for round_number in range(1, scheduled_rounds + 1):
        for side in ("RED", "BLUE"):
            selected = fight.loc[
                (fight["round"].astype(int) == round_number)
                & (fight["corner"].astype(str).str.upper() == side)
            ].copy()

            fighter_name = None
            if not selected.empty and "fighter_name" in selected.columns:
                fighter_name = str(selected["fighter_name"].iloc[0])

            rows.append(
                {
                    "round": round_number,
                    "side": side,
                    "fighter": fighter_name,
                    "actual_observed": not selected.empty,
                    "actual_sig_attempted": _numeric_sum(
                        selected,
                        "sig_str_attempted",
                    ),
                    "actual_sig_landed": _numeric_sum(
                        selected,
                        "sig_str_landed",
                    ),
                    "actual_distance_attempted": _numeric_sum(
                        selected,
                        "distance_attempted",
                        "distance_str_attempted",
                    ),
                    "actual_distance_landed": _numeric_sum(
                        selected,
                        "distance_landed",
                        "distance_str_landed",
                    ),
                    "actual_clinch_attempted": _numeric_sum(
                        selected,
                        "clinch_attempted",
                        "clinch_str_attempted",
                    ),
                    "actual_clinch_landed": _numeric_sum(
                        selected,
                        "clinch_landed",
                        "clinch_str_landed",
                    ),
                    "actual_ground_attempted": _numeric_sum(
                        selected,
                        "ground_attempted",
                        "ground_str_attempted",
                    ),
                    "actual_ground_landed": _numeric_sum(
                        selected,
                        "ground_landed",
                        "ground_str_landed",
                    ),
                    "actual_td_attempted": _numeric_sum(
                        selected,
                        "td_attempted",
                    ),
                    "actual_td_landed": _numeric_sum(
                        selected,
                        "td_landed",
                    ),
                    "actual_control_seconds": _numeric_sum(
                        selected,
                        "ctrl_sec",
                        "control_seconds",
                    ),
                    "actual_knockdowns": _numeric_sum(
                        selected,
                        "kd",
                        "knockdowns",
                    ),
                    "actual_submission_attempts": _numeric_sum(
                        selected,
                        "sub_att",
                        "submission_attempts",
                    ),
                }
            )

    return pd.DataFrame(rows)


def _print_actual_result(fight_id: str) -> None:
    """Print the realized fight result from the local master table when present."""

    if not MASTER_PATH.exists():
        return

    master = pd.read_parquet(MASTER_PATH)
    if "fight_id" not in master.columns:
        return

    master["fight_id"] = master["fight_id"].astype(str)
    rows = master.loc[master["fight_id"] == str(fight_id)].copy()
    if rows.empty:
        return

    row = rows.iloc[0]

    winner = row.get("winner", row.get("winner_name", None))
    method = row.get("method", None)
    finish_round = pd.to_numeric(
        pd.Series([row.get("finish_round", None)]),
        errors="coerce",
    ).iloc[0]
    elapsed = pd.to_numeric(
        pd.Series([row.get("match_time_sec", None)]),
        errors="coerce",
    ).iloc[0]

    time_text = ""
    if pd.notna(finish_round) and pd.notna(elapsed):
        round_number = int(finish_round)
        seconds_in_round = max(0.0, float(elapsed) - 300.0 * (round_number - 1))
        minutes = int(seconds_in_round // 60)
        seconds = int(round(seconds_in_round - 60 * minutes))
        time_text = f" | R{round_number} {minutes}:{seconds:02d}"

    print()
    print("=" * 120)
    print("ACTUAL FIGHT RESULT")
    print("=" * 120)
    print(f"Winner: {winner} | Method: {method}{time_text}")


def run_round_comparison_diagnostics(
    *,
    fight_id: str,
    red_name: str,
    blue_name: str,
    red_transition,
    blue_transition,
    red_phase,
    blue_phase,
    red_dynamic,
    blue_dynamic,
    red_card: dict[str, float],
    blue_card: dict[str, float],
    scheduled_rounds: int,
    simulation_count: int,
    seed_start: int,
) -> None:
    """Replay identical seeds and compare simulated vs actual round totals."""

    candidate = base.Candidate(
        landed_ko_hazard=base.V1_LANDED_KO_HAZARD,
        knockdown_bonus_hazard=base.V1_KNOCKDOWN_BONUS_HAZARD,
    )

    dynamic_cal = base.state_calibration(candidate)
    phase_cal = base.phase_effect_calibration(candidate)
    transition_cal = base.zero_transition_effect_calibration()
    finish_cal = base.finish_calibration(candidate)

    round_totals = {
        round_number: defaultdict(float)
        for round_number in range(1, scheduled_rounds + 1)
    }
    round_reached = Counter()

    for simulation_index in range(simulation_count):
        seed = seed_start + simulation_index

        path = _ORIGINAL_RUN_FINISH_PATH(
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
                base.intrinsic_power_multiplier(red_card["finishing_power"])
            ),
            blue_intrinsic_power_multiplier=(
                base.intrinsic_power_multiplier(blue_card["finishing_power"])
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
            shared_path_calibration=v1_4.V1_4_CALIBRATION,
        )

        segments_by_round: dict[int, list] = defaultdict(list)
        for segment in path.segments:
            segments_by_round[int(segment.state.round_number)].append(segment)

        for round_number, segments in segments_by_round.items():
            if not segments:
                continue

            round_reached[round_number] += 1
            totals = round_totals[round_number]

            for segment in segments:
                for side, activity in (
                    ("red", segment.activity.red),
                    ("blue", segment.activity.blue),
                ):
                    distance_attempted = float(
                        getattr(activity, "sig_str_attempted", 0)
                    )
                    distance_landed = float(
                        getattr(activity, "sig_str_landed", 0)
                    )
                    clinch_attempted = float(
                        getattr(activity, "clinch_str_attempted", 0)
                    )
                    clinch_landed = float(
                        getattr(activity, "clinch_str_landed", 0)
                    )
                    ground_attempted = float(
                        getattr(activity, "ground_str_attempted", 0)
                    )
                    ground_landed = float(
                        getattr(activity, "ground_str_landed", 0)
                    )

                    totals[f"{side}_distance_attempted"] += distance_attempted
                    totals[f"{side}_distance_landed"] += distance_landed
                    totals[f"{side}_clinch_attempted"] += clinch_attempted
                    totals[f"{side}_clinch_landed"] += clinch_landed
                    totals[f"{side}_ground_attempted"] += ground_attempted
                    totals[f"{side}_ground_landed"] += ground_landed
                    totals[f"{side}_sig_attempted"] += (
                        distance_attempted + clinch_attempted + ground_attempted
                    )
                    totals[f"{side}_sig_landed"] += (
                        distance_landed + clinch_landed + ground_landed
                    )
                    totals[f"{side}_control_seconds"] += float(
                        getattr(activity, "control_seconds", 0)
                    )
                    totals[f"{side}_knockdowns"] += float(
                        getattr(activity, "knockdowns", 0)
                    )
                    totals[f"{side}_submission_attempts"] += float(
                        getattr(activity, "submission_attempts", 0)
                    )

                transition = segment.transition
                if transition is None or transition.event not in TAKEDOWN_EVENTS:
                    continue

                if transition.actor is FighterSide.RED:
                    side = "red"
                elif transition.actor is FighterSide.BLUE:
                    side = "blue"
                else:
                    raise RuntimeError("takedown sequence has no actor")

                totals[f"{side}_td_attempted"] += float(transition.attempt_count)
                if transition.event is TransitionEvent.TAKEDOWN:
                    totals[f"{side}_td_landed"] += 1.0

    simulated_rows: list[dict[str, object]] = []

    for round_number in range(1, scheduled_rounds + 1):
        reached = round_reached[round_number]
        if reached <= 0:
            continue

        totals = round_totals[round_number]

        for side, fighter_name in (
            ("red", red_name),
            ("blue", blue_name),
        ):
            simulated_rows.append(
                {
                    "round": round_number,
                    "side": side.upper(),
                    "fighter": fighter_name,
                    "paths_reaching_round": reached,
                    "population_reaching_pct": (
                        100.0 * reached / float(simulation_count)
                    ),
                    "sim_sig_attempted": totals[f"{side}_sig_attempted"] / reached,
                    "sim_sig_landed": totals[f"{side}_sig_landed"] / reached,
                    "sim_distance_attempted": (
                        totals[f"{side}_distance_attempted"] / reached
                    ),
                    "sim_distance_landed": (
                        totals[f"{side}_distance_landed"] / reached
                    ),
                    "sim_clinch_attempted": (
                        totals[f"{side}_clinch_attempted"] / reached
                    ),
                    "sim_clinch_landed": (
                        totals[f"{side}_clinch_landed"] / reached
                    ),
                    "sim_ground_attempted": (
                        totals[f"{side}_ground_attempted"] / reached
                    ),
                    "sim_ground_landed": (
                        totals[f"{side}_ground_landed"] / reached
                    ),
                    "sim_td_attempted": totals[f"{side}_td_attempted"] / reached,
                    "sim_td_landed": totals[f"{side}_td_landed"] / reached,
                    "sim_control_seconds": (
                        totals[f"{side}_control_seconds"] / reached
                    ),
                    "sim_knockdowns": totals[f"{side}_knockdowns"] / reached,
                    "sim_submission_attempts": (
                        totals[f"{side}_submission_attempts"] / reached
                    ),
                }
            )

    simulated = pd.DataFrame(simulated_rows)
    actual = _actual_round_rows(
        fight_id,
        scheduled_rounds=scheduled_rounds,
    )

    comparison = simulated.merge(
        actual,
        on=["round", "side"],
        how="left",
        suffixes=("", "_actual"),
        validate="one_to_one",
    )

    # Prefer the simulator's authoritative fighter labels in terminal output.
    if "fighter_actual" in comparison.columns:
        comparison = comparison.drop(columns=["fighter_actual"])

    output_path = (
        base.OUTPUT_DIR
        / f"fsr_{fight_id}_v1_4_round_actual_comparison.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    _print_actual_result(fight_id)

    print()
    print("=" * 120)
    print("SIMULATED VS ACTUAL ROUND TOTALS")
    print("=" * 120)
    print(
        "Simulation values are population means conditional on reaching the "
        "round. Actual values are realized UFCStats totals."
    )

    compact = comparison[
        [
            "round",
            "side",
            "fighter",
            "population_reaching_pct",
            "sim_sig_attempted",
            "actual_sig_attempted",
            "sim_sig_landed",
            "actual_sig_landed",
            "sim_td_attempted",
            "actual_td_attempted",
            "sim_td_landed",
            "actual_td_landed",
            "sim_control_seconds",
            "actual_control_seconds",
            "sim_knockdowns",
            "actual_knockdowns",
            "sim_submission_attempts",
            "actual_submission_attempts",
        ]
    ].rename(
        columns={
            "round": "R",
            "side": "Side",
            "fighter": "Fighter",
            "population_reaching_pct": "Reach%",
            "sim_sig_attempted": "Sim SigAtt",
            "actual_sig_attempted": "Act SigAtt",
            "sim_sig_landed": "Sim SigLnd",
            "actual_sig_landed": "Act SigLnd",
            "sim_td_attempted": "Sim TDAtt",
            "actual_td_attempted": "Act TDAtt",
            "sim_td_landed": "Sim TDLnd",
            "actual_td_landed": "Act TDLnd",
            "sim_control_seconds": "Sim Ctrl",
            "actual_control_seconds": "Act Ctrl",
            "sim_knockdowns": "Sim KD",
            "actual_knockdowns": "Act KD",
            "sim_submission_attempts": "Sim SUB",
            "actual_submission_attempts": "Act SUB",
        }
    )

    print()
    print(
        compact.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print()
    print("=" * 120)
    print("SIMULATED VS ACTUAL PHASE STRIKING")
    print("=" * 120)

    phase = comparison[
        [
            "round",
            "side",
            "fighter",
            "sim_distance_attempted",
            "actual_distance_attempted",
            "sim_distance_landed",
            "actual_distance_landed",
            "sim_clinch_attempted",
            "actual_clinch_attempted",
            "sim_clinch_landed",
            "actual_clinch_landed",
            "sim_ground_attempted",
            "actual_ground_attempted",
            "sim_ground_landed",
            "actual_ground_landed",
        ]
    ].rename(
        columns={
            "round": "R",
            "side": "Side",
            "fighter": "Fighter",
            "sim_distance_attempted": "Sim DAtt",
            "actual_distance_attempted": "Act DAtt",
            "sim_distance_landed": "Sim DLnd",
            "actual_distance_landed": "Act DLnd",
            "sim_clinch_attempted": "Sim CAtt",
            "actual_clinch_attempted": "Act CAtt",
            "sim_clinch_landed": "Sim CLnd",
            "actual_clinch_landed": "Act CLnd",
            "sim_ground_attempted": "Sim GAtt",
            "actual_ground_attempted": "Act GAtt",
            "sim_ground_landed": "Sim GLnd",
            "actual_ground_landed": "Act GLnd",
        }
    )

    print()
    print(
        phase.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    print()
    print(f"Saved round comparison: {output_path}")


def install_v1_4() -> None:
    """Install current V1.4 behavior onto the existing fight-id runner."""

    global _V1_3_BUILD_FULL_CARD

    # Install the validated V1.1/V1.2/V1.3 adapter stack first.
    v1_3.install_overrides()

    # Capture the installed population-centered card builder before adding RFS
    # style fields required only by V1.4 transition tendency.
    _V1_3_BUILD_FULL_CARD = base.build_full_card

    # V1.2 rates are per active phase segment; recalculate their denominator
    # under the current V1.4 phase environment.
    v1_2.neutral_phase_exposure = v1_4.neutral_phase_exposure_v1_4

    base.build_full_card = build_full_card_v1_4
    base.build_transition = style.build_style_transition
    base.finish_calibration = finish_calibration_v1_4
    base.run_matchup_monte_carlo = run_matchup_monte_carlo_v1_4

    # Replace the old diagnostic replay with the requested actual-round
    # comparison. The authoritative population summary above it is unchanged.
    base.run_population_diagnostics = run_round_comparison_diagnostics


def main() -> None:
    install_v1_4()

    print()
    print("=" * 120)
    print("CURRENT SHADOW CHECKPOINT: FSR / MC V1.4 HISTORICAL FIGHT COMPARISON")
    print("=" * 120)
    print(
        f"TD scale={v1_4.ATTEMPT_SCALE:.2f} | "
        f"submission hazard={SUBMISSION_HAZARD:.2f} | "
        "round totals compared against realized UFCStats"
    )

    # Preserve the exact fight_id / --simulations / --seed CLI from the
    # existing historical runner.
    base.main()


if __name__ == "__main__":
    main()
