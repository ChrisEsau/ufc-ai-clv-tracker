"""Historical predictive backtest for RFS Monte Carlo V2.

This is a SHADOW diagnostic harness.

It:
- loads real historical UFC matchups,
- reconstructs each fighter's leakage-safe pre-fight RFS state,
- resolves fighter-specific simulator parameters,
- applies the frozen V1 global simulator calibration,
- runs Monte Carlo paths for each real matchup,
- stores predicted win/method probabilities,
- compares those predictions with the actual historical result.

Important:
The current V1 global calibration was tuned on an evenly spaced sample across
the historical dataset, so this script is NOT yet a pristine out-of-sample
validation. It is the first predictive diagnostic backtest.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner import (
    run_matchup_monte_carlo,
)

# Reuse the already-working leakage-safe bridge and frozen V1 calibration.
from scripts.calibrate_rfs_mc_v2_power_decay_v1 import (
    Candidate,
    MIN_PRIOR_FIGHTS,
    V1_KNOCKDOWN_BONUS_HAZARD,
    V1_LANDED_KO_HAZARD,
    build_matchup_inputs,
    finish_calibration,
    phase_effect_calibration,
    state_calibration,
    zero_transition_effect_calibration,
)


HISTORY_PATH = Path(
    "data/features/round_fighter_state_history.parquet"
)

MASTER_PATH = Path(
    "data/master/ufc_master.parquet"
)

OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "historical_predictive_validation_v1.csv"
)

# Keep the first diagnostic substantial but cheaper than the calibration grid.
SAMPLE_FIGHTS = 300
SIMULATIONS_PER_FIGHT = 10


def eligible_fight_ids(
    history: pd.DataFrame,
) -> list[str]:
    """Return deterministic evenly-spaced eligible historical fights."""

    frame = history.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["_prior"] = pd.to_numeric(
        frame["rfs_traj_prior_fight_count"],
        errors="coerce",
    )

    eligible = frame.loc[
        frame["_prior"] >= MIN_PRIOR_FIGHTS
    ].copy()

    # Both fighters must independently satisfy the prior-fight requirement.
    eligible = (
        eligible.groupby(
            "fight_id",
            group_keys=False,
        )
        .filter(
            lambda group: (
                len(group) == 2
                and group["fighter_id"].nunique() == 2
            )
        )
    )

    fights = (
        eligible[
            [
                "fight_id",
                "date",
            ]
        ]
        .drop_duplicates()
        .dropna(subset=["date"])
        .sort_values(
            [
                "date",
                "fight_id",
            ]
        )
        .reset_index(drop=True)
    )

    if len(fights) <= SAMPLE_FIGHTS:
        return (
            fights["fight_id"]
            .astype(str)
            .tolist()
        )

    # Deterministic evenly-spaced sample over UFC history.
    indices = [
        round(
            index
            * (len(fights) - 1)
            / (SAMPLE_FIGHTS - 1)
        )
        for index in range(SAMPLE_FIGHTS)
    ]

    return (
        fights.iloc[indices]["fight_id"]
        .astype(str)
        .tolist()
    )


def method_category(
    method: str | None,
) -> str | None:
    """Normalize historical master outcome text for evaluation only."""

    if method is None:
        return None

    value = method.strip().lower()

    if (
        "ko" in value
        or "tko" in value
    ):
        return "KO_TKO"

    if "sub" in value:
        return "SUBMISSION"

    if "decision" in value:
        return "DECISION"

    return "OTHER"


def clipped_probability(
    probability: float,
) -> float:
    """Protect log loss from log(0)."""

    epsilon = 1e-12

    return min(
        max(probability, epsilon),
        1.0 - epsilon,
    )


def main() -> None:
    """Run historical fighter-level predictive validation."""

    print("=" * 78)
    print("RFS MONTE CARLO V2 — HISTORICAL PREDICTIVE BACKTEST V1")
    print("=" * 78)

    history = pd.read_parquet(
        HISTORY_PATH
    )

    master = pd.read_parquet(
        MASTER_PATH
    )

    fight_ids = eligible_fight_ids(
        history
    )

    print(
        f"Historical fights selected: {len(fight_ids)}"
    )
    print(
        f"Simulations per matchup : {SIMULATIONS_PER_FIGHT}"
    )

    candidate = Candidate(
        landed_ko_hazard=(
            V1_LANDED_KO_HAZARD
        ),
        knockdown_bonus_hazard=(
            V1_KNOCKDOWN_BONUS_HAZARD
        ),
    )

    dynamic_state = state_calibration(
        candidate
    )

    phase_effects = phase_effect_calibration(
        candidate
    )

    transition_effects = (
        zero_transition_effect_calibration()
    )

    finish_effects = finish_calibration(
        candidate
    )

    rows: list[dict[str, object]] = []

    skipped_schedule = 0
    skipped_errors = 0

    for index, fight_id in enumerate(
        fight_ids,
        start=1,
    ):
        try:
            matchup, red, blue = (
                build_matchup_inputs(
                    history=history,
                    master=master,
                    fight_id=fight_id,
                )
            )

            if matchup.scheduled_rounds not in {
                3,
                5,
            }:
                skipped_schedule += 1
                continue

            # Unique deterministic seed block for each historical matchup.
            seed_start = (
                10_000_000
                + index * 100_000
            )

            summary = run_matchup_monte_carlo(
                red.transition,
                blue.transition,
                red.phase,
                blue.phase,
                red.dynamic,
                blue.dynamic,
                dynamic_state_calibration=(
                    dynamic_state
                ),
                phase_effect_calibration=(
                    phase_effects
                ),
                transition_effect_calibration=(
                    transition_effects
                ),
                finish_probability_calibration=(
                    finish_effects
                ),
                simulation_count=(
                    SIMULATIONS_PER_FIGHT
                ),
                seed_start=seed_start,
                scheduled_rounds=int(
                    matchup.scheduled_rounds
                ),
            )

        except Exception as exc:
            skipped_errors += 1

            print(
                f"  SKIP {fight_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        total = summary.simulation_count

        red_win_probability = (
            summary.red_win_count / total
        )

        blue_win_probability = (
            summary.blue_win_count / total
        )

        draw_probability = (
            summary.draw_count / total
        )

        red_ko_probability = (
            summary.red_ko_tko_count / total
        )

        blue_ko_probability = (
            summary.blue_ko_tko_count / total
        )

        red_submission_probability = (
            summary.red_submission_count / total
        )

        blue_submission_probability = (
            summary.blue_submission_count / total
        )

        red_decision_probability = (
            summary.red_decision_count / total
        )

        blue_decision_probability = (
            summary.blue_decision_count / total
        )

        actual_winner_side = None

        if (
            matchup.actual.winner_id
            == matchup.red.fighter_id
        ):
            actual_winner_side = "RED"

        elif (
            matchup.actual.winner_id
            == matchup.blue.fighter_id
        ):
            actual_winner_side = "BLUE"

        # For binary winner metrics, condition on a non-draw simulated result.
        decisive_probability = (
            red_win_probability
            + blue_win_probability
        )

        if decisive_probability > 0:
            red_win_probability_no_draw = (
                red_win_probability
                / decisive_probability
            )
        else:
            red_win_probability_no_draw = 0.5

        rows.append(
            {
                "fight_id": matchup.fight_id,
                "fight_date": matchup.date,
                "event_name": matchup.event_name,
                "division": matchup.division,
                "scheduled_rounds": (
                    matchup.scheduled_rounds
                ),
                "red_fighter_id": (
                    matchup.red.fighter_id
                ),
                "red_fighter_name": (
                    matchup.red.fighter_name
                ),
                "red_prior_fights": (
                    matchup.red.prior_fight_count
                ),
                "blue_fighter_id": (
                    matchup.blue.fighter_id
                ),
                "blue_fighter_name": (
                    matchup.blue.fighter_name
                ),
                "blue_prior_fights": (
                    matchup.blue.prior_fight_count
                ),
                "red_win_probability": (
                    red_win_probability
                ),
                "blue_win_probability": (
                    blue_win_probability
                ),
                "draw_probability": (
                    draw_probability
                ),
                "red_win_probability_no_draw": (
                    red_win_probability_no_draw
                ),
                "red_ko_probability": (
                    red_ko_probability
                ),
                "blue_ko_probability": (
                    blue_ko_probability
                ),
                "red_submission_probability": (
                    red_submission_probability
                ),
                "blue_submission_probability": (
                    blue_submission_probability
                ),
                "red_decision_probability": (
                    red_decision_probability
                ),
                "blue_decision_probability": (
                    blue_decision_probability
                ),
                "ko_probability": (
                    red_ko_probability
                    + blue_ko_probability
                ),
                "submission_probability": (
                    red_submission_probability
                    + blue_submission_probability
                ),
                "decision_probability": (
                    summary.scheduled_distance_count
                    / total
                ),
                "actual_winner_id": (
                    matchup.actual.winner_id
                ),
                "actual_winner_side": (
                    actual_winner_side
                ),
                "actual_method": (
                    matchup.actual.method
                ),
                "actual_method_category": (
                    method_category(
                        matchup.actual.method
                    )
                ),
                "actual_finish_round": (
                    matchup.actual.finish_round
                ),
            }
        )

        if (
            index % 25 == 0
            or index == len(fight_ids)
        ):
            print(
                f"  simulated {index}/{len(fight_ids)}"
            )

    results = pd.DataFrame(
        rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 78)
    print("PREDICTIVE RESULTS")
    print("=" * 78)

    print(
        f"Completed matchups      : {len(results)}"
    )
    print(
        f"Skipped bad schedule   : {skipped_schedule}"
    )
    print(
        f"Skipped resolver/errors: {skipped_errors}"
    )

    # ------------------------------------------------------------------
    # Winner metrics
    # ------------------------------------------------------------------

    winner_rows = results.loc[
        results["actual_winner_side"].isin(
            [
                "RED",
                "BLUE",
            ]
        )
    ].copy()

    if not winner_rows.empty:
        winner_rows["actual_red_win"] = (
            winner_rows[
                "actual_winner_side"
            ]
            == "RED"
        ).astype(float)

        winner_rows[
            "predicted_winner_side"
        ] = winner_rows.apply(
            lambda row: (
                "RED"
                if row["red_win_probability"]
                >= row["blue_win_probability"]
                else "BLUE"
            ),
            axis=1,
        )

        accuracy = (
            winner_rows[
                "predicted_winner_side"
            ]
            == winner_rows[
                "actual_winner_side"
            ]
        ).mean()

        brier = (
            (
                winner_rows[
                    "red_win_probability_no_draw"
                ]
                - winner_rows[
                    "actual_red_win"
                ]
            )
            ** 2
        ).mean()

        log_losses = []

        for row in winner_rows.itertuples():
            probability = clipped_probability(
                row.red_win_probability_no_draw
            )

            if row.actual_winner_side == "RED":
                log_losses.append(
                    -math.log(probability)
                )
            else:
                log_losses.append(
                    -math.log(
                        1.0 - probability
                    )
                )

        log_loss = sum(log_losses) / len(
            log_losses
        )

        print()
        print("Winner prediction")
        print(
            f"  evaluated fights : {len(winner_rows)}"
        )
        print(
            f"  accuracy         : {accuracy:.2%}"
        )
        print(
            f"  Brier score      : {brier:.4f}"
        )
        print(
            f"  log loss         : {log_loss:.4f}"
        )

        # Confidence calibration diagnostic.
        winner_rows[
            "predicted_win_probability"
        ] = winner_rows.apply(
            lambda row: max(
                row["red_win_probability_no_draw"],
                1.0
                - row["red_win_probability_no_draw"],
            ),
            axis=1,
        )

        winner_rows["correct"] = (
            winner_rows[
                "predicted_winner_side"
            ]
            == winner_rows[
                "actual_winner_side"
            ]
        ).astype(float)

        bins = [
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.000001,
        ]

        winner_rows["confidence_bucket"] = (
            pd.cut(
                winner_rows[
                    "predicted_win_probability"
                ],
                bins=bins,
                right=False,
            )
        )

        calibration = (
            winner_rows.groupby(
                "confidence_bucket",
                observed=True,
            )
            .agg(
                fights=(
                    "fight_id",
                    "count",
                ),
                mean_prediction=(
                    "predicted_win_probability",
                    "mean",
                ),
                actual_win_rate=(
                    "correct",
                    "mean",
                ),
            )
            .reset_index()
        )

        print()
        print("Winner confidence calibration")
        print(
            calibration.to_string(
                index=False,
            )
        )

    # ------------------------------------------------------------------
    # Aggregate method diagnostic
    # ------------------------------------------------------------------

    if not results.empty:
        actual_methods = (
            results[
                "actual_method_category"
            ]
            .value_counts(
                normalize=True
            )
        )

        print()
        print("Method probability diagnostic")

        print(
            "  simulated KO/TKO mean : "
            f"{results['ko_probability'].mean():.2%}"
        )
        print(
            "  actual KO/TKO rate    : "
            f"{actual_methods.get('KO_TKO', 0.0):.2%}"
        )

        print(
            "  simulated SUB mean    : "
            f"{results['submission_probability'].mean():.2%}"
        )
        print(
            "  actual SUB rate       : "
            f"{actual_methods.get('SUBMISSION', 0.0):.2%}"
        )

        print(
            "  simulated DEC mean    : "
            f"{results['decision_probability'].mean():.2%}"
        )
        print(
            "  actual DEC rate       : "
            f"{actual_methods.get('DECISION', 0.0):.2%}"
        )

    print()
    print(
        f"Saved shadow predictions: {OUTPUT_PATH}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
