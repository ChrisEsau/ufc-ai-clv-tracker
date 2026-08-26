"""Historical replay with absolute strike pace and trained finish hazards.

This is the first replay path that combines two learned/calibrated components:

- absolute pre-fight significant-strike attempt pace;
- mutually exclusive, conditional round finish hazards.

The mechanics engine is first run with its heuristic finish probability effectively
disabled so it produces a complete decision path. The finish provider then samples
one conditional terminal event per reached round. When a finish is sampled, the
full-path totals are thinned to the realized fight exposure. This preserves the
engine's decision scoring while removing the heuristic KO/submission logits.

The path is shadow-only. Uniform thinning of full-fight totals is an interim
approximation until all round-level activity components are provider-backed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from pipeline.simulation.contracts import (
    FightSimulationOutcome,
    FighterFightTotals,
    SimulatorConfig,
)
from pipeline.simulation.engine import summarize_outcomes
from pipeline.simulation.finish_hazard_provider import (
    FinishHazardKey,
    FinishHazardProvider,
    HistoricalFinishHazardProvider,
)
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
    _fight_level_baselines,
    aggregate_comparison,
    build_fighter_fight_history,
    build_holdout_matchups,
    calibration_tables,
    population_priors,
    score_historical_replay,
)
from pipeline.simulation.historical_strike_provider_replay import (
    PrefightStrikeCalibration,
    StaticPrefightStrikeProvider,
    estimate_prefight_strike_calibration,
)
from pipeline.simulation.provider_engine import run_simulation_with_strike_provider
from pipeline.simulation.terminal_round import RoundPerformance, thin_round_performance


TRAINED_COMPONENT_SIMULATOR_VERSION = (
    "round_simulator_v0_1_absolute_strike_trained_finish"
)
_HEURISTIC_FINISH_DISABLE_EPSILON = 1e-9
_FINISH_LABELS = (
    None,
    ("red", "ko_tko"),
    ("red", "submission"),
    ("blue", "ko_tko"),
    ("blue", "submission"),
)


@dataclass(frozen=True)
class HistoricalTrainedComponentReplayResult:
    fight_predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    aggregate_comparison: pd.DataFrame
    population_priors: Mapping[str, float]
    strike_calibration: PrefightStrikeCalibration
    finish_model_name: str


def _thin_totals(
    rng: np.random.Generator,
    totals: FighterFightTotals,
    exposure_fraction: float,
) -> FighterFightTotals:
    """Thin complete-fight totals to a sampled terminal exposure."""
    performance = thin_round_performance(
        rng,
        RoundPerformance(
            sig_attempted=int(totals.sig_attempted),
            sig_landed=int(totals.sig_landed),
            takedowns_attempted=int(totals.takedowns_attempted),
            takedowns_landed=int(totals.takedowns_landed),
            control_seconds=float(totals.control_seconds),
            knockdowns=int(totals.knockdowns),
        ),
        float(exposure_fraction),
    )
    return FighterFightTotals(
        sig_attempted=performance.sig_attempted,
        sig_landed=performance.sig_landed,
        takedowns_attempted=performance.takedowns_attempted,
        takedowns_landed=performance.takedowns_landed,
        control_seconds=round(performance.control_seconds, 3),
        knockdowns=performance.knockdowns,
    )


def apply_finish_provider_to_decision_path(
    outcome: FightSimulationOutcome,
    fight_id: str,
    scheduled_rounds: int,
    round_seconds: int,
    finish_provider: FinishHazardProvider,
    rng: np.random.Generator,
) -> FightSimulationOutcome:
    """Overlay conditional trained finish hazards on one complete decision path."""
    if outcome.method != "decision":
        raise HistoricalSimulatorReplayError(
            "Trained finish overlay requires a complete decision-path outcome"
        )

    selected: tuple[str, str] | None = None
    finish_round = int(scheduled_rounds)
    finish_time_seconds = float(round_seconds)

    for round_number in range(1, int(scheduled_rounds) + 1):
        probabilities = finish_provider.finish_hazards(
            FinishHazardKey(str(fight_id), int(round_number))
        ).as_array()
        class_index = int(rng.choice(len(_FINISH_LABELS), p=probabilities))
        selected = _FINISH_LABELS[class_index]
        if selected is not None:
            finish_round = int(round_number)
            time_fraction = float(rng.beta(1.5 + 0.15 * round_number, 1.35))
            finish_time_seconds = max(
                1.0,
                min(float(round_seconds), time_fraction * float(round_seconds)),
            )
            break

    if selected is None:
        return replace(
            outcome,
            finish_round=int(scheduled_rounds),
            finish_time_seconds=float(round_seconds),
            total_fight_seconds=float(scheduled_rounds * round_seconds),
        )

    winner_corner, method = selected
    total_fight_seconds = (
        (finish_round - 1) * int(round_seconds) + finish_time_seconds
    )
    full_fight_seconds = float(scheduled_rounds * round_seconds)
    exposure_fraction = float(total_fight_seconds / full_fight_seconds)
    return FightSimulationOutcome(
        winner_corner=winner_corner,
        method=method,
        finish_round=finish_round,
        finish_time_seconds=round(finish_time_seconds, 3),
        total_fight_seconds=round(total_fight_seconds, 3),
        red_rounds_won=outcome.red_rounds_won,
        blue_rounds_won=outcome.blue_rounds_won,
        red_totals=_thin_totals(rng, outcome.red_totals, exposure_fraction),
        blue_totals=_thin_totals(rng, outcome.blue_totals, exposure_fraction),
        regime=outcome.regime,
    )


def run_historical_trained_component_replay(
    training_df: pd.DataFrame,
    finish_predictions: pd.DataFrame,
    finish_model_name: str = "xgb_prefight_context",
    test_year: int = 2026,
    simulations_per_fight: int = 750,
    seed: int = 91,
    max_fights: int | None = None,
) -> HistoricalTrainedComponentReplayResult:
    """Replay a holdout with absolute strike pace and trained finish hazards."""
    if simulations_per_fight <= 0:
        raise HistoricalSimulatorReplayError("simulations_per_fight must be positive")

    history = build_fighter_fight_history(training_df)
    priors = population_priors(history, test_year=test_year)
    strike_calibration = estimate_prefight_strike_calibration(
        training_df,
        history,
        priors,
        test_year=test_year,
    )
    matchups = build_holdout_matchups(
        history,
        test_year=test_year,
        priors=priors,
        max_fights=max_fights,
    )
    baselines = _fight_level_baselines(history, test_year=test_year)
    finish_provider = HistoricalFinishHazardProvider(
        finish_predictions,
        model_name=finish_model_name,
    )

    rows: list[dict[str, object]] = []
    for index, record in enumerate(matchups):
        matchup = record["matchup"]
        strike_provider = StaticPrefightStrikeProvider(
            matchup,
            mean_calibration_factor=strike_calibration.mean_calibration_factor,
            gamma_poisson_alpha=strike_calibration.gamma_poisson_overdispersion,
        )
        mechanics_seed = int(seed + index * 9973)
        _, full_outcomes = run_simulation_with_strike_provider(
            matchup,
            strike_provider,
            SimulatorConfig(
                simulations=int(simulations_per_fight),
                seed=mechanics_seed,
                retain_outcomes=True,
                max_finish_probability_per_round=(
                    _HEURISTIC_FINISH_DISABLE_EPSILON
                ),
            ),
        )
        if full_outcomes is None or len(full_outcomes) != int(simulations_per_fight):
            raise HistoricalSimulatorReplayError(
                f"Decision-path simulator returned incomplete outcomes for {matchup.fight_id}"
            )
        if any(outcome.method != "decision" for outcome in full_outcomes):
            raise HistoricalSimulatorReplayError(
                "Heuristic finish-disable path unexpectedly produced a terminal event"
            )

        finish_rng = np.random.default_rng(mechanics_seed + 4_000_003)
        outcomes = [
            apply_finish_provider_to_decision_path(
                outcome,
                fight_id=matchup.fight_id,
                scheduled_rounds=matchup.scheduled_rounds,
                round_seconds=matchup.round_seconds,
                finish_provider=finish_provider,
                rng=finish_rng,
            )
            for outcome in full_outcomes
        ]
        summary = summarize_outcomes(
            matchup,
            SimulatorConfig(
                simulations=int(simulations_per_fight),
                seed=mechanics_seed,
                retain_outcomes=False,
            ),
            outcomes,
        )
        summary = replace(
            summary,
            simulator_version=TRAINED_COMPONENT_SIMULATOR_VERSION,
        )
        probabilities = summary.probabilities
        expectations = summary.expectations
        method_probabilities = {
            "decision": float(probabilities["goes_distance"]),
            "ko_tko": float(
                probabilities["red_by_ko_tko"]
                + probabilities["blue_by_ko_tko"]
            ),
            "submission": float(
                probabilities["red_by_submission"]
                + probabilities["blue_by_submission"]
            ),
        }
        method_total = sum(method_probabilities.values())
        if method_total <= 0:
            raise HistoricalSimulatorReplayError(
                f"Trained-component simulator returned zero method mass for {matchup.fight_id}"
            )
        method_probabilities = {
            key: value / method_total for key, value in method_probabilities.items()
        }
        baseline = baselines[int(matchup.scheduled_rounds)]
        baseline_time = float(baseline["fight_time_seconds"])

        rows.append(
            {
                "fight_id": matchup.fight_id,
                "event_id": matchup.event_id,
                "date": record["date"],
                "scheduled_rounds": matchup.scheduled_rounds,
                "red_fighter_id": matchup.red.fighter_id,
                "red_fighter_name": matchup.red.fighter_name,
                "blue_fighter_id": matchup.blue.fighter_id,
                "blue_fighter_name": matchup.blue.fighter_name,
                "red_prior_fights": record["red_prior_fights"],
                "blue_prior_fights": record["blue_prior_fights"],
                "actual_winner_corner": record["actual_winner_corner"],
                "actual_method": record["actual_method"],
                "actual_fight_time_seconds": record["actual_fight_time_seconds"],
                "actual_red_sig_attempted": record["actual_red_sig_attempted"],
                "actual_blue_sig_attempted": record["actual_blue_sig_attempted"],
                "sim_red_win_probability": float(probabilities["red_win"]),
                "sim_decision_probability": method_probabilities["decision"],
                "sim_ko_tko_probability": method_probabilities["ko_tko"],
                "sim_submission_probability": method_probabilities["submission"],
                "sim_fight_time_seconds": float(expectations["fight_time_seconds"]),
                "sim_red_sig_attempted": float(expectations["red_sig_attempted"]),
                "sim_blue_sig_attempted": float(expectations["blue_sig_attempted"]),
                "baseline_red_win_probability": float(
                    baseline["red_win_probability"]
                ),
                "baseline_decision_probability": float(
                    baseline["method_decision"]
                ),
                "baseline_ko_tko_probability": float(
                    baseline["method_ko_tko"]
                ),
                "baseline_submission_probability": float(
                    baseline["method_submission"]
                ),
                "baseline_fight_time_seconds": baseline_time,
                "baseline_red_sig_attempted": float(
                    matchup.red.sig_attempts_per_minute * baseline_time / 60.0
                ),
                "baseline_blue_sig_attempted": float(
                    matchup.blue.sig_attempts_per_minute * baseline_time / 60.0
                ),
                "strike_mean_calibration_factor": (
                    strike_calibration.mean_calibration_factor
                ),
                "strike_gamma_poisson_overdispersion": (
                    strike_calibration.gamma_poisson_overdispersion
                ),
                "finish_model_name": finish_model_name,
                "simulations": int(simulations_per_fight),
                "simulator_version": TRAINED_COMPONENT_SIMULATOR_VERSION,
            }
        )

    predictions = pd.DataFrame(rows)
    return HistoricalTrainedComponentReplayResult(
        fight_predictions=predictions,
        metrics=score_historical_replay(predictions),
        calibration=calibration_tables(predictions),
        aggregate_comparison=aggregate_comparison(predictions),
        population_priors=priors,
        strike_calibration=strike_calibration,
        finish_model_name=str(finish_model_name),
    )
