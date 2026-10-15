"""Historical replay for round-survival calibrated finish hazards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from pipeline.simulation.component_provider_engine import (
    run_simulation_with_component_providers,
)
from pipeline.simulation.contracts import SimulatorConfig
from pipeline.simulation.dynamic_strike_provider import DynamicPrefightStrikeProvider
from pipeline.simulation.engine import run_simulation
from pipeline.simulation.finish_hazard_holdout import (
    CounterfactualFinishHazardResult,
    build_counterfactual_finish_predictions,
)
from pipeline.simulation.finish_hazard_provider import HistoricalFinishHazardProvider
from pipeline.simulation.finish_survival_calibration import (
    FinishSurvivalCalibrationResult,
    apply_finish_survival_schedule,
    fit_finish_survival_schedule,
)
from pipeline.simulation.historical_component_provider_replay import (
    _combined_aggregate,
    _combined_metrics,
    _prediction_row,
)
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
    _fight_level_baselines,
    build_fighter_fight_history,
    build_holdout_matchups,
    population_priors,
)
from pipeline.simulation.historical_strike_provider_replay import (
    PrefightStrikeCalibration,
    StaticPrefightStrikeProvider,
    estimate_prefight_strike_calibration,
)


SURVIVAL_VARIANTS = (
    "heuristic_simulator",
    "class_finish_hazard_provider",
    "survival_finish_hazard_provider",
    "strike_and_survival_finish_providers",
    "dynamic_strike_and_survival_finish_providers",
)


@dataclass(frozen=True)
class HistoricalSurvivalReplayResult:
    fight_predictions: Mapping[str, pd.DataFrame]
    metrics: pd.DataFrame
    aggregate_comparison: pd.DataFrame
    strike_calibration: PrefightStrikeCalibration
    class_finish_predictions: CounterfactualFinishHazardResult
    survival_finish_predictions: FinishSurvivalCalibrationResult
    population_priors: Mapping[str, float]


def run_historical_survival_provider_replay(
    training_df: pd.DataFrame,
    finish_class_calibration_schedule: pd.DataFrame,
    finish_walk_forward_predictions: pd.DataFrame,
    test_year: int = 2026,
    simulations_per_fight: int = 500,
    seed: int = 91,
    max_fights: int | None = None,
    finish_model_name: str = "xgb_prefight_context",
    group_prior_rows: float = 200.0,
) -> HistoricalSurvivalReplayResult:
    """Compare class, survival, static-strike, and dynamic-strike paths."""
    if simulations_per_fight <= 0:
        raise HistoricalSimulatorReplayError("simulations_per_fight must be positive")

    history = build_fighter_fight_history(training_df)
    priors = population_priors(history, test_year=test_year)
    matchups = build_holdout_matchups(
        history,
        test_year=test_year,
        priors=priors,
        max_fights=max_fights,
    )
    baselines = _fight_level_baselines(history, test_year=test_year)
    strike_calibration = estimate_prefight_strike_calibration(
        training_df,
        history,
        priors,
        test_year=test_year,
    )

    class_predictions = build_counterfactual_finish_predictions(
        training_df,
        finish_class_calibration_schedule,
        test_year=test_year,
        model_name=finish_model_name,
    )
    survival_schedule = fit_finish_survival_schedule(
        finish_walk_forward_predictions,
        model_name=finish_model_name,
        target_year=test_year,
        group_prior_rows=group_prior_rows,
    )
    survival_predictions = apply_finish_survival_schedule(
        class_predictions.predictions,
        survival_schedule,
    )

    class_provider = HistoricalFinishHazardProvider(
        class_predictions.predictions,
        model_name=finish_model_name,
    )
    survival_provider = HistoricalFinishHazardProvider(
        survival_predictions.predictions,
        model_name=finish_model_name,
        model_version="finish_hazard_prefight_survival_v0",
    )

    rows: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in SURVIVAL_VARIANTS
    }
    for index, record in enumerate(matchups):
        matchup = record["matchup"]
        static_strike_provider = StaticPrefightStrikeProvider(
            matchup,
            mean_calibration_factor=strike_calibration.mean_calibration_factor,
            gamma_poisson_alpha=strike_calibration.gamma_poisson_overdispersion,
        )
        dynamic_strike_provider = DynamicPrefightStrikeProvider(
            matchup,
            mean_calibration_factor=strike_calibration.mean_calibration_factor,
            gamma_poisson_alpha=strike_calibration.gamma_poisson_overdispersion,
        )
        runtime = SimulatorConfig(
            simulations=int(simulations_per_fight),
            seed=int(seed + index * 9973),
            retain_outcomes=False,
        )
        summaries = {
            "heuristic_simulator": run_simulation(matchup, runtime)[0],
            "class_finish_hazard_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                finish_provider=class_provider,
            )[0],
            "survival_finish_hazard_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                finish_provider=survival_provider,
            )[0],
            "strike_and_survival_finish_providers": run_simulation_with_component_providers(
                matchup,
                runtime,
                strike_provider=static_strike_provider,
                finish_provider=survival_provider,
            )[0],
            "dynamic_strike_and_survival_finish_providers": run_simulation_with_component_providers(
                matchup,
                runtime,
                strike_provider=dynamic_strike_provider,
                finish_provider=survival_provider,
            )[0],
        }
        baseline = baselines[int(matchup.scheduled_rounds)]
        for variant, summary in summaries.items():
            rows[variant].append(
                _prediction_row(
                    matchup,
                    record,
                    summary,
                    baseline,
                    simulations_per_fight=simulations_per_fight,
                )
            )

    predictions = {
        variant: pd.DataFrame(variant_rows)
        for variant, variant_rows in rows.items()
    }
    return HistoricalSurvivalReplayResult(
        fight_predictions=predictions,
        metrics=_combined_metrics(predictions),
        aggregate_comparison=_combined_aggregate(predictions),
        strike_calibration=strike_calibration,
        class_finish_predictions=class_predictions,
        survival_finish_predictions=survival_predictions,
        population_priors=priors,
    )
