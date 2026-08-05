"""Historical simulator replay comparing flat and hierarchical finish providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from pipeline.simulation.component_provider_engine import (
    run_simulation_with_component_providers,
)
from pipeline.simulation.contracts import SimulatorConfig
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
from pipeline.simulation.hierarchical_finish_hazard_holdout import (
    HierarchicalCounterfactualFinishResult,
    build_hierarchical_counterfactual_finish_predictions,
)
from pipeline.simulation.hierarchical_finish_hazard_model import HIERARCHICAL_MODELS
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


HIERARCHICAL_REPLAY_VARIANTS = (
    "heuristic_simulator",
    "survival_finish_hazard_provider",
    "hierarchical_class_finish_hazard_provider",
    "hierarchical_survival_finish_hazard_provider",
)


@dataclass(frozen=True)
class HistoricalHierarchicalFinishReplayResult:
    fight_predictions: Mapping[str, pd.DataFrame]
    metrics: pd.DataFrame
    aggregate_comparison: pd.DataFrame
    current_class_predictions: CounterfactualFinishHazardResult
    current_survival_predictions: FinishSurvivalCalibrationResult
    hierarchical_class_predictions: HierarchicalCounterfactualFinishResult
    hierarchical_survival_predictions: FinishSurvivalCalibrationResult
    population_priors: Mapping[str, float]


def run_historical_hierarchical_finish_replay(
    training_df: pd.DataFrame,
    current_class_calibration_schedule: pd.DataFrame,
    current_walk_forward_predictions: pd.DataFrame,
    hierarchical_calibration_schedule: pd.DataFrame,
    hierarchical_walk_forward_predictions: pd.DataFrame,
    test_year: int = 2026,
    simulations_per_fight: int = 500,
    seed: int = 91,
    max_fights: int | None = None,
    current_model_name: str = "xgb_prefight_context",
    hierarchical_model_name: str = HIERARCHICAL_MODELS[0],
    group_prior_rows: float = 200.0,
) -> HistoricalHierarchicalFinishReplayResult:
    """Compare current survival hazards with hierarchical class and survival paths."""
    if simulations_per_fight <= 0:
        raise HistoricalSimulatorReplayError("simulations_per_fight must be positive")
    if hierarchical_model_name not in HIERARCHICAL_MODELS:
        raise HistoricalSimulatorReplayError(
            f"Unsupported hierarchical model: {hierarchical_model_name!r}"
        )

    history = build_fighter_fight_history(training_df)
    priors = population_priors(history, test_year=test_year)
    matchups = build_holdout_matchups(
        history,
        test_year=test_year,
        priors=priors,
        max_fights=max_fights,
    )
    baselines = _fight_level_baselines(history, test_year=test_year)

    current_class = build_counterfactual_finish_predictions(
        training_df,
        current_class_calibration_schedule,
        test_year=test_year,
        model_name=current_model_name,
    )
    current_survival_schedule = fit_finish_survival_schedule(
        current_walk_forward_predictions,
        model_name=current_model_name,
        target_year=test_year,
        group_prior_rows=group_prior_rows,
    )
    current_survival = apply_finish_survival_schedule(
        current_class.predictions,
        current_survival_schedule,
    )

    hierarchical_class = build_hierarchical_counterfactual_finish_predictions(
        training_df,
        hierarchical_calibration_schedule,
        test_year=test_year,
        model_name=hierarchical_model_name,
    )
    hierarchical_survival_schedule = fit_finish_survival_schedule(
        hierarchical_walk_forward_predictions,
        model_name=hierarchical_model_name,
        target_year=test_year,
        group_prior_rows=group_prior_rows,
    )
    hierarchical_survival = apply_finish_survival_schedule(
        hierarchical_class.predictions,
        hierarchical_survival_schedule,
    )

    current_provider = HistoricalFinishHazardProvider(
        current_survival.predictions,
        model_name=current_model_name,
        model_version="finish_hazard_prefight_survival_v0",
    )
    hierarchical_class_provider = HistoricalFinishHazardProvider(
        hierarchical_class.predictions,
        model_name=hierarchical_model_name,
        model_version="finish_hazard_hierarchical_prefight_v0",
    )
    hierarchical_survival_provider = HistoricalFinishHazardProvider(
        hierarchical_survival.predictions,
        model_name=hierarchical_model_name,
        model_version="finish_hazard_hierarchical_survival_v0",
    )

    rows: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in HIERARCHICAL_REPLAY_VARIANTS
    }
    for index, record in enumerate(matchups):
        matchup = record["matchup"]
        runtime = SimulatorConfig(
            simulations=int(simulations_per_fight),
            seed=int(seed + index * 9973),
            retain_outcomes=False,
        )
        summaries = {
            "heuristic_simulator": run_simulation(matchup, runtime)[0],
            "survival_finish_hazard_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                finish_provider=current_provider,
            )[0],
            "hierarchical_class_finish_hazard_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                finish_provider=hierarchical_class_provider,
            )[0],
            "hierarchical_survival_finish_hazard_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                finish_provider=hierarchical_survival_provider,
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
    return HistoricalHierarchicalFinishReplayResult(
        fight_predictions=predictions,
        metrics=_combined_metrics(predictions),
        aggregate_comparison=_combined_aggregate(predictions),
        current_class_predictions=current_class,
        current_survival_predictions=current_survival,
        hierarchical_class_predictions=hierarchical_class,
        hierarchical_survival_predictions=hierarchical_survival,
        population_priors=priors,
    )
