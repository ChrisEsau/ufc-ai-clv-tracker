"""Historical replay across heuristic and learned-component simulator variants."""

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
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
    _fight_level_baselines,
    aggregate_comparison,
    build_fighter_fight_history,
    build_holdout_matchups,
    population_priors,
    score_historical_replay,
)
from pipeline.simulation.historical_strike_provider_replay import (
    PrefightStrikeCalibration,
    StaticPrefightStrikeProvider,
    estimate_prefight_strike_calibration,
)


VARIANTS = (
    "heuristic_simulator",
    "absolute_strike_provider",
    "finish_hazard_provider",
    "strike_and_finish_providers",
)


@dataclass(frozen=True)
class HistoricalComponentReplayResult:
    fight_predictions: Mapping[str, pd.DataFrame]
    metrics: pd.DataFrame
    aggregate_comparison: pd.DataFrame
    strike_calibration: PrefightStrikeCalibration
    finish_predictions: CounterfactualFinishHazardResult
    population_priors: Mapping[str, float]


def _method_probabilities(summary) -> dict[str, float]:
    probabilities = summary.probabilities
    values = {
        "decision": float(probabilities["goes_distance"]),
        "ko_tko": float(
            probabilities["red_by_ko_tko"] + probabilities["blue_by_ko_tko"]
        ),
        "submission": float(
            probabilities["red_by_submission"]
            + probabilities["blue_by_submission"]
        ),
    }
    total = sum(values.values())
    if total <= 0:
        raise HistoricalSimulatorReplayError("Simulator returned zero method mass")
    return {key: value / total for key, value in values.items()}


def _prediction_row(
    matchup,
    record: Mapping[str, object],
    summary,
    baseline: Mapping[str, float],
    simulations_per_fight: int,
) -> dict[str, object]:
    method = _method_probabilities(summary)
    expectations = summary.expectations
    baseline_time = float(baseline["fight_time_seconds"])
    return {
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
        "sim_red_win_probability": float(summary.probabilities["red_win"]),
        "sim_decision_probability": method["decision"],
        "sim_ko_tko_probability": method["ko_tko"],
        "sim_submission_probability": method["submission"],
        "sim_fight_time_seconds": float(expectations["fight_time_seconds"]),
        "sim_red_sig_attempted": float(expectations["red_sig_attempted"]),
        "sim_blue_sig_attempted": float(expectations["blue_sig_attempted"]),
        "baseline_red_win_probability": float(baseline["red_win_probability"]),
        "baseline_decision_probability": float(baseline["method_decision"]),
        "baseline_ko_tko_probability": float(baseline["method_ko_tko"]),
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
        "simulations": int(simulations_per_fight),
        "simulator_version": summary.simulator_version,
    }


def _combined_metrics(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant, frame in predictions.items():
        metrics = score_historical_replay(frame)
        simulator = metrics.loc[metrics["model"].eq("simulator")].copy()
        simulator["model"] = variant
        frames.append(simulator)
    first = predictions[VARIANTS[0]]
    baseline = score_historical_replay(first)
    frames.append(baseline.loc[baseline["model"].eq("historical_baseline")].copy())
    return pd.concat(frames, ignore_index=True).sort_values(
        ["task", "metric", "model"]
    ).reset_index(drop=True)


def _combined_aggregate(predictions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    result: pd.DataFrame | None = None
    for variant, frame in predictions.items():
        aggregate = aggregate_comparison(frame)[
            ["quantity", "actual", "simulator"]
        ].rename(columns={"simulator": variant})
        if result is None:
            result = aggregate
        else:
            result = result.merge(
                aggregate[["quantity", variant]],
                on="quantity",
                how="inner",
                validate="one_to_one",
            )
    if result is None:
        raise HistoricalSimulatorReplayError("No aggregate replay frames were built")
    return result


def run_historical_component_provider_replay(
    training_df: pd.DataFrame,
    finish_calibration_schedule: pd.DataFrame,
    test_year: int = 2026,
    simulations_per_fight: int = 500,
    seed: int = 91,
    max_fights: int | None = None,
    finish_model_name: str = "xgb_prefight_context",
) -> HistoricalComponentReplayResult:
    """Compare heuristic, strike-only, finish-only, and combined variants."""
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
    finish_predictions = build_counterfactual_finish_predictions(
        training_df,
        finish_calibration_schedule,
        test_year=test_year,
        model_name=finish_model_name,
    )
    finish_provider = HistoricalFinishHazardProvider(
        finish_predictions.predictions,
        model_name=finish_model_name,
    )

    rows: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in VARIANTS
    }
    for index, record in enumerate(matchups):
        matchup = record["matchup"]
        strike_provider = StaticPrefightStrikeProvider(
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
            "absolute_strike_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                strike_provider=strike_provider,
            )[0],
            "finish_hazard_provider": run_simulation_with_component_providers(
                matchup,
                runtime,
                finish_provider=finish_provider,
            )[0],
            "strike_and_finish_providers": run_simulation_with_component_providers(
                matchup,
                runtime,
                strike_provider=strike_provider,
                finish_provider=finish_provider,
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
    return HistoricalComponentReplayResult(
        fight_predictions=predictions,
        metrics=_combined_metrics(predictions),
        aggregate_comparison=_combined_aggregate(predictions),
        strike_calibration=strike_calibration,
        finish_predictions=finish_predictions,
        population_priors=priors,
    )
