"""Historical ablation replay using absolute pre-fight strike pace.

The trained round-2+ strike rows include legitimate prior-round context. Reusing
those historical rows before a fight would leak actual target-fight rounds into a
pre-fight replay. This module instead performs the first mechanics ablation with
strictly pre-fight career pace:

1. reconstruct fighter pace from completed prior fights only;
2. estimate one mean correction and gamma-Poisson dispersion from pre-holdout
   fighter-round rows only;
3. supply that absolute rate to the provider-backed simulation path;
4. keep accuracy, wrestling, finish hazards, scoring, and every other component
   unchanged.

The result isolates whether heuristic strike-volume discounts caused the observed
activity deficit. It is shadow-only and is not the final dynamic trained provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from pipeline.simulation.contracts import MatchupSimulationInput, SimulatorConfig
from pipeline.simulation.historical_simulator_replay import (
    HistoricalSimulatorReplayError,
    _fight_level_baselines,
    aggregate_comparison,
    build_fighter_fight_history,
    build_holdout_matchups,
    calibration_tables,
    fighter_state_from_history,
    population_priors,
    score_historical_replay,
)
from pipeline.simulation.provider_engine import (
    PROVIDER_SIMULATOR_VERSION,
    run_simulation_with_strike_provider,
)
from pipeline.simulation.round_parameter_provider import (
    RoundParameterKey,
    RoundParameterProviderError,
    SignificantStrikeAttemptParameters,
)
from pipeline.simulation.sig_attempt_calibration import (
    gamma_poisson_overdispersion,
    multiplicative_mean_factor,
)


@dataclass(frozen=True)
class PrefightStrikeCalibration:
    test_year: int
    rows: int
    fights: int
    raw_predicted_mean: float
    actual_mean: float
    mean_calibration_factor: float
    gamma_poisson_overdispersion: float


@dataclass(frozen=True)
class HistoricalStrikeProviderReplayResult:
    fight_predictions: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    aggregate_comparison: pd.DataFrame
    population_priors: Mapping[str, float]
    strike_calibration: PrefightStrikeCalibration


class StaticPrefightStrikeProvider:
    """Expose one calibrated pre-fight pace for every scheduled fighter-round."""

    def __init__(
        self,
        matchup: MatchupSimulationInput,
        mean_calibration_factor: float,
        gamma_poisson_alpha: float,
    ) -> None:
        if mean_calibration_factor <= 0 or not np.isfinite(mean_calibration_factor):
            raise RoundParameterProviderError(
                "mean_calibration_factor must be finite and positive"
            )
        if gamma_poisson_alpha <= 0 or not np.isfinite(gamma_poisson_alpha):
            raise RoundParameterProviderError(
                "gamma_poisson_alpha must be finite and positive"
            )
        self.matchup = matchup
        self.factor = float(mean_calibration_factor)
        self.alpha = float(gamma_poisson_alpha)
        self._rates = {
            str(matchup.red.fighter_id): float(matchup.red.sig_attempts_per_minute),
            str(matchup.blue.fighter_id): float(matchup.blue.sig_attempts_per_minute),
        }

    def significant_strike_attempts(
        self,
        key: RoundParameterKey,
    ) -> SignificantStrikeAttemptParameters:
        if str(key.fight_id) != str(self.matchup.fight_id):
            raise RoundParameterProviderError(
                f"Provider fight mismatch: {key.fight_id!r}"
            )
        if int(key.round) > int(self.matchup.scheduled_rounds):
            raise RoundParameterProviderError(
                f"Round {key.round} exceeds scheduled rounds"
            )
        try:
            raw_rate = self._rates[str(key.fighter_id)]
        except KeyError as exc:
            raise RoundParameterProviderError(
                f"Unknown fighter for provider: {key.fighter_id!r}"
            ) from exc
        return SignificantStrikeAttemptParameters(
            key=key,
            mean_rate_per_minute=float(raw_rate * self.factor),
            gamma_poisson_overdispersion=self.alpha,
            model_name="prefight_career_pace_absolute",
            model_version="prefight_career_pace_ablation_v0",
            calibration_factor=self.factor,
            source="pre_holdout_career_pace_calibration",
        )


def estimate_prefight_strike_calibration(
    training_df: pd.DataFrame,
    history_df: pd.DataFrame,
    priors: Mapping[str, float],
    test_year: int,
) -> PrefightStrikeCalibration:
    """Estimate absolute-rate correction using only rows before the holdout."""
    required = (
        "fight_id",
        "fighter_id",
        "date",
        "target_sig_attempted",
        "target_finish_time_in_round_seconds",
    )
    missing = [column for column in required if column not in training_df.columns]
    if missing:
        raise HistoricalSimulatorReplayError(
            f"Strike calibration table is missing columns: {missing}"
        )

    rounds = training_df.copy()
    rounds["date"] = pd.to_datetime(rounds["date"], errors="coerce")
    rounds = rounds.loc[rounds["date"].dt.year.lt(int(test_year))].copy()
    if rounds.empty:
        raise HistoricalSimulatorReplayError(
            f"No pre-{test_year} rows are available for strike calibration"
        )

    pretest_history = history_df.loc[
        history_df["date"].dt.year.lt(int(test_year))
    ].copy()
    pace_rows: list[dict[str, object]] = []
    for _, row in pretest_history.iterrows():
        state = fighter_state_from_history(row, priors)
        pace_rows.append(
            {
                "fight_id": str(row["fight_id"]),
                "fighter_id": str(row["fighter_id"]),
                "prefight_rate_per_min": float(state.sig_attempts_per_minute),
            }
        )
    pace = pd.DataFrame(pace_rows)
    if pace.empty or pace.duplicated(["fight_id", "fighter_id"]).any():
        raise HistoricalSimulatorReplayError(
            "Pre-holdout fighter pace mapping is empty or duplicated"
        )

    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)
    joined = rounds.merge(
        pace,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != len(rounds):
        raise HistoricalSimulatorReplayError(
            "Not every pre-holdout round resolved a pre-fight pace state"
        )

    actual = pd.to_numeric(
        joined["target_sig_attempted"], errors="coerce"
    ).to_numpy(dtype=float)
    exposure = pd.to_numeric(
        joined["target_finish_time_in_round_seconds"], errors="coerce"
    ).to_numpy(dtype=float)
    raw_mean = (
        joined["prefight_rate_per_min"].to_numpy(dtype=float) * exposure / 60.0
    )
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(raw_mean).all()
        or np.any(actual < 0)
        or np.any(raw_mean <= 0)
    ):
        raise HistoricalSimulatorReplayError(
            "Pre-holdout strike calibration contains invalid values"
        )

    factor = multiplicative_mean_factor(actual, raw_mean)
    calibrated_mean = raw_mean * factor
    alpha = gamma_poisson_overdispersion(actual, calibrated_mean)
    return PrefightStrikeCalibration(
        test_year=int(test_year),
        rows=int(len(joined)),
        fights=int(joined["fight_id"].nunique()),
        raw_predicted_mean=float(raw_mean.mean()),
        actual_mean=float(actual.mean()),
        mean_calibration_factor=float(factor),
        gamma_poisson_overdispersion=float(alpha),
    )


def run_historical_strike_provider_replay(
    training_df: pd.DataFrame,
    test_year: int = 2026,
    simulations_per_fight: int = 750,
    seed: int = 91,
    max_fights: int | None = None,
) -> HistoricalStrikeProviderReplayResult:
    """Replay the holdout with only strike-attempt mechanics replaced."""
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

    rows: list[dict[str, object]] = []
    for index, record in enumerate(matchups):
        matchup = record["matchup"]
        provider = StaticPrefightStrikeProvider(
            matchup,
            mean_calibration_factor=strike_calibration.mean_calibration_factor,
            gamma_poisson_alpha=strike_calibration.gamma_poisson_overdispersion,
        )
        summary, _ = run_simulation_with_strike_provider(
            matchup,
            provider,
            SimulatorConfig(
                simulations=int(simulations_per_fight),
                seed=int(seed + index * 9973),
                retain_outcomes=False,
            ),
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
                f"Provider simulator returned zero method mass for {matchup.fight_id}"
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
                "simulations": int(simulations_per_fight),
                "simulator_version": PROVIDER_SIMULATOR_VERSION,
            }
        )

    predictions = pd.DataFrame(rows)
    return HistoricalStrikeProviderReplayResult(
        fight_predictions=predictions,
        metrics=score_historical_replay(predictions),
        calibration=calibration_tables(predictions),
        aggregate_comparison=aggregate_comparison(predictions),
        population_priors=priors,
        strike_calibration=strike_calibration,
    )
