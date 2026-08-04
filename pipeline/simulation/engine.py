"""Round-level Monte Carlo engine for UFC fight simulation.

This V0 engine is a mechanics and data-contract foundation. Its coefficients are
explicit heuristics and must not be treated as calibrated betting probabilities.
A later parameter model will be trained and validated against historical rounds,
closing prices, calibration, ROI, and CLV before production promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Iterable

import numpy as np

from pipeline.simulation.contracts import (
    FightSimulationOutcome,
    FighterFightTotals,
    FighterSimulationState,
    MatchupSimulationInput,
    SimulationSummary,
    SimulatorConfig,
)
from pipeline.simulation.terminal_round import (
    RoundPerformance,
    thin_round_performance,
)


SIMULATOR_VERSION = "round_simulator_v0_1"
REGIMES = (
    "distance_tactical",
    "high_volume",
    "chaos",
    "red_wrestling",
    "blue_wrestling",
    "mixed",
    "low_activity",
)


@dataclass
class _DynamicState:
    fatigue: float = 0.0
    damage: float = 0.0
    confidence: float = 0.0
    sig_attempted: int = 0
    sig_landed: int = 0
    takedowns_attempted: int = 0
    takedowns_landed: int = 0
    control_seconds: float = 0.0
    knockdowns: int = 0
    rounds_won: int = 0
    score_total: float = 0.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _gamma_poisson(rng: np.random.Generator, mean: float, overdispersion: float) -> int:
    """Sample an overdispersed nonnegative count using a gamma-Poisson mixture."""
    if mean <= 0:
        return 0
    if overdispersion <= 0:
        return int(rng.poisson(mean))
    shape = 1.0 / overdispersion
    scale = mean * overdispersion
    mixed_rate = float(rng.gamma(shape=shape, scale=scale))
    return int(rng.poisson(mixed_rate))


def _regime_probabilities(red: FighterSimulationState, blue: FighterSimulationState) -> np.ndarray:
    red_wrestle = (
        0.40 * red.td_attempts_per_15 / 10.0
        + 0.30 * red.phase_imposition
        + 0.30 * red.control_seconds_per_takedown / 180.0
    )
    blue_wrestle = (
        0.40 * blue.td_attempts_per_15 / 10.0
        + 0.30 * blue.phase_imposition
        + 0.30 * blue.control_seconds_per_takedown / 180.0
    )
    combined_pace = (red.sig_attempts_per_minute + blue.sig_attempts_per_minute) / 2.0
    combined_power = (red.power + blue.power) / 2.0
    combined_initiative = (red.initiative + blue.initiative) / 2.0

    weights = np.array(
        [
            1.3 + 0.12 * combined_pace + 0.5 * (1.0 - combined_initiative),
            0.7 + 0.22 * combined_pace + 0.4 * combined_initiative,
            0.35 + 1.3 * combined_power + 0.06 * combined_pace,
            0.25 + 2.0 * red_wrestle * (1.15 - blue.td_defense),
            0.25 + 2.0 * blue_wrestle * (1.15 - red.td_defense),
            1.0 + 0.7 * (red.adaptability + blue.adaptability),
            0.35 + 1.0 * max(0.0, 5.2 - combined_pace) / 5.2,
        ],
        dtype=float,
    )
    weights = np.clip(weights, 0.01, None)
    return weights / weights.sum()


def _sample_phase_shares(
    rng: np.random.Generator,
    regime: str,
    red: FighterSimulationState,
    blue: FighterSimulationState,
) -> tuple[float, float, float]:
    """Return distance, clinch, and ground shares that sum to one."""
    base = {
        "distance_tactical": np.array([0.78, 0.12, 0.10]),
        "high_volume": np.array([0.84, 0.09, 0.07]),
        "chaos": np.array([0.72, 0.15, 0.13]),
        "red_wrestling": np.array([0.43, 0.18, 0.39]),
        "blue_wrestling": np.array([0.43, 0.18, 0.39]),
        "mixed": np.array([0.58, 0.17, 0.25]),
        "low_activity": np.array([0.74, 0.16, 0.10]),
    }[regime].copy()

    imposition = abs(red.phase_imposition - blue.phase_imposition)
    if regime in ("red_wrestling", "blue_wrestling"):
        base[2] += 0.08 * imposition
        base[0] -= 0.06 * imposition
        base[1] -= 0.02 * imposition

    concentration = np.clip(base * 26.0, 0.25, None)
    shares = rng.dirichlet(concentration)
    return float(shares[0]), float(shares[1]), float(shares[2])


def _pace_multiplier(regime: str) -> float:
    return {
        "distance_tactical": 0.88,
        "high_volume": 1.22,
        "chaos": 1.12,
        "red_wrestling": 0.72,
        "blue_wrestling": 0.72,
        "mixed": 0.91,
        "low_activity": 0.66,
    }[regime]


def _fatigue_multiplier(fighter: FighterSimulationState, dynamic: _DynamicState) -> float:
    resilience = 0.50 * fighter.cardio + 0.35 * fighter.pace_sustainability + 0.15 * fighter.recovery
    return _clamp(1.0 - dynamic.fatigue * (0.78 - 0.48 * resilience), 0.38, 1.10)


def _initiative_share(red: FighterSimulationState, blue: FighterSimulationState) -> float:
    red_value = 0.60 * red.initiative + 0.40 * red.phase_imposition
    blue_value = 0.60 * blue.initiative + 0.40 * blue.phase_imposition
    return _sigmoid((red_value - blue_value) * 1.6)


def _strike_round(
    rng: np.random.Generator,
    fighter: FighterSimulationState,
    opponent: FighterSimulationState,
    dynamic: _DynamicState,
    opponent_dynamic: _DynamicState,
    effective_minutes: float,
    regime: str,
    initiative_multiplier: float,
    overdispersion: float,
) -> tuple[int, int]:
    fatigue_mult = _fatigue_multiplier(fighter, dynamic)
    damage_mult = _clamp(1.0 - 0.42 * dynamic.damage, 0.52, 1.0)
    confidence_mult = _clamp(1.0 + 0.12 * dynamic.confidence, 0.86, 1.14)
    suppression_mult = _clamp(1.08 - 0.28 * opponent.phase_imposition, 0.74, 1.08)

    expected_attempts = (
        fighter.sig_attempts_per_minute
        * effective_minutes
        * _pace_multiplier(regime)
        * initiative_multiplier
        * fatigue_mult
        * damage_mult
        * confidence_mult
        * suppression_mult
    )
    attempts = _gamma_poisson(rng, expected_attempts, overdispersion)

    defense_effect = 0.52 + 0.48 * (1.0 - opponent.sig_defense)
    fatigue_accuracy = 1.0 - 0.12 * dynamic.fatigue
    opponent_damage_opening = 1.0 + 0.08 * opponent_dynamic.damage
    accuracy = _clamp(
        fighter.sig_accuracy * defense_effect * fatigue_accuracy * opponent_damage_opening,
        0.08,
        0.78,
    )
    landed = int(rng.binomial(attempts, accuracy)) if attempts else 0
    return attempts, landed


def _takedown_round(
    rng: np.random.Generator,
    fighter: FighterSimulationState,
    opponent: FighterSimulationState,
    dynamic: _DynamicState,
    opponent_dynamic: _DynamicState,
    regime: str,
    initiative_multiplier: float,
    ground_share: float,
    round_seconds: int,
    overdispersion: float,
) -> tuple[int, int, float]:
    regime_mult = {
        "distance_tactical": 0.62,
        "high_volume": 0.54,
        "chaos": 0.72,
        "red_wrestling": 1.38,
        "blue_wrestling": 1.38,
        "mixed": 1.02,
        "low_activity": 0.70,
    }[regime]
    fatigue_mult = _fatigue_multiplier(fighter, dynamic)
    expected_attempts = (
        fighter.td_attempts_per_15
        * (round_seconds / 900.0)
        * regime_mult
        * initiative_multiplier
        * fatigue_mult
    )
    attempts = _gamma_poisson(rng, expected_attempts, overdispersion)

    defense_effect = 0.42 + 0.58 * (1.0 - opponent.td_defense)
    fatigue_opening = 1.0 + 0.22 * opponent_dynamic.fatigue
    success_p = _clamp(fighter.td_accuracy * defense_effect * fatigue_opening, 0.03, 0.82)
    landed = int(rng.binomial(attempts, success_p)) if attempts else 0

    available_seconds = max(0.0, ground_share * round_seconds)
    if landed == 0 or available_seconds <= 0:
        return attempts, landed, 0.0

    mean_control = fighter.control_seconds_per_takedown
    shape = max(0.8, landed * 1.4)
    sampled_control = float(rng.gamma(shape=shape, scale=max(1.0, mean_control / shape)))
    control = min(available_seconds, sampled_control)
    return attempts, landed, control


def _knockdowns(
    rng: np.random.Generator,
    attacker: FighterSimulationState,
    defender: FighterSimulationState,
    attacker_dynamic: _DynamicState,
    defender_dynamic: _DynamicState,
    sig_landed: int,
    regime: str,
) -> int:
    chaos_mult = 1.45 if regime == "chaos" else (1.12 if regime == "high_volume" else 1.0)
    effective_power = 0.35 + 1.25 * attacker.power
    vulnerability = 0.35 + 0.75 * (1.0 - defender.durability)
    accumulated = 1.0 + 0.85 * defender_dynamic.damage + 0.25 * defender_dynamic.fatigue
    tired_power = 1.0 - 0.22 * attacker_dynamic.fatigue
    expected = sig_landed * 0.012 * effective_power * vulnerability * accumulated * tired_power * chaos_mult
    return min(3, int(rng.poisson(max(0.0, expected))))


def _finish_hazards(
    attacker: FighterSimulationState,
    defender: FighterSimulationState,
    attacker_dynamic: _DynamicState,
    defender_dynamic: _DynamicState,
    sig_landed: int,
    knockdowns: int,
    takedowns_landed: int,
    control_seconds: float,
    regime: str,
) -> tuple[float, float]:
    chaos = 0.28 if regime == "chaos" else 0.0
    ko_logit = (
        -5.15
        + 1.45 * attacker.power
        + 1.25 * (1.0 - defender.durability)
        + 1.70 * defender_dynamic.damage
        + 0.24 * sig_landed
        + 1.35 * knockdowns
        + 0.45 * defender_dynamic.fatigue
        + chaos
    )
    ko_hazard = _sigmoid(ko_logit)

    control_minutes = control_seconds / 60.0
    sub_logit = (
        -5.35
        + 2.05 * attacker.submission_threat
        + 1.55 * (1.0 - defender.submission_defense)
        + 0.62 * takedowns_landed
        + 0.36 * control_minutes
        + 0.55 * defender_dynamic.fatigue
        + 0.35 * defender_dynamic.damage
    )
    sub_hazard = _sigmoid(sub_logit)
    return ko_hazard, sub_hazard


def _sample_finish(
    rng: np.random.Generator,
    hazards: dict[tuple[str, str], float],
    max_probability: float,
) -> tuple[str, str] | None:
    labels = list(hazards)
    rates = np.array([max(0.0, hazards[label]) for label in labels], dtype=float)
    total_rate = float(rates.sum())
    if total_rate <= 0:
        return None

    finish_probability = min(max_probability, 1.0 - exp(-total_rate))
    if float(rng.random()) >= finish_probability:
        return None

    probabilities = rates / total_rate
    selected = int(rng.choice(len(labels), p=probabilities))
    return labels[selected]


def _round_score(
    sig_landed: int,
    knockdowns: int,
    takedowns_landed: int,
    control_seconds: float,
    damage_created: float,
) -> float:
    return (
        1.00 * sig_landed
        + 13.0 * knockdowns
        + 2.4 * takedowns_landed
        + 0.018 * control_seconds
        + 18.0 * damage_created
    )


def _update_dynamic_state(
    fighter: FighterSimulationState,
    dynamic: _DynamicState,
    attempts: int,
    td_attempts: int,
    control_seconds: float,
    damage_received: float,
    round_won: bool,
) -> None:
    workload = 0.0062 * attempts + 0.025 * td_attempts + 0.00055 * control_seconds
    cardio_buffer = 0.36 * fighter.cardio + 0.34 * fighter.pace_sustainability
    fatigue_gain = workload * (1.18 - cardio_buffer)
    between_round_recovery = 0.10 + 0.22 * fighter.recovery

    dynamic.fatigue = _clamp(dynamic.fatigue + fatigue_gain - between_round_recovery, 0.0, 1.0)
    dynamic.damage = _clamp(dynamic.damage + damage_received * (1.10 - 0.38 * fighter.recovery), 0.0, 1.0)
    confidence_change = 0.16 if round_won else -0.12
    confidence_change *= 0.60 + 0.40 * fighter.adaptability
    dynamic.confidence = _clamp(dynamic.confidence + confidence_change, -1.0, 1.0)


def _totals(dynamic: _DynamicState) -> FighterFightTotals:
    return FighterFightTotals(
        sig_attempted=dynamic.sig_attempted,
        sig_landed=dynamic.sig_landed,
        takedowns_attempted=dynamic.takedowns_attempted,
        takedowns_landed=dynamic.takedowns_landed,
        control_seconds=round(dynamic.control_seconds, 3),
        knockdowns=dynamic.knockdowns,
    )


def simulate_fight(
    matchup: MatchupSimulationInput,
    rng: np.random.Generator,
    config: SimulatorConfig,
) -> FightSimulationOutcome:
    """Simulate one complete fight path."""
    red = matchup.red
    blue = matchup.blue
    red_dynamic = _DynamicState()
    blue_dynamic = _DynamicState()

    regime_probs = _regime_probabilities(red, blue)
    regime = str(rng.choice(REGIMES, p=regime_probs))
    initiative_red = _initiative_share(red, blue)

    winner_corner: str | None = None
    method = "decision"
    finish_round = matchup.scheduled_rounds
    finish_time_seconds = float(matchup.round_seconds)
    total_fight_seconds = float(matchup.scheduled_rounds * matchup.round_seconds)

    for round_number in range(1, matchup.scheduled_rounds + 1):
        distance_share, clinch_share, ground_share = _sample_phase_shares(rng, regime, red, blue)
        active_share = _clamp(distance_share + 0.65 * clinch_share + 0.35 * ground_share, 0.25, 1.0)
        effective_minutes = matchup.round_seconds / 60.0 * active_share

        red_initiative_mult = 0.78 + 0.44 * initiative_red
        blue_initiative_mult = 0.78 + 0.44 * (1.0 - initiative_red)

        red_attempts, red_landed = _strike_round(
            rng,
            red,
            blue,
            red_dynamic,
            blue_dynamic,
            effective_minutes,
            regime,
            red_initiative_mult,
            config.strike_overdispersion,
        )
        blue_attempts, blue_landed = _strike_round(
            rng,
            blue,
            red,
            blue_dynamic,
            red_dynamic,
            effective_minutes,
            regime,
            blue_initiative_mult,
            config.strike_overdispersion,
        )

        red_td_attempts, red_td_landed, red_control = _takedown_round(
            rng,
            red,
            blue,
            red_dynamic,
            blue_dynamic,
            regime,
            red_initiative_mult,
            ground_share,
            matchup.round_seconds,
            config.takedown_overdispersion,
        )
        blue_td_attempts, blue_td_landed, blue_control = _takedown_round(
            rng,
            blue,
            red,
            blue_dynamic,
            red_dynamic,
            regime,
            blue_initiative_mult,
            ground_share,
            matchup.round_seconds,
            config.takedown_overdispersion,
        )

        max_control = ground_share * matchup.round_seconds
        total_control = red_control + blue_control
        if total_control > max_control > 0:
            scale = max_control / total_control
            red_control *= scale
            blue_control *= scale

        red_kd = _knockdowns(
            rng, red, blue, red_dynamic, blue_dynamic, red_landed, regime
        )
        blue_kd = _knockdowns(
            rng, blue, red, blue_dynamic, red_dynamic, blue_landed, regime
        )

        red_damage_created = _clamp(
            red_landed * (0.0018 + 0.0028 * red.power) + 0.095 * red_kd,
            0.0,
            0.55,
        )
        blue_damage_created = _clamp(
            blue_landed * (0.0018 + 0.0028 * blue.power) + 0.095 * blue_kd,
            0.0,
            0.55,
        )

        red_ko, red_sub = _finish_hazards(
            red,
            blue,
            red_dynamic,
            blue_dynamic,
            red_landed,
            red_kd,
            red_td_landed,
            red_control,
            regime,
        )
        blue_ko, blue_sub = _finish_hazards(
            blue,
            red,
            blue_dynamic,
            red_dynamic,
            blue_landed,
            blue_kd,
            blue_td_landed,
            blue_control,
            regime,
        )

        red_dynamic.sig_attempted += red_attempts
        red_dynamic.sig_landed += red_landed
        red_dynamic.takedowns_attempted += red_td_attempts
        red_dynamic.takedowns_landed += red_td_landed
        red_dynamic.control_seconds += red_control
        red_dynamic.knockdowns += red_kd

        blue_dynamic.sig_attempted += blue_attempts
        blue_dynamic.sig_landed += blue_landed
        blue_dynamic.takedowns_attempted += blue_td_attempts
        blue_dynamic.takedowns_landed += blue_td_landed
        blue_dynamic.control_seconds += blue_control
        blue_dynamic.knockdowns += blue_kd

        finish = _sample_finish(
            rng,
            {
                ("red", "ko_tko"): red_ko,
                ("red", "submission"): red_sub,
                ("blue", "ko_tko"): blue_ko,
                ("blue", "submission"): blue_sub,
            },
            config.max_finish_probability_per_round,
        )

        red_score = _round_score(
            red_landed, red_kd, red_td_landed, red_control, red_damage_created
        )
        blue_score = _round_score(
            blue_landed, blue_kd, blue_td_landed, blue_control, blue_damage_created
        )
        judge_noise = float(rng.normal(0.0, max(1.0, 0.06 * (red_score + blue_score))))
        red_won_round = red_score + judge_noise >= blue_score
        if red_won_round:
            red_dynamic.rounds_won += 1
        else:
            blue_dynamic.rounds_won += 1
        red_dynamic.score_total += red_score
        blue_dynamic.score_total += blue_score

        if finish is not None:
            winner_corner, method = finish
            finish_round = round_number
            time_fraction = float(rng.beta(1.5 + 0.15 * round_number, 1.35))
            finish_time_seconds = max(
                1.0,
                min(matchup.round_seconds, time_fraction * matchup.round_seconds),
            )
            exposure_fraction = finish_time_seconds / matchup.round_seconds

            red_partial = thin_round_performance(
                rng,
                RoundPerformance(
                    sig_attempted=red_attempts,
                    sig_landed=red_landed,
                    takedowns_attempted=red_td_attempts,
                    takedowns_landed=red_td_landed,
                    control_seconds=red_control,
                    knockdowns=red_kd,
                ),
                exposure_fraction,
            )
            blue_partial = thin_round_performance(
                rng,
                RoundPerformance(
                    sig_attempted=blue_attempts,
                    sig_landed=blue_landed,
                    takedowns_attempted=blue_td_attempts,
                    takedowns_landed=blue_td_landed,
                    control_seconds=blue_control,
                    knockdowns=blue_kd,
                ),
                exposure_fraction,
            )

            # Full latent round lines were already added above. Replace them
            # with only the portion realized before the terminal event.
            red_dynamic.sig_attempted += red_partial.sig_attempted - red_attempts
            red_dynamic.sig_landed += red_partial.sig_landed - red_landed
            red_dynamic.takedowns_attempted += (
                red_partial.takedowns_attempted - red_td_attempts
            )
            red_dynamic.takedowns_landed += (
                red_partial.takedowns_landed - red_td_landed
            )
            red_dynamic.control_seconds += red_partial.control_seconds - red_control
            red_dynamic.knockdowns += red_partial.knockdowns - red_kd

            blue_dynamic.sig_attempted += blue_partial.sig_attempted - blue_attempts
            blue_dynamic.sig_landed += blue_partial.sig_landed - blue_landed
            blue_dynamic.takedowns_attempted += (
                blue_partial.takedowns_attempted - blue_td_attempts
            )
            blue_dynamic.takedowns_landed += (
                blue_partial.takedowns_landed - blue_td_landed
            )
            blue_dynamic.control_seconds += blue_partial.control_seconds - blue_control
            blue_dynamic.knockdowns += blue_partial.knockdowns - blue_kd

            # An unfinished terminal round is not a completed judge-scored
            # round. Undo the score bookkeeping performed for the latent line.
            if red_won_round:
                red_dynamic.rounds_won -= 1
            else:
                blue_dynamic.rounds_won -= 1
            red_dynamic.score_total -= red_score
            blue_dynamic.score_total -= blue_score

            total_fight_seconds = (
                (round_number - 1) * matchup.round_seconds
                + finish_time_seconds
            )
            break

        _update_dynamic_state(
            red,
            red_dynamic,
            red_attempts,
            red_td_attempts,
            red_control,
            blue_damage_created,
            red_won_round,
        )
        _update_dynamic_state(
            blue,
            blue_dynamic,
            blue_attempts,
            blue_td_attempts,
            blue_control,
            red_damage_created,
            not red_won_round,
        )

        if red_won_round:
            initiative_red -= 0.035 * blue.adaptability
        else:
            initiative_red += 0.035 * red.adaptability
        initiative_red = _clamp(initiative_red, 0.15, 0.85)

    if winner_corner is None:
        if red_dynamic.rounds_won > blue_dynamic.rounds_won:
            winner_corner = "red"
        elif blue_dynamic.rounds_won > red_dynamic.rounds_won:
            winner_corner = "blue"
        else:
            score_noise = float(rng.normal(0.0, 2.0))
            winner_corner = (
                "red"
                if red_dynamic.score_total + score_noise >= blue_dynamic.score_total
                else "blue"
            )

    return FightSimulationOutcome(
        winner_corner=winner_corner,
        method=method,
        finish_round=finish_round,
        finish_time_seconds=round(finish_time_seconds, 3),
        total_fight_seconds=round(total_fight_seconds, 3),
        red_rounds_won=red_dynamic.rounds_won,
        blue_rounds_won=blue_dynamic.rounds_won,
        red_totals=_totals(red_dynamic),
        blue_totals=_totals(blue_dynamic),
        regime=regime,
    )


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return float(np.mean(values_list)) if values_list else 0.0


def summarize_outcomes(
    matchup: MatchupSimulationInput,
    config: SimulatorConfig,
    outcomes: list[FightSimulationOutcome],
) -> SimulationSummary:
    """Aggregate simulated paths into market and expectation probabilities."""
    n = len(outcomes)
    if n == 0:
        raise ValueError("At least one outcome is required")

    def probability(predicate) -> float:
        return float(sum(1 for outcome in outcomes if predicate(outcome)) / n)

    probabilities: dict[str, float] = {
        "red_win": probability(lambda x: x.winner_corner == "red"),
        "blue_win": probability(lambda x: x.winner_corner == "blue"),
        "goes_distance": probability(lambda x: x.method == "decision"),
        "inside_distance": probability(lambda x: x.method != "decision"),
        "red_by_decision": probability(
            lambda x: x.winner_corner == "red" and x.method == "decision"
        ),
        "blue_by_decision": probability(
            lambda x: x.winner_corner == "blue" and x.method == "decision"
        ),
        "red_by_ko_tko": probability(
            lambda x: x.winner_corner == "red" and x.method == "ko_tko"
        ),
        "blue_by_ko_tko": probability(
            lambda x: x.winner_corner == "blue" and x.method == "ko_tko"
        ),
        "red_by_submission": probability(
            lambda x: x.winner_corner == "red" and x.method == "submission"
        ),
        "blue_by_submission": probability(
            lambda x: x.winner_corner == "blue" and x.method == "submission"
        ),
    }

    for round_number in range(2, matchup.scheduled_rounds + 1):
        threshold = (round_number - 1) * matchup.round_seconds
        probabilities[f"reaches_round_{round_number}"] = probability(
            lambda x, threshold=threshold: x.total_fight_seconds > threshold
        )

    for half_round in range(1, matchup.scheduled_rounds):
        line = half_round + 0.5
        threshold = line * matchup.round_seconds
        probabilities[f"over_{line:.1f}_rounds"] = probability(
            lambda x, threshold=threshold: x.total_fight_seconds > threshold
        )
        probabilities[f"under_{line:.1f}_rounds"] = 1.0 - probabilities[
            f"over_{line:.1f}_rounds"
        ]

    expectations = {
        "fight_time_seconds": _mean(x.total_fight_seconds for x in outcomes),
        "red_sig_attempted": _mean(x.red_totals.sig_attempted for x in outcomes),
        "red_sig_landed": _mean(x.red_totals.sig_landed for x in outcomes),
        "red_takedowns_attempted": _mean(
            x.red_totals.takedowns_attempted for x in outcomes
        ),
        "red_takedowns_landed": _mean(x.red_totals.takedowns_landed for x in outcomes),
        "red_control_seconds": _mean(x.red_totals.control_seconds for x in outcomes),
        "red_knockdowns": _mean(x.red_totals.knockdowns for x in outcomes),
        "blue_sig_attempted": _mean(x.blue_totals.sig_attempted for x in outcomes),
        "blue_sig_landed": _mean(x.blue_totals.sig_landed for x in outcomes),
        "blue_takedowns_attempted": _mean(
            x.blue_totals.takedowns_attempted for x in outcomes
        ),
        "blue_takedowns_landed": _mean(x.blue_totals.takedowns_landed for x in outcomes),
        "blue_control_seconds": _mean(x.blue_totals.control_seconds for x in outcomes),
        "blue_knockdowns": _mean(x.blue_totals.knockdowns for x in outcomes),
    }

    joint_probabilities = {
        "red_win_and_over_1_5": probability(
            lambda x: x.winner_corner == "red"
            and x.total_fight_seconds > 1.5 * matchup.round_seconds
        ),
        "blue_win_and_over_1_5": probability(
            lambda x: x.winner_corner == "blue"
            and x.total_fight_seconds > 1.5 * matchup.round_seconds
        ),
        "red_win_and_2plus_takedowns": probability(
            lambda x: x.winner_corner == "red" and x.red_totals.takedowns_landed >= 2
        ),
        "blue_win_and_2plus_takedowns": probability(
            lambda x: x.winner_corner == "blue" and x.blue_totals.takedowns_landed >= 2
        ),
        "red_win_and_goes_distance": probability(
            lambda x: x.winner_corner == "red" and x.method == "decision"
        ),
        "blue_win_and_goes_distance": probability(
            lambda x: x.winner_corner == "blue" and x.method == "decision"
        ),
    }

    regime_probabilities = {
        regime: probability(lambda x, regime=regime: x.regime == regime)
        for regime in REGIMES
    }

    return SimulationSummary(
        fight_id=matchup.fight_id,
        event_id=matchup.event_id,
        red_fighter_id=matchup.red.fighter_id,
        red_fighter_name=matchup.red.fighter_name,
        blue_fighter_id=matchup.blue.fighter_id,
        blue_fighter_name=matchup.blue.fighter_name,
        scheduled_rounds=matchup.scheduled_rounds,
        simulations=n,
        seed=config.seed,
        probabilities=probabilities,
        expectations=expectations,
        joint_probabilities=joint_probabilities,
        regime_probabilities=regime_probabilities,
        source_snapshot_id=matchup.source_snapshot_id,
        simulator_version=SIMULATOR_VERSION,
    )


def run_simulation(
    matchup: MatchupSimulationInput,
    config: SimulatorConfig | None = None,
) -> tuple[SimulationSummary, list[FightSimulationOutcome] | None]:
    """Run Monte Carlo simulation and return summary plus optional raw outcomes."""
    runtime = config or SimulatorConfig()
    rng = np.random.default_rng(runtime.seed)
    outcomes = [simulate_fight(matchup, rng, runtime) for _ in range(runtime.simulations)]
    summary = summarize_outcomes(matchup, runtime, outcomes)
    return summary, outcomes if runtime.retain_outcomes else None
