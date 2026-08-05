"""Shadow simulator path using absolute trained/provider strike-attempt means.

This module intentionally leaves the existing heuristic engine unchanged. It
reuses the same regime, wrestling, damage, finish, scoring, and terminal-round
mechanics, but replaces significant-strike attempt generation with the public
``SignificantStrikeParameterProvider`` contract.

Provider means are absolute exposure-adjusted rates. The strike count sampler
therefore does not reapply regime pace, active phase share, initiative, fatigue,
suppression, damage, or confidence multipliers. Accuracy and every non-strike
component remain heuristic in this ablation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pipeline.simulation.contracts import (
    FightSimulationOutcome,
    MatchupSimulationInput,
    SimulationSummary,
    SimulatorConfig,
)
from pipeline.simulation.engine import (
    REGIMES,
    SIMULATOR_VERSION,
    _DynamicState,
    _clamp,
    _finish_hazards,
    _initiative_share,
    _knockdowns,
    _regime_probabilities,
    _round_score,
    _sample_finish,
    _sample_phase_shares,
    _takedown_round,
    _totals,
    _update_dynamic_state,
    summarize_outcomes,
)
from pipeline.simulation.round_parameter_provider import (
    RoundParameterKey,
    SignificantStrikeParameterProvider,
)
from pipeline.simulation.terminal_round import RoundPerformance, thin_round_performance


PROVIDER_SIMULATOR_VERSION = f"{SIMULATOR_VERSION}_absolute_strike_provider"


def _provider_strike_round(
    rng: np.random.Generator,
    provider: SignificantStrikeParameterProvider,
    key: RoundParameterKey,
    fighter,
    opponent,
    dynamic: _DynamicState,
    opponent_dynamic: _DynamicState,
    round_seconds: int,
) -> tuple[int, int]:
    """Sample attempts directly from the provider's absolute round distribution."""
    parameters = provider.significant_strike_attempts(key)
    attempts = parameters.sample_count(rng, exposure_seconds=float(round_seconds))

    # Attempt volume comes exclusively from the provider. Accuracy remains the
    # current heuristic contract so this replay isolates one component change.
    defense_effect = 0.52 + 0.48 * (1.0 - opponent.sig_defense)
    fatigue_accuracy = 1.0 - 0.12 * dynamic.fatigue
    opponent_damage_opening = 1.0 + 0.08 * opponent_dynamic.damage
    accuracy = _clamp(
        fighter.sig_accuracy
        * defense_effect
        * fatigue_accuracy
        * opponent_damage_opening,
        0.08,
        0.78,
    )
    landed = int(rng.binomial(attempts, accuracy)) if attempts else 0
    return attempts, landed


def simulate_fight_with_strike_provider(
    matchup: MatchupSimulationInput,
    rng: np.random.Generator,
    config: SimulatorConfig,
    strike_provider: SignificantStrikeParameterProvider,
) -> FightSimulationOutcome:
    """Simulate one fight while sourcing strike attempts from a provider."""
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
        distance_share, clinch_share, ground_share = _sample_phase_shares(
            rng, regime, red, blue
        )

        red_initiative_mult = 0.78 + 0.44 * initiative_red
        blue_initiative_mult = 0.78 + 0.44 * (1.0 - initiative_red)

        red_attempts, red_landed = _provider_strike_round(
            rng,
            strike_provider,
            RoundParameterKey(matchup.fight_id, red.fighter_id, round_number),
            red,
            blue,
            red_dynamic,
            blue_dynamic,
            matchup.round_seconds,
        )
        blue_attempts, blue_landed = _provider_strike_round(
            rng,
            strike_provider,
            RoundParameterKey(matchup.fight_id, blue.fighter_id, round_number),
            blue,
            red,
            blue_dynamic,
            red_dynamic,
            matchup.round_seconds,
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
        judge_noise = float(
            rng.normal(0.0, max(1.0, 0.06 * (red_score + blue_score)))
        )
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

            if red_won_round:
                red_dynamic.rounds_won -= 1
            else:
                blue_dynamic.rounds_won -= 1
            red_dynamic.score_total -= red_score
            blue_dynamic.score_total -= blue_score

            total_fight_seconds = (
                (round_number - 1) * matchup.round_seconds + finish_time_seconds
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


def run_simulation_with_strike_provider(
    matchup: MatchupSimulationInput,
    strike_provider: SignificantStrikeParameterProvider,
    config: SimulatorConfig | None = None,
) -> tuple[SimulationSummary, list[FightSimulationOutcome] | None]:
    """Run provider-backed strike simulation without changing the base engine."""
    runtime = config or SimulatorConfig()
    rng = np.random.default_rng(runtime.seed)
    outcomes = [
        simulate_fight_with_strike_provider(matchup, rng, runtime, strike_provider)
        for _ in range(runtime.simulations)
    ]
    summary = summarize_outcomes(matchup, runtime, outcomes)
    summary = replace(summary, simulator_version=PROVIDER_SIMULATOR_VERSION)
    return summary, outcomes if runtime.retain_outcomes else None
