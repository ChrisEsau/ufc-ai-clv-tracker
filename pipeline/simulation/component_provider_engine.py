"""Shadow engine that can replace strike volume and finish hazards independently.

The production-neutral mechanics kernel remains unchanged. This module supports
controlled historical ablations with optional providers:

- no providers: use the original engine instead;
- strike provider only: absolute significant-strike attempt distributions;
- finish provider only: calibrated mutually exclusive fight-round hazards;
- both providers: both component replacements in the same simulated path.

Strike providers may optionally expose a context-aware method. That method is
called with state created only by earlier simulated rounds, allowing round-two-plus
pace to react to the current Monte Carlo path without reading realized fight data.
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
from pipeline.simulation.dynamic_strike_provider import DynamicStrikeRoundContext
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
    _strike_round,
    _takedown_round,
    _totals,
    _update_dynamic_state,
    summarize_outcomes,
)
from pipeline.simulation.finish_hazard_provider import (
    FinishHazardKey,
    FinishHazardProvider,
)
from pipeline.simulation.provider_engine import _provider_strike_round
from pipeline.simulation.round_parameter_provider import (
    RoundParameterKey,
    SignificantStrikeParameterProvider,
)
from pipeline.simulation.terminal_round import RoundPerformance, thin_round_performance


COMPONENT_PROVIDER_VERSION = f"{SIMULATOR_VERSION}_component_providers"


def _sample_provider_finish(
    rng: np.random.Generator,
    finish_provider: FinishHazardProvider,
    fight_id: str,
    round_number: int,
) -> tuple[str, str] | None:
    probabilities = finish_provider.finish_hazards(
        FinishHazardKey(fight_id=str(fight_id), round=int(round_number))
    ).as_array()
    selected = int(rng.choice(len(probabilities), p=probabilities))
    if selected == 0:
        return None
    return {
        1: ("red", "ko_tko"),
        2: ("red", "submission"),
        3: ("blue", "ko_tko"),
        4: ("blue", "submission"),
    }[selected]


def _component_provider_strike_round(
    rng: np.random.Generator,
    provider: SignificantStrikeParameterProvider,
    key: RoundParameterKey,
    fighter,
    opponent,
    dynamic: _DynamicState,
    opponent_dynamic: _DynamicState,
    matchup: MatchupSimulationInput,
) -> tuple[int, int]:
    """Sample static or path-context strike parameters without changing accuracy."""
    contextual_method = getattr(
        provider,
        "significant_strike_attempts_with_context",
        None,
    )
    if contextual_method is None:
        return _provider_strike_round(
            rng,
            provider,
            key,
            fighter,
            opponent,
            dynamic,
            opponent_dynamic,
            matchup.round_seconds,
        )

    context = DynamicStrikeRoundContext(
        key=key,
        opponent_id=str(opponent.fighter_id),
        scheduled_rounds=int(matchup.scheduled_rounds),
        round_seconds=int(matchup.round_seconds),
        fighter_fatigue=float(dynamic.fatigue),
        fighter_damage=float(dynamic.damage),
        fighter_confidence=float(dynamic.confidence),
        opponent_fatigue=float(opponent_dynamic.fatigue),
        opponent_damage=float(opponent_dynamic.damage),
        opponent_confidence=float(opponent_dynamic.confidence),
        fighter_sig_attempted=int(dynamic.sig_attempted),
        fighter_sig_landed=int(dynamic.sig_landed),
        opponent_sig_attempted=int(opponent_dynamic.sig_attempted),
        opponent_sig_landed=int(opponent_dynamic.sig_landed),
        fighter_control_seconds=float(dynamic.control_seconds),
        opponent_control_seconds=float(opponent_dynamic.control_seconds),
        fighter_knockdowns=int(dynamic.knockdowns),
        opponent_knockdowns=int(opponent_dynamic.knockdowns),
        fighter_rounds_won=int(dynamic.rounds_won),
        opponent_rounds_won=int(opponent_dynamic.rounds_won),
    )
    parameters = contextual_method(context)
    attempts = parameters.sample_count(
        rng,
        exposure_seconds=float(matchup.round_seconds),
    )

    # The provider owns attempt volume. Accuracy remains the existing heuristic
    # contract so this ablation isolates dynamic volume generation.
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


def simulate_fight_with_component_providers(
    matchup: MatchupSimulationInput,
    rng: np.random.Generator,
    config: SimulatorConfig,
    strike_provider: SignificantStrikeParameterProvider | None = None,
    finish_provider: FinishHazardProvider | None = None,
) -> FightSimulationOutcome:
    """Simulate one path with independently optional component providers."""
    if strike_provider is None and finish_provider is None:
        raise ValueError("At least one component provider is required")

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
        active_share = _clamp(
            distance_share + 0.65 * clinch_share + 0.35 * ground_share,
            0.25,
            1.0,
        )
        effective_minutes = matchup.round_seconds / 60.0 * active_share

        red_initiative_mult = 0.78 + 0.44 * initiative_red
        blue_initiative_mult = 0.78 + 0.44 * (1.0 - initiative_red)

        if strike_provider is None:
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
        else:
            red_attempts, red_landed = _component_provider_strike_round(
                rng,
                strike_provider,
                RoundParameterKey(matchup.fight_id, red.fighter_id, round_number),
                red,
                blue,
                red_dynamic,
                blue_dynamic,
                matchup,
            )
            blue_attempts, blue_landed = _component_provider_strike_round(
                rng,
                strike_provider,
                RoundParameterKey(matchup.fight_id, blue.fighter_id, round_number),
                blue,
                red,
                blue_dynamic,
                red_dynamic,
                matchup,
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

        if finish_provider is None:
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
        else:
            finish = _sample_provider_finish(
                rng,
                finish_provider,
                fight_id=matchup.fight_id,
                round_number=round_number,
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


def run_simulation_with_component_providers(
    matchup: MatchupSimulationInput,
    config: SimulatorConfig | None = None,
    strike_provider: SignificantStrikeParameterProvider | None = None,
    finish_provider: FinishHazardProvider | None = None,
) -> tuple[SimulationSummary, list[FightSimulationOutcome] | None]:
    """Run component-provider paths and return the normal simulator summary."""
    runtime = config or SimulatorConfig()
    rng = np.random.default_rng(runtime.seed)
    outcomes = [
        simulate_fight_with_component_providers(
            matchup,
            rng,
            runtime,
            strike_provider=strike_provider,
            finish_provider=finish_provider,
        )
        for _ in range(runtime.simulations)
    ]
    summary = summarize_outcomes(matchup, runtime, outcomes)
    suffixes = []
    if strike_provider is not None:
        suffixes.append(str(getattr(strike_provider, "simulator_suffix", "strike")))
    if finish_provider is not None:
        suffixes.append("finish")
    version = f"{COMPONENT_PROVIDER_VERSION}_{'_'.join(suffixes)}"
    summary = replace(summary, simulator_version=version)
    return summary, outcomes if runtime.retain_outcomes else None
