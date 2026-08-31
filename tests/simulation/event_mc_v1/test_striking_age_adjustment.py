import numpy as np
import pytest

from pipeline.simulation.event_mc_v1.components.action_rates import (
    FSRV2ActionRateProvider,
)
from pipeline.simulation.event_mc_v1.components.fsr_v2 import (
    FSRV2FighterInput,
    FSRV2Matchup,
    FSR_V2_TRAIT_FIELDS,
)
from pipeline.simulation.event_mc_v1.components.fsr_v2_actions import (
    FSRV2Candidate,
)
from pipeline.simulation.event_mc_v1.components.fsr_v2_mechanics import (
    STANDING_STRIKE_ACCURACY_LOGIT_PER_YEAR,
    STANDING_STRIKE_ATTACKER_AGE_CENTER_YEARS,
    STANDING_STRIKE_RATE_LOG_PER_YEAR,
    standing_strike_age_accuracy_logit_offset,
    standing_strike_age_rate_multiplier,
)
from pipeline.simulation.event_mc_v1.config import FightConfig
from pipeline.simulation.event_mc_v1.contracts import FightContext
from pipeline.simulation.event_mc_v1.state import FightState
from pipeline.simulation.event_mc_v1.components.profiles import Side


def row(fighter_id="f", age_years=30.0):
    values = {
        name: 0.0
        for name in FSR_V2_TRAIT_FIELDS
    }

    values.update(
        {
            "fighter_id": fighter_id,
            "fighter_name": fighter_id,

            "standing_striking_tendency": 0.08,
            "standing_striking_suppression": 0.01,

            "takedown_tendency": 0.02,
            "ground_striking_tendency": 0.05,
            "submission_tendency": 0.01,

            "head_strike_tendency": 0.7,
            "body_strike_tendency": 0.3,
            "leg_strike_tendency": 0.2,

            "stamina_capacity": 100.0,
            "stamina_depletion_resistance": 61.0,
            "stamina_performance_resilience": 62.0,
            "striking_power": 63.0,
            "damage_durability": 64.0,
            "knockdown_resistance": 65.0,

            "standing_accuracy_baseline": 0.47,
            "takedown_completion_baseline": 0.38,
            "ground_accuracy_baseline": 0.56,
            "submission_conversion_baseline": 0.21,
            "escape_population_mean_seconds": 42.0,

            "age_years": age_years,
        }
    )

    return values


def matchup(red_age, blue_age=30.0):
    red = FSRV2FighterInput.from_mapping(
        row("red", red_age)
    )
    blue = FSRV2FighterInput.from_mapping(
        row("blue", blue_age)
    )
    return FSRV2Matchup(red, blue)


def red_standing_rate(red_age, blue_age=30.0):
    m = matchup(red_age, blue_age)

    provider = FSRV2ActionRateProvider(
        m,
        m.physical_profiles(),
    )

    context = FightContext(
        FightConfig(3),
        0,
        1,
    )

    candidates = provider.candidates(
        FightState(),
        context,
    )

    candidate = next(
        item
        for item in candidates
        if item.candidate.side is Side.RED
        and item.candidate.action_family
        == "standing_strike"
    )

    return candidate.rate_per_second


def test_standing_age_constants_match_validated_coefficients():
    assert (
        STANDING_STRIKE_ATTACKER_AGE_CENTER_YEARS
        == pytest.approx(30.0)
    )
    assert (
        STANDING_STRIKE_RATE_LOG_PER_YEAR
        == pytest.approx(-0.014547)
    )
    assert (
        STANDING_STRIKE_ACCURACY_LOGIT_PER_YEAR
        == pytest.approx(-0.017472)
    )


def test_standing_rate_age_is_centered_and_bidirectional():
    assert standing_strike_age_rate_multiplier(
        30.0
    ) == pytest.approx(1.0)

    assert (
        standing_strike_age_rate_multiplier(25.0)
        > 1.0
    )

    assert (
        standing_strike_age_rate_multiplier(40.0)
        < 1.0
    )

    neutral = red_standing_rate(30.0)

    assert (
        red_standing_rate(25.0) / neutral
        == pytest.approx(
            standing_strike_age_rate_multiplier(
                25.0
            )
        )
    )

    assert (
        red_standing_rate(40.0) / neutral
        == pytest.approx(
            standing_strike_age_rate_multiplier(
                40.0
            )
        )
    )


def test_defender_age_does_not_change_standing_attempt_rate():
    assert red_standing_rate(
        30.0,
        25.0,
    ) == pytest.approx(
        red_standing_rate(
            30.0,
            40.0,
        )
    )


def test_standing_accuracy_age_offset_is_centered_and_bidirectional():
    assert standing_strike_age_accuracy_logit_offset(
        30.0
    ) == pytest.approx(0.0)

    assert (
        standing_strike_age_accuracy_logit_offset(
            25.0
        )
        > 0.0
    )

    assert (
        standing_strike_age_accuracy_logit_offset(
            40.0
        )
        < 0.0
    )


def test_standing_resolver_adds_attacker_age_to_accuracy_logit(
    monkeypatch,
):
    import pipeline.simulation.event_mc_v1.components.fsr_v2_actions as actions

    m = matchup(40.0, 30.0)

    candidate = FSRV2Candidate(
        Side.RED,
        "standing_strike",
        m,
        m.physical_profiles(),
    )

    captured = []

    def fake_matchup_probability(
        baseline,
        offense,
        defense,
        logit_offset=0.0,
    ):
        captured.append(logit_offset)
        return 0.0

    monkeypatch.setattr(
        actions,
        "matchup_probability",
        fake_matchup_probability,
    )

    candidate.resolve(
        FightState(),
        FightContext(FightConfig(3), 0, 1),
        np.random.default_rng(123),
    )

    c = candidate.calibration.section(
        "fsr_v2_calibration"
    )

    expected = (
        c["standing_accuracy_logit_offset"]
        + standing_strike_age_accuracy_logit_offset(
            40.0
        )
    )

    assert captured == [pytest.approx(expected)]
