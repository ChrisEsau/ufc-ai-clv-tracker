import pytest

from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles
from pipeline.simulation.event_mc_v1.diagnostics.distance_parity import matched_exposure_summary


def balanced(name: str) -> FighterProfile:
    return FighterProfile(name, name, 50, 50, 50, 50, 50, 50, 50, 50)


def test_matched_exposure_diagnostic_is_distributionally_close() -> None:
    summary = matched_exposure_summary(
        MatchupProfiles(balanced("red"), balanced("blue")),
        paths=8_000,
        exposure_seconds=900,
        seed=20260811,
    )
    for side in ("red", "blue"):
        legacy = summary[side]["legacy"]
        event = summary[side]["event_mc"]
        assert event["strike_attempts_per_minute"] == pytest.approx(
            legacy["strike_attempts_per_minute"], rel=0.02
        )
        assert event["strike_landing_percentage"] == pytest.approx(
            legacy["strike_landing_percentage"], abs=0.01
        )
        assert event["td_success_percentage"] == pytest.approx(
            legacy["td_success_percentage"], abs=0.02
        )
        # Continuous clocks allow separate actions that V0's transition sampler
        # suppresses after the first transition inside a ten-second segment.
        assert event["td_attempts_per_15_minutes"] >= legacy["td_attempts_per_15_minutes"]
