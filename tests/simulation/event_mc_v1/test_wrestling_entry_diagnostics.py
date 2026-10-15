import pytest

from pipeline.simulation.event_mc_v1.components.profiles import FighterProfile, MatchupProfiles
from pipeline.simulation.event_mc_v1.diagnostics.wrestling_entry_ontology import (
    matched_ontology_summary,
    ontology_audit_rows,
)


def fighter(name: str, entry: float) -> FighterProfile:
    return FighterProfile(name, name, 50, 50, 50, 50, entry, 50, 50, 50)


def test_audit_exposes_both_semantics_without_affecting_simulation() -> None:
    profiles = MatchupProfiles(fighter("red", 58), fighter("blue", 42))
    rows = ontology_audit_rows(profiles)
    assert len(rows) == 2
    assert all(row.context_multiplier == 1 for row in rows)
    assert all(row.phase_2a_hazard_per_second > 0 for row in rows)
    assert all(row.phase_2b_hazard_per_second > 0 for row in rows)


def test_matched_exposure_preserves_success_and_moves_attempts() -> None:
    profiles = MatchupProfiles(fighter("red", 58), fighter("blue", 42))
    summary = matched_ontology_summary(profiles, paths=20_000)
    for side in ("red", "blue"):
        assert summary[side]["phase_2b"]["td_success_percentage"] == pytest.approx(
            summary[side]["phase_2a"]["td_success_percentage"], abs=0.01
        )
        assert summary[side]["unchanged_phase_2a_2b_striking"] == {
                "strike_attempts_per_minute": pytest.approx(12.0),
            "strike_landing_probability": pytest.approx(0.4),
        }
