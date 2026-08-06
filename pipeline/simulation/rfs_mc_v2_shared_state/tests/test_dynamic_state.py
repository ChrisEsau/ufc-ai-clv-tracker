"""Tests for V2 dynamic fighter-state contracts."""

from dataclasses import FrozenInstanceError

import pytest

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_state import (
    FightDynamicState,
    FighterDynamicState,
)


def fighter_state(
    *,
    fatigue: float = 0.20,
    damage: float = 0.30,
    acute_stress: float = 0.40,
) -> FighterDynamicState:
    """Build a valid fighter dynamic state."""

    return FighterDynamicState(
        fatigue=fatigue,
        damage=damage,
        acute_stress=acute_stress,
    )


def test_opening_fighter_state_is_fresh() -> None:
    assert FighterDynamicState.opening_state() == (
        FighterDynamicState(
            fatigue=0.0,
            damage=0.0,
            acute_stress=0.0,
        )
    )


def test_opening_fight_state_is_fresh() -> None:
    state = FightDynamicState.opening_state()

    assert state.red == FighterDynamicState.opening_state()
    assert state.blue == FighterDynamicState.opening_state()


def test_boundary_values_are_allowed() -> None:
    low = FighterDynamicState(
        fatigue=0.0,
        damage=0.0,
        acute_stress=0.0,
    )
    high = FighterDynamicState(
        fatigue=1.0,
        damage=1.0,
        acute_stress=1.0,
    )

    assert low.fatigue == 0.0
    assert high.fatigue == 1.0


@pytest.mark.parametrize(
    "field_name",
    [
        "fatigue",
        "damage",
        "acute_stress",
    ],
)
def test_dynamic_values_must_be_numeric(
    field_name: str,
) -> None:
    values = {
        "fatigue": 0.20,
        "damage": 0.30,
        "acute_stress": 0.40,
    }
    values[field_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        FighterDynamicState(**values)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("fatigue", float("nan")),
        ("damage", float("nan")),
        ("acute_stress", float("nan")),
        ("fatigue", float("inf")),
        ("damage", float("-inf")),
        ("acute_stress", float("inf")),
    ],
)
def test_dynamic_values_must_be_finite(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "fatigue": 0.20,
        "damage": 0.30,
        "acute_stress": 0.40,
    }
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        FighterDynamicState(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "fatigue",
        "damage",
        "acute_stress",
    ],
)
def test_dynamic_values_cannot_be_negative(
    field_name: str,
) -> None:
    values = {
        "fatigue": 0.20,
        "damage": 0.30,
        "acute_stress": 0.40,
    }
    values[field_name] = -0.01

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        FighterDynamicState(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "fatigue",
        "damage",
        "acute_stress",
    ],
)
def test_dynamic_values_cannot_exceed_one(
    field_name: str,
) -> None:
    values = {
        "fatigue": 0.20,
        "damage": 0.30,
        "acute_stress": 0.40,
    }
    values[field_name] = 1.01

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0 and 1",
    ):
        FighterDynamicState(**values)


def test_fighter_dynamic_state_is_immutable() -> None:
    state = fighter_state()

    with pytest.raises(FrozenInstanceError):
        state.fatigue = 0.90


def test_fight_dynamic_state_is_immutable() -> None:
    state = FightDynamicState.opening_state()

    with pytest.raises(FrozenInstanceError):
        state.red = fighter_state()


@pytest.mark.parametrize(
    ("side", "expected_attribute"),
    [
        (FighterSide.RED, "red"),
        (FighterSide.BLUE, "blue"),
    ],
)
def test_for_side_returns_requested_state(
    side: FighterSide,
    expected_attribute: str,
) -> None:
    state = FightDynamicState(
        red=fighter_state(fatigue=0.10),
        blue=fighter_state(fatigue=0.80),
    )

    assert state.for_side(side) == getattr(
        state,
        expected_attribute,
    )


def test_replace_red_side() -> None:
    original = FightDynamicState.opening_state()
    replacement = fighter_state(
        fatigue=0.60,
        damage=0.20,
        acute_stress=0.10,
    )

    updated = original.replace_side(
        FighterSide.RED,
        replacement,
    )

    assert updated.red == replacement
    assert updated.blue == original.blue


def test_replace_blue_side() -> None:
    original = FightDynamicState.opening_state()
    replacement = fighter_state(
        fatigue=0.70,
        damage=0.40,
        acute_stress=0.30,
    )

    updated = original.replace_side(
        FighterSide.BLUE,
        replacement,
    )

    assert updated.red == original.red
    assert updated.blue == replacement


def test_replace_side_does_not_mutate_original() -> None:
    original = FightDynamicState.opening_state()

    updated = original.replace_side(
        FighterSide.RED,
        fighter_state(fatigue=0.75),
    )

    assert original == FightDynamicState.opening_state()
    assert updated != original


def test_replace_side_requires_dynamic_state() -> None:
    state = FightDynamicState.opening_state()

    with pytest.raises(
        TypeError,
        match="state must be FighterDynamicState",
    ):
        state.replace_side(
            FighterSide.RED,
            "invalid",
        )


def test_for_side_rejects_unsupported_side() -> None:
    state = FightDynamicState.opening_state()

    with pytest.raises(
        ValueError,
        match="unsupported fighter side",
    ):
        state.for_side("red")


def test_replace_side_rejects_unsupported_side() -> None:
    state = FightDynamicState.opening_state()

    with pytest.raises(
        ValueError,
        match="unsupported fighter side",
    ):
        state.replace_side(
            "red",
            fighter_state(),
        )
