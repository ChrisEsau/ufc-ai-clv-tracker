"""Tests for the RFS-to-simulator feature contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import get_type_hints

import pytest

from pipeline.round_stats.rfs_simulator_feature_contracts import (
    MappingSupportLevel,
    ReliabilityShrunkEstimate,
    SIMULATOR_TARGET_BY_NAME,
    SIMULATOR_TARGET_SPECS,
    SimulatorFeatureFamily,
    SimulatorTargetSpec,
    SimulatorValueKind,
    validate_simulator_target_registry,
)
from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    FighterPhaseParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)


def nested_field_names(
    contract: type,
    *,
    prefix: str,
) -> set[str]:
    """Return fully qualified leaf fields from one dataclass contract."""

    names: set[str] = set()
    type_hints = get_type_hints(contract)

    for field in fields(contract):
        field_type = type_hints.get(
            field.name,
            field.type,
        )
        full_name = f"{prefix}.{field.name}"

        if (
            isinstance(field_type, type)
            and is_dataclass(field_type)
        ):
            names.update(
                nested_field_names(
                    field_type,
                    prefix=full_name,
                )
            )
        else:
            names.add(full_name)

    return names


def actual_simulator_targets() -> set[str]:
    """Return all authoritative fighter parameter targets."""

    return (
        nested_field_names(
            FighterTransitionParameters,
            prefix="transition",
        )
        | nested_field_names(
            FighterPhaseParameters,
            prefix="phase",
        )
        | nested_field_names(
            FighterDynamicParameters,
            prefix="dynamic",
        )
    )


def valid_estimate() -> ReliabilityShrunkEstimate:
    """Build one valid reliability-shrunk estimate."""

    return ReliabilityShrunkEstimate(
        raw_estimate=0.70,
        population_prior=0.50,
        sample_size=8,
        effective_sample_size=5.5,
        reliability=0.60,
        shrunk_estimate=0.62,
        used_fallback=False,
        source_columns=(
            "td_landed",
            "td_attempted",
        ),
    )


def test_registry_contains_exactly_37_targets() -> None:
    assert len(SIMULATOR_TARGET_SPECS) == 37


def test_registry_exactly_matches_simulator_contracts() -> None:
    registry_targets = {
        spec.target_parameter
        for spec in SIMULATOR_TARGET_SPECS
    }

    assert registry_targets == actual_simulator_targets()


def test_actual_simulator_contracts_contain_37_targets() -> None:
    assert len(actual_simulator_targets()) == 37


def test_registry_names_are_unique() -> None:
    names = [
        spec.target_parameter
        for spec in SIMULATOR_TARGET_SPECS
    ]

    assert len(names) == len(set(names))


def test_registry_lookup_contains_every_spec() -> None:
    assert len(SIMULATOR_TARGET_BY_NAME) == 37

    for spec in SIMULATOR_TARGET_SPECS:
        assert (
            SIMULATOR_TARGET_BY_NAME[
                spec.target_parameter
            ]
            is spec
        )


def test_registry_validation_runs_cleanly() -> None:
    validate_simulator_target_registry()


def test_every_feature_family_is_used() -> None:
    observed = {
        spec.primary_family
        for spec in SIMULATOR_TARGET_SPECS
    }

    assert observed == set(SimulatorFeatureFamily)


def test_every_value_kind_is_used() -> None:
    observed = {
        spec.value_kind
        for spec in SIMULATOR_TARGET_SPECS
    }

    assert observed == set(SimulatorValueKind)


def test_every_support_level_is_used() -> None:
    observed = {
        spec.support_level
        for spec in SIMULATOR_TARGET_SPECS
    }

    assert observed == set(MappingSupportLevel)


def test_locked_family_counts() -> None:
    counts = Counter(
        spec.primary_family
        for spec in SIMULATOR_TARGET_SPECS
    )

    assert counts == {
        SimulatorFeatureFamily.PHASE_INTERACTION: 17,
        SimulatorFeatureFamily.PHASE_BASELINE: 10,
        SimulatorFeatureFamily.FINISH_STATE: 6,
        SimulatorFeatureFamily.DYNAMIC_RESPONSE: 4,
    }


def test_locked_support_counts() -> None:
    counts = Counter(
        spec.support_level
        for spec in SIMULATOR_TARGET_SPECS
    )

    assert counts == {
        MappingSupportLevel.LATENT_CALIBRATED: 21,
        MappingSupportLevel.DERIVED_OBSERVED: 10,
        MappingSupportLevel.DIRECT_OBSERVED: 6,
    }


def test_every_target_has_description() -> None:
    for spec in SIMULATOR_TARGET_SPECS:
        assert spec.description.strip()


def test_target_spec_is_immutable() -> None:
    spec = SIMULATOR_TARGET_SPECS[0]

    with pytest.raises(FrozenInstanceError):
        spec.description = "changed"


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        1,
        True,
    ],
)
def test_target_parameter_requires_string(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="target_parameter must be a string",
    ):
        SimulatorTargetSpec(
            target_parameter=invalid_value,
            primary_family=(
                SimulatorFeatureFamily.PHASE_BASELINE
            ),
            value_kind=(
                SimulatorValueKind.UNIT_INTERVAL_STRENGTH
            ),
            support_level=(
                MappingSupportLevel.DERIVED_OBSERVED
            ),
            description="Valid description.",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "",
        "   ",
    ],
)
def test_target_parameter_cannot_be_empty(
    invalid_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="target_parameter cannot be empty",
    ):
        SimulatorTargetSpec(
            target_parameter=invalid_value,
            primary_family=(
                SimulatorFeatureFamily.PHASE_BASELINE
            ),
            value_kind=(
                SimulatorValueKind.UNIT_INTERVAL_STRENGTH
            ),
            support_level=(
                MappingSupportLevel.DERIVED_OBSERVED
            ),
            description="Valid description.",
        )


def test_target_spec_requires_family_enum() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "primary_family must be "
            "SimulatorFeatureFamily"
        ),
    ):
        SimulatorTargetSpec(
            target_parameter="transition.example",
            primary_family="phase_baseline",
            value_kind=(
                SimulatorValueKind.UNIT_INTERVAL_STRENGTH
            ),
            support_level=(
                MappingSupportLevel.DERIVED_OBSERVED
            ),
            description="Valid description.",
        )


def test_target_spec_requires_value_kind_enum() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "value_kind must be "
            "SimulatorValueKind"
        ),
    ):
        SimulatorTargetSpec(
            target_parameter="transition.example",
            primary_family=(
                SimulatorFeatureFamily.PHASE_BASELINE
            ),
            value_kind="probability",
            support_level=(
                MappingSupportLevel.DERIVED_OBSERVED
            ),
            description="Valid description.",
        )


def test_target_spec_requires_support_level_enum() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "support_level must be "
            "MappingSupportLevel"
        ),
    ):
        SimulatorTargetSpec(
            target_parameter="transition.example",
            primary_family=(
                SimulatorFeatureFamily.PHASE_BASELINE
            ),
            value_kind=(
                SimulatorValueKind.UNIT_INTERVAL_STRENGTH
            ),
            support_level="derived_observed",
            description="Valid description.",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        None,
        1,
        True,
    ],
)
def test_target_description_requires_string(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="description must be a string",
    ):
        SimulatorTargetSpec(
            target_parameter="transition.example",
            primary_family=(
                SimulatorFeatureFamily.PHASE_BASELINE
            ),
            value_kind=(
                SimulatorValueKind.UNIT_INTERVAL_STRENGTH
            ),
            support_level=(
                MappingSupportLevel.DERIVED_OBSERVED
            ),
            description=invalid_value,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "",
        "   ",
    ],
)
def test_target_description_cannot_be_empty(
    invalid_value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="description cannot be empty",
    ):
        SimulatorTargetSpec(
            target_parameter="transition.example",
            primary_family=(
                SimulatorFeatureFamily.PHASE_BASELINE
            ),
            value_kind=(
                SimulatorValueKind.UNIT_INTERVAL_STRENGTH
            ),
            support_level=(
                MappingSupportLevel.DERIVED_OBSERVED
            ),
            description=invalid_value,
        )


def test_valid_reliability_shrunk_estimate() -> None:
    estimate = valid_estimate()

    assert estimate.raw_estimate == pytest.approx(0.70)
    assert estimate.population_prior == pytest.approx(0.50)
    assert estimate.sample_size == 8
    assert estimate.effective_sample_size == pytest.approx(5.5)
    assert estimate.reliability == pytest.approx(0.60)
    assert estimate.shrunk_estimate == pytest.approx(0.62)
    assert estimate.used_fallback is False


def test_valid_fallback_estimate() -> None:
    estimate = ReliabilityShrunkEstimate(
        raw_estimate=None,
        population_prior=0.50,
        sample_size=0,
        effective_sample_size=0.0,
        reliability=0.0,
        shrunk_estimate=0.50,
        used_fallback=True,
        source_columns=(),
    )

    assert estimate.raw_estimate is None
    assert estimate.used_fallback is True
    assert estimate.shrunk_estimate == estimate.population_prior


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "0.5",
    ],
)
def test_raw_estimate_must_be_finite_or_none(
    invalid_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="raw_estimate must be finite or None",
    ):
        ReliabilityShrunkEstimate(
            raw_estimate=invalid_value,
            population_prior=0.50,
            sample_size=1,
            effective_sample_size=1.0,
            reliability=0.50,
            shrunk_estimate=0.50,
            used_fallback=False,
            source_columns=(),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "population_prior",
        "effective_sample_size",
        "reliability",
        "shrunk_estimate",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        "0.5",
        None,
    ],
)
def test_numeric_estimate_fields_require_numeric_values(
    field_name: str,
    invalid_value: object,
) -> None:
    values = {
        "raw_estimate": 0.50,
        "population_prior": 0.50,
        "sample_size": 1,
        "effective_sample_size": 1.0,
        "reliability": 0.50,
        "shrunk_estimate": 0.50,
        "used_fallback": False,
        "source_columns": (),
    }
    values[field_name] = invalid_value

    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        ReliabilityShrunkEstimate(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "population_prior",
        "effective_sample_size",
        "reliability",
        "shrunk_estimate",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_numeric_estimate_fields_must_be_finite(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {
        "raw_estimate": 0.50,
        "population_prior": 0.50,
        "sample_size": 1,
        "effective_sample_size": 1.0,
        "reliability": 0.50,
        "shrunk_estimate": 0.50,
        "used_fallback": False,
        "source_columns": (),
    }
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        ReliabilityShrunkEstimate(**values)


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_sample_size_requires_exact_integer(
    invalid_value: object,
) -> None:
    values = vars(valid_estimate()).copy()
    values["sample_size"] = invalid_value

    with pytest.raises(
        TypeError,
        match="sample_size must be an integer",
    ):
        ReliabilityShrunkEstimate(**values)


def test_sample_size_cannot_be_negative() -> None:
    values = vars(valid_estimate()).copy()
    values["sample_size"] = -1

    with pytest.raises(
        ValueError,
        match="sample_size cannot be negative",
    ):
        ReliabilityShrunkEstimate(**values)


def test_effective_sample_size_cannot_be_negative() -> None:
    values = vars(valid_estimate()).copy()
    values["effective_sample_size"] = -0.1

    with pytest.raises(
        ValueError,
        match=(
            "effective_sample_size cannot be negative"
        ),
    ):
        ReliabilityShrunkEstimate(**values)


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.01,
        1.01,
    ],
)
def test_reliability_must_be_unit_interval(
    invalid_value: float,
) -> None:
    values = vars(valid_estimate()).copy()
    values["reliability"] = invalid_value

    with pytest.raises(
        ValueError,
        match="reliability must be between 0 and 1",
    ):
        ReliabilityShrunkEstimate(**values)


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        1,
        "false",
        None,
    ],
)
def test_used_fallback_requires_boolean(
    invalid_value: object,
) -> None:
    values = vars(valid_estimate()).copy()
    values["used_fallback"] = invalid_value

    with pytest.raises(
        TypeError,
        match="used_fallback must be boolean",
    ):
        ReliabilityShrunkEstimate(**values)


def test_source_columns_must_be_tuple() -> None:
    values = vars(valid_estimate()).copy()
    values["source_columns"] = [
        "td_landed",
        "td_attempted",
    ]

    with pytest.raises(
        TypeError,
        match="source_columns must be a tuple",
    ):
        ReliabilityShrunkEstimate(**values)


def test_source_columns_must_contain_strings() -> None:
    values = vars(valid_estimate()).copy()
    values["source_columns"] = (
        "td_landed",
        1,
    )

    with pytest.raises(
        TypeError,
        match="source_columns must contain strings",
    ):
        ReliabilityShrunkEstimate(**values)


@pytest.mark.parametrize(
    "invalid_value",
    [
        "",
        "   ",
    ],
)
def test_source_columns_cannot_contain_empty_names(
    invalid_value: str,
) -> None:
    values = vars(valid_estimate()).copy()
    values["source_columns"] = (
        "td_landed",
        invalid_value,
    )

    with pytest.raises(
        ValueError,
        match=(
            "source_columns cannot contain empty names"
        ),
    ):
        ReliabilityShrunkEstimate(**values)


def test_missing_raw_estimate_requires_fallback() -> None:
    values = vars(valid_estimate()).copy()
    values["raw_estimate"] = None
    values["used_fallback"] = False

    with pytest.raises(
        ValueError,
        match=(
            "missing raw_estimate requires "
            "used_fallback=True"
        ),
    ):
        ReliabilityShrunkEstimate(**values)


def test_reliability_shrunk_estimate_is_immutable() -> None:
    estimate = valid_estimate()

    with pytest.raises(FrozenInstanceError):
        estimate.shrunk_estimate = 0.75
