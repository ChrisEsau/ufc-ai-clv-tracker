"""Tests for deterministic V2 round-evidence aggregation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, replace
from types import SimpleNamespace

import pytest

import pipeline.simulation.rfs_mc_v2_shared_state.round_evidence as evidence_module
from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.clinch_activity_engine import (
    ClinchSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.distance_activity_engine import (
    DistanceSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.ground_activity_engine import (
    GroundSegmentActivity,
)
from pipeline.simulation.rfs_mc_v2_shared_state.round_evidence import (
    FighterRoundEvidence,
    RoundEvidence,
    calculate_round_evidence,
)
from scripts.audit_rfs_mc_v2_finish_paths import (
    run_path,
    zero_finish_calibration,
)


INTEGER_FIELDS = (
    "distance_strikes_attempted",
    "distance_strikes_landed",
    "clinch_strikes_attempted",
    "clinch_strikes_landed",
    "ground_strikes_attempted",
    "ground_strikes_landed",
    "knockdowns",
    "damaging_clinch_strikes",
    "control_seconds",
    "submission_attempts",
    "position_advancements",
    "escape_attempts",
    "reversal_attempts",
    "scramble_attempts",
)

FLOAT_FIELDS = (
    "persistent_damage_inflicted",
    "acute_stress_inflicted",
)


def fighter_evidence(
    **overrides: int | float,
) -> FighterRoundEvidence:
    """Build valid fighter evidence with optional field overrides."""

    values: dict[str, int | float] = {
        name: 0
        for name in INTEGER_FIELDS
    }
    values.update(
        {
            name: 0.0
            for name in FLOAT_FIELDS
        }
    )
    values.update(overrides)

    return FighterRoundEvidence(**values)


@dataclass(frozen=True)
class FakeRecord:
    """Minimal segment record used to isolate aggregation logic."""

    state: object
    finish: object
    activity: object
    exposure: object


class FakeDistanceActivity:
    """Marker for a synthetic distance segment."""


class FakeClinchActivity:
    """Marker for a synthetic clinch segment."""


class FakeGroundActivity:
    """Marker for a synthetic ground segment."""


def fighter_activity(**values: int) -> SimpleNamespace:
    """Build a phase-specific fighter activity namespace."""

    return SimpleNamespace(**values)


def exposure(
    *,
    red_damage: float = 0.2,
    red_stress: float = 0.3,
    blue_damage: float = 0.5,
    blue_stress: float = 0.7,
) -> SimpleNamespace:
    """Build adversity received by each fighter."""

    return SimpleNamespace(
        red=SimpleNamespace(
            persistent_damage_exposure=red_damage,
            acute_stress_exposure=red_stress,
        ),
        blue=SimpleNamespace(
            persistent_damage_exposure=blue_damage,
            acute_stress_exposure=blue_stress,
        ),
    )


def distance_activity() -> FakeDistanceActivity:
    """Build one synthetic distance segment."""

    selected = FakeDistanceActivity()
    selected.red = fighter_activity(
        sig_str_attempted=10,
        sig_str_landed=5,
        knockdowns=0,
    )
    selected.blue = fighter_activity(
        sig_str_attempted=8,
        sig_str_landed=3,
        knockdowns=0,
    )

    return selected


def clinch_activity() -> FakeClinchActivity:
    """Build one synthetic clinch segment."""

    selected = FakeClinchActivity()
    selected.red = fighter_activity(
        clinch_str_attempted=4,
        clinch_str_landed=2,
        damaging_clinch_strikes=1,
        control_seconds=6,
    )
    selected.blue = fighter_activity(
        clinch_str_attempted=3,
        clinch_str_landed=1,
        damaging_clinch_strikes=0,
        control_seconds=4,
    )

    return selected


def ground_activity() -> FakeGroundActivity:
    """Build one synthetic ground segment."""

    selected = FakeGroundActivity()
    selected.red = fighter_activity(
        ground_str_attempted=5,
        ground_str_landed=3,
        control_seconds=15,
        submission_attempts=1,
        position_advancements=1,
        escape_attempts=0,
        reversal_attempts=1,
        scramble_attempts=0,
    )
    selected.blue = fighter_activity(
        ground_str_attempted=2,
        ground_str_landed=1,
        control_seconds=3,
        submission_attempts=0,
        position_advancements=0,
        escape_attempts=1,
        reversal_attempts=0,
        scramble_attempts=1,
    )

    return selected


def fake_completed_round(
    *,
    round_number: int = 2,
) -> tuple[FakeRecord, ...]:
    """Build a completed round containing all three phases."""

    activities = (
        distance_activity(),
        distance_activity(),
        distance_activity(),
        clinch_activity(),
        clinch_activity(),
        clinch_activity(),
        ground_activity(),
        ground_activity(),
        ground_activity(),
        ground_activity(),
    )

    # Add one knockdown to the opening distance segment.
    activities[0].red.knockdowns = 1

    return tuple(
        FakeRecord(
            state=SimpleNamespace(
                round_number=round_number,
                segment_number=segment_number,
            ),
            finish=None,
            activity=activity,
            exposure=exposure(),
        )
        for segment_number, activity in enumerate(
            activities,
            start=1,
        )
    )


def install_fake_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install synthetic record and activity classes."""

    monkeypatch.setattr(
        evidence_module,
        "FinishEvaluatedPathSegment",
        FakeRecord,
    )
    monkeypatch.setattr(
        evidence_module,
        "DistanceSegmentActivity",
        FakeDistanceActivity,
    )
    monkeypatch.setattr(
        evidence_module,
        "ClinchSegmentActivity",
        FakeClinchActivity,
    )
    monkeypatch.setattr(
        evidence_module,
        "GroundSegmentActivity",
        FakeGroundActivity,
    )


@pytest.mark.parametrize(
    "field_name",
    INTEGER_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_integer_evidence_fields_require_exact_integer(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be an integer",
    ):
        fighter_evidence(
            **{
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    INTEGER_FIELDS,
)
def test_integer_evidence_fields_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        fighter_evidence(
            **{
                field_name: -1,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    FLOAT_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        "1.0",
        None,
    ],
)
def test_float_evidence_fields_require_numeric_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=f"{field_name} must be numeric",
    ):
        fighter_evidence(
            **{
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    FLOAT_FIELDS,
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_float_evidence_fields_must_be_finite(
    field_name: str,
    invalid_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be finite",
    ):
        fighter_evidence(
            **{
                field_name: invalid_value,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    FLOAT_FIELDS,
)
def test_float_evidence_fields_cannot_be_negative(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} cannot be negative",
    ):
        fighter_evidence(
            **{
                field_name: -0.01,
            }
        )


def test_fighter_evidence_derived_properties() -> None:
    selected = fighter_evidence(
        distance_strikes_attempted=10,
        distance_strikes_landed=5,
        clinch_strikes_attempted=4,
        clinch_strikes_landed=2,
        ground_strikes_attempted=6,
        ground_strikes_landed=3,
        submission_attempts=2,
        position_advancements=3,
        reversal_attempts=1,
    )

    assert selected.total_strikes_attempted == 20
    assert selected.total_strikes_landed == 10
    assert selected.striking_accuracy == pytest.approx(0.50)
    assert selected.offensive_grappling_actions == 6


def test_zero_attempts_produce_zero_accuracy() -> None:
    assert fighter_evidence().striking_accuracy == 0.0


def test_fighter_evidence_is_immutable() -> None:
    selected = fighter_evidence()

    with pytest.raises(FrozenInstanceError):
        selected.knockdowns = 1


@pytest.mark.parametrize(
    "invalid_value",
    [
        1.0,
        True,
        "1",
    ],
)
def test_round_number_requires_exact_integer(
    invalid_value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="round_number must be an integer",
    ):
        RoundEvidence(
            round_number=invalid_value,
            red=fighter_evidence(),
            blue=fighter_evidence(),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        6,
    ],
)
def test_round_number_must_be_between_one_and_five(
    invalid_value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="round_number must be between 1 and 5",
    ):
        RoundEvidence(
            round_number=invalid_value,
            red=fighter_evidence(),
            blue=fighter_evidence(),
        )


@pytest.mark.parametrize(
    "side_name",
    [
        "red",
        "blue",
    ],
)
def test_round_evidence_requires_fighter_evidence(
    side_name: str,
) -> None:
    values = {
        "round_number": 1,
        "red": fighter_evidence(),
        "blue": fighter_evidence(),
    }
    values[side_name] = "invalid"

    with pytest.raises(
        TypeError,
        match=(
            f"{side_name} must be "
            "FighterRoundEvidence"
        ),
    ):
        RoundEvidence(**values)


def test_round_evidence_returns_selected_side() -> None:
    red = fighter_evidence(
        knockdowns=1,
    )
    blue = fighter_evidence(
        control_seconds=20,
    )
    selected = RoundEvidence(
        round_number=1,
        red=red,
        blue=blue,
    )

    assert selected.for_side(FighterSide.RED) is red
    assert selected.for_side(FighterSide.BLUE) is blue


def test_round_evidence_side_requires_fighter_side() -> None:
    selected = RoundEvidence(
        round_number=1,
        red=fighter_evidence(),
        blue=fighter_evidence(),
    )

    with pytest.raises(
        TypeError,
        match="side must be FighterSide",
    ):
        selected.for_side("red")


def test_round_evidence_is_immutable() -> None:
    selected = RoundEvidence(
        round_number=1,
        red=fighter_evidence(),
        blue=fighter_evidence(),
    )

    with pytest.raises(FrozenInstanceError):
        selected.round_number = 2


def test_exact_mixed_phase_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    selected = calculate_round_evidence(
        fake_completed_round()
    )

    assert selected.round_number == 2

    assert selected.red.distance_strikes_attempted == 30
    assert selected.red.distance_strikes_landed == 15
    assert selected.red.clinch_strikes_attempted == 12
    assert selected.red.clinch_strikes_landed == 6
    assert selected.red.ground_strikes_attempted == 20
    assert selected.red.ground_strikes_landed == 12
    assert selected.red.knockdowns == 1
    assert selected.red.damaging_clinch_strikes == 3
    assert selected.red.control_seconds == 78
    assert selected.red.submission_attempts == 4
    assert selected.red.position_advancements == 4
    assert selected.red.escape_attempts == 0
    assert selected.red.reversal_attempts == 4
    assert selected.red.scramble_attempts == 0

    assert selected.blue.distance_strikes_attempted == 24
    assert selected.blue.distance_strikes_landed == 9
    assert selected.blue.clinch_strikes_attempted == 9
    assert selected.blue.clinch_strikes_landed == 3
    assert selected.blue.ground_strikes_attempted == 8
    assert selected.blue.ground_strikes_landed == 4
    assert selected.blue.knockdowns == 0
    assert selected.blue.damaging_clinch_strikes == 0
    assert selected.blue.control_seconds == 24
    assert selected.blue.submission_attempts == 0
    assert selected.blue.position_advancements == 0
    assert selected.blue.escape_attempts == 4
    assert selected.blue.reversal_attempts == 0
    assert selected.blue.scramble_attempts == 4


def test_exposure_received_maps_to_opponent_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    selected = calculate_round_evidence(
        fake_completed_round()
    )

    # Blue received 0.5 damage and 0.7 stress per segment,
    # so those values were inflicted by red.
    assert (
        selected.red.persistent_damage_inflicted
        == pytest.approx(5.0)
    )
    assert (
        selected.red.acute_stress_inflicted
        == pytest.approx(7.0)
    )

    # Red received 0.2 damage and 0.3 stress per segment,
    # so those values were inflicted by blue.
    assert (
        selected.blue.persistent_damage_inflicted
        == pytest.approx(2.0)
    )
    assert (
        selected.blue.acute_stress_inflicted
        == pytest.approx(3.0)
    )


def test_real_path_aggregation_matches_phase_activity() -> None:
    """Verify aggregation against one real simulated completed round."""

    path = run_path(
        seed=2026,
        finish_calibration=zero_finish_calibration(),
    )
    segments = tuple(path.segments[:10])

    distance_segments = tuple(
        record
        for record in segments
        if isinstance(
            record.activity,
            DistanceSegmentActivity,
        )
    )
    clinch_segments = tuple(
        record
        for record in segments
        if isinstance(
            record.activity,
            ClinchSegmentActivity,
        )
    )
    ground_segments = tuple(
        record
        for record in segments
        if isinstance(
            record.activity,
            GroundSegmentActivity,
        )
    )

    assert (
        len(distance_segments)
        + len(clinch_segments)
        + len(ground_segments)
        == 10
    )

    selected = calculate_round_evidence(segments)

    assert selected.round_number == 1

    assert selected.red.distance_strikes_attempted == sum(
        record.activity.red.sig_str_attempted
        for record in distance_segments
    )
    assert selected.red.distance_strikes_landed == sum(
        record.activity.red.sig_str_landed
        for record in distance_segments
    )
    assert selected.red.knockdowns == sum(
        record.activity.red.knockdowns
        for record in distance_segments
    )

    assert selected.blue.distance_strikes_attempted == sum(
        record.activity.blue.sig_str_attempted
        for record in distance_segments
    )
    assert selected.blue.distance_strikes_landed == sum(
        record.activity.blue.sig_str_landed
        for record in distance_segments
    )
    assert selected.blue.knockdowns == sum(
        record.activity.blue.knockdowns
        for record in distance_segments
    )

    assert selected.red.clinch_strikes_attempted == sum(
        record.activity.red.clinch_str_attempted
        for record in clinch_segments
    )
    assert selected.red.clinch_strikes_landed == sum(
        record.activity.red.clinch_str_landed
        for record in clinch_segments
    )
    assert selected.red.damaging_clinch_strikes == sum(
        record.activity.red.damaging_clinch_strikes
        for record in clinch_segments
    )

    assert selected.blue.clinch_strikes_attempted == sum(
        record.activity.blue.clinch_str_attempted
        for record in clinch_segments
    )
    assert selected.blue.clinch_strikes_landed == sum(
        record.activity.blue.clinch_str_landed
        for record in clinch_segments
    )
    assert selected.blue.damaging_clinch_strikes == sum(
        record.activity.blue.damaging_clinch_strikes
        for record in clinch_segments
    )

    assert selected.red.ground_strikes_attempted == sum(
        record.activity.red.ground_str_attempted
        for record in ground_segments
    )
    assert selected.red.ground_strikes_landed == sum(
        record.activity.red.ground_str_landed
        for record in ground_segments
    )
    assert selected.red.submission_attempts == sum(
        record.activity.red.submission_attempts
        for record in ground_segments
    )
    assert selected.red.position_advancements == sum(
        record.activity.red.position_advancements
        for record in ground_segments
    )

    assert selected.blue.ground_strikes_attempted == sum(
        record.activity.blue.ground_str_attempted
        for record in ground_segments
    )
    assert selected.blue.ground_strikes_landed == sum(
        record.activity.blue.ground_str_landed
        for record in ground_segments
    )
    assert selected.blue.submission_attempts == sum(
        record.activity.blue.submission_attempts
        for record in ground_segments
    )
    assert selected.blue.position_advancements == sum(
        record.activity.blue.position_advancements
        for record in ground_segments
    )

    assert selected.red.control_seconds == sum(
        (
            record.activity.red.control_seconds
            if isinstance(
                record.activity,
                (
                    ClinchSegmentActivity,
                    GroundSegmentActivity,
                ),
            )
            else 0
        )
        for record in segments
    )
    assert selected.blue.control_seconds == sum(
        (
            record.activity.blue.control_seconds
            if isinstance(
                record.activity,
                (
                    ClinchSegmentActivity,
                    GroundSegmentActivity,
                ),
            )
            else 0
        )
        for record in segments
    )

    assert selected.red.persistent_damage_inflicted == pytest.approx(
        sum(
            record.exposure.blue.persistent_damage_exposure
            for record in segments
        )
    )
    assert selected.blue.persistent_damage_inflicted == pytest.approx(
        sum(
            record.exposure.red.persistent_damage_exposure
            for record in segments
        )
    )


def test_segments_must_be_tuple() -> None:
    with pytest.raises(
        TypeError,
        match="segments must be a tuple",
    ):
        calculate_round_evidence([])


def test_completed_round_requires_exactly_ten_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    with pytest.raises(
        ValueError,
        match="completed round must contain exactly 10 segments",
    ):
        calculate_round_evidence(
            fake_completed_round()[:-1]
        )


def test_segments_require_finish_path_records() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "segments must contain "
            "FinishEvaluatedPathSegment values"
        ),
    ):
        calculate_round_evidence(
            tuple(
                "invalid"
                for _ in range(10)
            )
        )


def test_finishing_round_cannot_be_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    segments = list(
        fake_completed_round()
    )
    segments[-1] = replace(
        segments[-1],
        finish=object(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "round evidence requires a completed "
            "non-finishing round"
        ),
    ):
        calculate_round_evidence(
            tuple(segments)
        )


def test_all_segments_must_share_round_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    segments = list(
        fake_completed_round()
    )
    segments[5] = replace(
        segments[5],
        state=SimpleNamespace(
            round_number=3,
            segment_number=6,
        ),
    )

    with pytest.raises(
        ValueError,
        match="all segments must belong to the same round",
    ):
        calculate_round_evidence(
            tuple(segments)
        )


def test_round_segments_must_be_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    segments = list(
        fake_completed_round()
    )
    segments[4] = replace(
        segments[4],
        state=SimpleNamespace(
            round_number=2,
            segment_number=6,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "round segments must be sequential "
            "from one through ten"
        ),
    ):
        calculate_round_evidence(
            tuple(segments)
        )


def test_unsupported_activity_contract_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_contracts(monkeypatch)

    segments = list(
        fake_completed_round()
    )
    segments[0] = replace(
        segments[0],
        activity=object(),
    )

    with pytest.raises(
        TypeError,
        match=(
            "segment activity must be a supported "
            "phase activity contract"
        ),
    ):
        calculate_round_evidence(
            tuple(segments)
        )
