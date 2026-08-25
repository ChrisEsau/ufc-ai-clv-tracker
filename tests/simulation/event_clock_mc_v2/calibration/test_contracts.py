from dataclasses import asdict
import pandas as pd
import pytest
from pipeline.simulation.event_clock_mc_v2.calibration.cohort import (
    select_split,
    validate_manifest,
)
from pipeline.simulation.event_clock_mc_v2.calibration.config import (
    config_hash,
    resolve_overrides,
)
from pipeline.simulation.event_clock_mc_v2.calibration.ledger import (
    stable_experiment_id,
)
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    DEFAULT_MECHANICS_CALIBRATION_CONFIG,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows
from pipeline.common.paths import EVENT_CLOCK_V2_COHORT_MANIFEST_PATH


def manifest():
    rows = []
    for i, split in enumerate(
        ("development", "calibration", "validation", "final_holdout")
    ):
        rows.append(
            {
                "bout_id": str(i),
                "date": f"202{i+1}-01-01",
                "cohort_split": split,
                "red_fighter_id": f"r{i}",
                "blue_fighter_id": f"b{i}",
            }
        )
    return pd.DataFrame(rows)


def test_splits_are_unique_chronological_and_holdout_dark():
    frame = manifest()
    validate_manifest(frame)
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(PermissionError, match="dark"):
        select_split(frame, "final_holdout")


def test_matched_rng_excludes_candidate_config():
    baseline = derive_path_seed("v1", "bout-a", 7)
    candidate = derive_path_seed("v1", "bout-a", 7)  # config deliberately absent
    assert baseline == candidate
    assert baseline != derive_path_seed("v1", "bout-b", 7)
    assert baseline != derive_path_seed("v1", "bout-a", 8)


def test_overrides_are_allowlisted_and_do_not_mutate_defaults():
    before = asdict(DEFAULT_MECHANICS_CALIBRATION_CONFIG)
    resolved, explicit = resolve_overrides({"kd_slope": 2.5})
    assert resolved.kd_slope == 2.5 and explicit == {"kd_slope": 2.5}
    assert asdict(DEFAULT_MECHANICS_CALIBRATION_CONFIG) == before
    with pytest.raises(ValueError, match="unknown or frozen"):
        resolve_overrides({"standing_cadence": 3})


def test_hashes_and_experiment_ids_are_stable():
    value = {"b": 2, "a": 1}
    assert config_hash(value) == config_hash({"a": 1, "b": 2})
    assert stable_experiment_id(value) == stable_experiment_id({"a": 1, "b": 2})


def test_capability_reference_excludes_cutoff_and_future(monkeypatch):
    rows = []
    for i in range(60):
        rows.append(
            {
                "fighter_id": str(i),
                "event_date": "2020-01-01",
                "fight_id": f"old{i}",
                "x": float(i),
            }
        )
        rows.append(
            {
                "fighter_id": str(i),
                "event_date": "2030-01-01",
                "fight_id": f"future{i}",
                "x": 9999.0,
            }
        )
    seen = {}

    def fake(cls, frame):
        seen["max_date"] = pd.to_datetime(frame.event_date).max()
        seen["xmax"] = frame.x.max()
        return cls(pd.DataFrame({"standing_rate": [1] * 60}))

    monkeypatch.setattr(CapabilityReference, "from_frame", classmethod(fake))
    CapabilityReference.from_prefight_before(pd.DataFrame(rows), "2025-01-01")
    assert seen["max_date"] < pd.Timestamp("2025-01-01") and seen["xmax"] < 9999


def test_historical_state_is_exact_prefight_not_latest():
    snapshots = pd.DataFrame(
        [
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fight_id": "bout",
                "fighter_id": "red",
                "trait": 1.0,
            },
            {
                "event_date": pd.Timestamp("2020-01-01"),
                "fight_id": "bout",
                "fighter_id": "blue",
                "trait": 2.0,
            },
            {
                "event_date": pd.Timestamp("2030-01-01"),
                "fight_id": "future",
                "fighter_id": "red",
                "trait": 999.0,
            },
        ]
    )
    red, blue = historical_fighter_rows(
        snapshots, event_date="2020-01-01", fight_id="bout", fighter_ids=("red", "blue")
    )
    assert red.trait == 1.0 and blue.trait == 2.0


def test_frozen_manifest_contract():
    frame = pd.read_csv(EVENT_CLOCK_V2_COHORT_MANIFEST_PATH, dtype={"bout_id": str})
    validate_manifest(frame)
    assert frame.groupby("cohort_split").size().to_dict() == {
        "development": 400,
        "calibration": 200,
        "validation": 200,
        "final_holdout": 200,
    }
