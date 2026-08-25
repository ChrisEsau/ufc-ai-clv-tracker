from dataclasses import asdict
import inspect
import json
import pandas as pd
import pytest
from pipeline.simulation.event_clock_mc_v2.calibration.cohort import (
    build_manifest,
    select_split,
    validate_manifest,
    validate_manifest_prefight_contract,
)
from pipeline.simulation.event_clock_mc_v2.calibration.config import (
    config_hash,
    resolve_overrides,
)
from pipeline.simulation.event_clock_mc_v2.calibration.ledger import (
    build_record,
    metrics_fingerprint,
    stable_experiment_id,
)
from pipeline.simulation.event_clock_mc_v2.calibration import runner
from pipeline.simulation.event_clock_mc_v2.calibration.seeds import derive_path_seed
from pipeline.simulation.event_clock_mc_v2.mechanics.config import (
    DEFAULT_MECHANICS_CALIBRATION_CONFIG,
)
from pipeline.simulation.event_clock_mc_v2.standard_fighter_v1.capability_translation import (
    CapabilityReference,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import historical_fighter_rows
from pipeline.common.paths import (
    EVENT_CLOCK_V2_COHORT_MANIFEST_PATH,
    EVENT_CLOCK_V2_HISTORICAL_TARGETS_PATH,
)


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
    baseline, explicit = resolve_overrides({})
    assert baseline is DEFAULT_MECHANICS_CALIBRATION_CONFIG
    assert explicit == {}


def test_hashes_and_experiment_ids_are_stable():
    value = {"b": 2, "a": 1}
    assert config_hash(value) == config_hash({"a": 1, "b": 2})
    assert stable_experiment_id(value) == stable_experiment_id({"a": 1, "b": 2})
    assert metrics_fingerprint(value, {"status": "PASS"}) == metrics_fingerprint(
        {"a": 1, "b": 2}, {"status": "PASS"}
    )


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


def test_historical_runner_never_loads_latest_profiles():
    source = inspect.getsource(runner)
    assert "load_latest_profiles" not in source
    assert "CapabilityReference.from_prefight_before" in source


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


def test_historical_targets_are_frozen_grouped_and_digest_bound():
    targets = json.loads(EVENT_CLOCK_V2_HISTORICAL_TARGETS_PATH.read_text())
    digest = targets.pop("target_digest")
    assert digest == config_hash(targets)
    assert set(targets["metric_groups"]) == {
        "structural_targets",
        "physiology_targets",
        "discrimination_diagnostics",
        "predictive_diagnostics",
        "invariants",
        "diagnostic_only",
    }
    assert targets["historical_comparator_split"] == "calibration"
    assert targets["acceptance_bands"]["submission_fight_share"]["diagnostic_only"]


def test_scheduled_horizon_does_not_use_realized_finish_time():
    fight = pd.Series({"total_rounds": 3, "match_time_sec": 75})
    assert runner.scheduled_horizon_seconds(fight, "bout") == 900.0


def test_exact_prefight_contract_rejects_missing_wrong_date_and_duplicates():
    frame = manifest().iloc[[0]].copy()
    snapshots = pd.DataFrame(
        [
            {"event_date": "2021-01-01", "fight_id": "0", "fighter_id": "r0"},
            {"event_date": "2021-01-01", "fight_id": "0", "fighter_id": "b0"},
        ]
    )
    validate_manifest_prefight_contract(frame, snapshots)
    with pytest.raises(ValueError, match="exact historical prefight"):
        validate_manifest_prefight_contract(frame, snapshots.iloc[[0]])
    with pytest.raises(ValueError, match="exact historical prefight"):
        validate_manifest_prefight_contract(
            frame, pd.concat([snapshots, snapshots.iloc[[0]]], ignore_index=True)
        )
    wrong_date = snapshots.assign(event_date="2021-01-02")
    with pytest.raises(ValueError, match="exact historical prefight"):
        validate_manifest_prefight_contract(frame, wrong_date)


def test_acceptance_failures_control_ledger_status(monkeypatch):
    monkeypatch.setattr(
        "pipeline.simulation.event_clock_mc_v2.calibration.ledger.git_sha",
        lambda: "abc",
    )
    record = build_record(
        identity={"run": 1},
        config={},
        metrics={"acceptance_results": {"rate": {"status": "FAIL"}}},
        invariants={"status": "PASS", "counts": {}},
    )
    assert record["run_status"] == "FAIL"


def test_maturity_counts_exclude_all_same_date_fights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.simulation.event_clock_mc_v2.calibration.cohort.SPLIT_COUNTS",
        {"development": 1, "calibration": 1, "validation": 1, "final_holdout": 1},
    )
    raw = [
        ("same-a1", "2020-01-01", "a", "x1"),
        ("same-a2", "2020-01-01", "a", "x2"),
        ("same-b1", "2020-01-01", "b", "y1"),
        ("same-b2", "2020-01-01", "b", "y2"),
        ("same-leak", "2020-01-01", "a", "b"),
        ("eligible-1", "2020-02-01", "a", "b"),
        ("eligible-2", "2020-03-01", "a", "b"),
        ("eligible-3", "2020-04-01", "a", "b"),
        ("eligible-4", "2020-05-01", "a", "b"),
    ]
    master = pd.DataFrame(
        [
            {
                "fight_id": fight_id,
                "date": date,
                "r_id": red,
                "b_id": blue,
                "r_name": red,
                "b_name": blue,
            }
            for fight_id, date, red, blue in raw
        ]
    )
    rounds = pd.DataFrame({"fight_id": [row[0] for row in raw]})
    snapshots = pd.DataFrame(
        [
            {"fight_id": fight_id, "event_date": date, "fighter_id": fighter}
            for fight_id, date, red, blue in raw
            for fighter in (red, blue)
        ]
    )
    paths = [
        tmp_path / name for name in ("master.parquet", "rounds.parquet", "fsr.parquet")
    ]
    for frame, path in zip((master, rounds, snapshots), paths):
        frame.to_parquet(path, index=False)
    result, audit = build_manifest(*paths)
    assert "same-leak" not in set(result.bout_id)
    first = result.loc[result.bout_id.eq("eligible-1")].iloc[0]
    assert first.red_prior_ufc_fights == 3
    assert first.blue_prior_ufc_fights == 3
    assert audit["eligible_fights_threshold_2"] == 4
