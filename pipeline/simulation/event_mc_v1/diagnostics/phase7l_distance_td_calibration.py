"""One-parameter temporal calibration of DISTANCE takedown initiation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from pipeline.common.paths import ROUND_STATS_PATH

from ..calibration import DEFAULT_CONFIG_PATH, EventMCCalibration
from .phase7b_kd_calibration import temporal_cohorts
from .phase7k_takedown_decomposition import historical_takedowns, modeled_takedowns

COARSE_GRID = (0.10, 0.12, 0.14, 0.16, 0.18)
EXPECTED_ANCHORS = {
    "train": {
        "attempts_per_fight": 5.190,
        "attempts_per_15min": 6.169105605156109,
        "landed_per_fight": 1.810,
        "landed_per_15min": 2.1514607216440385,
        "success_percentage": 0.348747591522158,
    },
    "holdout": {
        "attempts_per_fight": 6.220,
        "attempts_per_15min": 7.2816670568953406,
        "landed_per_fight": 1.880,
        "landed_per_15min": 2.200889721376727,
        "success_percentage": 0.3022508038585209,
    },
}


def calibration_for_distance_td(base_30s: float) -> EventMCCalibration:
    """Return an isolated candidate changing only the authorized TD base."""
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["distance"]["td_attempt_base_30s"] = float(base_30s)
    return EventMCCalibration(values, f"in-memory distance TD base={base_30s}")


def validate_historical_anchors(name: str, observed: dict) -> None:
    expected = EXPECTED_ANCHORS[name]
    mismatches = {
        key: (observed[key], value)
        for key, value in expected.items()
        if abs(observed[key] - value) > 1e-9
    }
    if mismatches:
        raise RuntimeError(f"{name} historical TD anchors changed: {mismatches}")


def run(
    grid=COARSE_GRID,
    paths=3,
    train_limit=100,
    holdout_limit=50,
    seed=20260813,
    output=Path("data/diagnostics/event_mc_v1_phase7l.json"),
):
    train, holdout, fsr = temporal_cohorts(train_limit, holdout_limit)
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    cohorts = {"train": train, "holdout": holdout}
    historical = {
        name: historical_takedowns(cohort, rounds) for name, cohort in cohorts.items()
    }
    # The promotion cohort must reproduce the exact established Phase 7K anchors.
    if train_limit == 100 and holdout_limit == 50:
        for name, values in historical.items():
            validate_historical_anchors(name, values)
    results = []
    for base in grid:
        calibration = calibration_for_distance_td(base)
        results.append({
            "distance_td_base_30s": float(base),
            **{
                name: modeled_takedowns(cohort, fsr, paths, seed, calibration)
                for name, cohort in cohorts.items()
            },
        })
    report = {
        "train_dates": [str(train.event_date.min().date()), str(train.event_date.max().date())],
        "holdout_dates": [str(holdout.event_date.min().date()), str(holdout.event_date.max().date())],
        "historical": historical,
        "results_in_grid_order": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", nargs="+", type=float, default=COARSE_GRID)
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--holdout-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7l.json"))
    run(**vars(parser.parse_args()))
