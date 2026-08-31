"""One-parameter temporal calibration of submission-finish incidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from ..calibration import DEFAULT_CONFIG_PATH, EventMCCalibration
from .phase7b_kd_calibration import temporal_cohorts
from .phase7g_submission_attempt_calibration import evaluate as evaluate_attempts
from .population_validation import METHODS, normalize_method


GRID = (-2.20, -1.90, -1.60, -1.30, -1.00, -0.70, -0.40)


def calibration_for_intercept(intercept):
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["submission_finish"]["intercept"] = float(intercept)
    return EventMCCalibration(values, f"in-memory submission intercept={intercept}")


def historical_methods(cohort):
    methods = cohort.method.map(normalize_method).value_counts(normalize=True)
    return {method: float(methods.get(method, 0)) for method in METHODS}


def evaluate(cohort, fsr, intercept, paths, seed=20260813):
    # Phase 7G supplies every required sufficient statistic; inject only the
    # in-memory intercept candidate while leaving attempt generation frozen.
    result = evaluate_attempts(cohort, fsr, .045, paths, seed, calibration=calibration_for_intercept(intercept))
    result["intercept"] = float(intercept)
    result.pop("base_30s", None)
    result["historical_method_shares"] = historical_methods(cohort)
    result["sub_share_error"] = abs(result["method_shares"]["SUB"] - result["historical_method_shares"]["SUB"])
    top = result["conversion_by_position"]["top"]["probability"]
    bottom = result["conversion_by_position"]["bottom"]["probability"]
    result["top_minus_bottom_conversion"] = None if top is None or bottom is None else top - bottom
    return result


def run(grid=GRID, paths=3, train_limit=100, holdout_limit=50, seed=20260813,
        output=Path("data/diagnostics/event_mc_v1_phase7h.json")):
    train, holdout, fsr = temporal_cohorts(train_limit, holdout_limit)
    results = [{"train": evaluate(train, fsr, value, paths, seed), "holdout": evaluate(holdout, fsr, value, paths, seed)} for value in grid]
    report = {
        "train_dates": [str(train.event_date.min().date()), str(train.event_date.max().date())],
        "holdout_dates": [str(holdout.event_date.min().date()), str(holdout.event_date.max().date())],
        "results_in_grid_order": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--holdout-limit", type=int, default=50)
    parser.add_argument("--grid", nargs="+", type=float, default=GRID)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7h.json"))
    run(**vars(parser.parse_args()))
