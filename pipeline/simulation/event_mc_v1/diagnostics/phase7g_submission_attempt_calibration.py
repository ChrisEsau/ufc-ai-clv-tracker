"""One-parameter temporal calibration of global submission-attempt exposure."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from ..calibration import DEFAULT_CONFIG_PATH, EventMCCalibration
from .phase7b_kd_calibration import engine_for, temporal_cohorts
from .phase7d_submission_decomposition import SubmissionDecompositionSink
from .population_validation import METHODS, _fight, normalize_method, observed_duration_seconds


GRID = (0.045, 0.050, 0.055, 0.060, 0.065, 0.070)


def calibration_for_base(base_30s):
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["submission_attempts"]["base_30s"] = float(base_30s)
    return EventMCCalibration(values, f"in-memory submission base={base_30s}")


def historical_anchors(cohort):
    attempts = cohort.r_sub_att + cohort.b_sub_att
    exposure = cohort.apply(observed_duration_seconds, axis=1).sum()
    return {
        "attempts_per_fight": float(attempts.mean()),
        "attempts_per_15min": float(attempts.sum() / exposure * 900),
        "fights_with_attempt_share": float((attempts > 0).mean()),
    }


def evaluate(cohort, fsr, base_30s, paths, seed=20260813, *, calibration=None):
    calibration = calibration or calibration_for_base(base_30s); started = time.perf_counter(); rows = []
    rounds = {}; positions = {"top": 0, "bottom": 0}; position_finishes = {"top": 0, "bottom": 0}
    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = _fight(historical, fsr)
        for path_index in range(paths):
            sink = SubmissionDecompositionSink()
            result = engine_for(fight, seed + fight_index * 100000 + path_index, sink, calibration).run()
            stats = result.sink_result; attempts = sum(stats["attempts_by_round"].values())
            for key, value in stats["attempts_by_round"].items(): rounds[key] = rounds.get(key, 0) + value
            for key in positions:
                positions[key] += stats["attempts_by_position"].get(key, 0)
                position_finishes[key] += stats["submission_finishes_by_position"].get(key, 0)
            rows.append({"fight": fight_index, "method": result.state.finish_method, "seconds": stats["exposure_seconds"], "ground": stats["ground_seconds"], "attempts": attempts, "sub_finishes": stats["submission_finishes"], "landed": stats["landed_strikes"], "kds": stats["knockdowns"]})
    frame = pd.DataFrame(rows); exposure = frame.seconds.sum(); ground = frame.ground.sum(); methods = frame.method.value_counts(normalize=True); nondec = frame[frame.method != "DEC"]
    finish_rounds = (np.maximum(nondec.seconds.to_numpy() - 1e-12, 0) // 300 + 1).astype(int); total_attempts = int(frame.attempts.sum()); total_subs = int(frame.sub_finishes.sum())
    return {
        "base_30s": float(base_30s), "fights": len(cohort), "paths_per_fight": paths, "historical": historical_anchors(cohort),
        "attempts_per_path": float(frame.attempts.mean()), "attempts_per_15min": float(total_attempts / exposure * 900), "paths_with_attempt_share": float((frame.attempts > 0).mean()), "total_attempts": total_attempts,
        "attempts_by_round": {str(key): int(value) for key, value in sorted(rounds.items())}, "attempts_by_position": positions,
        "attempts_per_15_position_ground_minutes": {key: float(value / ground * 900) if ground else None for key, value in positions.items()},
        "p_sub_given_attempt": float(total_subs / total_attempts) if total_attempts else None,
        "conversion_by_position": {key: {"attempts": positions[key], "finishes": position_finishes[key], "probability": float(position_finishes[key] / positions[key]) if positions[key] else None} for key in positions},
        "ground_seconds_per_path": float(frame.ground.mean()), "ground_seconds_per_15min": float(ground / exposure * 900),
        "method_shares": {method: float(methods.get(method, 0)) for method in METHODS}, "mean_fight_duration": float(frame.seconds.mean()), "mean_nondecision_finish_time": float(nondec.seconds.mean()),
        "finish_round_shares": {str(number): float(np.mean(finish_rounds == number)) for number in range(1, 6)},
        "kd_per_path": float(frame.kds.mean()), "kd_per_100_landed": float(frame.kds.sum() / frame.landed.sum() * 100), "kd_per_15min": float(frame.kds.sum() / exposure * 900), "runtime_seconds": time.perf_counter() - started,
    }


def run(grid=GRID, paths=3, train_limit=100, holdout_limit=50, seed=20260813, output=Path("data/diagnostics/event_mc_v1_phase7g.json")):
    train, holdout, fsr = temporal_cohorts(train_limit, holdout_limit); results = []
    for base in grid: results.append({"train": evaluate(train, fsr, base, paths, seed), "holdout": evaluate(holdout, fsr, base, paths, seed)})
    report = {"train_dates": [str(train.event_date.min().date()), str(train.event_date.max().date())], "holdout_dates": [str(holdout.event_date.min().date()), str(holdout.event_date.max().date())], "results_in_grid_order": results}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True)); print(json.dumps(report, indent=2, sort_keys=True)); return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--paths", type=int, default=3); parser.add_argument("--train-limit", type=int, default=100); parser.add_argument("--holdout-limit", type=int, default=50); parser.add_argument("--grid", nargs="+", type=float, default=GRID); parser.add_argument("--seed", type=int, default=20260813); parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7g.json")); run(**vars(parser.parse_args()))
