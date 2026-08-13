"""Two-parameter temporal calibration of global strike-attempt exposure."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from ..calibration import DEFAULT_CONFIG_PATH, EventMCCalibration
from ..components.actions import ActionAttempt
from ..events import ConsequenceEvent, PrimaryEvent
from ..physiology import PhysiologyOutcome
from ..submission_finishes import SubmissionFinishOutcome
from .phase7b_kd_calibration import engine_for, temporal_cohorts
from .phase7i_strike_exposure_audit import historical_strikes
from .population_validation import METHODS, _fight

DISTANCE_GRID = (5.0, 5.5, 6.0, 6.5)
CLINCH_GRID = (1.2, 2.0, 2.8, 3.6)
PHASES = ("distance", "clinch", "ground")
STRIKE_PHASE = {"strike": "distance", "clinch_strike": "clinch", "ground_strike": "ground"}


def _ratio(numerator, denominator):
    """Return a diagnostic ratio without failing on sparse smoke cohorts."""
    return float(numerator / denominator) if denominator else 0.0


def calibration_for_rates(distance, clinch):
    """Return an isolated candidate; never mutate the load-once defaults."""
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["distance"]["strike_attempts_per_30s"] = float(distance)
    values["clinch"]["strike_attempts_per_30s"] = float(clinch)
    return EventMCCalibration(values, f"in-memory strike rates={distance}/{clinch}")


@dataclass
class StrikeCalibrationSink:
    attempts: Counter = field(default_factory=Counter)
    landed: Counter = field(default_factory=Counter)
    submission_attempts: int = 0
    submission_finishes: int = 0
    knockdowns: int = 0
    exposure_seconds: float = 0.0

    def on_time_advance(self, dt, before, after):
        self.exposure_seconds += dt

    def on_event(self, event, before, after):
        if isinstance(event, PrimaryEvent) and isinstance(event.payload, ActionAttempt):
            phase = STRIKE_PHASE.get(event.payload.action_family)
            if phase:
                self.attempts[phase] += 1
            elif event.payload.action_family == "submission_attempt":
                self.submission_attempts += 1
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, PhysiologyOutcome):
            self.landed[event.payload.phase] += 1
            self.knockdowns += int(event.payload.knockdown)
        elif isinstance(event, ConsequenceEvent) and isinstance(event.payload, SubmissionFinishOutcome):
            self.submission_finishes += int(event.payload.finished)

    def finalize(self):
        return {
            "attempts": dict(self.attempts), "landed": dict(self.landed),
            "submission_attempts": self.submission_attempts,
            "submission_finishes": self.submission_finishes,
            "knockdowns": self.knockdowns, "exposure_seconds": self.exposure_seconds,
        }


def evaluate(cohort, fsr, distance, clinch, paths, seed=20260813):
    calibration = calibration_for_rates(distance, clinch)
    historical = historical_strikes(cohort)
    target = historical["significant"]["attempts_per_15min"]
    started = time.perf_counter()
    rows = []
    for fight_index, (_, historical_row) in enumerate(cohort.iterrows()):
        fight = _fight(historical_row, fsr)
        for path_index in range(paths):
            sink = StrikeCalibrationSink()
            result = engine_for(
                fight, seed + fight_index * 100000 + path_index, sink, calibration
            ).run()
            stats = result.sink_result
            rows.append({
                "method": result.state.finish_method, "seconds": stats["exposure_seconds"],
                "finish_round": min(fight.rounds, int(stats["exposure_seconds"] // 300) + 1),
                "sub_attempts": stats["submission_attempts"], "sub_finishes": stats["submission_finishes"],
                "kds": stats["knockdowns"],
                **{f"{phase}_attempts": stats["attempts"].get(phase, 0) for phase in PHASES},
                **{f"{phase}_landed": stats["landed"].get(phase, 0) for phase in PHASES},
            })
    frame = pd.DataFrame(rows)
    exposure = frame.seconds.sum()
    attempts = sum(frame[f"{phase}_attempts"].sum() for phase in PHASES)
    landed = sum(frame[f"{phase}_landed"].sum() for phase in PHASES)
    attempts_15 = attempts / exposure * 900
    methods = frame.method.value_counts(normalize=True)
    nondecision = frame[frame.method != "DEC"]
    phase = {}
    for name in PHASES:
        phase_attempts = int(frame[f"{name}_attempts"].sum())
        phase_landed = int(frame[f"{name}_landed"].sum())
        phase[name] = {
            "attempts_per_path": phase_attempts / len(frame),
            "attempts_per_15min": phase_attempts / exposure * 900,
            "attempt_share": _ratio(phase_attempts, attempts),
            "landed_per_path": phase_landed / len(frame),
            "landed_per_15min": phase_landed / exposure * 900,
            "landed_share": _ratio(phase_landed, landed),
            "accuracy": _ratio(phase_landed, phase_attempts),
        }
    submissions = int(frame.sub_attempts.sum())
    return {
        "distance_rate": float(distance), "clinch_rate": float(clinch),
        "fights": len(cohort), "paths_per_fight": paths,
        "historical_significant_attempts_per_15min": target,
        "attempts_per_path": attempts / len(frame), "attempts_per_15min": attempts_15,
        "absolute_attempt_error": abs(attempts_15 - target), "exposure_ratio": attempts_15 / target,
        "phase": phase, "landed_per_path": landed / len(frame),
        "landed_per_15min": landed / exposure * 900,
        "landing_percentage": _ratio(landed, attempts),
        "method_shares": {method: float(methods.get(method, 0)) for method in METHODS},
        "kd_per_path": float(frame.kds.mean()), "kd_per_100_landed": _ratio(frame.kds.sum(), landed) * 100,
        "kd_per_15min": frame.kds.sum() / exposure * 900,
        "mean_fight_duration": float(frame.seconds.mean()),
        "mean_nondecision_finish_time": float(nondecision.seconds.mean()),
        "submission_attempts_per_path": float(frame.sub_attempts.mean()),
        "p_sub_given_attempt": float(frame.sub_finishes.sum() / submissions) if submissions else None,
        "finish_round_distribution": {
            str(round_number): _ratio(count, len(nondecision))
            for round_number, count in sorted(nondecision.finish_round.value_counts().items())
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def run(distance_grid=DISTANCE_GRID, clinch_grid=CLINCH_GRID, paths=3,
        train_limit=100, holdout_limit=50, seed=20260813,
        output=Path("data/diagnostics/event_mc_v1_phase7j.json")):
    train, holdout, fsr = temporal_cohorts(train_limit, holdout_limit)
    results = []
    for distance in distance_grid:
        for clinch in clinch_grid:
            results.append({
                "train": evaluate(train, fsr, distance, clinch, paths, seed),
                "holdout": evaluate(holdout, fsr, distance, clinch, paths, seed),
            })
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
    parser.add_argument("--distance-grid", nargs="+", type=float, default=DISTANCE_GRID)
    parser.add_argument("--clinch-grid", nargs="+", type=float, default=CLINCH_GRID)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7j.json"))
    run(**vars(parser.parse_args()))
