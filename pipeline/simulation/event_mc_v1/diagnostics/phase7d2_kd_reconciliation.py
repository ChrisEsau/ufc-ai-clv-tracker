"""Measurement-only reconciliation of incompatible historical KD targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .phase7b_kd_calibration import evaluate
from .population_validation import build_cohort, observed_duration_seconds


MIDPOINTS = (32, 36, 40, 44, 48)


def historical_anchors(cohort):
    exposure = cohort.apply(observed_duration_seconds, axis=1).sum()
    landed = (cohort.r_total_str_landed + cohort.b_total_str_landed).sum()
    knockdowns = (cohort.r_kd + cohort.b_kd).sum()
    return {
        "kd_per_fight": float(knockdowns / len(cohort)),
        "kd_per_100_landed": float(knockdowns / landed * 100),
        "kd_per_15min": float(knockdowns / exposure * 900),
        "landed_per_fight": float(landed / len(cohort)),
        "landed_per_15min": float(landed / exposure * 900),
        "mean_fight_duration": float(exposure / len(cohort)),
    }


def run(paths=10, start_year=2020, limit=100, seed=20260813,
        midpoints=MIDPOINTS, output=Path("data/diagnostics/event_mc_v1_phase7d2.json")):
    cohort, fsr = build_cohort(start_year, limit)
    results = []
    for midpoint in midpoints:
        result = evaluate(cohort, fsr, midpoint, paths, seed)
        result.pop("normalized_error", None)  # Phase 7D2 explicitly forbids a combined objective.
        results.append(result)
    report = {
        "cohort_dates": [str(cohort.event_date.min().date()), str(cohort.event_date.max().date())],
        "historical": historical_anchors(cohort),
        "candidates_in_requested_order": results,
        "ranking": None,
        "ranking_note": "No combined optimization score or candidate ranking is authorized.",
        "submission_position_lock": "Top and bottom conversion must be 1:1 in future submission work; this measurement phase does not alter the currently committed model.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--midpoints", nargs="+", type=float, default=MIDPOINTS)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7d2.json"))
    run(**vars(parser.parse_args()))
