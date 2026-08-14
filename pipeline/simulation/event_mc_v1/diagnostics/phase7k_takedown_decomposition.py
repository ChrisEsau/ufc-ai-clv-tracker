"""Measurement-only historical/model takedown decomposition for Phase 7K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..flow_stats import FlowStatsSink
from ..calibration import DEFAULT_CALIBRATION, EventMCCalibration
from .phase7b_kd_calibration import engine_for
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import METHODS, _fight, observed_duration_seconds

TD_SOURCES = {"distance": "takedown", "clinch": "clinch_takedown"}
STRIKE_FAMILIES = ("strike", "clinch_strike", "ground_strike")


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def source_totals(distance_attempts, clinch_attempts, distance_landed, clinch_landed):
    """Keep the public total/source accounting invariant explicit and testable."""
    return distance_attempts + clinch_attempts, distance_landed + clinch_landed


def historical_takedowns(cohort: pd.DataFrame, rounds: pd.DataFrame) -> dict:
    """Aggregate both corners' round-level UFCStats TD fields by fight."""
    required = {"fight_id", "corner", "round", "td_attempted", "td_landed"}
    missing = required.difference(rounds.columns)
    if missing:
        raise ValueError(f"round stats missing required TD columns: {sorted(missing)}")
    selected = rounds[rounds.fight_id.astype(str).isin(cohort.fight_id.astype(str))].copy()
    selected["td_attempted"] = pd.to_numeric(selected.td_attempted, errors="coerce").fillna(0)
    selected["td_landed"] = pd.to_numeric(selected.td_landed, errors="coerce").fillna(0)
    if (selected.td_landed > selected.td_attempted).any():
        raise ValueError("round stats contain TD landed greater than attempted")
    fight_totals = selected.groupby("fight_id", as_index=False)[["td_attempted", "td_landed"]].sum()
    # Retain zero-action cohort fights even if their round rows are unexpectedly absent.
    fight_totals = cohort[["fight_id"]].merge(fight_totals, on="fight_id", how="left").fillna(0)
    exposure = float(cohort.apply(observed_duration_seconds, axis=1).sum())
    attempts, landed = float(fight_totals.td_attempted.sum()), float(fight_totals.td_landed.sum())
    return {
        "source": str(ROUND_STATS_PATH),
        "field_mapping": {"attempts": "td_attempted", "landed": "td_landed"},
        "round_rows": int(len(selected)),
        "fights": int(len(cohort)),
        "exposure_seconds": exposure,
        "total_attempts": attempts,
        "total_landed": landed,
        "attempts_per_fight": attempts / len(cohort),
        "landed_per_fight": landed / len(cohort),
        "attempts_per_15min": _ratio(attempts * 900, exposure),
        "landed_per_15min": _ratio(landed * 900, exposure),
        "success_percentage": _ratio(landed, attempts),
        "fights_with_attempt_share": float((fight_totals.td_attempted >= 1).mean()),
        "fights_with_landed_share": float((fight_totals.td_landed >= 1).mean()),
        "zero_attempt_share": float((fight_totals.td_attempted == 0).mean()),
        "multi_attempt_share": float((fight_totals.td_attempted >= 2).mean()),
        "attempt_quantiles": {str(q): float(fight_totals.td_attempted.quantile(q)) for q in (.25, .5, .75)},
        "landed_quantiles": {str(q): float(fight_totals.td_landed.quantile(q)) for q in (.25, .5, .75)},
    }


def modeled_takedowns(
    cohort: pd.DataFrame,
    fsr: pd.DataFrame,
    paths: int,
    seed: int,
    calibration: EventMCCalibration = DEFAULT_CALIBRATION,
) -> dict:
    rows = []
    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = _fight(historical, fsr)
        for path_index in range(paths):
            result = engine_for(
                fight, seed + fight_index * 100000 + path_index,
                FlowStatsSink(), calibration,
            ).run()
            stats = result.sink_result
            attempts = {
                source: sum(stats["attempts"][side].get(family, 0) for side in ("red", "blue"))
                for source, family in TD_SOURCES.items()
            }
            landed = {
                source: sum(stats["outcomes"][side].get(f"{family}_landed", 0) for side in ("red", "blue"))
                for source, family in TD_SOURCES.items()
            }
            strike_attempts = {
                phase: sum(stats["attempts"][side].get(family, 0) for side in ("red", "blue"))
                for phase, family in zip(("distance", "clinch", "ground"), STRIKE_FAMILIES)
            }
            strike_landed = sum(
                stats["outcomes"][side].get(f"{family}_landed", 0)
                for side in ("red", "blue") for family in STRIKE_FAMILIES
            )
            strike_landed_by_phase = {
                phase: sum(stats["outcomes"][side].get(f"{family}_landed", 0) for side in ("red", "blue"))
                for phase, family in zip(("distance", "clinch", "ground"), STRIKE_FAMILIES)
            }
            sub_attempts = sum(stats["attempts"][side].get("submission_attempt", 0) for side in ("red", "blue"))
            rows.append({
                "seconds": result.state.fight_time_seconds,
                "method": result.state.finish_method,
                "distance_attempts": attempts["distance"], "clinch_attempts": attempts["clinch"],
                "distance_landed": landed["distance"], "clinch_landed": landed["clinch"],
                "strike_attempts": sum(strike_attempts.values()), "strike_landed": strike_landed,
                **{f"strike_{phase}_attempts": value for phase, value in strike_attempts.items()},
                **{f"strike_{phase}_landed": value for phase, value in strike_landed_by_phase.items()},
                "kds": sum(int(item.knockdown) for item in stats["physiology"]),
                "ground_seconds": stats["phase_seconds"]["ground"],
                "clinch_seconds": stats["phase_seconds"]["clinch"],
                "distance_seconds": stats["phase_seconds"]["distance"],
                "sub_attempts": sub_attempts,
                "sub_finishes": sum(int(item.finished) for item in stats["submission_checks"]),
            })
    frame = pd.DataFrame(rows)
    exposure = float(frame.seconds.sum())
    total_attempts, total_landed = source_totals(
        frame.distance_attempts, frame.clinch_attempts,
        frame.distance_landed, frame.clinch_landed,
    )

    def metrics(attempts: pd.Series, landed: pd.Series) -> dict:
        a, l = float(attempts.sum()), float(landed.sum())
        return {
            "attempts": a, "landed": l,
            "attempts_per_path": a / len(frame), "landed_per_path": l / len(frame),
            "attempts_per_15min": _ratio(a * 900, exposure),
            "landed_per_15min": _ratio(l * 900, exposure),
            "success_percentage": _ratio(l, a),
            "attempt_share": _ratio(a, float(total_attempts.sum())),
            "landed_share": _ratio(l, float(total_landed.sum())),
        }

    methods = frame.method.value_counts(normalize=True)
    nondecision = frame[frame.method != "DEC"]
    strike_attempts = float(frame.strike_attempts.sum())
    strike_landed = float(frame.strike_landed.sum())
    submissions = float(frame.sub_attempts.sum())
    return {
        "paths": int(len(frame)), "paths_per_fight": paths, "exposure_seconds": exposure,
        "total": {**metrics(total_attempts, total_landed),
                  "paths_with_attempt_share": float((total_attempts >= 1).mean()),
                  "paths_with_landed_share": float((total_landed >= 1).mean()),
                  "zero_attempt_share": float((total_attempts == 0).mean()),
                  "multi_attempt_share": float((total_attempts >= 2).mean()),
                  "attempt_quantiles": {
                      str(q): float(total_attempts.quantile(q)) for q in (.25, .5, .75)
                  },
                  "landed_quantiles": {
                      str(q): float(total_landed.quantile(q)) for q in (.25, .5, .75)
                  }},
        "entry_source": {
            "distance": metrics(frame.distance_attempts, frame.distance_landed),
            "clinch": metrics(frame.clinch_attempts, frame.clinch_landed),
        },
        "guardrails": {
            "strike_attempts_per_path": strike_attempts / len(frame),
            "strike_attempts_per_15min": _ratio(strike_attempts * 900, exposure),
            "strike_landed_per_path": strike_landed / len(frame),
            "strike_landed_per_15min": _ratio(strike_landed * 900, exposure),
            "strike_landing_percentage": _ratio(strike_landed, strike_attempts),
            "strike_attempt_shares": {
                phase: _ratio(float(frame[f"strike_{phase}_attempts"].sum()), strike_attempts)
                for phase in ("distance", "clinch", "ground")
            },
            "strike_phase": {
                phase: {
                    "attempts_per_path": float(frame[f"strike_{phase}_attempts"].mean()),
                    "attempts_per_15min": _ratio(float(frame[f"strike_{phase}_attempts"].sum()) * 900, exposure),
                    "landed_per_path": float(frame[f"strike_{phase}_landed"].mean()),
                    "landed_per_15min": _ratio(float(frame[f"strike_{phase}_landed"].sum()) * 900, exposure),
                    "accuracy": _ratio(float(frame[f"strike_{phase}_landed"].sum()), float(frame[f"strike_{phase}_attempts"].sum())),
                    "attempt_share": _ratio(float(frame[f"strike_{phase}_attempts"].sum()), strike_attempts),
                    "landed_share": _ratio(float(frame[f"strike_{phase}_landed"].sum()), strike_landed),
                }
                for phase in ("distance", "clinch", "ground")
            },
            "ground_seconds_per_path": float(frame.ground_seconds.mean()),
            "ground_seconds_per_15min": _ratio(float(frame.ground_seconds.sum()) * 900, exposure),
            "clinch_seconds_per_path": float(frame.clinch_seconds.mean()),
            "clinch_seconds_per_15min": _ratio(float(frame.clinch_seconds.sum()) * 900, exposure),
            "distance_seconds_per_path": float(frame.distance_seconds.mean()),
            "submission_attempts_per_path": float(frame.sub_attempts.mean()),
            "submission_attempts_per_15min": _ratio(submissions * 900, exposure),
            "paths_with_submission_attempt_share": float((frame.sub_attempts > 0).mean()),
            "p_sub_given_attempt": _ratio(float(frame.sub_finishes.sum()), submissions),
            "method_shares": {method: float(methods.get(method, 0)) for method in METHODS},
            "kd_per_path": float(frame.kds.mean()),
            "kd_per_15min": _ratio(float(frame.kds.sum()) * 900, exposure),
            "kd_per_100_landed": _ratio(float(frame.kds.sum()) * 100, strike_landed),
            "zero_kd_share": float((frame.kds == 0).mean()),
            "multi_kd_share": float((frame.kds >= 2).mean()),
            "mean_fight_duration": float(frame.seconds.mean()),
            "mean_nondecision_finish_time": float(nondecision.seconds.mean()),
        },
    }


def run(paths=10, seed=20260813, output=Path("data/diagnostics/event_mc_v1_phase7k.json")):
    train, holdout, fsr = temporal_cohorts(100, 50)
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    report = {}
    for name, cohort in (("train", train), ("holdout", holdout)):
        report[name] = {
            "dates": [str(cohort.event_date.min().date()), str(cohort.event_date.max().date())],
            "historical": historical_takedowns(cohort, rounds),
            "simulated": modeled_takedowns(cohort, fsr, paths, seed),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7k.json"))
    run(**vars(parser.parse_args()))
