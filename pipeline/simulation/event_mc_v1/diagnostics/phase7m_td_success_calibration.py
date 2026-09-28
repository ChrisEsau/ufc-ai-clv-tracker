"""Shared TD-success calibration and complete Phase 7 global comparison."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from pipeline.common.paths import ROUND_STATS_PATH

from ..calibration import DEFAULT_CALIBRATION, DEFAULT_CONFIG_PATH, EventMCCalibration
from .phase7b_kd_calibration import temporal_cohorts
from .phase7i_strike_exposure_audit import historical_strikes
from .phase7k_takedown_decomposition import historical_takedowns, modeled_takedowns
from .phase7l_distance_td_calibration import validate_historical_anchors
from .population_validation import METHODS, normalize_method, observed_duration_seconds

COARSE_GRID = (-0.40, -0.55, -0.70, -0.85, -1.00)
UNAVAILABLE = "historical comparator unavailable"
REQUIRED_REPORT_FAMILIES = (
    "fight.", "outcomes.", "strikes.", "takedowns.", "knockdowns.",
    "submissions.", "phase_residence.",
)


def calibration_for_td_success(offset: float) -> EventMCCalibration:
    """Change only the shared TD success offset in an isolated candidate."""
    values = deepcopy(yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())["defaults"])
    values["distance"]["td_success_logit_offset"] = float(offset)
    return EventMCCalibration(values, f"in-memory TD success offset={offset}")


def current_calibration_values(calibration: EventMCCalibration) -> dict:
    distance, clinch, ground = (calibration.section(x) for x in ("distance", "clinch", "ground"))
    attempts, submission = calibration.section("submission_attempts"), calibration.section("submission_finish")
    return {
        "distance_strike_attempts_per_30s": distance["strike_attempts_per_30s"],
        "clinch_strike_attempts_per_30s": clinch["strike_attempts_per_30s"],
        "ground_strike_attempts_per_30s": ground["strike_attempts_per_30s"],
        "distance_strike_accuracy": distance["strike_accuracy"],
        "clinch_strike_accuracy": clinch["strike_accuracy"],
        "ground_strike_accuracy": ground["strike_accuracy"],
        "distance_td_attempt_base_30s": distance["td_attempt_base_30s"],
        "clinch_td_attempt_base_30s": clinch["td_attempt_base_30s"],
        "td_success_logit_offset": distance["td_success_logit_offset"],
        "submission_attempt_base_30s": attempts["base_30s"],
        "submission_bottom_multiplier": attempts["bottom_multiplier"],
        "submission_conversion_intercept": submission["intercept"],
        "submission_top_bonus": submission["top_position_bonus"],
        "submission_bottom_bonus": submission["bottom_position_bonus"],
        "kd_midpoint": calibration.section("knockdown")["midpoint_impact_ratio"],
        "finish_midpoint": calibration.section("finish")["midpoint_impact_ratio"],
    }


def historical_global(cohort: pd.DataFrame, rounds: pd.DataFrame) -> dict:
    exposure = float(cohort.apply(observed_duration_seconds, axis=1).sum())
    strikes = historical_strikes(cohort)
    td = historical_takedowns(cohort, rounds)
    methods = cohort.method.map(normalize_method).value_counts(normalize=True)
    nondecision = cohort[cohort.method.map(normalize_method) != "DEC"]
    kd = float((cohort.r_kd + cohort.b_kd).sum())
    sig_landed = float((cohort.r_sig_str_landed + cohort.b_sig_str_landed).sum())
    sub_attempts = cohort.r_sub_att + cohort.b_sub_att
    return {
        "fights": len(cohort), "exposure_seconds": exposure,
        "mean_fight_duration": exposure / len(cohort),
        "mean_nondecision_finish_time": float(nondecision.apply(observed_duration_seconds, axis=1).mean()),
        "method_shares": {method: float(methods.get(method, 0)) for method in METHODS},
        "strikes": strikes, "takedowns": td,
        "knockdowns": {
            "per_fight": kd / len(cohort), "per_15min": kd / exposure * 900,
            "per_100_landed": kd / sig_landed * 100,
            "zero_share": float(((cohort.r_kd + cohort.b_kd) == 0).mean()),
            "multi_share": float(((cohort.r_kd + cohort.b_kd) >= 2).mean()),
            "semantic_note": "KD/100 uses historical UFCStats significant landed versus modeled meaningful landed strikes; comparator is close but not definition-identical.",
        },
        "submissions": {
            "attempts_per_fight": float(sub_attempts.mean()),
            "attempts_per_15min": float(sub_attempts.sum() / exposure * 900),
            "fights_with_attempt_share": float((sub_attempts > 0).mean()),
            "p_sub_given_attempt": UNAVAILABLE,
            "sub_outcome_share": float(methods.get("SUB", 0)),
        },
        "phase_residence": {phase: UNAVAILABLE for phase in ("distance", "clinch", "ground")},
    }


def _row(metric: str, historical, simulated, percentage=False) -> dict:
    if isinstance(historical, str) or historical is None:
        return {"metric": metric, "historical": historical or UNAVAILABLE, "event_mc": simulated,
                "absolute_difference": None, "relative_difference_percent": None,
                "percentage_point_difference": None}
    difference = float(simulated - historical)
    return {"metric": metric, "historical": float(historical), "event_mc": float(simulated),
            "absolute_difference": difference,
            "relative_difference_percent": float(difference / historical * 100) if historical else None,
            "percentage_point_difference": difference * 100 if percentage else None}


def global_comparison(cohort: pd.DataFrame, rounds: pd.DataFrame, simulated: dict) -> dict:
    historical = historical_global(cohort, rounds)
    total, guard = simulated["total"], simulated["guardrails"]
    sig, phases = historical["strikes"]["significant"], historical["strikes"]["significant_by_phase"]
    rows = [
        _row("fight.historical_fights", historical["fights"], historical["fights"]),
        _row("fight.simulated_paths", UNAVAILABLE, simulated["paths"]),
        _row("fight.mean_duration_seconds", historical["mean_fight_duration"], guard["mean_fight_duration"]),
        _row("fight.mean_nondecision_finish_seconds", historical["mean_nondecision_finish_time"], guard["mean_nondecision_finish_time"]),
    ]
    rows += [_row(f"outcomes.{method}", historical["method_shares"][method], guard["method_shares"][method], True) for method in METHODS]
    rows += [
        _row("strikes.attempts_per_fight_or_path", sig["attempts_per_fight"], guard["strike_attempts_per_path"]),
        _row("strikes.attempts_per_15min", sig["attempts_per_15min"], guard["strike_attempts_per_15min"]),
        _row("strikes.landed_per_fight_or_path", sig["landed_per_fight"], guard["strike_landed_per_path"]),
        _row("strikes.landed_per_15min", sig["landed_per_15min"], guard["strike_landed_per_15min"]),
        _row("strikes.accuracy", sig["landing_percentage"], guard["strike_landing_percentage"], True),
    ]
    for phase in ("distance", "clinch", "ground"):
        h, m = phases[phase], guard["strike_phase"][phase]
        for key in ("attempts_per_15min", "landed_per_15min", "accuracy", "attempt_share", "landed_share"):
            historical_key = "landing_percentage" if key == "accuracy" else key
            rows.append(_row(f"strikes.{phase}.{key}", h[historical_key], m[key], key in {"accuracy", "attempt_share", "landed_share"}))
    htd = historical["takedowns"]
    for metric, hkey, mkey, percent in (
        ("attempts_per_fight_or_path", "attempts_per_fight", "attempts_per_path", False),
        ("attempts_per_15min", "attempts_per_15min", "attempts_per_15min", False),
        ("completed_per_fight_or_path", "landed_per_fight", "landed_per_path", False),
        ("completed_per_15min", "landed_per_15min", "landed_per_15min", False),
        ("success", "success_percentage", "success_percentage", True),
        ("with_attempt_share", "fights_with_attempt_share", "paths_with_attempt_share", True),
        ("with_completion_share", "fights_with_landed_share", "paths_with_landed_share", True),
        ("zero_attempt_share", "zero_attempt_share", "zero_attempt_share", True),
        ("multi_attempt_share", "multi_attempt_share", "multi_attempt_share", True),
    ): rows.append(_row(f"takedowns.{metric}", htd[hkey], total[mkey], percent))
    rows += [
        _row("takedowns.attempt_q25", htd["attempt_quantiles"]["0.25"], total["attempt_quantiles"]["0.25"]),
        _row("takedowns.attempt_median", htd["attempt_quantiles"]["0.5"], total["attempt_quantiles"]["0.5"]),
        _row("takedowns.attempt_q75", htd["attempt_quantiles"]["0.75"], total["attempt_quantiles"]["0.75"]),
        _row("takedowns.completion_q25", htd["landed_quantiles"]["0.25"], total["landed_quantiles"]["0.25"]),
        _row("takedowns.completion_median", htd["landed_quantiles"]["0.5"], total["landed_quantiles"]["0.5"]),
        _row("takedowns.completion_q75", htd["landed_quantiles"]["0.75"], total["landed_quantiles"]["0.75"]),
    ]
    hk = historical["knockdowns"]
    rows += [
        _row("knockdowns.per_fight_or_path", hk["per_fight"], guard["kd_per_path"]),
        _row("knockdowns.per_15min", hk["per_15min"], guard["kd_per_15min"]),
        _row("knockdowns.per_100_comparable_landed", hk["per_100_landed"], guard["kd_per_100_landed"]),
        _row("knockdowns.zero_share", hk["zero_share"], guard["zero_kd_share"], True),
        _row("knockdowns.multi_share", hk["multi_share"], guard["multi_kd_share"], True),
    ]
    hs = historical["submissions"]
    rows += [
        _row("submissions.attempts_per_fight_or_path", hs["attempts_per_fight"], guard["submission_attempts_per_path"]),
        _row("submissions.attempts_per_15min", hs["attempts_per_15min"], guard["submission_attempts_per_15min"]),
        _row("submissions.with_attempt_share", hs["fights_with_attempt_share"], guard["paths_with_submission_attempt_share"], True),
        _row("submissions.p_sub_given_attempt", hs["p_sub_given_attempt"], guard["p_sub_given_attempt"], True),
        _row("submissions.outcome_share", hs["sub_outcome_share"], guard["method_shares"]["SUB"], True),
    ]
    for phase in ("distance", "clinch", "ground"):
        rows.append(_row(f"phase_residence.{phase}_seconds_per_path", UNAVAILABLE, guard[f"{phase}_seconds_per_path"]))
    return {"rows": rows, "historical": historical,
            "semantic_notes": {"phase_time": UNAVAILABLE, "kd_per_100_landed": hk["semantic_note"]}}


def render_comparison(name: str, comparison: dict) -> str:
    lines = [f"\n{name.upper()} HISTORICAL VS EVENT MC", "Metric | Historical | EVENT MC | Abs Diff | Rel Diff % | PP Diff"]
    for row in comparison["rows"]:
        lines.append(" | ".join(str(row[key]) for key in ("metric", "historical", "event_mc", "absolute_difference", "relative_difference_percent", "percentage_point_difference")))
    return "\n".join(lines)


def validate_global_comparison(comparison: dict) -> None:
    metrics = tuple(row["metric"] for row in comparison["rows"])
    missing = [family for family in REQUIRED_REPORT_FAMILIES if not any(metric.startswith(family) for metric in metrics)]
    if missing: raise RuntimeError(f"global comparison missing metric families: {missing}")
    residence = [row for row in comparison["rows"] if row["metric"].startswith("phase_residence.")]
    if not residence or any(row["historical"] != UNAVAILABLE for row in residence):
        raise RuntimeError("historical phase residence must be explicitly unavailable")


def run(grid=COARSE_GRID, paths=3, train_limit=100, holdout_limit=50, seed=20260813,
        final_report=False, output=Path("data/diagnostics/event_mc_v1_phase7m.json")):
    train, holdout, fsr = temporal_cohorts(train_limit, holdout_limit)
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    cohorts = {"train": train, "holdout": holdout}
    historical = {name: historical_takedowns(cohort, rounds) for name, cohort in cohorts.items()}
    if train_limit == 100 and holdout_limit == 50:
        for name, values in historical.items(): validate_historical_anchors(name, values)
    results = []
    for offset in grid:
        calibration = calibration_for_td_success(offset)
        results.append({"td_success_logit_offset": float(offset), **{
            name: modeled_takedowns(cohort, fsr, paths, seed, calibration)
            for name, cohort in cohorts.items()}})
    report = {"train_dates": [str(train.event_date.min().date()), str(train.event_date.max().date())],
              "holdout_dates": [str(holdout.event_date.min().date()), str(holdout.event_date.max().date())],
              "historical_takedowns": historical, "results_in_grid_order": results}
    if final_report:
        if len(results) != 1: raise ValueError("final report requires exactly one current offset")
        report["global_comparison"] = {
            name: global_comparison(cohort, rounds, results[0][name]) for name, cohort in cohorts.items()}
        for comparison in report["global_comparison"].values(): validate_global_comparison(comparison)
        report["current_calibration"] = current_calibration_values(calibration_for_td_success(grid[0]))
        for name in cohorts: print(render_comparison(name, report["global_comparison"][name]))
        print("\nCURRENT CALIBRATION\n" + json.dumps(report["current_calibration"], indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True)); print(json.dumps(report, indent=2, sort_keys=True)); return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--grid",nargs="+",type=float,default=COARSE_GRID); p.add_argument("--paths",type=int,default=3); p.add_argument("--train-limit",type=int,default=100); p.add_argument("--holdout-limit",type=int,default=50); p.add_argument("--seed",type=int,default=20260813); p.add_argument("--final-report",action="store_true"); p.add_argument("--output",type=Path,default=Path("data/diagnostics/event_mc_v1_phase7m.json")); run(**vars(p.parse_args()))
