"""Measurement-only Phase 7N coupled global re-audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..calibration import DEFAULT_CALIBRATION
from .phase7b_kd_calibration import temporal_cohorts
from .phase7i_strike_exposure_audit import historical_strikes
from .phase7k_takedown_decomposition import historical_takedowns, modeled_takedowns
from .phase7l_distance_td_calibration import validate_historical_anchors
from .phase7m_td_success_calibration import (
    UNAVAILABLE,
    current_calibration_values,
    global_comparison,
    render_comparison,
    validate_global_comparison,
)

EXPECTED_SIGNIFICANT_ATTEMPTS_15 = {
    "train": 238.17026784299225,
    "holdout": 256.5909623039101,
}
READINESS_LINE = "GLOBAL ENVIRONMENT READY FOR ROUND-SPECIFIC VALIDATION: NO"
NEXT_GLOBAL_SUBSYSTEM = "significant-strike attempt generation and phase composition"


def classify_relative_error(relative_difference_percent) -> str | None:
    """Classify magnitude deterministically using the Phase 7N bands."""
    if relative_difference_percent is None:
        return None
    magnitude = abs(float(relative_difference_percent))
    if magnitude <= 5.0:
        return "CLOSE"
    if magnitude <= 10.0:
        return "MODERATE"
    if magnitude <= 20.0:
        return "MATERIAL"
    return "LARGE"


def validate_significant_anchor(name: str, historical: dict) -> None:
    observed = historical["significant"]["attempts_per_15min"]
    expected = EXPECTED_SIGNIFICANT_ATTEMPTS_15[name]
    if abs(observed - expected) > 1e-9:
        raise RuntimeError(
            f"{name} historical significant-strike anchor changed: {observed} != {expected}"
        )


def classify_comparison(comparison: dict) -> list[dict]:
    classified = []
    for row in comparison["rows"]:
        enriched = dict(row)
        enriched["classification"] = classify_relative_error(
            row["relative_difference_percent"]
        )
        classified.append(enriched)
    return classified


def coupled_diagnosis(classified: dict[str, list[dict]]) -> dict:
    """Record evidence-bounded family decisions; no calibration is performed."""
    families = {
        "strike_attempt_generation": "Residual underexposure, especially holdout; upstream and definitionally strong.",
        "strike_landing_accuracy": "Train is moderately low while holdout is close; partly coupled to phase mix.",
        "strike_phase_composition": "DISTANCE remains high and CLINCH/GROUND shares remain split-dependent.",
        "td_attempt_generation": "Train is high and holdout is close after residence coupling; defer retuning.",
        "td_success_conversion": "Completed exposure is close on both splits; conversion errors oppose temporally.",
        "knockdown_generation": "KD exposure is not consistently excessive after coupled calibration.",
        "submission_attempt_generation": "Ground-residence coupling reduced attempts; compare but do not retune here.",
        "submission_conversion_outcomes": "Method shares remain close-to-moderate and temporally mixed.",
        "ko_tko_outcomes": "Close on both splits.",
        "decision_outcomes": "Close on both splits.",
        "fight_timing": "Train is close; holdout finish timing remains a downstream diagnostic.",
        "phase_residence_control": "Historical same-definition comparator unavailable; MC-only diagnostic.",
    }
    metric_selectors = {
        "strike_attempt_generation": ("strikes.attempts_per_15min",),
        "strike_landing_accuracy": ("strikes.accuracy",),
        "strike_phase_composition": (".attempt_share", ".landed_share"),
        "td_attempt_generation": ("takedowns.attempts_per_15min",),
        "td_success_conversion": ("takedowns.success", "takedowns.completed_per_15min"),
        "knockdown_generation": ("knockdowns.per_15min", "knockdowns.per_100"),
        "submission_attempt_generation": ("submissions.attempts_per_15min",),
        "submission_conversion_outcomes": ("submissions.outcome_share",),
        "ko_tko_outcomes": ("outcomes.KO_TKO",),
        "decision_outcomes": ("outcomes.DEC",),
        "fight_timing": ("fight.mean_duration", "fight.mean_nondecision"),
        "phase_residence_control": ("phase_residence.",),
    }
    order = {None: -1, "CLOSE": 0, "MODERATE": 1, "MATERIAL": 2, "LARGE": 3}

    def status(split, family):
        selectors = metric_selectors[family]
        matches = [
            row["classification"] for row in classified[split]
            if any(selector in row["metric"] for selector in selectors)
        ]
        return max(matches, key=lambda item: order[item]) if matches else None

    output = {}
    for family, note in families.items():
        train_status, holdout_status = status("train", family), status("holdout", family)
        output[family] = {
            "train_status": train_status or UNAVAILABLE,
            "holdout_status": holdout_status or UNAVAILABLE,
            "same_story": train_status == holdout_status,
            "coupling_assessment": note,
            "needs_global_work_before_round_validation": family in {
                "strike_attempt_generation", "strike_phase_composition"
            },
        }
    return output


def run(
    paths=10,
    seed=20260813,
    output=Path("data/diagnostics/event_mc_v1_phase7n.json"),
):
    train, holdout, fsr = temporal_cohorts(100, 50)
    rounds = pd.read_parquet(ROUND_STATS_PATH)
    cohorts = {"train": train, "holdout": holdout}
    comparisons, simulated, classified = {}, {}, {}
    for name, cohort in cohorts.items():
        historical_td = historical_takedowns(cohort, rounds)
        validate_historical_anchors(name, historical_td)
        validate_significant_anchor(name, historical_strikes(cohort))
        simulated[name] = modeled_takedowns(
            cohort, fsr, paths, seed, DEFAULT_CALIBRATION
        )
        comparisons[name] = global_comparison(cohort, rounds, simulated[name])
        validate_global_comparison(comparisons[name])
        classified[name] = classify_comparison(comparisons[name])
        print(render_comparison(name, comparisons[name]))

    material_large = {
        name: [
            row for row in rows
            if row["classification"] in {"MATERIAL", "LARGE"}
        ]
        for name, rows in classified.items()
    }
    report = {
        "phase": "7N",
        "measurement_only": True,
        "dates": {
            name: [str(cohort.event_date.min().date()), str(cohort.event_date.max().date())]
            for name, cohort in cohorts.items()
        },
        "paths_per_fight": paths,
        "seed": seed,
        "global_comparison": comparisons,
        "classified_metrics": classified,
        "material_and_large": material_large,
        "coupled_diagnosis": coupled_diagnosis(classified),
        "priority_ranking": [
            "significant-strike attempt generation and phase composition",
            "submission attempt exposure after reduced ground residence",
            "temporally asymmetric TD attempt/success residuals",
            "holdout fight and finish timing",
        ],
        "readiness_decision": READINESS_LINE,
        "next_global_subsystem": NEXT_GLOBAL_SUBSYSTEM,
        "next_parameter_family_to_examine": (
            "DISTANCE-versus-CLINCH strike opportunity allocation; measurement/sensitivity first, no Phase 7N tuning"
        ),
        "current_calibration": current_calibration_values(DEFAULT_CALIBRATION),
    }
    print("\nMATERIAL / LARGE MISMATCHES")
    for name, rows in material_large.items():
        print(name.upper())
        for row in rows:
            print(f"{row['classification']}: {row['metric']} ({row['relative_difference_percent']:.2f}%)")
    print("\n" + READINESS_LINE)
    print("NEXT GLOBAL SUBSYSTEM: " + NEXT_GLOBAL_SUBSYSTEM)
    print("\nCURRENT CALIBRATION\n" + json.dumps(report["current_calibration"], indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/diagnostics/event_mc_v1_phase7n.json"),
    )
    run(**vars(parser.parse_args()))
