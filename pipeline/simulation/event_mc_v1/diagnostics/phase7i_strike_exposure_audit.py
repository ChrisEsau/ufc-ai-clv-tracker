"""Measurement-only historical/model strike-definition reconciliation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .phase7a_decomposition import DecompositionSink
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import (
    METHODS,
    _fight,
    normalize_method,
    observed_duration_seconds,
)
from ..single_fight import build_engine

PHASE_COLUMNS = {
    "clinch": ("clinch_atmpted", "clinch_landed"),
    "ground": ("ground_atmpted", "ground_landed"),
}


def _metrics(attempts, landed, fights, exposure):
    return {
        "attempts_per_fight": float(attempts / fights),
        "landed_per_fight": float(landed / fights),
        "attempts_per_15min": float(attempts / exposure * 900),
        "landed_per_15min": float(landed / exposure * 900),
        "landing_percentage": float(landed / attempts) if attempts else None,
    }


def historical_strikes(cohort):
    exposure = cohort.apply(observed_duration_seconds, axis=1).sum()
    fights = len(cohort)

    def total(prefix):
        return (cohort[f"r_{prefix}"] + cohort[f"b_{prefix}"]).sum()

    total_att, total_land = total("total_str_atmpted"), total("total_str_landed")
    sig_att, sig_land = total("sig_str_atmpted"), total("sig_str_landed")
    output = {
        "exposure_seconds": float(exposure),
        "total": _metrics(total_att, total_land, fights, exposure),
        "significant": _metrics(sig_att, sig_land, fights, exposure),
        "significant_by_phase": {},
    }
    phase_values = {}
    for phase, (att_col, land_col) in PHASE_COLUMNS.items():
        phase_values[phase] = (total(att_col), total(land_col))
    phase_values["distance"] = (
        sig_att - sum(x[0] for x in phase_values.values()),
        sig_land - sum(x[1] for x in phase_values.values()),
    )
    for phase, (att, land) in phase_values.items():
        row = _metrics(att, land, fights, exposure)
        row.update(
            {
                "attempt_share": float(att / sig_att),
                "landed_share": float(land / sig_land),
            }
        )
        output["significant_by_phase"][phase] = row
    output["field_note"] = (
        "Master has phase-specific significant-strike clinch/ground fields; distance is the significant-strike residual. It has no phase-specific TOTAL-strike fields or trustworthy phase-time denominators."
    )
    return output


def reconcile(historical, simulated):
    model = simulated["simulated"]
    return {
        name: {
            "historical_attempts_per_15min": values["attempts_per_15min"],
            "historical_landed_per_15min": values["landed_per_15min"],
            "historical_landing_percentage": values["landing_percentage"],
            "simulated_attempts_per_15min": model["attempts_per_15min"],
            "simulated_landed_per_15min": model["landed_per_15min"],
            "simulated_landing_percentage": model["landing_rate"],
            "attempt_ratio": model["attempts_per_15min"] / values["attempts_per_15min"],
            "landed_ratio": model["landed_per_15min"] / values["landed_per_15min"],
        }
        for name, values in (
            ("total", historical["total"]),
            ("significant", historical["significant"]),
        )
    }


def modeled_strikes(cohort, fsr, paths, seed):
    rows = []
    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = _fight(historical, fsr)
        for path_index in range(paths):
            sink = DecompositionSink()
            result = build_engine(
                fight, seed + fight_index * 100000 + path_index, sink
            )[0].run()
            stats = result.sink_result
            attempts = sum(stats["attempts"].values())
            landed = sum(stats["landed"].values())
            rows.append(
                {
                    "seconds": stats["exposure_seconds"],
                    "method": result.state.finish_method,
                    "attempts": attempts,
                    "landed": landed,
                    "kds": sum(x["kd"] for x in stats["impacts"]),
                    **{
                        f"{p}_attempts": stats["attempts"].get(p, 0)
                        for p in ("distance", "clinch", "ground")
                    },
                    **{
                        f"{p}_landed": stats["landed"].get(p, 0)
                        for p in ("distance", "clinch", "ground")
                    },
                }
            )
    frame = pd.DataFrame(rows)
    exposure = frame.seconds.sum()
    methods = frame.method.value_counts(normalize=True)
    nondec = frame[frame.method != "DEC"]
    out = {
        "simulated": _metrics(
            frame.attempts.sum(), frame.landed.sum(), len(frame), exposure
        ),
        "phase": {},
        "method_shares": {m: float(methods.get(m, 0)) for m in METHODS},
        "kd_per_path": float(frame.kds.mean()),
        "kd_per_100_landed": float(frame.kds.sum() / frame.landed.sum() * 100),
        "kd_per_15min": float(frame.kds.sum() / exposure * 900),
        "mean_fight_duration": float(frame.seconds.mean()),
        "mean_nondecision_finish_time": float(nondec.seconds.mean()),
    }
    out["simulated"]["landing_rate"] = out["simulated"].pop("landing_percentage")
    for phase in ("distance", "clinch", "ground"):
        a = frame[f"{phase}_attempts"].sum()
        l = frame[f"{phase}_landed"].sum()
        row = _metrics(a, l, len(frame), exposure)
        row.update(
            {
                "attempt_share": float(a / frame.attempts.sum()),
                "landed_share": float(l / frame.landed.sum()),
            }
        )
        out["phase"][phase] = row
    return out


def run(
    paths=10, seed=20260813, output=Path("data/diagnostics/event_mc_v1_phase7i.json")
):
    train, holdout, fsr = temporal_cohorts(100, 50)
    report = {}
    for name, cohort in (("train", train), ("holdout", holdout)):
        sim = modeled_strikes(cohort, fsr, paths, seed)
        hist = historical_strikes(cohort)
        report[name] = {
            "dates": [
                str(cohort.event_date.min().date()),
                str(cohort.event_date.max().date()),
            ],
            "historical": hist,
            "simulated": sim,
            "comparison": reconcile(hist, sim),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260813)
    p.add_argument(
        "--output", type=Path, default=Path("data/diagnostics/event_mc_v1_phase7i.json")
    )
    run(**vars(p.parse_args()))
