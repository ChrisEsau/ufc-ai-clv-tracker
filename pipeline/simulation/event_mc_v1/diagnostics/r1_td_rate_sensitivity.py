"""R1 takedown-attempt sensitivity for canonical FSR V2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..scheduler import EventRate
from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight
from .r1_control_baseline import (
    R1ControlSink,
    historical_rows,
)
from .r1_strike_rate_sensitivity import RateScaledProvider


STANDING_MULT = 0.80
GROUND_MULT = 2.25
TD_GRID = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


@dataclass(frozen=True)
class ControlRateProvider:
    base: object
    td_multiplier: float

    def candidates(self, state, context):
        rows = []

        for item in self.base.candidates(state, context):
            family = getattr(item.candidate, "action_family", "")
            mult = self.td_multiplier if family == "takedown" else 1.0

            rows.append(
                EventRate(
                    item.candidate,
                    item.rate_per_second * mult,
                )
            )

        return tuple(rows)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def hist_metrics(frame):
    exposure = frame["exposure_seconds"].sum()

    att = (
        frame["red_td_attempts"].sum()
        + frame["blue_td_attempts"].sum()
    )
    landed = (
        frame["red_td_landed"].sum()
        + frame["blue_td_landed"].sum()
    )

    return {
        "td_att_5": safe_div(att, exposure) * 300.0,
        "td_land_5": safe_div(landed, exposure) * 300.0,
        "td_success": safe_div(landed, att),
    }


def simulate(cohort, fsr, paths, seed, td_mult):
    output = []

    for fight_index, (_, row) in enumerate(cohort.iterrows()):
        fight = replace(_fight(row, fsr), rounds=1)

        for path_index in range(paths):
            sink = R1ControlSink()

            engine, _, _ = build_engine(
                fight,
                seed + fight_index * 100000 + path_index,
                sink,
            )

            # Preserve provisional R1 striking opportunity calibration.
            strike_scaled = RateScaledProvider(
                engine.rate_provider,
                standing_multiplier=STANDING_MULT,
                ground_multiplier=GROUND_MULT,
            )

            engine.rate_provider = ControlRateProvider(
                strike_scaled,
                td_multiplier=td_mult,
            )

            result = engine.run()
            stats = result.sink_result

            output.append({
                "exposure_seconds": stats["exposure_seconds"],
                "red_td_attempts": stats["td_attempts"].get("red", 0),
                "blue_td_attempts": stats["td_attempts"].get("blue", 0),
                "red_td_landed": stats["td_landed"].get("red", 0),
                "blue_td_landed": stats["td_landed"].get("blue", 0),
            })

    return pd.DataFrame(output)


def pct(sim, hist):
    return (sim / hist - 1.0) * 100.0


def evaluate(name, cohort, fsr, rounds, paths, seed):
    hist = hist_metrics(historical_rows(cohort, rounds))

    rows = []

    for td_mult in TD_GRID:
        sim = hist_metrics(
            simulate(
                cohort,
                fsr,
                paths,
                seed,
                td_mult,
            )
        )

        rows.append({
            "split": name,
            "td_mult": td_mult,

            "hist_td_att_5": hist["td_att_5"],
            "sim_td_att_5": sim["td_att_5"],
            "att_err_%": pct(
                sim["td_att_5"],
                hist["td_att_5"],
            ),

            "hist_td_land_5": hist["td_land_5"],
            "sim_td_land_5": sim["td_land_5"],
            "land_err_%": pct(
                sim["td_land_5"],
                hist["td_land_5"],
            ),

            "hist_success": hist["td_success"],
            "sim_success": sim["td_success"],
            "success_err_%": pct(
                sim["td_success"],
                hist["td_success"],
            ),
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 130)
    print(name.upper())
    print("=" * 130)

    print(
        out.to_string(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
        )
    )


def run(paths=10, train_limit=100, holdout_limit=50, seed=20260814):
    train, holdout, fsr = temporal_cohorts(
        train_limit,
        holdout_limit,
    )

    rounds = pd.read_parquet(ROUND_STATS_PATH)

    evaluate("train", train, fsr, rounds, paths, seed)
    evaluate(
        "holdout",
        holdout,
        fsr,
        rounds,
        paths,
        seed + 50000000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--holdout-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)

    run(**vars(parser.parse_args()))
