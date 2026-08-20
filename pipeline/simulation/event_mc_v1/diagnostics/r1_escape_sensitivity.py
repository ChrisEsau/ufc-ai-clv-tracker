"""R1 escape-rate/control-duration sensitivity for canonical FSR V2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..scheduler import EventRate
from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight
from .r1_control_baseline import R1ControlSink, historical_rows
from .r1_strike_rate_sensitivity import RateScaledProvider
from .r1_td_rate_sensitivity import ControlRateProvider


STANDING_MULT = 0.80
GROUND_MULT = 2.25
TD_MULT = 0.83

ESCAPE_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.00)


@dataclass(frozen=True)
class EscapeScaledProvider:
    base: object
    escape_multiplier: float

    def candidates(self, state, context):
        output = []

        for item in self.base.candidates(state, context):
            family = getattr(item.candidate, "action_family", "")
            mult = (
                self.escape_multiplier
                if family == "ground_escape"
                else 1.0
            )

            output.append(
                EventRate(
                    item.candidate,
                    item.rate_per_second * mult,
                )
            )

        return tuple(output)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def historical_metrics(frame):
    exposure = frame["exposure_seconds"].sum()

    td_attempts = (
        frame["red_td_attempts"].sum()
        + frame["blue_td_attempts"].sum()
    )

    td_landed = (
        frame["red_td_landed"].sum()
        + frame["blue_td_landed"].sum()
    )

    control = (
        frame["red_control_seconds"].sum()
        + frame["blue_control_seconds"].sum()
    )

    landed_by_row = (
        frame["red_td_landed"]
        + frame["blue_td_landed"]
    )

    control_by_row = (
        frame["red_control_seconds"]
        + frame["blue_control_seconds"]
    )

    # Control in zero-landed-TD R1s is definitely unreachable by the
    # current two-state EVENT MC, which only establishes ground control
    # after a completed takedown.
    unreachable_control = control_by_row[landed_by_row == 0].sum()
    td_compatible_control = control - unreachable_control

    return {
        "td_att_5": safe_div(td_attempts, exposure) * 300.0,
        "td_land_5": safe_div(td_landed, exposure) * 300.0,

        "raw_control_5":
            safe_div(control, exposure) * 300.0,

        "td_compatible_control_5":
            safe_div(td_compatible_control, exposure) * 300.0,

        "raw_control_per_td":
            safe_div(control, td_landed),

        "td_compatible_control_per_td":
            safe_div(td_compatible_control, td_landed),

        "unreachable_control_share":
            safe_div(unreachable_control, control),
    }


def simulate(cohort, fsr, paths, seed, escape_mult):
    rows = []

    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = replace(_fight(historical, fsr), rounds=1)

        for path_index in range(paths):
            sink = R1ControlSink()

            engine, _, _ = build_engine(
                fight,
                seed + fight_index * 100000 + path_index,
                sink,
            )

            provider = RateScaledProvider(
                engine.rate_provider,
                standing_multiplier=STANDING_MULT,
                ground_multiplier=GROUND_MULT,
            )

            provider = ControlRateProvider(
                provider,
                td_multiplier=TD_MULT,
            )

            provider = EscapeScaledProvider(
                provider,
                escape_multiplier=escape_mult,
            )

            engine.rate_provider = provider

            result = engine.run()
            stats = result.sink_result

            rows.append({
                "exposure_seconds":
                    float(stats["exposure_seconds"]),

                "red_td_attempts":
                    stats["td_attempts"].get("red", 0),
                "blue_td_attempts":
                    stats["td_attempts"].get("blue", 0),

                "red_td_landed":
                    stats["td_landed"].get("red", 0),
                "blue_td_landed":
                    stats["td_landed"].get("blue", 0),

                "red_control":
                    stats["control_seconds"].get("red", 0.0),
                "blue_control":
                    stats["control_seconds"].get("blue", 0.0),

                "red_escapes":
                    stats["escapes"].get("red", 0),
                "blue_escapes":
                    stats["escapes"].get("blue", 0),
            })

    return pd.DataFrame(rows)


def simulated_metrics(frame):
    exposure = frame["exposure_seconds"].sum()

    td_attempts = (
        frame["red_td_attempts"].sum()
        + frame["blue_td_attempts"].sum()
    )

    td_landed = (
        frame["red_td_landed"].sum()
        + frame["blue_td_landed"].sum()
    )

    control = (
        frame["red_control"].sum()
        + frame["blue_control"].sum()
    )

    escapes = (
        frame["red_escapes"].sum()
        + frame["blue_escapes"].sum()
    )

    return {
        "td_att_5":
            safe_div(td_attempts, exposure) * 300.0,

        "td_land_5":
            safe_div(td_landed, exposure) * 300.0,

        "control_5":
            safe_div(control, exposure) * 300.0,

        "control_per_td":
            safe_div(control, td_landed),

        "escape_per_td":
            safe_div(escapes, td_landed),

        "escape_5":
            safe_div(escapes, exposure) * 300.0,
    }


def pct(sim, hist):
    return (sim / hist - 1.0) * 100.0


def evaluate(name, cohort, fsr, rounds, paths, seed):
    hist = historical_metrics(
        historical_rows(cohort, rounds)
    )

    print("\nHistorical targets:")
    print(
        f"  TD-compatible control / 5 min: "
        f"{hist['td_compatible_control_5']:.4f}"
    )
    print(
        f"  TD-compatible control / landed TD: "
        f"{hist['td_compatible_control_per_td']:.4f}"
    )
    print(
        f"  structurally unreachable control share: "
        f"{hist['unreachable_control_share']:.4f}"
    )

    rows = []

    for escape_mult in ESCAPE_GRID:
        sim = simulated_metrics(
            simulate(
                cohort,
                fsr,
                paths,
                seed,
                escape_mult,
            )
        )

        rows.append({
            "split": name,
            "escape_mult": escape_mult,

            "hist_td_att_5":
                hist["td_att_5"],
            "sim_td_att_5":
                sim["td_att_5"],

            "hist_td_land_5":
                hist["td_land_5"],
            "sim_td_land_5":
                sim["td_land_5"],

            "hist_ctrl_td":
                hist["td_compatible_control_per_td"],
            "sim_ctrl_td":
                sim["control_per_td"],
            "ctrl_td_err_%":
                pct(
                    sim["control_per_td"],
                    hist["td_compatible_control_per_td"],
                ),

            "hist_ctrl_5":
                hist["td_compatible_control_5"],
            "sim_ctrl_5":
                sim["control_5"],
            "ctrl_5_err_%":
                pct(
                    sim["control_5"],
                    hist["td_compatible_control_5"],
                ),

            "escape_per_td":
                sim["escape_per_td"],
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 150)
    print(name.upper())
    print("=" * 150)

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

    evaluate(
        "train",
        train,
        fsr,
        rounds,
        paths,
        seed,
    )

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
