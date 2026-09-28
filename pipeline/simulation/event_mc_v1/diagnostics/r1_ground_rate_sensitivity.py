"""R1 ground-strike opportunity sensitivity with standing multiplier fixed at 0.80."""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight
from .r1_striking_calibration import R1StrikeSink, historical_rows
from .r1_strike_rate_sensitivity import RateScaledProvider


STANDING_MULT = 0.80
GROUND_GRID = (1.50, 1.75, 2.00, 2.25, 2.50, 2.75)


def safe_div(a, b):
    return a / b if b else float("nan")


def phase_metrics(frame):
    exposure = frame["exposure_seconds"].sum()

    standing = (
        frame["red_standing_attempts"].sum()
        + frame["blue_standing_attempts"].sum()
    )
    ground = (
        frame["red_ground_attempts"].sum()
        + frame["blue_ground_attempts"].sum()
    )

    landed = (
        frame["red_landed"].sum()
        + frame["blue_landed"].sum()
    )

    attempts = standing + ground

    return {
        "standing_att_5": safe_div(standing, exposure) * 300.0,
        "ground_att_5": safe_div(ground, exposure) * 300.0,
        "total_att_5": safe_div(attempts, exposure) * 300.0,
        "ground_share": safe_div(ground, attempts),
        "accuracy": safe_div(landed, attempts),
    }


def historical_metrics(frame):
    exposure = frame["exposure_seconds"].sum()

    standing = (
        frame["red_standing_attempts"].sum()
        + frame["blue_standing_attempts"].sum()
    )
    ground = (
        frame["red_ground_attempts"].sum()
        + frame["blue_ground_attempts"].sum()
    )
    landed = (
        frame["red_landed"].sum()
        + frame["blue_landed"].sum()
    )

    attempts = standing + ground

    return {
        "standing_att_5": safe_div(standing, exposure) * 300.0,
        "ground_att_5": safe_div(ground, exposure) * 300.0,
        "total_att_5": safe_div(attempts, exposure) * 300.0,
        "ground_share": safe_div(ground, attempts),
        "accuracy": safe_div(landed, attempts),
    }


def simulate(cohort, fsr, paths, seed, ground_mult):
    rows = []

    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = replace(_fight(historical, fsr), rounds=1)

        for path_index in range(paths):
            sink = R1StrikeSink()

            from ..single_fight import build_engine

            engine, _, _ = build_engine(
                fight,
                seed + fight_index * 100000 + path_index,
                sink,
            )

            engine.rate_provider = RateScaledProvider(
                engine.rate_provider,
                standing_multiplier=STANDING_MULT,
                ground_multiplier=ground_mult,
            )

            result = engine.run()
            stats = result.sink_result

            row = {
                "exposure_seconds": float(stats["exposure_seconds"]),
            }

            for side in ("red", "blue"):
                sa = stats["attempts"][side].get("standing", 0)
                ga = stats["attempts"][side].get("ground", 0)
                sl = stats["landed"][side].get("standing", 0)
                gl = stats["landed"][side].get("ground", 0)

                row[f"{side}_standing_attempts"] = sa
                row[f"{side}_ground_attempts"] = ga
                row[f"{side}_landed"] = sl + gl

            rows.append(row)

    return pd.DataFrame(rows)


def err(sim, hist):
    return (sim / hist - 1.0) * 100.0


def evaluate(name, cohort, fsr, rounds, paths, seed):
    hist_frame = historical_rows(cohort, rounds)
    hist = historical_metrics(hist_frame)

    rows = []

    for ground_mult in GROUND_GRID:
        sim_frame = simulate(
            cohort,
            fsr,
            paths,
            seed,
            ground_mult,
        )

        sim = phase_metrics(sim_frame)

        rows.append({
            "split": name,
            "standing_mult": STANDING_MULT,
            "ground_mult": ground_mult,

            "hist_stand_5": hist["standing_att_5"],
            "sim_stand_5": sim["standing_att_5"],
            "stand_err_%": err(
                sim["standing_att_5"],
                hist["standing_att_5"],
            ),

            "hist_ground_5": hist["ground_att_5"],
            "sim_ground_5": sim["ground_att_5"],
            "ground_err_%": err(
                sim["ground_att_5"],
                hist["ground_att_5"],
            ),

            "hist_total_5": hist["total_att_5"],
            "sim_total_5": sim["total_att_5"],
            "total_err_%": err(
                sim["total_att_5"],
                hist["total_att_5"],
            ),

            "hist_ground_share": hist["ground_share"],
            "sim_ground_share": sim["ground_share"],

            "hist_accuracy": hist["accuracy"],
            "sim_accuracy": sim["accuracy"],
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
        "train", train, fsr, rounds,
        paths, seed,
    )

    evaluate(
        "holdout", holdout, fsr, rounds,
        paths, seed + 50000000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--holdout-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)

    run(**vars(parser.parse_args()))
