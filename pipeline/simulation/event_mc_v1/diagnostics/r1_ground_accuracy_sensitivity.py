"""R1 ground-strike accuracy sensitivity for canonical FSR V2."""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight
from .r1_striking_calibration import R1StrikeSink, historical_rows
from .r1_strike_rate_sensitivity import RateScaledProvider
from .r1_td_rate_sensitivity import ControlRateProvider
from .r1_escape_sensitivity import EscapeScaledProvider
from .r1_standing_accuracy_sensitivity import logit_shift


STANDING_MULT = 0.80
GROUND_MULT = 2.35
STANDING_ACCURACY_OFFSET = 0.22
TD_MULT = 0.85
ESCAPE_MULT = 0.85

GROUND_OFFSET_GRID = (0.45,)


def safe_div(a, b):
    return float(a / b) if b else float("nan")


def shifted_fight(fight, ground_offset):
    matchup = fight.fsr_v2_matchup

    shifted_matchup = replace(
        matchup,
        red=replace(
            matchup.red,
            standing_accuracy_baseline=logit_shift(
                matchup.red.standing_accuracy_baseline,
                STANDING_ACCURACY_OFFSET,
            ),
            ground_accuracy_baseline=logit_shift(
                matchup.red.ground_accuracy_baseline,
                ground_offset,
            ),
        ),
        blue=replace(
            matchup.blue,
            standing_accuracy_baseline=logit_shift(
                matchup.blue.standing_accuracy_baseline,
                STANDING_ACCURACY_OFFSET,
            ),
            ground_accuracy_baseline=logit_shift(
                matchup.blue.ground_accuracy_baseline,
                ground_offset,
            ),
        ),
    )

    return replace(
        fight,
        rounds=1,
        fsr_v2_matchup=shifted_matchup,
    )


def metrics(frame):
    exposure = frame["exposure_seconds"].sum()

    standing_a = (
        frame["red_standing_attempts"].sum()
        + frame["blue_standing_attempts"].sum()
    )
    standing_l = (
        frame["red_standing_landed"].sum()
        + frame["blue_standing_landed"].sum()
    )

    ground_a = (
        frame["red_ground_attempts"].sum()
        + frame["blue_ground_attempts"].sum()
    )
    ground_l = (
        frame["red_ground_landed"].sum()
        + frame["blue_ground_landed"].sum()
    )

    total_a = standing_a + ground_a
    total_l = standing_l + ground_l

    return {
        "standing_acc": safe_div(standing_l, standing_a),
        "ground_acc": safe_div(ground_l, ground_a),
        "total_acc": safe_div(total_l, total_a),
        "attempts_5": safe_div(total_a, exposure) * 300.0,
        "landed_5": safe_div(total_l, exposure) * 300.0,
    }


def historical_metrics(frame):
    return metrics(frame)


def simulate(cohort, fsr, paths, seed, ground_offset):
    rows = []

    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = shifted_fight(
            _fight(historical, fsr),
            ground_offset,
        )

        for path_index in range(paths):
            sink = R1StrikeSink()

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
                escape_multiplier=ESCAPE_MULT,
            )
            engine.rate_provider = provider

            result = engine.run()
            stats = result.sink_result

            row = {
                "exposure_seconds": float(stats["exposure_seconds"]),
                "completed_r1": (
                    float(stats["exposure_seconds"]) >= 300.0 - 1e-9
                ),
            }

            for side in ("red", "blue"):
                row[f"{side}_standing_attempts"] = (
                    stats["attempts"][side].get("standing", 0)
                )
                row[f"{side}_standing_landed"] = (
                    stats["landed"][side].get("standing", 0)
                )
                row[f"{side}_ground_attempts"] = (
                    stats["attempts"][side].get("ground", 0)
                )
                row[f"{side}_ground_landed"] = (
                    stats["landed"][side].get("ground", 0)
                )

            rows.append(row)

    frame = pd.DataFrame(rows)
    out = metrics(frame)
    out["r1_complete"] = float(frame["completed_r1"].mean())
    return out


def pct(sim, hist):
    return (sim / hist - 1.0) * 100.0


def evaluate(name, cohort, fsr, rounds, paths, seed):
    hist = historical_metrics(
        historical_rows(cohort, rounds)
    )

    rows = []

    for offset in GROUND_OFFSET_GRID:
        sim = simulate(
            cohort,
            fsr,
            paths,
            seed,
            offset,
        )

        rows.append({
            "split": name,
            "ground_offset": offset,

            "hist_stand_acc": hist["standing_acc"],
            "sim_stand_acc": sim["standing_acc"],

            "hist_ground_acc": hist["ground_acc"],
            "sim_ground_acc": sim["ground_acc"],
            "ground_acc_err_%": pct(
                sim["ground_acc"],
                hist["ground_acc"],
            ),

            "hist_total_acc": hist["total_acc"],
            "sim_total_acc": sim["total_acc"],
            "total_acc_err_%": pct(
                sim["total_acc"],
                hist["total_acc"],
            ),

            "hist_att_5": hist["attempts_5"],
            "sim_att_5": sim["attempts_5"],

            "hist_land_5": hist["landed_5"],
            "sim_land_5": sim["landed_5"],
            "land_err_%": pct(
                sim["landed_5"],
                hist["landed_5"],
            ),

            "r1_complete": sim["r1_complete"],
        })

    out = pd.DataFrame(rows)

    print("\n" + "=" * 170)
    print(name.upper())
    print("=" * 170)

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
