"""R1 standing-strike opportunity sensitivity for canonical FSR V2.

Measurement only. Wraps the live rate provider without changing production code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..scheduler import EventRate
from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight
from .r1_striking_calibration import (
    R1StrikeSink,
    aggregate,
    historical_rows,
)


GRID = (0.75, 0.80, 0.85, 0.90, 1.00)


@dataclass(frozen=True)
class RateScaledProvider:
    base: object
    standing_multiplier: float
    ground_multiplier: float = 1.0

    def candidates(self, state, context):
        output = []

        for item in self.base.candidates(state, context):
            family = getattr(item.candidate, "action_family", "")

            if family == "standing_strike":
                multiplier = self.standing_multiplier
            elif family == "ground_strike":
                multiplier = self.ground_multiplier
            else:
                multiplier = 1.0

            output.append(
                EventRate(
                    item.candidate,
                    item.rate_per_second * multiplier,
                )
            )

        return tuple(output)


def simulated_rows(cohort, fsr, paths, seed, standing_multiplier):
    rows = []

    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = replace(_fight(historical, fsr), rounds=1)

        for path_index in range(paths):
            sink = R1StrikeSink()

            engine, _, _ = build_engine(
                fight,
                seed + fight_index * 100000 + path_index,
                sink,
            )

            engine.rate_provider = RateScaledProvider(
                engine.rate_provider,
                standing_multiplier=standing_multiplier,
            )

            result = engine.run()
            stats = result.sink_result

            row = {
                "fight_id": str(fight.fight_id),
                "exposure_seconds": float(stats["exposure_seconds"]),
                "completed_r1": (
                    float(stats["exposure_seconds"]) >= 300.0 - 1e-9
                ),
            }

            for side in ("red", "blue"):
                standing_a = stats["attempts"][side].get("standing", 0)
                ground_a = stats["attempts"][side].get("ground", 0)

                standing_l = stats["landed"][side].get("standing", 0)
                ground_l = stats["landed"][side].get("ground", 0)

                row[f"{side}_standing_attempts"] = standing_a
                row[f"{side}_ground_attempts"] = ground_a
                row[f"{side}_standing_landed"] = standing_l
                row[f"{side}_ground_landed"] = ground_l

                row[f"{side}_attempts"] = standing_a + ground_a
                row[f"{side}_landed"] = standing_l + ground_l

            rows.append(row)

    return pd.DataFrame(rows)


def pct(sim, hist):
    return (sim / hist - 1.0) * 100.0


def evaluate(name, cohort, fsr, rounds, paths, seed):
    hist_frame = historical_rows(cohort, rounds)
    hist = aggregate(hist_frame, historical=True)

    rows = []

    for multiplier in GRID:
        sim_frame = simulated_rows(
            cohort,
            fsr,
            paths,
            seed,
            multiplier,
        )
        sim = aggregate(sim_frame, historical=False)

        rows.append(
            {
                "split": name,
                "standing_mult": multiplier,

                "hist_att_5": hist["attempts_per_5min_exposure"],
                "sim_att_5": sim["attempts_per_5min_exposure"],
                "att_err_%": pct(
                    sim["attempts_per_5min_exposure"],
                    hist["attempts_per_5min_exposure"],
                ),

                "hist_land_5": hist["landed_per_5min_exposure"],
                "sim_land_5": sim["landed_per_5min_exposure"],
                "land_err_%": pct(
                    sim["landed_per_5min_exposure"],
                    hist["landed_per_5min_exposure"],
                ),

                "hist_acc": hist["accuracy"],
                "sim_acc": sim["accuracy"],
                "acc_err_%": pct(
                    sim["accuracy"],
                    hist["accuracy"],
                ),

                "hist_stand_share": hist["standing_attempt_share"],
                "sim_stand_share": sim["standing_attempt_share"],

                "hist_ground_share": hist["ground_attempt_share"],
                "sim_ground_share": sim["ground_attempt_share"],

                "r1_complete": sim["r1_completion_rate"],
            }
        )

    frame = pd.DataFrame(rows)

    print("\n" + "=" * 130)
    print(name.upper())
    print("=" * 130)

    print(
        frame.to_string(
            index=False,
            float_format=lambda x: f"{x:8.4f}",
        )
    )

    return frame


def run(paths=5, train_limit=100, holdout_limit=50, seed=20260814):
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
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--holdout-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)

    run(**vars(parser.parse_args()))
