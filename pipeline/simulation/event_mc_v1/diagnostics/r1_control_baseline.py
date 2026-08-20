"""R1 TD / control / escape baseline for canonical FSR V2.

Measurement only.

Control model:
    successful takedown -> ground controller established
    ground escape       -> return to standing

Current provisional R1 strike opportunity calibration:
    standing strike multiplier = 0.80
    ground strike multiplier   = 2.25
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..components.actions import ActionAttempt
from ..events import PrimaryEvent
from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight, observed_duration_seconds
from .r1_strike_rate_sensitivity import RateScaledProvider


ROUND_SECONDS = 300.0
STANDING_MULT = 0.80
GROUND_MULT = 2.25


@dataclass
class R1ControlSink:
    td_attempts: dict = field(
        default_factory=lambda: defaultdict(int)
    )
    td_landed: dict = field(
        default_factory=lambda: defaultdict(int)
    )
    escapes: dict = field(
        default_factory=lambda: defaultdict(int)
    )
    control_seconds: dict = field(
        default_factory=lambda: defaultdict(float)
    )
    exposure_seconds: float = 0.0

    def on_time_advance(self, dt, before, after):
        start = float(before.fight_time_seconds)
        end = min(float(after.fight_time_seconds), ROUND_SECONDS)

        if end <= start:
            return

        used = end - start
        self.exposure_seconds += used

        controller = before.ground_controller
        if controller in ("red", "blue"):
            self.control_seconds[controller] += used

    def on_event(self, event, before, after):
        if not isinstance(event, PrimaryEvent):
            return

        attempt = event.payload
        if not isinstance(attempt, ActionAttempt):
            return

        side = attempt.side.value

        if attempt.action_family == "takedown":
            self.td_attempts[side] += 1

            if attempt.landed:
                self.td_landed[side] += 1

        elif attempt.action_family == "ground_escape":
            self.escapes[side] += 1

    def finalize(self):
        return {
            "td_attempts": dict(self.td_attempts),
            "td_landed": dict(self.td_landed),
            "escapes": dict(self.escapes),
            "control_seconds": dict(self.control_seconds),
            "exposure_seconds": self.exposure_seconds,
        }


def safe_div(a, b):
    return float(a / b) if b else np.nan


def val(x):
    if pd.isna(x):
        return 0.0
    return float(x)


def historical_rows(cohort, rounds):
    fight_ids = set(cohort["fight_id"].astype(str))

    r1 = rounds[
        (rounds["round"] == 1)
        & (rounds["fight_id"].astype(str).isin(fight_ids))
    ].copy()

    r1["fight_id"] = r1["fight_id"].astype(str)
    r1["corner"] = r1["corner"].astype(str).str.lower()

    output = []

    for _, fight in cohort.iterrows():
        fight_id = str(fight["fight_id"])
        rows = r1[r1["fight_id"] == fight_id]

        red = rows[rows["corner"].str.startswith("r")]
        blue = rows[rows["corner"].str.startswith("b")]

        if len(red) != 1 or len(blue) != 1:
            raise RuntimeError(
                f"R1 stats must resolve one red and one blue row "
                f"for {fight_id}: red={len(red)} blue={len(blue)}"
            )

        red = red.iloc[0]
        blue = blue.iloc[0]

        exposure = min(
            float(observed_duration_seconds(fight)),
            ROUND_SECONDS,
        )

        row = {
            "fight_id": fight_id,
            "exposure_seconds": exposure,
        }

        for side, source in (("red", red), ("blue", blue)):
            row[f"{side}_td_attempts"] = val(source["td_attempted"])
            row[f"{side}_td_landed"] = val(source["td_landed"])
            row[f"{side}_control_seconds"] = val(source["ctrl_sec"])

        output.append(row)

    return pd.DataFrame(output)


def simulated_rows(cohort, fsr, paths, seed):
    output = []

    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = replace(_fight(historical, fsr), rounds=1)

        for path_index in range(paths):
            sink = R1ControlSink()

            engine, _, _ = build_engine(
                fight,
                seed + fight_index * 100000 + path_index,
                sink,
            )

            # Preserve the R1 striking opportunity environment
            # we already calibrated.
            engine.rate_provider = RateScaledProvider(
                engine.rate_provider,
                standing_multiplier=STANDING_MULT,
                ground_multiplier=GROUND_MULT,
            )

            result = engine.run()
            stats = result.sink_result

            row = {
                "fight_id": str(fight.fight_id),
                "path": path_index,
                "exposure_seconds": float(stats["exposure_seconds"]),
            }

            for side in ("red", "blue"):
                row[f"{side}_td_attempts"] = (
                    stats["td_attempts"].get(side, 0)
                )
                row[f"{side}_td_landed"] = (
                    stats["td_landed"].get(side, 0)
                )
                row[f"{side}_control_seconds"] = (
                    stats["control_seconds"].get(side, 0.0)
                )
                row[f"{side}_escapes"] = (
                    stats["escapes"].get(side, 0)
                )

            output.append(row)

    return pd.DataFrame(output)


def aggregate(frame, simulated=False):
    exposure = frame["exposure_seconds"].sum()
    n = len(frame)

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

    total_td_per_row = (
        frame["red_td_landed"]
        + frame["blue_td_landed"]
    )

    total_control_per_row = (
        frame["red_control_seconds"]
        + frame["blue_control_seconds"]
    )

    zero_td = total_td_per_row == 0

    out = {
        "observations": int(n),

        "td_attempts_per_5min_exposure":
            safe_div(td_attempts, exposure) * ROUND_SECONDS,

        "td_landed_per_5min_exposure":
            safe_div(td_landed, exposure) * ROUND_SECONDS,

        "td_success_rate":
            safe_div(td_landed, td_attempts),

        "control_seconds_per_r1":
            safe_div(control, n),

        "control_seconds_per_5min_exposure":
            safe_div(control, exposure) * ROUND_SECONDS,

        "control_share_of_exposure":
            safe_div(control, exposure),

        "control_seconds_per_td_landed":
            safe_div(control, td_landed),

        "share_rows_with_td_landed":
            float((total_td_per_row > 0).mean()),

        "share_rows_with_control":
            float((total_control_per_row > 0).mean()),

        "control_seconds_when_zero_td_landed":
            float(total_control_per_row[zero_td].sum()),

        "share_control_from_zero_td_rows":
            safe_div(
                total_control_per_row[zero_td].sum(),
                control,
            ),
    }

    if simulated:
        escapes = (
            frame["red_escapes"].sum()
            + frame["blue_escapes"].sum()
        )

        out["escapes_per_5min_exposure"] = (
            safe_div(escapes, exposure) * ROUND_SECONDS
        )

        out["escapes_per_td_landed"] = safe_div(
            escapes,
            td_landed,
        )

    return out


METRICS = (
    "td_attempts_per_5min_exposure",
    "td_landed_per_5min_exposure",
    "td_success_rate",
    "control_seconds_per_r1",
    "control_seconds_per_5min_exposure",
    "control_share_of_exposure",
    "control_seconds_per_td_landed",
    "share_rows_with_td_landed",
    "share_rows_with_control",
)


def compare(hist, sim):
    rows = []

    for metric in METRICS:
        h = hist[metric]
        s = sim[metric]

        error = (
            (s / h - 1.0) * 100.0
            if h and not pd.isna(h)
            else np.nan
        )

        rows.append({
            "metric": metric,
            "historical": h,
            "simulated": s,
            "sim_vs_hist_%": error,
        })

    return pd.DataFrame(rows)


def evaluate(name, cohort, fsr, rounds, paths, seed):
    hist_frame = historical_rows(cohort, rounds)
    sim_frame = simulated_rows(cohort, fsr, paths, seed)

    hist = aggregate(hist_frame)
    sim = aggregate(sim_frame, simulated=True)

    print("\n" + "=" * 110)
    print(name.upper())
    print("=" * 110)

    print(
        compare(hist, sim).to_string(
            index=False,
            float_format=lambda x: f"{x:10.4f}",
        )
    )

    print("\nHistorical control that exists with ZERO landed TDs:")
    print(
        f"  seconds: {hist['control_seconds_when_zero_td_landed']:.1f}"
    )
    print(
        f"  share of historical control: "
        f"{hist['share_control_from_zero_td_rows']:.4f}"
    )

    print("\nSimulation escape diagnostics:")
    print(
        f"  escapes / 5 min exposure: "
        f"{sim['escapes_per_5min_exposure']:.4f}"
    )
    print(
        f"  escapes / landed TD: "
        f"{sim['escapes_per_td_landed']:.4f}"
    )


def run(
    paths=10,
    train_limit=100,
    holdout_limit=50,
    seed=20260814,
):
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
