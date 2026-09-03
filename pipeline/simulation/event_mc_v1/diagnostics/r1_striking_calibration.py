"""R1-only striking calibration diagnostic for canonical FSR V2 EVENT MC.

Measurement only. Does not alter calibration or simulator mechanics.

Historical mapping:
    UFCStats distance + clinch significant strikes -> model standing strikes
    UFCStats ground significant strikes            -> model ground strikes
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import ROUND_STATS_PATH

from ..components.actions import ActionAttempt
from ..events import PrimaryEvent
from ..single_fight import build_engine
from .phase7b_kd_calibration import temporal_cohorts
from .population_validation import _fight, observed_duration_seconds


ROUND_SECONDS = 300.0
STRIKE_PHASE = {
    "standing_strike": "standing",
    "ground_strike": "ground",
}


@dataclass
class R1StrikeSink:
    attempts: dict = field(
        default_factory=lambda: {
            "red": defaultdict(int),
            "blue": defaultdict(int),
        }
    )
    landed: dict = field(
        default_factory=lambda: {
            "red": defaultdict(int),
            "blue": defaultdict(int),
        }
    )
    phase_seconds: dict = field(
        default_factory=lambda: defaultdict(float)
    )
    exposure_seconds: float = 0.0

    def on_time_advance(self, dt, before, after):
        # Diagnostic engines are one-round engines, but clip defensively.
        start = float(before.fight_time_seconds)
        end = min(float(after.fight_time_seconds), ROUND_SECONDS)

        if end <= start:
            return

        used = end - start
        self.exposure_seconds += used
        self.phase_seconds[str(before.phase)] += used

    def on_event(self, event, before, after):
        if not isinstance(event, PrimaryEvent):
            return

        if float(event.timestamp_seconds) > ROUND_SECONDS + 1e-9:
            return

        attempt = event.payload
        if not isinstance(attempt, ActionAttempt):
            return

        phase = STRIKE_PHASE.get(attempt.action_family)
        if phase is None:
            return

        side = attempt.side.value
        self.attempts[side][phase] += 1

        if attempt.landed:
            self.landed[side][phase] += 1

    def finalize(self):
        return {
            "attempts": {
                side: dict(values)
                for side, values in self.attempts.items()
            },
            "landed": {
                side: dict(values)
                for side, values in self.landed.items()
            },
            "phase_seconds": dict(self.phase_seconds),
            "exposure_seconds": self.exposure_seconds,
        }


def safe_div(a, b):
    return float(a / b) if b else np.nan


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


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
                f"R1 round stats must resolve exactly one red and one blue row "
                f"for fight {fight_id}; red={len(red)} blue={len(blue)}"
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
            "completed_r1": exposure >= ROUND_SECONDS - 1e-9,
        }

        for side, source in (("red", red), ("blue", blue)):
            row[f"{side}_attempts"] = float(source["sig_str_attempted"])
            row[f"{side}_landed"] = float(source["sig_str_landed"])

            row[f"{side}_standing_attempts"] = float(
                source["distance_attempted"] + source["clinch_attempted"]
            )
            row[f"{side}_standing_landed"] = float(
                source["distance_landed"] + source["clinch_landed"]
            )

            row[f"{side}_ground_attempts"] = float(source["ground_attempted"])
            row[f"{side}_ground_landed"] = float(source["ground_landed"])

            row[f"{side}_distance_attempts"] = float(source["distance_attempted"])
            row[f"{side}_distance_landed"] = float(source["distance_landed"])
            row[f"{side}_clinch_attempts"] = float(source["clinch_attempted"])
            row[f"{side}_clinch_landed"] = float(source["clinch_landed"])

        output.append(row)

    return pd.DataFrame(output)


def simulated_rows(cohort, fsr, paths, seed):
    output = []

    for fight_index, (_, historical) in enumerate(cohort.iterrows()):
        fight = _fight(historical, fsr)

        # R1 diagnostic only. Same R1 physics, hard horizon at 5:00.
        fight = replace(fight, rounds=1)

        for path_index in range(paths):
            sink = R1StrikeSink()

            engine, _, _ = build_engine(
                fight,
                seed + fight_index * 100000 + path_index,
                sink,
            )

            result = engine.run()
            stats = result.sink_result

            row = {
                "fight_id": str(fight.fight_id),
                "path": path_index,
                "exposure_seconds": float(stats["exposure_seconds"]),
                "completed_r1": (
                    float(stats["exposure_seconds"])
                    >= ROUND_SECONDS - 1e-9
                ),
                "standing_seconds": float(
                    stats["phase_seconds"].get("standing", 0.0)
                ),
                "ground_seconds": float(
                    stats["phase_seconds"].get("ground", 0.0)
                ),
                "finish_method": result.state.finish_method,
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

            output.append(row)

    return pd.DataFrame(output)


def aggregate(frame, historical=False):
    fights_or_paths = len(frame)
    exposure = frame["exposure_seconds"].sum()

    attempts = frame["red_attempts"].sum() + frame["blue_attempts"].sum()
    landed = frame["red_landed"].sum() + frame["blue_landed"].sum()

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

    output = {
        "observations": int(fights_or_paths),
        "exposure_seconds": float(exposure),
        "r1_completion_rate": float(frame["completed_r1"].mean()),

        "sig_attempts_per_r1": safe_div(attempts, fights_or_paths),
        "sig_landed_per_r1": safe_div(landed, fights_or_paths),

        "attempts_per_fighter_r1": safe_div(
            attempts, fights_or_paths * 2
        ),
        "landed_per_fighter_r1": safe_div(
            landed, fights_or_paths * 2
        ),

        "attempts_per_5min_exposure": safe_div(
            attempts, exposure
        ) * ROUND_SECONDS,
        "landed_per_5min_exposure": safe_div(
            landed, exposure
        ) * ROUND_SECONDS,

        "accuracy": safe_div(landed, attempts),

        "standing_attempt_share": safe_div(standing_a, attempts),
        "ground_attempt_share": safe_div(ground_a, attempts),

        "standing_accuracy": safe_div(standing_l, standing_a),
        "ground_accuracy": safe_div(ground_l, ground_a),

        "red_attempts_per_r1": safe_div(
            frame["red_attempts"].sum(), fights_or_paths
        ),
        "blue_attempts_per_r1": safe_div(
            frame["blue_attempts"].sum(), fights_or_paths
        ),
        "red_landed_per_r1": safe_div(
            frame["red_landed"].sum(), fights_or_paths
        ),
        "blue_landed_per_r1": safe_div(
            frame["blue_landed"].sum(), fights_or_paths
        ),
    }

    if historical:
        distance_a = (
            frame["red_distance_attempts"].sum()
            + frame["blue_distance_attempts"].sum()
        )
        clinch_a = (
            frame["red_clinch_attempts"].sum()
            + frame["blue_clinch_attempts"].sum()
        )

        output["distance_attempt_share"] = safe_div(distance_a, attempts)
        output["clinch_attempt_share"] = safe_div(clinch_a, attempts)

        # UFCStats significant-strike phase categories should reconcile.
        output["phase_attempt_reconciliation_error"] = float(
            attempts - (standing_a + ground_a)
        )

    return output


def differential_report(hist, sim):
    sim_fight = sim.groupby("fight_id", as_index=False).agg(
        red_attempts=("red_attempts", "mean"),
        blue_attempts=("blue_attempts", "mean"),
        red_landed=("red_landed", "mean"),
        blue_landed=("blue_landed", "mean"),
    )

    merged = hist.merge(
        sim_fight,
        on="fight_id",
        suffixes=("_hist", "_sim"),
        validate="one_to_one",
    )

    hist_att = (
        merged["red_attempts_hist"] - merged["blue_attempts_hist"]
    )
    sim_att = (
        merged["red_attempts_sim"] - merged["blue_attempts_sim"]
    )

    hist_land = (
        merged["red_landed_hist"] - merged["blue_landed_hist"]
    )
    sim_land = (
        merged["red_landed_sim"] - merged["blue_landed_sim"]
    )

    return {
        "attempt_differential": {
            "historical_mean_red_minus_blue": float(hist_att.mean()),
            "simulated_mean_red_minus_blue": float(sim_att.mean()),
            "fight_level_correlation": corr(hist_att, sim_att),
            "fight_level_mae": float(np.mean(np.abs(hist_att - sim_att))),
        },
        "landed_differential": {
            "historical_mean_red_minus_blue": float(hist_land.mean()),
            "simulated_mean_red_minus_blue": float(sim_land.mean()),
            "fight_level_correlation": corr(hist_land, sim_land),
            "fight_level_mae": float(np.mean(np.abs(hist_land - sim_land))),
        },
    }


DISPLAY_METRICS = [
    "r1_completion_rate",
    "sig_attempts_per_r1",
    "sig_landed_per_r1",
    "attempts_per_fighter_r1",
    "landed_per_fighter_r1",
    "attempts_per_5min_exposure",
    "landed_per_5min_exposure",
    "accuracy",
    "standing_attempt_share",
    "ground_attempt_share",
    "standing_accuracy",
    "ground_accuracy",
    "red_attempts_per_r1",
    "blue_attempts_per_r1",
    "red_landed_per_r1",
    "blue_landed_per_r1",
]


def comparison_table(hist, sim):
    rows = []

    for metric in DISPLAY_METRICS:
        h = hist.get(metric, np.nan)
        s = sim.get(metric, np.nan)

        if pd.isna(h) or h == 0:
            pct = np.nan
        else:
            pct = (s / h - 1.0) * 100.0

        rows.append({
            "metric": metric,
            "historical": h,
            "simulated": s,
            "sim_vs_hist_%": pct,
        })

    return pd.DataFrame(rows)


def evaluate_split(name, cohort, fsr, rounds, paths, seed):
    hist = historical_rows(cohort, rounds)
    sim = simulated_rows(cohort, fsr, paths, seed)

    hist_all = aggregate(hist, historical=True)
    sim_all = aggregate(sim, historical=False)

    hist_full = aggregate(
        hist[hist["completed_r1"]].copy(),
        historical=True,
    )
    sim_full = aggregate(
        sim[sim["completed_r1"]].copy(),
        historical=False,
    )

    print("\n" + "=" * 100)
    print(f"{name.upper()} — ALL R1 EXPOSURE")
    print("=" * 100)
    print(
        comparison_table(hist_all, sim_all)
        .to_string(index=False, float_format=lambda x: f"{x:10.4f}")
    )

    print("\nHistorical phase detail:")
    print(
        f"  distance attempt share: {hist_all['distance_attempt_share']:.4f}\n"
        f"  clinch attempt share:   {hist_all['clinch_attempt_share']:.4f}\n"
        f"  ground attempt share:   {hist_all['ground_attempt_share']:.4f}\n"
        f"  phase reconcile error:  {hist_all['phase_attempt_reconciliation_error']:.1f}"
    )

    print("\n" + "-" * 100)
    print(f"{name.upper()} — COMPLETED 5:00 R1 ONLY")
    print("-" * 100)
    print(
        comparison_table(hist_full, sim_full)
        .to_string(index=False, float_format=lambda x: f"{x:10.4f}")
    )

    diff = differential_report(hist, sim)

    print("\nFight-level matchup differential:")
    print(json.dumps(diff, indent=2))

    return {
        "historical_all": hist_all,
        "simulated_all": sim_all,
        "historical_completed_r1": hist_full,
        "simulated_completed_r1": sim_full,
        "differential": diff,
    }


def run(
    paths=10,
    train_limit=100,
    holdout_limit=50,
    seed=20260814,
    output=Path("data/diagnostics/event_mc_v1_r1_striking_baseline.json"),
):
    train, holdout, fsr = temporal_cohorts(
        train_limit,
        holdout_limit,
    )

    rounds = pd.read_parquet(ROUND_STATS_PATH)

    report = {
        "description": "R1 striking baseline; measurement only",
        "paths_per_fight": paths,
        "seed": seed,
        "mapping": {
            "historical_distance_plus_clinch": "simulated_standing",
            "historical_ground": "simulated_ground",
            "historical_total": "UFCStats significant strikes",
        },
    }

    report["train"] = evaluate_split(
        "train",
        train,
        fsr,
        rounds,
        paths,
        seed,
    )

    report["holdout"] = evaluate_split(
        "holdout",
        holdout,
        fsr,
        rounds,
        paths,
        seed + 50000000,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True)
    )

    print("\nSaved:", output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--holdout-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/diagnostics/event_mc_v1_r1_striking_baseline.json"
        ),
    )

    run(**vars(parser.parse_args()))
