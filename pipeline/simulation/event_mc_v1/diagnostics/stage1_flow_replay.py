"""Stage 1 EVENT MC flow-only historical replay.

Purpose:
    Measure whether canonical FSR V2 predicts fight-specific phase flow.

Deliberately excluded:
    damage / KD / KO
    submission conversion
    judging
    winner prediction

Preserved:
    canonical FSR V2 action rates
    takedown mechanics
    ground escapes
    current frozen stamina mechanics
    round recovery
    normal scheduled-round context

Each path is stopped at the ACTUAL historical fight duration so downstream
finish mechanics cannot truncate the flow comparison.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..calibration import DEFAULT_RESOLVER
from ..components.action_rates import FSRV2ActionRateProvider
from ..config import FightConfig
from ..engine import SimulationEngine
from ..flow_stats import FlowStatsSink
from ..modifiers import DynamicModifierProvider
from ..physiology import PhysiologyTimeAdvanceModel
from ..rng import RNGManager
from ..stamina import StaminaModel

from .fresh_100_fight_predictive_replay import (
    build_simulation_inputs,
    select_fresh_cohort,
)


def stage1_observed_duration_seconds(row) -> float:
    """Normalize master fight time exactly like FSR V2 round-stat history.

    The master contains mixed semantics:
    - some rows store total elapsed fight seconds;
    - newer/legacy-style rows store seconds within the finishing round.

    For round 2+, a value <= 300 cannot be total elapsed fight time, so
    completed rounds are added. This matches the FSR V2 round-stat builder.
    """
    finish_round = int(row["finish_round"])
    match_time = float(row["match_time_sec"])

    if finish_round > 1 and match_time <= 300:
        return (finish_round - 1) * 300.0 + match_time

    return match_time


def build_flow_engine(fight, seed):
    key = (
        fight.division
        if fight.division in DEFAULT_RESOLVER.weight_classes
        else None
    )
    calibration = DEFAULT_RESOLVER.for_weight_class(key)

    if fight.fsr_v2_matchup is None:
        raise RuntimeError(
            "Stage 1 flow replay requires canonical FSR V2 matchup"
        )

    stamina = StaminaModel(
        fight.profiles,
        calibration=calibration,
    )

    rate_provider = FSRV2ActionRateProvider(
        fight.fsr_v2_matchup,
        fight.profiles,
        stamina,
        DynamicModifierProvider(calibration),
        calibration,
    )

    # IMPORTANT:
    # No physiology model.
    # No KO finish model.
    # No submission finish model.
    # No judging model.
    #
    # PhysiologyTimeAdvanceModel is retained only because it owns the
    # continuous positional stamina update used by the current simulator.
    return SimulationEngine(
        FightConfig(fight.rounds),
        rate_provider,
        PhysiologyTimeAdvanceModel(
            stamina,
            calibration,
        ),
        RNGManager(seed),
        FlowStatsSink(),
        round_recovery_model=stamina,
    )


def simulate_one(args):
    fight_index, fight, duration, paths, seed = args

    side_values = {
        side: {
            "td_attempts": [],
            "td_landed": [],
            "ground_entries": [],
            "ground_control_seconds": [],
        }
        for side in ("red", "blue")
    }

    ground_seconds = []
    standing_seconds = []
    elapsed_seconds = []

    for path_index in range(paths):
        path_seed = (
            seed
            + fight_index * 100000
            + path_index
        )

        engine = build_flow_engine(
            fight,
            path_seed,
        )

        result = engine.run(
            stop_at_seconds=duration,
        )

        stats = result.sink_result

        elapsed_seconds.append(
            float(result.state.fight_time_seconds)
        )

        ground_seconds.append(
            float(
                stats["phase_seconds"].get(
                    "ground",
                    0.0,
                )
            )
        )

        standing_seconds.append(
            float(
                stats["phase_seconds"].get(
                    "standing",
                    0.0,
                )
            )
        )

        for side in ("red", "blue"):
            side_values[side]["td_attempts"].append(
                stats["attempts"][side].get(
                    "takedown",
                    0,
                )
            )

            side_values[side]["td_landed"].append(
                stats["outcomes"][side].get(
                    "takedown_landed",
                    0,
                )
            )

            side_values[side][
                "ground_control_seconds"
            ].append(
                float(
                    stats[
                        "ground_control_seconds"
                    ].get(side, 0.0)
                )
            )

            side_values[side][
                "ground_entries"
            ].append(
                sum(
                    1
                    for transition
                    in stats["transitions"]
                    if (
                        transition["to_phase"]
                        == "ground"
                        and transition["from_phase"]
                        != "ground"
                        and transition["to_controller"]
                        == side
                    )
                )
            )

    return {
        "fight_index": fight_index,
        "actual_elapsed_seconds": float(duration),
        "simulated_mean_elapsed_seconds": float(
            np.mean(elapsed_seconds)
        ),
        "simulated_mean_ground_seconds": float(
            np.mean(ground_seconds)
        ),
        "simulated_mean_standing_seconds": float(
            np.mean(standing_seconds)
        ),
        **{
            f"simulated_mean_{side}_{metric}":
                float(np.mean(values))
            for side in ("red", "blue")
            for metric, values
            in side_values[side].items()
        },
    }


def heartbeat(done, total, started):
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0.0

    print(
        f"[heartbeat] fights {done}/{total} "
        f"({done / total:.0%}) | "
        f"elapsed={elapsed:.1f}s | "
        f"rate={rate:.2f} fights/s",
        flush=True,
    )


def run(
    fights=500,
    paths=10,
    seed=20260813,
    workers=2,
    heartbeat_every=25,
    offset=0,
    hide_fight_ids=False,
    output=Path(
        "data/diagnostics/event_mc/"
        "stage1_flow_control_500x10.json"
    ),
    csv=Path(
        "data/diagnostics/event_mc/"
        "stage1_flow_control_500x10.csv"
    ),
):
    cohort, fsr, selection = (
        select_fresh_cohort(
            fights,
            offset,
        )
    )

    simulation_inputs = build_simulation_inputs(
        cohort,
        fsr,
    )

    durations = [
        stage1_observed_duration_seconds(row)
        for _, row in cohort.iterrows()
    ]

    terminal_selection = (
        {
            k: v
            for k, v in selection.items()
            if k != "bout_ids"
        }
        if hide_fight_ids
        else selection
    )

    print("=" * 100)
    print(
        "EVENT MC V1 — STAGE 1 FLOW-ONLY REPLAY"
    )
    print("=" * 100)
    print(
        json.dumps(
            terminal_selection,
            indent=2,
        )
    )
    print(
        f"fights={fights} | "
        f"paths/fight={paths} | "
        f"seed={seed}"
    )
    print(
        "Downstream finishes/judging: DISABLED"
    )
    print(
        "Exposure: ACTUAL HISTORICAL FIGHT DURATION"
    )

    tasks = [
        (
            i,
            simulation_inputs[i],
            durations[i],
            paths,
            seed,
        )
        for i in range(len(simulation_inputs))
    ]

    raw = [None] * len(tasks)
    started = time.perf_counter()
    done = 0

    if workers == 1:
        for task in tasks:
            raw[task[0]] = simulate_one(task)
            done += 1

            if (
                done == 1
                or done % heartbeat_every == 0
                or done == len(tasks)
            ):
                heartbeat(
                    done,
                    len(tasks),
                    started,
                )
    else:
        with ProcessPoolExecutor(
            max_workers=workers
        ) as pool:
            futures = {
                pool.submit(
                    simulate_one,
                    task,
                ): task[0]
                for task in tasks
            }

            for future in as_completed(futures):
                index = futures[future]
                raw[index] = future.result()
                done += 1

                if (
                    done == 1
                    or done % heartbeat_every == 0
                    or done == len(tasks)
                ):
                    heartbeat(
                        done,
                        len(tasks),
                        started,
                    )

    rows = []

    for index, (_, actual) in enumerate(
        cohort.iterrows()
    ):
        row = {
            "event_date": str(
                actual["event_date"].date()
            ),
            "bout_id": str(
                actual["fight_id"]
            ),
            "red_fighter": str(
                actual["r_name"]
            ),
            "blue_fighter": str(
                actual["b_name"]
            ),
            **{
                k: v
                for k, v in raw[index].items()
                if k != "fight_index"
            },
        }

        rows.append(row)

    frame = pd.DataFrame(rows)

    report = {
        "selection": selection,
        "fights": fights,
        "paths_per_fight": paths,
        "seed": seed,
        "workers": workers,
        "runtime_seconds": (
            time.perf_counter() - started
        ),
        "flow_only": True,
        "actual_duration_conditioned": True,
        "duration_semantics": (
            "mixed master match_time normalized identically to "
            "FSR V2 round-stat history"
        ),
        "downstream_models_disabled": [
            "physiology",
            "knockdown",
            "ko_tko_finish",
            "submission_finish",
            "judging",
        ],
        "rows": (
            frame.replace(
                {np.nan: None}
            ).to_dict("records")
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    frame.to_csv(
        csv,
        index=False,
    )

    print("\n" + "=" * 100)
    print("FLOW-ONLY POPULATION CHECK")
    print("=" * 100)

    print(
        "mean historical exposure: "
        f"{frame.actual_elapsed_seconds.mean():.1f}s"
    )
    print(
        "mean simulated exposure:  "
        f"{frame.simulated_mean_elapsed_seconds.mean():.1f}s"
    )
    print(
        "mean simulated ground:    "
        f"{frame.simulated_mean_ground_seconds.mean():.1f}s"
    )
    print(
        "mean simulated standing:  "
        f"{frame.simulated_mean_standing_seconds.mean():.1f}s"
    )

    for side in ("red", "blue"):
        print(
            f"{side.upper()} TD attempts/fight: "
            f"{frame[f'simulated_mean_{side}_td_attempts'].mean():.3f}"
        )
        print(
            f"{side.upper()} TD landed/fight:   "
            f"{frame[f'simulated_mean_{side}_td_landed'].mean():.3f}"
        )
        print(
            f"{side.upper()} ground control:    "
            f"{frame[f'simulated_mean_{side}_ground_control_seconds'].mean():.1f}s"
        )

    print(f"\nwrote JSON: {output}")
    print(f"wrote CSV : {csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fights",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260813,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hide-fight-ids",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/diagnostics/event_mc/"
            "stage1_flow_control_500x10.json"
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/diagnostics/event_mc/"
            "stage1_flow_control_500x10.csv"
        ),
    )

    run(**vars(parser.parse_args()))
