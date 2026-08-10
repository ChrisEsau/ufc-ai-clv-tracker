"""Population audit for the shadow KO/TKO V3 hybrid finish engine.

Uses the same mature 2020+ leakage-safe cohort and fixed seed stream as the
existing 10-path population audits. The goal is architectural diagnosis, not
final probability calibration.

Primary questions
-----------------
- Does the acute-KO route raise R1 finishes?
- Does between-round recovery plus removal of KD-collapse reduce R2/R3 inflation?
- How often do finishes come from acute KO, post-KD TKO, or cumulative exhaustion?
- Does winner direction / occurrence discrimination improve or deteriorate?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as baseline
from scripts.experimental import fsr_mature_2020plus_mc_10path_round_recovery_audit as recovery_audit
from scripts.experimental import fsr_static_mc_ko_tko_v3_hybrid as hybrid


DEFAULT_PATHS = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_hybrid_v3_audit.csv"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _actual_round_flag(bout: pd.Series, round_no: int) -> int:
    if int(bout["actual_ko_tko"]) != 1 or pd.isna(bout["actual_finish_round"]):
        return 0
    return int(int(bout["actual_finish_round"]) == round_no)


def _simulate_one_bout(
    bout: pd.Series,
    pair: tuple[pd.Series, pd.Series],
    *,
    paths: int,
    rounds: int,
    seeds: np.ndarray,
) -> dict[str, object]:
    red, blue = pair
    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    winner_id = str(bout.get("winner_id", ""))
    r_age = float(bout["r_age"]) if pd.notna(bout["r_age"]) else None
    b_age = float(bout["b_age"]) if pd.notna(bout["b_age"]) else None

    ko_count = 0
    r1_ko_count = 0
    r2_ko_count = 0
    r3_ko_count = 0
    r_ko = 0
    b_ko = 0
    actual_winner_ko = 0
    finish_rounds: list[int] = []
    route_counts = {
        "acute_ko": 0,
        "post_kd_tko": 0,
        "cumulative_exhaustion": 0,
    }
    finish_reservoir_fraction: list[float] = []
    finish_shock_fraction: list[float] = []
    total_kds = 0
    red_recovered = 0.0
    blue_recovered = 0.0

    for seed in seeds:
        sim = hybrid.StaticFSRMCKOTKOV3Hybrid(
            red,
            blue,
            rounds=rounds,
            seed=int(seed),
            red_age=r_age,
            blue_age=b_age,
        )
        result = sim.run()
        red_recovered += sim.total_round_recovery[0]
        blue_recovered += sim.total_round_recovery[1]
        total_kds += sim.stats[0].knockdowns_scored + sim.stats[1].knockdowns_scored

        finish = result.finish
        if finish is None:
            continue

        ko_count += 1
        finish_round = int(finish.round)
        finish_rounds.append(finish_round)
        if finish_round == 1:
            r1_ko_count += 1
        elif finish_round == 2:
            r2_ko_count += 1
        elif finish_round == 3:
            r3_ko_count += 1

        route = str(getattr(finish, "finish_route", "unknown"))
        if route in route_counts:
            route_counts[route] += 1
        finish_reservoir_fraction.append(
            float(getattr(finish, "reservoir_fraction_after", np.nan))
        )
        finish_shock_fraction.append(float(getattr(finish, "shock_fraction", np.nan)))

        if finish.winner == 0:
            r_ko += 1
            if winner_id == r_id:
                actual_winner_ko += 1
        else:
            b_ko += 1
            if winner_id == b_id:
                actual_winner_ko += 1

    p_r_ko = r_ko / paths
    p_b_ko = b_ko / paths
    predicted_ko_winner = "tie"
    if p_r_ko > p_b_ko:
        predicted_ko_winner = r_id
    elif p_b_ko > p_r_ko:
        predicted_ko_winner = b_id

    return {
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "r_id": r_id,
        "b_id": b_id,
        "winner_id": winner_id,
        "r_age": r_age,
        "b_age": b_age,
        "max_age": np.nanmax(
            [
                r_age if r_age is not None else np.nan,
                b_age if b_age is not None else np.nan,
            ]
        ),
        "actual_ko_tko": int(bout["actual_ko_tko"]),
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "actual_r2_ko": _actual_round_flag(bout, 2),
        "actual_r3_ko": _actual_round_flag(bout, 3),
        "actual_finish_round": bout["actual_finish_round"],
        "p_any_ko": ko_count / paths,
        "p_r1_ko": r1_ko_count / paths,
        "p_r2_ko": r2_ko_count / paths,
        "p_r3_ko": r3_ko_count / paths,
        "p_r_ko": p_r_ko,
        "p_b_ko": p_b_ko,
        "p_actual_winner_ko": (
            actual_winner_ko / paths if winner_id in {r_id, b_id} else np.nan
        ),
        "predicted_ko_winner": predicted_ko_winner,
        "ko_winner_direction_hit": (
            int(predicted_ko_winner == winner_id)
            if int(bout["actual_ko_tko"]) == 1
            and predicted_ko_winner != "tie"
            and winner_id in {r_id, b_id}
            else np.nan
        ),
        "ko_winner_direction_tie": (
            int(predicted_ko_winner == "tie")
            if int(bout["actual_ko_tko"]) == 1 and winner_id in {r_id, b_id}
            else np.nan
        ),
        "mean_sim_finish_round": (
            float(np.mean(finish_rounds)) if finish_rounds else np.nan
        ),
        "p_acute_ko": route_counts["acute_ko"] / paths,
        "p_post_kd_tko": route_counts["post_kd_tko"] / paths,
        "p_cumulative_exhaustion": route_counts["cumulative_exhaustion"] / paths,
        "mean_finish_reservoir_fraction": (
            float(np.nanmean(finish_reservoir_fraction))
            if finish_reservoir_fraction
            else np.nan
        ),
        "mean_finish_shock_fraction": (
            float(np.nanmean(finish_shock_fraction))
            if finish_shock_fraction
            else np.nan
        ),
        "mean_kds_per_path": total_kds / paths,
        "mean_red_round_recovery": red_recovered / paths,
        "mean_blue_round_recovery": blue_recovered / paths,
    }


def _print_occurrence_metrics(frame: pd.DataFrame) -> None:
    rows = []
    for label, target, probability in [
        ("Any KO/TKO", "actual_ko_tko", "p_any_ko"),
        ("R1 KO/TKO", "actual_r1_ko", "p_r1_ko"),
        ("R2 KO/TKO", "actual_r2_ko", "p_r2_ko"),
        ("R3 KO/TKO", "actual_r3_ko", "p_r3_ko"),
    ]:
        rows.append(
            {
                "target": label,
                **baseline._binary_metrics(frame, target, probability),
            }
        )
    print("\nKO OCCURRENCE METRICS")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_route_decomposition(frame: pd.DataFrame) -> None:
    route_means = {
        "acute_ko": frame["p_acute_ko"].mean(),
        "post_kd_tko": frame["p_post_kd_tko"].mean(),
        "cumulative_exhaustion": frame["p_cumulative_exhaustion"].mean(),
    }
    total = sum(route_means.values())
    rows = []
    for route, rate in route_means.items():
        rows.append(
            {
                "finish_route": route,
                "population_path_rate": rate,
                "share_of_sim_finishes": rate / total if total > 0 else np.nan,
            }
        )
    print("\nSIMULATED FINISH-ROUTE DECOMPOSITION")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "mean reservoir remaining at finish: "
        f"{frame['mean_finish_reservoir_fraction'].mean():.2%}"
    )
    print(
        "mean finish-strike shock/capacity: "
        f"{frame['mean_finish_shock_fraction'].mean():.2%}"
    )
    print(f"mean KDs per simulated path: {frame['mean_kds_per_path'].mean():.3f}")


def _print_age_calibration(frame: pd.DataFrame) -> None:
    work = frame.copy()
    work["oldest_age_band"] = pd.cut(
        work["max_age"],
        baseline.AGE_BINS,
        labels=baseline.AGE_LABELS,
    )
    by_age = (
        work.groupby("oldest_age_band", observed=True)
        .agg(
            bouts=("bout_id", "size"),
            actual_ko_rate=("actual_ko_tko", "mean"),
            sim_ko_rate=("p_any_ko", "mean"),
            actual_r1_ko_rate=("actual_r1_ko", "mean"),
            sim_r1_ko_rate=("p_r1_ko", "mean"),
        )
        .reset_index()
    )
    print("\nCALIBRATION BY OLDEST FIGHTER AGE")
    print(by_age.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_saved_comparison(frame: pd.DataFrame) -> None:
    candidates = [
        ("no_recovery_strong_collapse", baseline.OUTPUT_PATH),
        ("strong_collapse_plus_recovery", recovery_audit.OUTPUT_PATH),
    ]
    rows = [
        {
            "engine": "hybrid_v3",
            "any_ko": frame["p_any_ko"].mean(),
            "r1_ko": frame["p_r1_ko"].mean(),
            "mean_finish_round": frame["mean_sim_finish_round"].mean(),
        }
    ]
    for label, path in candidates:
        if not path.exists():
            continue
        old = pd.read_csv(path)
        if len(old) != len(frame):
            continue
        if set(old["bout_id"].astype(str)) != set(frame["bout_id"].astype(str)):
            continue
        rows.append(
            {
                "engine": label,
                "any_ko": old["p_any_ko"].mean(),
                "r1_ko": old["p_r1_ko"].mean(),
                "mean_finish_round": old["mean_sim_finish_round"].mean(),
            }
        )
    if len(rows) > 1:
        print("\nSAVED ENGINE COMPARISON")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_summary(frame: pd.DataFrame, paths: int, rounds: int) -> None:
    print("\n" + "=" * 124)
    print("MATURE 2020+ MC AUDIT — KO/TKO V3 HYBRID SHADOW")
    print("=" * 124)
    print(f"bouts: {len(frame):,}")
    print(f"paths per bout: {paths}")
    print(f"total paths: {len(frame) * paths:,}")
    print(f"simulation horizon: {rounds} rounds")
    print(f"age coverage: {frame[['r_age', 'b_age']].notna().all(axis=1).mean():.2%}")
    print(f"actual KO/TKO rate: {frame['actual_ko_tko'].mean():.2%}")
    print(f"simulated any-KO rate: {frame['p_any_ko'].mean():.2%}")
    print(f"actual R1 KO/TKO rate: {frame['actual_r1_ko'].mean():.2%}")
    print(f"simulated R1-KO rate: {frame['p_r1_ko'].mean():.2%}")
    print(f"actual R2 KO/TKO rate: {frame['actual_r2_ko'].mean():.2%}")
    print(f"simulated R2-KO rate: {frame['p_r2_ko'].mean():.2%}")
    print(f"actual R3 KO/TKO rate: {frame['actual_r3_ko'].mean():.2%}")
    print(f"simulated R3-KO rate: {frame['p_r3_ko'].mean():.2%}")

    _print_occurrence_metrics(frame)

    ko_bouts = frame.loc[frame["actual_ko_tko"].eq(1)].copy()
    valid_dir = ko_bouts["ko_winner_direction_hit"].notna()
    print("\nACTUAL KO/TKO WINNER DIRECTION")
    print(f"actual KO bouts: {len(ko_bouts):,}")
    print(f"non-tie directional calls: {int(valid_dir.sum()):,}")
    print(
        "direction hit rate among non-ties: "
        f"{ko_bouts.loc[valid_dir, 'ko_winner_direction_hit'].mean():.2%}"
    )
    print(f"direction tie rate: {ko_bouts['ko_winner_direction_tie'].mean():.2%}")
    print(
        "mean P(actual KO winner scores KO): "
        f"{ko_bouts['p_actual_winner_ko'].mean():.2%}"
    )

    _print_route_decomposition(frame)
    _print_age_calibration(frame)

    actual_ko_rounds = frame.loc[
        frame["actual_ko_tko"].eq(1), "actual_finish_round"
    ].dropna()
    sim_finish_rounds = frame["mean_sim_finish_round"].dropna()
    print("\nFINISH ROUND DIAGNOSTIC")
    print(f"actual KO mean finish round: {actual_ko_rounds.mean():.3f}")
    print(f"mean bout-level simulated finish round: {sim_finish_rounds.mean():.3f}")
    print(
        "mean reservoir restored per bout-path: "
        f"red={frame['mean_red_round_recovery'].mean():.2f}, "
        f"blue={frame['mean_blue_round_recovery'].mean():.2f}"
    )

    _print_saved_comparison(frame)
    print(
        "\nNOTE: V3 hazard constants are provisional. Diagnose route mix and round "
        "shape before tuning them. Stored FSR profiles and prior engines are unchanged."
    )


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    cohort, pairs = baseline._build_cohort()
    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {len(cohort) * args.paths:,}")
    print("engine: KO/TKO V3 hybrid shadow")
    print("finish routes: acute KO + post-KD TKO + cumulative exhaustion safeguard")
    print("KD collapse: disabled")
    print("post-KD strike damage multiplier: disabled")
    print("between-round recovery: enabled from existing recovery_ability curve")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    completed_paths = 0
    total_paths = len(cohort) * args.paths

    for bout_no, (_, bout) in enumerate(cohort.iterrows(), start=1):
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        rows.append(
            _simulate_one_bout(
                bout,
                pairs[str(bout["bout_id"])],
                paths=args.paths,
                rounds=args.rounds,
                seeds=seeds,
            )
        )
        completed_paths += args.paths
        if completed_paths % 1000 == 0 or bout_no == len(cohort):
            print(
                f"[hybrid V3 MC] paths {completed_paths:,}/{total_paths:,}; "
                f"bouts {bout_no:,}/{len(cohort):,}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    _print_summary(result, args.paths, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {len(result):,} bout rows to {args.output}")


if __name__ == "__main__":
    main()
