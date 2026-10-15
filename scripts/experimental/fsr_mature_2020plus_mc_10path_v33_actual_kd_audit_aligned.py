"""Historical KD-vs-KO audit for the aligned mature 2020+ V3.3 cohort.

Adds actual UFCStats knockdown totals to the existing V3.3 population simulation
and separates simulated reservoir-exhaustion finishes by whether the finishing
strike itself also scored a knockdown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_mature_2020plus_mc_10path_population_audit as baseline
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse
from scripts.experimental import fsr_static_mc_ko_tko_v3_3_global_recovery as v33

DEFAULT_PATHS = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_v33_actual_kd_audit_aligned.csv"
)
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _actual_kd_metadata() -> pd.DataFrame:
    raw = pd.read_parquet(modern.MASTER_PATH).copy()
    raw["fight_id"] = raw["fight_id"].astype(str)
    for col in ("r_kd", "b_kd"):
        if col not in raw.columns:
            raise RuntimeError(f"master missing required KD column: {col}")
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.sort_values("fight_id").drop_duplicates("fight_id", keep="last")
    out = raw[["fight_id", "r_kd", "b_kd"]].rename(columns={"fight_id": "bout_id"})
    out["actual_total_kd"] = out[["r_kd", "b_kd"]].sum(axis=1, min_count=1)
    out["actual_any_kd"] = np.where(
        out["actual_total_kd"].notna(),
        (out["actual_total_kd"] > 0).astype(float),
        np.nan,
    )
    return out


def _simulate_one_bout(bout, pair, *, paths, rounds, seeds):
    red, blue = pair
    r_id = str(bout["r_id"])
    b_id = str(bout["b_id"])
    winner_id = str(bout.get("winner_id", ""))
    r_age = float(bout["r_age"]) if pd.notna(bout["r_age"]) else None
    b_age = float(bout["b_age"]) if pd.notna(bout["b_age"]) else None

    ko_count = r1_ko_count = r_ko = b_ko = actual_winner_ko = 0
    any_kd_count = 0
    r_kd_total = b_kd_total = 0.0
    ko_finish_strike_kd = 0
    ko_finish_strike_no_kd = 0
    ko_paths_with_any_prior_or_finish_kd = 0
    finish_rounds: list[int] = []

    for seed in seeds:
        sim = v33.StaticFSRMCKOTKOV33GlobalRecovery(
            red,
            blue,
            collapse=STRONG_COLLAPSE,
            rounds=rounds,
            seed=int(seed),
            red_age=r_age,
            blue_age=b_age,
        )
        result = sim.run()

        r_kd = int(sim.stats[0].knockdowns_scored)
        b_kd = int(sim.stats[1].knockdowns_scored)
        total_kd = r_kd + b_kd
        r_kd_total += r_kd
        b_kd_total += b_kd
        any_kd_count += int(total_kd > 0)

        finish = result.finish
        if finish is None:
            continue

        ko_count += 1
        finish_rounds.append(int(finish.round))
        r1_ko_count += int(int(finish.round) == 1)
        ko_finish_strike_kd += int(bool(finish.knockdown_on_strike))
        ko_finish_strike_no_kd += int(not bool(finish.knockdown_on_strike))
        ko_paths_with_any_prior_or_finish_kd += int(total_kd > 0)

        if finish.winner == 0:
            r_ko += 1
            actual_winner_ko += int(winner_id == r_id)
        else:
            b_ko += 1
            actual_winner_ko += int(winner_id == b_id)

    p_r_ko = r_ko / paths
    p_b_ko = b_ko / paths
    predicted = "tie"
    if p_r_ko > p_b_ko:
        predicted = r_id
    elif p_b_ko > p_r_ko:
        predicted = b_id

    return {
        "bout_id": str(bout["bout_id"]),
        "event_date": bout["event_date"],
        "r_id": r_id,
        "b_id": b_id,
        "winner_id": winner_id,
        "actual_ko_tko": int(bout["actual_ko_tko"]),
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "actual_finish_round": bout["actual_finish_round"],
        "p_any_ko": ko_count / paths,
        "p_r1_ko": r1_ko_count / paths,
        "p_r_ko": p_r_ko,
        "p_b_ko": p_b_ko,
        "p_actual_winner_ko": actual_winner_ko / paths if winner_id in {r_id, b_id} else np.nan,
        "predicted_ko_winner": predicted,
        "ko_winner_direction_hit": (
            int(predicted == winner_id)
            if int(bout["actual_ko_tko"]) == 1 and predicted != "tie" and winner_id in {r_id, b_id}
            else np.nan
        ),
        "ko_winner_direction_tie": (
            int(predicted == "tie")
            if int(bout["actual_ko_tko"]) == 1 and winner_id in {r_id, b_id}
            else np.nan
        ),
        "p_any_kd": any_kd_count / paths,
        "mean_r_kd": r_kd_total / paths,
        "mean_b_kd": b_kd_total / paths,
        "mean_total_kd": (r_kd_total + b_kd_total) / paths,
        "p_ko_finish_strike_kd": ko_finish_strike_kd / paths,
        "p_ko_finish_strike_no_kd": ko_finish_strike_no_kd / paths,
        "p_ko_path_with_any_kd": ko_paths_with_any_prior_or_finish_kd / paths,
        "mean_sim_finish_round": float(np.mean(finish_rounds)) if finish_rounds else np.nan,
    }


def _print_summary(frame: pd.DataFrame, paths: int, rounds: int) -> None:
    print("\n" + "=" * 132)
    print("MATURE 2020+ V3.3 — HISTORICAL KD / RESERVOIR-KO AUDIT")
    print("=" * 132)
    print(f"bouts: {len(frame):,}")
    print(f"paths per bout: {paths}")
    print(f"total paths: {len(frame) * paths:,}")
    print(f"simulation horizon: {rounds} rounds")

    print("\nACTUAL VS SIMULATED KO")
    print(f"actual KO/TKO rate: {frame['actual_ko_tko'].mean():.2%}")
    print(f"simulated any-KO rate: {frame['p_any_ko'].mean():.2%}")
    print(f"actual R1 KO/TKO rate: {frame['actual_r1_ko'].mean():.2%}")
    print(f"simulated R1-KO rate: {frame['p_r1_ko'].mean():.2%}")

    kd = frame.dropna(subset=["actual_total_kd", "actual_any_kd"]).copy()
    print("\nACTUAL VS SIMULATED KD")
    print(f"historical KD coverage: {len(kd):,}/{len(frame):,} ({len(kd)/len(frame):.2%})")
    print(f"actual fights with >=1 KD: {kd['actual_any_kd'].mean():.2%}")
    print(f"simulated paths with >=1 KD: {frame['p_any_kd'].mean():.2%}")
    print(f"actual mean total KDs per fight: {kd['actual_total_kd'].mean():.4f}")
    print(f"simulated mean total KDs per fight: {frame['mean_total_kd'].mean():.4f}")

    actual_ko = kd.loc[kd["actual_ko_tko"].eq(1)]
    actual_nonko = kd.loc[kd["actual_ko_tko"].eq(0)]
    print("\nHISTORICAL KD BY OUTCOME")
    print(f"actual KO/TKO fights with >=1 recorded KD: {actual_ko['actual_any_kd'].mean():.2%}")
    print(f"actual KO/TKO mean KDs: {actual_ko['actual_total_kd'].mean():.4f}")
    print(f"actual non-KO fights with >=1 recorded KD: {actual_nonko['actual_any_kd'].mean():.2%}")
    print(f"actual non-KO mean KDs: {actual_nonko['actual_total_kd'].mean():.4f}")

    sim_ko_rate = frame["p_any_ko"].mean()
    finish_kd_rate = frame["p_ko_finish_strike_kd"].mean()
    finish_no_kd_rate = frame["p_ko_finish_strike_no_kd"].mean()
    any_kd_ko_rate = frame["p_ko_path_with_any_kd"].mean()
    print("\nSIMULATED RESERVOIR-EXHAUSTION FINISHES")
    print(f"all KO paths: {sim_ko_rate:.2%}")
    print(f"KO paths where finishing strike also KD'd: {finish_kd_rate:.2%}")
    print(f"KO paths where finishing strike did NOT KD: {finish_no_kd_rate:.2%}")
    if sim_ko_rate > 0:
        print(f"share of simulated KOs with finishing-strike KD: {finish_kd_rate/sim_ko_rate:.2%}")
        print(f"share of simulated KOs without finishing-strike KD: {finish_no_kd_rate/sim_ko_rate:.2%}")
        print(f"share of simulated KOs with any KD somewhere in path: {any_kd_ko_rate/sim_ko_rate:.2%}")

    metrics = []
    for label, target, prob in [
        ("Any KO/TKO", "actual_ko_tko", "p_any_ko"),
        ("R1 KO/TKO", "actual_r1_ko", "p_r1_ko"),
        ("Any KD", "actual_any_kd", "p_any_kd"),
    ]:
        metrics.append({"target": label, **baseline._binary_metrics(frame, target, prob)})
    print("\nOCCURRENCE METRICS")
    print(pd.DataFrame(metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nINTERPRETATION GUIDE")
    print("- If simulated KD is far below actual KD, tune strike->KD / acute-shock generation first.")
    print("- If simulated KD is near actual but KO remains low, tune reservoir depletion / KO conversion first.")
    print("- Reservoir exhaustion already causes deterministic KO/TKO even when the finishing strike is not a KD.")


def main() -> None:
    args = _parse_args()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.merge(_actual_kd_metadata(), on="bout_id", how="left", validate="one_to_one")

    rng = np.random.default_rng(args.seed)
    rows = []
    total = len(cohort) * args.paths
    completed = 0

    print(f"aligned FSR-32 mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {total:,}")

    for bout_no, (_, bout) in enumerate(cohort.iterrows(), start=1):
        seeds = rng.integers(0, 2**31 - 1, size=args.paths, dtype=np.int64)
        row = _simulate_one_bout(
            bout,
            pairs[str(bout["bout_id"])],
            paths=args.paths,
            rounds=args.rounds,
            seeds=seeds,
        )
        row["actual_r_kd"] = bout["r_kd"]
        row["actual_b_kd"] = bout["b_kd"]
        row["actual_total_kd"] = bout["actual_total_kd"]
        row["actual_any_kd"] = bout["actual_any_kd"]
        rows.append(row)

        completed += args.paths
        if completed % 1000 == 0 or bout_no == len(cohort):
            print(
                f"[V3.3 actual-KD audit] paths {completed:,}/{total:,}; "
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
