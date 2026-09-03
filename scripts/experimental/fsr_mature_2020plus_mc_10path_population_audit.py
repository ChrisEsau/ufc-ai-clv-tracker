"""Run a low-path-count population audit of the current shadow FSR Monte Carlo.

Purpose
-------
Run the full modern mature-fighter historical cohort through the current
strong-KD-collapse KO/TKO simulator with the locked age mechanic enabled.

This is intentionally a broad diagnostic, not final calibration:
- 2020-01-01+ UFC bouts
- both fighters had >=3 prior UFC fights before the bout
- leakage-safe pre-fight FSR profiles
- fighter age resolved on the historical fight date
- strong KD-collapse candidate
- locked age adjustment applies only to knockdown_resistance and damage_durability
- default 10 Monte Carlo paths per bout
- fixed 3-round simulation horizon for this first population pass

Because only 10 paths are used, per-bout probabilities move in 0.10 increments.
The useful outputs are aggregate ranking/calibration diagnostics, not fine-grained
individual probabilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as age_study
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern
from scripts.experimental import fsr_static_mc_ko_tko_v2_kd_collapse_sweep as collapse


DEFAULT_PATHS = 10
DEFAULT_ROUNDS = 3
DEFAULT_SEED = 20260810
OUTPUT_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/"
    "mature_2020plus_mc_10path_population_audit.csv"
)
STRONG_COLLAPSE = collapse.CollapseCandidate("strong", 5.0, 2.0)

AGE_BINS = [-np.inf, 30.999, 33.999, 36.999, 39.999, np.inf]
AGE_LABELS = ["<=30", "31-33", "34-36", "37-39", "40+"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return p.parse_args()


def _master_metadata() -> pd.DataFrame:
    raw = pd.read_parquet(modern.MASTER_PATH).copy()
    date_col = modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    raw["fight_id"] = raw["fight_id"].astype(str)
    raw["r_id"] = raw["r_id"].astype(str)
    raw["b_id"] = raw["b_id"].astype(str)
    if "winner_id" in raw.columns:
        raw["winner_id"] = raw["winner_id"].astype(str)
    else:
        raw["winner_id"] = ""
    raw = raw.sort_values(["event_date", "fight_id"]).drop_duplicates("fight_id", keep="last")

    raw["r_age"] = age_study._resolve_corner_age(raw, "r")
    raw["b_age"] = age_study._resolve_corner_age(raw, "b")

    keep = ["fight_id", "winner_id", "r_age", "b_age"]
    return raw[keep].rename(columns={"fight_id": "bout_id"})


def _build_cohort() -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, pd.Series]]]:
    master = modern._load_master(modern.MASTER_PATH)
    cohort = modern._build_outcome_cohort(master)
    cohort, pairs = modern._load_fsr_pairs_for_cohort(modern.FSR_PATH, cohort)
    cohort = cohort.merge(_master_metadata(), on="bout_id", how="left", validate="one_to_one")
    return cohort.reset_index(drop=True), pairs


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
    r_ko = 0
    b_ko = 0
    actual_winner_ko = 0
    finish_rounds: list[int] = []

    for seed in seeds:
        sim = collapse.StaticFSRMCKOTKOV2KDCollapse(
            red,
            blue,
            collapse=STRONG_COLLAPSE,
            rounds=rounds,
            seed=int(seed),
            red_age=r_age,
            blue_age=b_age,
        )
        result = sim.run()
        finish = result.finish
        if finish is None:
            continue

        ko_count += 1
        finish_rounds.append(int(finish.round))
        if finish.round == 1:
            r1_ko_count += 1
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
    p_any_ko = ko_count / paths
    p_r1_ko = r1_ko_count / paths
    p_actual_winner_ko = actual_winner_ko / paths if winner_id in {r_id, b_id} else np.nan

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
        "max_age": np.nanmax([r_age if r_age is not None else np.nan, b_age if b_age is not None else np.nan]),
        "actual_ko_tko": int(bout["actual_ko_tko"]),
        "actual_r1_ko": int(bout["actual_r1_ko"]),
        "actual_finish_round": bout["actual_finish_round"],
        "p_any_ko": p_any_ko,
        "p_r1_ko": p_r1_ko,
        "p_r_ko": p_r_ko,
        "p_b_ko": p_b_ko,
        "p_actual_winner_ko": p_actual_winner_ko,
        "predicted_ko_winner": predicted_ko_winner,
        "ko_winner_direction_hit": (
            int(predicted_ko_winner == winner_id)
            if int(bout["actual_ko_tko"]) == 1 and predicted_ko_winner != "tie" and winner_id in {r_id, b_id}
            else np.nan
        ),
        "ko_winner_direction_tie": (
            int(predicted_ko_winner == "tie")
            if int(bout["actual_ko_tko"]) == 1 and winner_id in {r_id, b_id}
            else np.nan
        ),
        "mean_sim_finish_round": float(np.mean(finish_rounds)) if finish_rounds else np.nan,
    }


def _binary_metrics(frame: pd.DataFrame, target: str, probability: str) -> dict[str, float]:
    work = frame[[target, probability]].dropna().copy()
    y = work[target].astype(int)
    p = work[probability].astype(float).clip(1e-6, 1 - 1e-6)
    out = {
        "n": len(work),
        "actual_rate": float(y.mean()),
        "mean_pred": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
    }
    out["auc"] = float(roc_auc_score(y, p)) if y.nunique() == 2 else np.nan
    return out


def _print_summary(frame: pd.DataFrame, paths: int, rounds: int) -> None:
    print("\n" + "=" * 124)
    print("MATURE 2020+ POPULATION MC AUDIT — STRONG KD COLLAPSE + LOCKED AGE MECHANIC")
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

    metrics = []
    for label, target, prob in [
        ("Any KO/TKO", "actual_ko_tko", "p_any_ko"),
        ("R1 KO/TKO", "actual_r1_ko", "p_r1_ko"),
    ]:
        metrics.append({"target": label, **_binary_metrics(frame, target, prob)})
    print("\nKO OCCURRENCE METRICS")
    print(pd.DataFrame(metrics).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ko_bouts = frame.loc[frame["actual_ko_tko"].eq(1)].copy()
    valid_dir = ko_bouts["ko_winner_direction_hit"].notna()
    print("\nACTUAL KO/TKO WINNER DIRECTION")
    print(f"actual KO bouts: {len(ko_bouts):,}")
    print(f"non-tie directional calls: {int(valid_dir.sum()):,}")
    print(f"direction hit rate among non-ties: {ko_bouts.loc[valid_dir, 'ko_winner_direction_hit'].mean():.2%}")
    print(f"direction tie rate: {ko_bouts['ko_winner_direction_tie'].mean():.2%}")
    print(f"mean P(actual KO winner scores KO): {ko_bouts['p_actual_winner_ko'].mean():.2%}")

    frame["oldest_age_band"] = pd.cut(frame["max_age"], AGE_BINS, labels=AGE_LABELS)
    by_age = frame.groupby("oldest_age_band", observed=True).agg(
        bouts=("bout_id", "size"),
        actual_ko_rate=("actual_ko_tko", "mean"),
        sim_ko_rate=("p_any_ko", "mean"),
        actual_r1_ko_rate=("actual_r1_ko", "mean"),
        sim_r1_ko_rate=("p_r1_ko", "mean"),
    ).reset_index()
    print("\nCALIBRATION BY OLDEST FIGHTER AGE")
    print(by_age.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    actual_ko_rounds = frame.loc[frame["actual_ko_tko"].eq(1), "actual_finish_round"].dropna()
    sim_finish_rounds = frame["mean_sim_finish_round"].dropna()
    print("\nFINISH ROUND DIAGNOSTIC")
    if len(actual_ko_rounds):
        print(f"actual KO mean finish round: {actual_ko_rounds.mean():.3f}")
    if len(sim_finish_rounds):
        print(f"mean bout-level simulated finish round: {sim_finish_rounds.mean():.3f}")

    print("\nNOTE: 10 paths/bout gives coarse 0.10 probability increments. Use this pass to identify structural bias, not final probability calibration.")


def main() -> None:
    args = _parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    cohort, pairs = _build_cohort()
    print(f"mature 2020+ cohort: {len(cohort):,} bouts")
    print(f"paths per bout: {args.paths}")
    print(f"total paths: {len(cohort) * args.paths:,}")
    print("engine: strong KD collapse + locked age mechanic")
    print("age affects only knockdown_resistance and damage_durability")

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
                f"[population MC] paths {completed_paths:,}/{total_paths:,}; bouts {bout_no:,}/{len(cohort):,}",
                flush=True,
            )

    result = pd.DataFrame(rows)
    _print_summary(result, args.paths, args.rounds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nWrote {len(result):,} bout rows to {args.output}")
    print("Stored FSR ratings remain unchanged.")


if __name__ == "__main__":
    main()
