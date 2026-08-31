"""Run the frozen 34-fight validation baseline with the active age-modifier config.

This is a controlled A/B against:
  data/experimental/validation_baselines/fsr_mc_card_validation_prechange_v1.csv

Locks:
- exact frozen 34 bout IDs; no card rebuilding
- 1,000 paths per bout by default
- same per-bout seed stream as run_single_historical_full_fight_bout.py
- same current full-fight simulator
- only the active external age-modifier YAML differs from the frozen V1 baseline

Outputs a bout-level comparison CSV plus an aggregate summary CSV.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base


BASELINE_PATH = Path(
    "data/experimental/validation_baselines/fsr_mc_card_validation_prechange_v1.csv"
)
OUTPUT_DIR = Path("data/experimental/validation_age_modifier_ab")
OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_age_modifier_v1.csv"
SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_age_modifier_v1_summary.csv"
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811


def _age(row: pd.Series, col: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _winner_name(winner: int, red_name: str, blue_name: str) -> str:
    return red_name if int(winner) == 0 else blue_name


def _actual_probability(row: pd.Series, red_name: str, blue_name: str, p_red: float, p_blue: float) -> float:
    actual = str(row["actual_winner"])
    if actual == red_name:
        return float(p_red)
    if actual == blue_name:
        return float(p_blue)
    raise RuntimeError(
        f"actual winner {actual!r} does not match aligned names {red_name!r}/{blue_name!r}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if not args.baseline.exists():
        raise FileNotFoundError(args.baseline)

    started = time.perf_counter()
    baseline = pd.read_csv(args.baseline)
    if len(baseline) != 34:
        raise RuntimeError(f"frozen validation baseline must contain 34 fights, found {len(baseline)}")
    baseline["bout_id"] = baseline["bout_id"].astype(str)
    if baseline["bout_id"].duplicated().any():
        raise RuntimeError("frozen validation baseline contains duplicate bout_id values")

    print("[34-fight age A/B] building aligned historical cohort once...", flush=True)
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    cohort_by_id = cohort.set_index("bout_id", drop=False)

    missing = [bout_id for bout_id in baseline["bout_id"] if bout_id not in pairs or bout_id not in cohort_by_id.index]
    if missing:
        raise RuntimeError(f"baseline bouts missing from aligned cohort: {missing}")

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    rows: list[dict[str, object]] = []
    total = len(baseline)
    for index, old in baseline.iterrows():
        bout_started = time.perf_counter()
        bout_id = str(old["bout_id"])
        bout = cohort_by_id.loc[bout_id]
        if isinstance(bout, pd.DataFrame):
            if len(bout) != 1:
                raise RuntimeError(f"expected one cohort row for {bout_id}, found {len(bout)}")
            bout = bout.iloc[0]
        red, blue = pairs[bout_id]
        red_name = base._display_name(red)
        blue_name = base._display_name(blue)
        red_age = _age(bout, "r_age")
        blue_age = _age(bout, "b_age")

        if red_name != str(old["red"]) or blue_name != str(old["blue"]):
            raise RuntimeError(
                f"baseline/aligned name mismatch for {bout_id}: "
                f"baseline={old['red']} vs {old['blue']} | aligned={red_name} vs {blue_name}"
            )

        wins = [0, 0]
        methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
        for seed in seeds:
            sim = full.StaticFSRMCFullFightV1(
                red,
                blue,
                rounds=3,
                seed=int(seed),
                red_age=red_age,
                blue_age=blue_age,
            )
            path = sim.run()
            wins[int(path.winner)] += 1
            methods[path.method] += 1

        n = float(args.paths)
        p_red = wins[0] / n
        p_blue = wins[1] / n
        p_ko = methods["KO/TKO"] / n
        p_sub = methods["SUB"] / n
        p_dec = methods["DEC"] / n
        new_favorite = red_name if p_red >= p_blue else blue_name
        old_actual_p = _actual_probability(
            old, red_name, blue_name, float(old["p_red_win"]), float(old["p_blue_win"])
        )
        new_actual_p = _actual_probability(old, red_name, blue_name, p_red, p_blue)
        new_correct = int(new_favorite == str(old["actual_winner"]))

        rows.append({
            "baseline_version": old["baseline_version"],
            "card_no": old["card_no"],
            "event_date": old["event_date"],
            "event_name": old["event_name"],
            "bout_id": bout_id,
            "red": red_name,
            "blue": blue_name,
            "red_age": red_age,
            "blue_age": blue_age,
            "paths": args.paths,
            "old_p_red_win": float(old["p_red_win"]),
            "new_p_red_win": p_red,
            "delta_p_red_win": p_red - float(old["p_red_win"]),
            "old_p_blue_win": float(old["p_blue_win"]),
            "new_p_blue_win": p_blue,
            "delta_p_blue_win": p_blue - float(old["p_blue_win"]),
            "old_p_ko_tko": float(old["p_ko_tko"]),
            "new_p_ko_tko": p_ko,
            "delta_p_ko_tko": p_ko - float(old["p_ko_tko"]),
            "old_p_sub": float(old["p_sub"]),
            "new_p_sub": p_sub,
            "delta_p_sub": p_sub - float(old["p_sub"]),
            "old_p_dec": float(old["p_dec"]),
            "new_p_dec": p_dec,
            "delta_p_dec": p_dec - float(old["p_dec"]),
            "old_mc_favorite": old["mc_favorite"],
            "new_mc_favorite": new_favorite,
            "favorite_flipped": int(str(old["mc_favorite"]) != new_favorite),
            "actual_winner": old["actual_winner"],
            "actual_method": old["actual_method"],
            "old_mc_correct": int(old["mc_correct"]),
            "new_mc_correct": new_correct,
            "old_actual_winner_probability": old_actual_p,
            "new_actual_winner_probability": new_actual_p,
            "delta_actual_winner_probability": new_actual_p - old_actual_p,
        })

        elapsed = time.perf_counter() - bout_started
        total_elapsed = time.perf_counter() - started
        print(
            f"[34-fight age A/B] {index + 1:02d}/{total} {red_name} vs {blue_name} | "
            f"old {float(old['p_red_win']):.1%}/{float(old['p_blue_win']):.1%} -> "
            f"new {p_red:.1%}/{p_blue:.1%} | actual Δ {new_actual_p - old_actual_p:+.1%} | "
            f"{elapsed:.1f}s bout | {total_elapsed:.1f}s total",
            flush=True,
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    old_accuracy = float(out["old_mc_correct"].mean())
    new_accuracy = float(out["new_mc_correct"].mean())
    summary = pd.DataFrame([{
        "bouts": len(out),
        "paths_per_bout": args.paths,
        "old_correct": int(out["old_mc_correct"].sum()),
        "new_correct": int(out["new_mc_correct"].sum()),
        "old_accuracy": old_accuracy,
        "new_accuracy": new_accuracy,
        "accuracy_delta": new_accuracy - old_accuracy,
        "favorite_flips": int(out["favorite_flipped"].sum()),
        "mean_delta_actual_winner_probability": float(out["delta_actual_winner_probability"].mean()),
        "median_delta_actual_winner_probability": float(out["delta_actual_winner_probability"].median()),
        "old_mean_p_ko_tko": float(out["old_p_ko_tko"].mean()),
        "new_mean_p_ko_tko": float(out["new_p_ko_tko"].mean()),
        "delta_mean_p_ko_tko": float((out["new_p_ko_tko"] - out["old_p_ko_tko"]).mean()),
        "old_mean_p_sub": float(out["old_p_sub"].mean()),
        "new_mean_p_sub": float(out["new_p_sub"].mean()),
        "delta_mean_p_sub": float((out["new_p_sub"] - out["old_p_sub"]).mean()),
        "old_mean_p_dec": float(out["old_p_dec"].mean()),
        "new_mean_p_dec": float(out["new_p_dec"].mean()),
        "delta_mean_p_dec": float((out["new_p_dec"] - out["old_p_dec"]).mean()),
    }])
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 110)
    print("34-FIGHT AGE MODIFIER A/B SUMMARY")
    print("=" * 110)
    print(f"old winner accuracy: {int(out['old_mc_correct'].sum())}/34 = {old_accuracy:.1%}")
    print(f"new winner accuracy: {int(out['new_mc_correct'].sum())}/34 = {new_accuracy:.1%}")
    print(f"favorite flips: {int(out['favorite_flipped'].sum())}")
    print(
        "actual-winner probability: "
        f"mean Δ={out['delta_actual_winner_probability'].mean():+.2%} | "
        f"median Δ={out['delta_actual_winner_probability'].median():+.2%}"
    )
    print(
        "method means: "
        f"KO {out['old_p_ko_tko'].mean():.1%}->{out['new_p_ko_tko'].mean():.1%} | "
        f"SUB {out['old_p_sub'].mean():.1%}->{out['new_p_sub'].mean():.1%} | "
        f"DEC {out['old_p_dec'].mean():.1%}->{out['new_p_dec'].mean():.1%}"
    )

    flips = out.loc[out["favorite_flipped"].eq(1), [
        "red", "blue", "old_mc_favorite", "new_mc_favorite", "actual_winner",
        "old_mc_correct", "new_mc_correct", "delta_actual_winner_probability",
    ]]
    if not flips.empty:
        print("\nFAVORITE FLIPS")
        print(flips.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nwrote: {args.output}")
    print(f"wrote: {SUMMARY_PATH}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
