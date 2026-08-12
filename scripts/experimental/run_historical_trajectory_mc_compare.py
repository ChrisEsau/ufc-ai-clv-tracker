"""Generic historical Current vs Raw vs Shrunk FSR-32 MC comparison.

For any eligible historical bout_id:
- resolves the two canonical FSR-32 fighter profiles,
- builds leakage-safe linear trajectory forecasts using only prior snapshots,
- runs CURRENT, RAW, and SHRUNK profiles through the same full-fight MC,
- reuses the exact same path seeds across all three variants,
- prints heartbeat/progress output and a compact comparison table.

Research/shadow only.  Does not modify canonical FSR or simulator code.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental.fsr_32_trajectory_forecast_audit import (
    FSR_PATH,
    build_forecasts,
)
from scripts.experimental.run_single_historical_full_fight_bout import (
    _age,
    _elapsed_seconds,
    _select_bout,
)

DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
DEFAULT_OUTPUT_DIR = Path("data/experimental/trajectory_mc_compare")
VARIANTS = ("current", "raw", "shrunk")


def _target_profile(frame: pd.DataFrame, bout_id: str, fighter_id: str) -> pd.Series:
    rows = frame.loc[
        frame["fight_id"].astype(str).eq(str(bout_id))
        & frame["fighter_id"].astype(str).eq(str(fighter_id))
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one target FSR row for bout={bout_id}, fighter={fighter_id}; found {len(rows)}"
        )
    return rows.iloc[0].copy()


def _run_variant(
    *,
    label: str,
    red: pd.Series,
    blue: pd.Series,
    red_age: float | None,
    blue_age: float | None,
    seeds: np.ndarray,
    heartbeat_every: int,
) -> dict[str, object]:
    started = time.perf_counter()
    red_name = base._display_name(red)
    blue_name = base._display_name(blue)

    winners = {0: 0, 1: 0}
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    winner_methods = {
        0: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
        1: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
    }
    decision_winners = {0: 0, 1: 0}
    elapsed_sum = 0.0

    total = len(seeds)
    print(f"[{label}] starting {total:,} paths", flush=True)

    for i, seed in enumerate(seeds, start=1):
        sim = full.StaticFSRMCFullFightV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()
        winner = int(path.winner)
        method = str(path.method)

        winners[winner] += 1
        methods[method] = methods.get(method, 0) + 1
        winner_methods[winner][method] = winner_methods[winner].get(method, 0) + 1
        if method == "DEC":
            decision_winners[winner] += 1
        elapsed_sum += _elapsed_seconds(path)

        if i == 1 or i % heartbeat_every == 0 or i == total:
            elapsed = time.perf_counter() - started
            pct = 100.0 * i / total
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total - i) / rate if rate > 0 else 0.0
            print(
                f"[{label}] {i:,}/{total:,} | {pct:5.1f}% | "
                f"elapsed={elapsed:6.1f}s | eta={eta:6.1f}s",
                flush=True,
            )

    n = float(total)
    dec_n = methods.get("DEC", 0)
    runtime = time.perf_counter() - started
    print(f"[{label}] complete in {runtime:.2f}s", flush=True)

    return {
        "variant": label,
        "red_name": red_name,
        "blue_name": blue_name,
        "paths": total,
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods.get("KO/TKO", 0) / n,
        "p_sub": methods.get("SUB", 0) / n,
        "p_dec": methods.get("DEC", 0) / n,
        "p_red_ko_tko": winner_methods[0].get("KO/TKO", 0) / n,
        "p_blue_ko_tko": winner_methods[1].get("KO/TKO", 0) / n,
        "p_red_sub": winner_methods[0].get("SUB", 0) / n,
        "p_blue_sub": winner_methods[1].get("SUB", 0) / n,
        "p_red_dec": winner_methods[0].get("DEC", 0) / n,
        "p_blue_dec": winner_methods[1].get("DEC", 0) / n,
        "p_red_win_given_decision": (decision_winners[0] / dec_n) if dec_n else np.nan,
        "p_blue_win_given_decision": (decision_winners[1] / dec_n) if dec_n else np.nan,
        "mean_elapsed_seconds": elapsed_sum / n,
        "runtime_seconds": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bout-id", required=True, help="Historical mature-cohort bout_id")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=0,
        help="Print progress every N paths per variant; default is about 10 heartbeats.",
    )
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    overall_started = time.perf_counter()
    bout_id = str(args.bout_id)

    print("=" * 112)
    print("HISTORICAL FSR-32 TRAJECTORY MC COMPARISON")
    print("=" * 112)
    print(f"bout_id: {bout_id}")
    print(f"paths per variant: {args.paths:,}")
    print(f"seed: {args.seed}")
    print("variants: CURRENT / RAW / SHRUNK")
    print("same path seeds reused across all variants", flush=True)

    print("\n[setup] resolving historical bout + canonical current FSR...", flush=True)
    bout, current_pair = _select_bout(bout_id)
    current_red, current_blue = current_pair
    red_id = str(current_red["fighter_id"])
    blue_id = str(current_blue["fighter_id"])
    red_name = base._display_name(current_red)
    blue_name = base._display_name(current_blue)
    red_age = _age(bout, "r_age")
    blue_age = _age(bout, "b_age")
    print(f"[setup] RED : {red_name} ({red_id})", flush=True)
    print(f"[setup] BLUE: {blue_name} ({blue_id})", flush=True)

    print("[setup] loading FSR-32 history...", flush=True)
    fsr = pd.read_parquet(FSR_PATH)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    subset = fsr.loc[fsr["fighter_id"].isin({red_id, blue_id})].copy()
    print(
        f"[setup] retained {len(subset):,} prefight snapshots for the two fighters",
        flush=True,
    )

    print("[setup] fitting leakage-safe raw + shrunk trajectory forecasts...", flush=True)
    raw_frame, shrunk_frame, audit, traits = build_forecasts(subset)
    raw_red = _target_profile(raw_frame, bout_id, red_id)
    raw_blue = _target_profile(raw_frame, bout_id, blue_id)
    shrunk_red = _target_profile(shrunk_frame, bout_id, red_id)
    shrunk_blue = _target_profile(shrunk_frame, bout_id, blue_id)
    print(f"[setup] forecasted {len(traits)} FSR parameters", flush=True)

    target_audit = audit.loc[audit["fight_id"].astype(str).eq(bout_id)].copy()
    trajectory = target_audit.groupby("fighter_name", as_index=False).agg(
        mean_raw_delta=("raw_delta", "mean"),
        mean_shrunk_delta=("shrunk_delta", "mean"),
        mean_slope_per_year=("trend_slope_per_year", "mean"),
    )
    print("\nTARGET TRAJECTORY SUMMARY")
    print(trajectory.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(
        1,
        np.iinfo(np.int32).max,
        size=args.paths,
        dtype=np.int64,
    )
    heartbeat_every = args.heartbeat_every or max(1, args.paths // 10)

    variant_pairs = {
        "current": (current_red, current_blue),
        "raw": (raw_red, raw_blue),
        "shrunk": (shrunk_red, shrunk_blue),
    }

    results: list[dict[str, object]] = []
    for label in VARIANTS:
        print("\n" + "-" * 112, flush=True)
        red, blue = variant_pairs[label]
        results.append(
            _run_variant(
                label=label,
                red=red,
                blue=blue,
                red_age=red_age,
                blue_age=blue_age,
                seeds=seeds,
                heartbeat_every=heartbeat_every,
            )
        )

    result = pd.DataFrame(results)

    print("\n" + "=" * 112)
    print("CURRENT vs RAW vs SHRUNK — MC RESULTS")
    print("=" * 112)
    display_cols = [
        "variant",
        "p_red_win",
        "p_blue_win",
        "p_ko_tko",
        "p_sub",
        "p_dec",
        "p_red_win_given_decision",
        "p_blue_win_given_decision",
        "mean_elapsed_seconds",
    ]
    display = result[display_cols].copy()
    for col in [
        "p_red_win",
        "p_blue_win",
        "p_ko_tko",
        "p_sub",
        "p_dec",
        "p_red_win_given_decision",
        "p_blue_win_given_decision",
    ]:
        display[col] = display[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.2%}")
    display["mean_elapsed_seconds"] = display["mean_elapsed_seconds"].map(lambda x: f"{x:.1f}")
    print(display.to_string(index=False))

    current = result.loc[result["variant"].eq("current")].iloc[0]
    print("\nCHANGE FROM CURRENT")
    for label in ("raw", "shrunk"):
        row = result.loc[result["variant"].eq(label)].iloc[0]
        print(
            f"{label:6s}: {red_name} win {row['p_red_win'] - current['p_red_win']:+.2%} | "
            f"{blue_name} win {row['p_blue_win'] - current['p_blue_win']:+.2%} | "
            f"KO {row['p_ko_tko'] - current['p_ko_tko']:+.2%} | "
            f"SUB {row['p_sub'] - current['p_sub']:+.2%} | "
            f"DEC {row['p_dec'] - current['p_dec']:+.2%}"
        )

    out_dir = args.output_dir / bout_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "trajectory_mc_compare.csv"
    audit_path = out_dir / "target_trajectory_traits.csv"
    result.to_csv(result_path, index=False)
    target_audit.to_csv(audit_path, index=False)

    print("\nOUTPUTS")
    print(result_path)
    print(audit_path)
    print(f"\n[done] total elapsed={time.perf_counter() - overall_started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
