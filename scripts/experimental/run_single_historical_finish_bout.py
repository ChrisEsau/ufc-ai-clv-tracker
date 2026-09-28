"""Run one mature-cohort historical bout through the combined KO + SUB shadow MC.

Research-only convenience runner. The current locked KO/damage/stamina/phase
configuration is preserved; submission finishes are added only after existing
submission-attempt events.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_v1 as combined
from scripts.experimental import fsr_static_mc_v0 as base


DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
DEFAULT_OUTPUT_DIR = Path("data/experimental/single_historical_finish_bout")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one historical bout through current KO + submission shadow MC"
    )
    p.add_argument("--bout-id", required=True)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--save-paths", action="store_true")
    return p.parse_args()


def _select_bout(bout_id: str):
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    selected = cohort.loc[cohort["bout_id"].eq(str(bout_id))]
    if selected.empty:
        raise ValueError(f"bout_id {bout_id} is not in aligned mature 2020+ FSR-32 cohort")
    if len(selected) != 1:
        raise ValueError(f"expected one row for {bout_id}, found {len(selected)}")
    return selected.iloc[0], pairs[str(bout_id)]


def _age(row: pd.Series, column: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def main() -> None:
    args = parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    combined.configure_current_finish_candidate()
    bout, pair = _select_bout(str(args.bout_id))
    red, blue = pair
    red_name = base._display_name(red)
    blue_name = base._display_name(blue)
    red_age = _age(bout, "r_age")
    blue_age = _age(bout, "b_age")

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    rows: list[dict[str, object]] = []
    method_counts = {"KO/TKO": 0, "SUB": 0, "NONE": 0}
    winner_counts = {0: 0, 1: 0}
    sub_finish_by_side = {0: 0, 1: 0}
    total_sub_attempts = {0: 0, 1: 0}
    sub_paths_with_attempt = 0
    sub_finish_without_attempt = 0
    sub_round_counts = {1: 0, 2: 0, 3: 0}

    for seed in seeds:
        sim = combined.StaticFSRMCKOSUBV1(
            red,
            blue,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        )
        path = sim.run()

        red_sub_att = int(sim.stats[0].sub_att)
        blue_sub_att = int(sim.stats[1].sub_att)
        total_sub_attempts[0] += red_sub_att
        total_sub_attempts[1] += blue_sub_att
        any_sub_attempt = red_sub_att + blue_sub_att > 0
        sub_paths_with_attempt += int(any_sub_attempt)

        method = "NONE"
        winner = -1
        finish_round = 0
        if path.finish is not None:
            method = str(path.finish.method)
            winner = int(path.finish.winner)
            finish_round = int(path.finish.round or 0)
            method_counts[method] = method_counts.get(method, 0) + 1
            winner_counts[winner] += 1
            if method == "SUB":
                sub_finish_by_side[winner] += 1
                if finish_round in sub_round_counts:
                    sub_round_counts[finish_round] += 1
                if not any_sub_attempt:
                    sub_finish_without_attempt += 1
        else:
            method_counts["NONE"] += 1

        rows.append(
            {
                "seed": int(seed),
                "finish_method": method,
                "winner_side": "red" if winner == 0 else "blue" if winner == 1 else "none",
                "finish_round": finish_round if finish_round else np.nan,
                "red_sub_attempts": red_sub_att,
                "blue_sub_attempts": blue_sub_att,
                "red_sig_landed": int(sim.stats[0].sig_landed),
                "blue_sig_landed": int(sim.stats[1].sig_landed),
            }
        )

    n = float(args.paths)
    red_conversion = float(red["submission_conversion"])
    blue_conversion = float(blue["submission_conversion"])
    red_resistance = float(red["submission_resistance"])
    blue_resistance = float(blue["submission_resistance"])

    # Use one simulator instance only to expose deterministic matchup-level
    # per-attempt probabilities; no path is run from this object.
    probe = combined.StaticFSRMCKOSUBV1(
        red,
        blue,
        rounds=3,
        seed=args.seed,
        red_age=red_age,
        blue_age=blue_age,
    )
    red_p_attempt = probe._submission_finish_probability(0)
    blue_p_attempt = probe._submission_finish_probability(1)

    print("\n" + "=" * 108)
    print("HISTORICAL BOUT COMBINED FINISH AUDIT — CURRENT KO + SUB SHADOW MC")
    print("=" * 108)
    print(f"bout_id: {args.bout_id}")
    print(f"RED : {red_name}")
    print(f"BLUE: {blue_name}")
    print(f"paths: {args.paths:,}")
    print("\nSUBMISSION FSR")
    print(f"RED  {red_name}: conversion={red_conversion:.2f}, resistance={red_resistance:.2f}, P(SUB|attempt vs blue)={red_p_attempt:.3%}")
    print(f"BLUE {blue_name}: conversion={blue_conversion:.2f}, resistance={blue_resistance:.2f}, P(SUB|attempt vs red)={blue_p_attempt:.3%}")

    print("\nPATH OUTCOMES")
    print(f"KO/TKO: {method_counts.get('KO/TKO', 0) / n:.2%} ({method_counts.get('KO/TKO', 0):,}/{args.paths:,})")
    print(f"SUB:    {method_counts.get('SUB', 0) / n:.2%} ({method_counts.get('SUB', 0):,}/{args.paths:,})")
    print(f"NONE:   {method_counts.get('NONE', 0) / n:.2%} ({method_counts.get('NONE', 0):,}/{args.paths:,})")
    print(f"RED SUB:  {sub_finish_by_side[0] / n:.2%} ({sub_finish_by_side[0]:,})")
    print(f"BLUE SUB: {sub_finish_by_side[1] / n:.2%} ({sub_finish_by_side[1]:,})")

    print("\nSUBMISSION ATTEMPTS")
    print(f"RED attempts/path:  {total_sub_attempts[0] / n:.4f}")
    print(f"BLUE attempts/path: {total_sub_attempts[1] / n:.4f}")
    print(f"paths with >=1 SUB attempt: {sub_paths_with_attempt / n:.2%}")
    print(f"SUB finishes without an observed attempt: {sub_finish_without_attempt}")
    print("SUB finish round counts: " + ", ".join(f"R{r}={sub_round_counts[r]}" for r in (1, 2, 3)))

    out_dir = args.output_dir / str(args.bout_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "bout_id": str(args.bout_id),
                "red_name": red_name,
                "blue_name": blue_name,
                "paths": args.paths,
                "p_ko_tko": method_counts.get("KO/TKO", 0) / n,
                "p_sub": method_counts.get("SUB", 0) / n,
                "p_none": method_counts.get("NONE", 0) / n,
                "p_red_sub": sub_finish_by_side[0] / n,
                "p_blue_sub": sub_finish_by_side[1] / n,
                "red_sub_attempts_per_path": total_sub_attempts[0] / n,
                "blue_sub_attempts_per_path": total_sub_attempts[1] / n,
                "red_submission_conversion": red_conversion,
                "blue_submission_conversion": blue_conversion,
                "red_submission_resistance": red_resistance,
                "blue_submission_resistance": blue_resistance,
                "red_p_sub_per_attempt": red_p_attempt,
                "blue_p_sub_per_attempt": blue_p_attempt,
                "sub_finish_without_attempt": sub_finish_without_attempt,
            }
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    if args.save_paths:
        pd.DataFrame(rows).to_csv(out_dir / "paths.csv", index=False)

    print(f"\nsummary: {out_dir / 'summary.csv'}")
    if args.save_paths:
        print(f"paths:   {out_dir / 'paths.csv'}")
    print("Research-only. Existing KO benchmark files and production artifacts are unchanged.")


if __name__ == "__main__":
    main()
