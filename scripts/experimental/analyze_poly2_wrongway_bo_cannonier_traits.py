"""Leave-one-trait-out attribution for the two fights broken by the corrected poly2 test.

Targets:
- Bo Nickal vs Kyle Daukaus
- Jared Cannonier vs Christian Leroy Duncan

For each fight, reconstruct the corrected include-target-prefight poly2 profiles from
its saved fighter_trait_forecasts.csv. Then, for each materially changed fighter/trait,
revert ONLY that trait to the stored prefight FSR and rerun the fight with the same
seed stream. The change in actual-winner probability versus the full-poly2 profile is
used as the trait's MC attribution.

Positive recovery_if_reverted means that poly2 movement in that trait was hurting the
actual winner. Negative means the poly2 movement in that trait was helping the actual
winner.

Research only. No stored FSR or simulator configuration is changed.
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

RESULT_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "fsr_mc_card_validation_poly2_include_target_v1.csv"
)
FORECAST_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "fighter_trait_forecasts.csv"
)
OUTPUT_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "bo_cannonier_leave_one_trait_out.csv"
)

TARGET_MATCHUPS = {
    ("Bo Nickal", "Kyle Daukaus"),
    ("Jared Cannonier", "Christian Leroy Duncan"),
}
DEFAULT_PATHS = 250
DEFAULT_SEED = 20260811
DEFAULT_MIN_ABS_DELTA = 0.25


def _winner_probability(actual: str, red_name: str, blue_name: str, wins: list[int], n: int) -> float:
    if actual == red_name:
        return wins[0] / float(n)
    if actual == blue_name:
        return wins[1] / float(n)
    raise RuntimeError(f"actual winner {actual!r} does not match {red_name!r}/{blue_name!r}")


def _simulate(red: pd.Series, blue: pd.Series, seeds: np.ndarray) -> tuple[float, float, dict[str, int]]:
    wins = [0, 0]
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    for seed in seeds:
        sim = full.StaticFSRMCFullFightV1(red, blue, rounds=3, seed=int(seed))
        path = sim.run()
        wins[int(path.winner)] += 1
        methods[path.method] += 1
    n = len(seeds)
    return wins[0] / n, wins[1] / n, methods


def _build_poly_profile(profile: pd.Series, audit: pd.DataFrame, fighter_name: str) -> pd.Series:
    out = profile.copy(deep=True)
    rows = audit.loc[audit["fighter_name"].eq(fighter_name)]
    if rows.empty:
        raise RuntimeError(f"no saved poly2 forecasts for {fighter_name}")
    for _, r in rows.iterrows():
        out[str(r["trait"])] = float(r["mc_fsr"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-abs-delta", type=float, default=DEFAULT_MIN_ABS_DELTA)
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    for p in (RESULT_PATH, FORECAST_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    started = time.perf_counter()
    results = pd.read_csv(RESULT_PATH)
    forecasts = pd.read_csv(FORECAST_PATH)
    results["bout_id"] = results["bout_id"].astype(str)
    forecasts["target_fight_id"] = forecasts["target_fight_id"].astype(str)

    selected = results.loc[
        results.apply(lambda r: (str(r["red"]), str(r["blue"])) in TARGET_MATCHUPS, axis=1)
    ].copy()
    if len(selected) != 2:
        raise RuntimeError(f"expected 2 target fights, found {len(selected)}")

    print("[trait-attribution] building aligned cohort once...", flush=True)
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    out_rows: list[dict[str, object]] = []

    for fight_no, (_, fight) in enumerate(selected.iterrows(), start=1):
        bout_id = str(fight["bout_id"])
        red_base, blue_base = pairs[bout_id]
        red_name = base._display_name(red_base)
        blue_name = base._display_name(blue_base)
        actual = str(fight["actual_winner"])

        fa = forecasts.loc[forecasts["target_fight_id"].eq(bout_id)].copy()
        red_poly = _build_poly_profile(red_base, fa, red_name)
        blue_poly = _build_poly_profile(blue_base, fa, blue_name)

        full_red_p, full_blue_p, _ = _simulate(red_poly, blue_poly, seeds)
        full_actual_p = full_red_p if actual == red_name else full_blue_p

        print("\n" + "=" * 120)
        print(f"{fight_no}/2 {red_name} vs {blue_name} | actual={actual}")
        print(
            f"saved 1000-path poly2: {float(fight['new_p_red_win']):.1%}/{float(fight['new_p_blue_win']):.1%} | "
            f"attribution baseline ({args.paths} paths): {full_red_p:.1%}/{full_blue_p:.1%}"
        )
        print("=" * 120, flush=True)

        candidates = fa.loc[fa["mc_delta"].abs() >= args.min_abs_delta].copy()
        candidates = candidates.sort_values("mc_delta", key=lambda s: s.abs(), ascending=False)
        print(
            f"[trait-attribution] testing {len(candidates)} changed traits "
            f"(|delta| >= {args.min_abs_delta:.2f})...",
            flush=True,
        )

        fight_rows: list[dict[str, object]] = []
        for idx, (_, tr) in enumerate(candidates.iterrows(), start=1):
            fighter_name = str(tr["fighter_name"])
            trait = str(tr["trait"])
            red_test = red_poly.copy(deep=True)
            blue_test = blue_poly.copy(deep=True)
            if fighter_name == red_name:
                red_test[trait] = float(tr["aligned_latest_fsr"])
            elif fighter_name == blue_name:
                blue_test[trait] = float(tr["aligned_latest_fsr"])
            else:
                raise RuntimeError(f"unexpected fighter {fighter_name}")

            p_red, p_blue, _ = _simulate(red_test, blue_test, seeds)
            reverted_actual_p = p_red if actual == red_name else p_blue
            recovery = reverted_actual_p - full_actual_p

            row = {
                "bout_id": bout_id,
                "red": red_name,
                "blue": blue_name,
                "actual_winner": actual,
                "fighter_name": fighter_name,
                "trait": trait,
                "stored_prefight_fsr": float(tr["aligned_latest_fsr"]),
                "poly2_fight_fsr": float(tr["mc_fsr"]),
                "poly2_delta": float(tr["mc_delta"]),
                "full_poly2_actual_p": full_actual_p,
                "reverted_trait_actual_p": reverted_actual_p,
                "recovery_if_reverted": recovery,
                "hurting_actual_winner": int(recovery > 0),
                "paths": args.paths,
            }
            fight_rows.append(row)
            out_rows.append(row)

            if idx % 5 == 0 or idx == len(candidates):
                print(f"  completed {idx}/{len(candidates)} trait reversions", flush=True)

        ranked = pd.DataFrame(fight_rows).sort_values("recovery_if_reverted", ascending=False)
        print("\nTOP TRAITS HURTING ACTUAL WINNER")
        print(
            ranked.head(12)[[
                "fighter_name", "trait", "stored_prefight_fsr", "poly2_fight_fsr",
                "poly2_delta", "recovery_if_reverted"
            ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )
        print("\nTOP TRAITS HELPING ACTUAL WINNER")
        print(
            ranked.tail(8).sort_values("recovery_if_reverted")[ [
                "fighter_name", "trait", "stored_prefight_fsr", "poly2_fight_fsr",
                "poly2_delta", "recovery_if_reverted"
            ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    out = pd.DataFrame(out_rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwrote: {OUTPUT_PATH}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")
    print("Positive recovery_if_reverted = that poly2 trait movement hurt the actual winner.")


if __name__ == "__main__":
    main()
