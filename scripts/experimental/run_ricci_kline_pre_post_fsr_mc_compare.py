"""Compare Ricci/Kline MC using actual target pre-fight vs observed post-fight FSR.

Diagnostic only. The post-fight arm is intentionally leaky/counterfactual: it
uses evidence from the target fight itself to ask how much the FSR update would
move the simulator's view of the same matchup.

No FSR replay occurs here. This script reads the already-generated canonical
pre/post trait CSV and overlays those values onto the existing aligned FSR-32
historical profiles, preserving non-FSR metadata and simulator compatibility
fields. Both arms use identical Monte Carlo seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_single_historical_full_fight_bout as historical


DEFAULT_BOUT_ID = "52ddf20a10890b41"
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
DEFAULT_INPUT = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_canonical_shadow/"
    "fsr_canonical_52ddf20a10890b41_pre_post_traits.csv"
)
DEFAULT_OUTPUT = Path(
    "data/experimental/ricci_kline_pre_post_fsr_mc_compare.csv"
)


def _load_trait_map(path: Path, bout_id: str) -> dict[str, dict[str, dict[str, float]]]:
    if not path.exists():
        raise RuntimeError(f"canonical pre/post trait CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"fighter_id", "trait", "pre_fsr", "post_fsr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"pre/post trait CSV missing columns: {missing}")

    df["fighter_id"] = df["fighter_id"].astype(str)
    if "fight_id" in df.columns:
        selected = df.loc[df["fight_id"].astype(str).eq(str(bout_id))].copy()
        if not selected.empty:
            df = selected

    out: dict[str, dict[str, dict[str, float]]] = {}
    for fighter_id, fighter_rows in df.groupby("fighter_id", sort=False):
        pre: dict[str, float] = {}
        post: dict[str, float] = {}
        for row in fighter_rows.itertuples(index=False):
            trait = str(row.trait)
            if trait not in canonical.CANONICAL_RATINGS:
                continue
            pre[trait] = float(row.pre_fsr)
            post[trait] = float(row.post_fsr)
        missing_traits = sorted(set(canonical.CANONICAL_RATINGS) - set(pre))
        if missing_traits:
            raise RuntimeError(
                f"fighter {fighter_id} missing canonical traits in pre/post CSV: {missing_traits}"
            )
        out[str(fighter_id)] = {"pre": pre, "post": post}
    if len(out) != 2:
        raise RuntimeError(f"expected exactly two fighters in pre/post CSV, found {len(out)}")
    return out


def _overlay(profile: pd.Series, values: dict[str, float]) -> pd.Series:
    out = profile.copy()
    for trait, value in values.items():
        out[trait] = float(value)

    # Keep compatibility aliases synchronized with canonical learned ratings.
    out["distance_precision"] = float(out["distance_striking_precision"])
    out["distance_defense"] = float(out["distance_striking_defense"])
    out["stamina_depletion_resistance"] = float(out["fatigue_accumulation_resistance"])
    out["stamina_performance_resilience"] = float(out["fatigue_performance_resilience"])
    out["stamina_capacity"] = float(canonical.STAMINA_CAPACITY)
    return out


def _run_arm(
    label: str,
    red: pd.Series,
    blue: pd.Series,
    *,
    seeds: np.ndarray,
    red_age: float | None,
    blue_age: float | None,
) -> dict[str, object]:
    winners = [0, 0]
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    winner_methods = {
        0: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
        1: {"KO/TKO": 0, "SUB": 0, "DEC": 0},
    }

    total = len(seeds)
    heartbeat = max(1, total // 5)
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
        methods[method] += 1
        winner_methods[winner][method] += 1
        if i == 1 or i % heartbeat == 0 or i == total:
            print(f"[{label}] {i:,}/{total:,} paths ({100.0*i/total:.0f}%)", flush=True)

    n = float(total)
    return {
        "arm": label,
        "paths": total,
        "p_red_win": winners[0] / n,
        "p_blue_win": winners[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n,
        "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
        "p_red_ko_tko": winner_methods[0]["KO/TKO"] / n,
        "p_blue_ko_tko": winner_methods[1]["KO/TKO"] / n,
        "p_red_sub": winner_methods[0]["SUB"] / n,
        "p_blue_sub": winner_methods[1]["SUB"] / n,
        "p_red_dec": winner_methods[0]["DEC"] / n,
        "p_blue_dec": winner_methods[1]["DEC"] / n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bout-id", default=DEFAULT_BOUT_ID)
    p.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--pre-post-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")

    bout, pair = historical._select_bout(str(args.bout_id))
    original_red, original_blue = pair
    red_id = str(original_red["fighter_id"])
    blue_id = str(original_blue["fighter_id"])
    values = _load_trait_map(args.pre_post_csv, str(args.bout_id))
    if red_id not in values or blue_id not in values:
        raise RuntimeError(
            f"profile IDs do not match pre/post CSV: red={red_id}, blue={blue_id}, "
            f"csv={sorted(values)}"
        )

    red_name = base._display_name(original_red)
    blue_name = base._display_name(original_blue)
    red_age = historical._age(bout, "r_age")
    blue_age = historical._age(bout, "b_age")

    pre_red = _overlay(original_red, values[red_id]["pre"])
    pre_blue = _overlay(original_blue, values[blue_id]["pre"])
    post_red = _overlay(original_red, values[red_id]["post"])
    post_blue = _overlay(original_blue, values[blue_id]["post"])

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    print("=" * 100)
    print("RICCI / KLINE — PRE-FIGHT VS OBSERVED POST-FIGHT FSR MC")
    print("=" * 100)
    print(f"bout_id: {args.bout_id}")
    print(f"RED : {red_name} ({red_id})")
    print(f"BLUE: {blue_name} ({blue_id})")
    print(f"paths per arm: {args.paths:,} | identical seeds")
    print("POST arm is intentionally leaky/counterfactual for diagnostic use only.\n")

    pre = _run_arm(
        "PRE",
        pre_red,
        pre_blue,
        seeds=seeds,
        red_age=red_age,
        blue_age=blue_age,
    )
    post = _run_arm(
        "POST",
        post_red,
        post_blue,
        seeds=seeds,
        red_age=red_age,
        blue_age=blue_age,
    )

    result = pd.DataFrame([pre, post])
    delta = {"arm": "POST_MINUS_PRE", "paths": args.paths}
    for col in result.columns:
        if col.startswith("p_"):
            delta[col] = float(post[col]) - float(pre[col])
    result = pd.concat([result, pd.DataFrame([delta])], ignore_index=True)

    display = result.copy()
    prob_cols = [c for c in display.columns if c.startswith("p_")]
    for col in prob_cols:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{100.0*x:+.1f} pp" if display.loc[display[col].eq(x), "arm"].iloc[0] == "POST_MINUS_PRE" else f"{100.0*x:.1f}%")

    # Explicit compact report avoids relying on the formatting helper above.
    print("\nRESULT")
    for arm in (pre, post):
        print(
            f"{arm['arm']:4s} | {red_name} {arm['p_red_win']:.1%} | "
            f"{blue_name} {arm['p_blue_win']:.1%} | "
            f"KO {arm['p_ko_tko']:.1%} | SUB {arm['p_sub']:.1%} | DEC {arm['p_dec']:.1%}"
        )
    print(
        "DELTA POST-PRE | "
        f"{red_name} {100*(post['p_red_win']-pre['p_red_win']):+.1f} pp | "
        f"{blue_name} {100*(post['p_blue_win']-pre['p_blue_win']):+.1f} pp | "
        f"KO {100*(post['p_ko_tko']-pre['p_ko_tko']):+.1f} pp | "
        f"SUB {100*(post['p_sub']-pre['p_sub']):+.1f} pp | "
        f"DEC {100*(post['p_dec']-pre['p_dec']):+.1f} pp"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
