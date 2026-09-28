"""Run stored-prefight-FSR BASELINE MC with fight-night age modifiers.

Purpose
-------
Clean 2026 control for the trajectory experiments:
- same mature leakage-safe FSR-32 cohort
- same 2026 eligibility rules
- same exclusion of the already-audited 47 bout_ids
- same StaticFSRMCFullFightV1 engine
- same 1000-path default
- same deterministic per-bout seeds
- same per-fight persistence / 20-fight checkpoints

Age modes
---------
Default:
- no trajectory adjustment
- stored prefight FSR is passed unchanged into the simulator
- r_age / b_age are supplied so config/fsr_age_modifiers.yaml applies its
  enabled+calibrated fight-night age traits
- striking-power age term remains OFF

With --age-power:
- everything above remains identical
- additionally apply the previously calibrated striking-power residual age term:
    modifier = -0.66104720 * (age - 30)
- the temporary fight-night profile is clamped to the standard 10-90 FSR range
- stored FSR artifacts remain immutable

The two modes use separate output directories so progress/checkpoints cannot be
mixed accidentally.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_age_modifiers as age_modifiers
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_2026_all_but_full_cohort as allbut

YEAR = 2026
POWER_AGE_CENTER = 30.0
POWER_AGE_INTERCEPT = 0.0
POWER_AGE_LINEAR = -0.66104720
FSR_MIN = 10.0
FSR_MAX = 90.0


def _paths(age_power: bool) -> tuple[Path, Path, Path, Path]:
    mode = "power_on" if age_power else "power_off"
    out_dir = Path("data/experimental/2026_baseline_age_full_cohort") / mode
    return (
        out_dir,
        out_dir / "progress.csv",
        out_dir / "baseline_age_2026_results.csv",
        out_dir / "baseline_age_2026_summary.csv",
    )


def _age_value(row: pd.Series, col: str) -> float | None:
    if col not in row.index or pd.isna(row[col]):
        return None
    value = float(row[col])
    return value if np.isfinite(value) else None


def _apply_optional_power_age(
    profile: pd.Series,
    age: float | None,
    *,
    enabled: bool,
) -> tuple[pd.Series, float]:
    """Return a temporary profile plus applied striking-power modifier."""
    effective = profile.copy(deep=True)
    if not enabled or age is None:
        return effective, 0.0
    if "striking_power" not in effective.index or pd.isna(effective["striking_power"]):
        raise ValueError("profile missing striking_power required by --age-power")
    modifier = POWER_AGE_INTERCEPT + POWER_AGE_LINEAR * (float(age) - POWER_AGE_CENTER)
    effective["striking_power"] = float(
        np.clip(float(effective["striking_power"]) + modifier, FSR_MIN, FSR_MAX)
    )
    return effective, float(modifier)


def _run_exact_age(
    red: pd.Series,
    blue: pd.Series,
    seeds: np.ndarray,
    *,
    red_age: float | None,
    blue_age: float | None,
    age_power: bool,
) -> dict[str, float]:
    # Power is applied to temporary profile copies here. The simulator then
    # applies the normal YAML-driven age layer to all other configured traits.
    red_effective, _ = _apply_optional_power_age(red, red_age, enabled=age_power)
    blue_effective, _ = _apply_optional_power_age(blue, blue_age, enabled=age_power)

    counts = {
        "red_ko": 0, "red_sub": 0, "red_dec": 0,
        "blue_ko": 0, "blue_sub": 0, "blue_dec": 0,
    }
    for seed in seeds:
        path = full.StaticFSRMCFullFightV1(
            red_effective,
            blue_effective,
            rounds=3,
            seed=int(seed),
            red_age=red_age,
            blue_age=blue_age,
        ).run()
        side = "red" if int(path.winner) == 0 else "blue"
        method = {"KO/TKO": "ko", "SUB": "sub", "DEC": "dec"}[path.method]
        counts[f"{side}_{method}"] += 1
    n = float(len(seeds))
    return {k: v / n for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--include-audited-47", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument(
        "--age-power",
        action="store_true",
        help=(
            "Also apply calibrated striking-power age term: "
            "-0.66104720 FSR points per year relative to age 30"
        ),
    )
    args = ap.parse_args()

    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    out_dir, progress_path, final_path, summary_path = _paths(args.age_power)
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and progress_path.exists():
        progress_path.unlink()

    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    date_col = allbut.replay._event_date_col(cohort)
    cohort["_event_date"] = pd.to_datetime(cohort[date_col], errors="raise").dt.normalize()

    for age_col in ("r_age", "b_age"):
        if age_col not in cohort.columns:
            raise RuntimeError(
                f"aligned cohort missing required age column {age_col!r}; "
                f"available columns: {list(cohort.columns)}"
            )

    eligible = cohort[cohort["_event_date"].dt.year.eq(YEAR)].copy()
    excluded: set[str] = set()
    if not args.include_audited_47:
        excluded = allbut._load_excluded_bout_ids()
        eligible = eligible[~eligible["bout_id"].isin(excluded)].copy()

    eligible = eligible[eligible["bout_id"].isin(set(map(str, pairs.keys())))].copy()
    eligible = eligible.sort_values(["_event_date", "bout_id"]).reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No eligible 2026 mature FSR bouts remain after exclusions.")

    completed = pd.DataFrame()
    completed_ids: set[str] = set()
    if progress_path.exists():
        completed = pd.read_csv(progress_path, dtype={"bout_id": str})
        if "bout_id" not in completed.columns:
            raise RuntimeError(f"{progress_path} missing bout_id")
        completed_ids = set(completed["bout_id"].astype(str))

    remaining = eligible[~eligible["bout_id"].isin(completed_ids)].copy().reset_index(drop=True)
    master_by_id = allbut._master_map()
    enabled_age_traits = age_modifiers.enabled_calibrated_traits()

    print("=" * 142)
    print("2026 BASELINE + AGE FULL-COHORT RUNNER — EXCLUDING CURRENT 47")
    print("=" * 142)
    print(
        f"eligible={len(eligible)} | already complete={len(set(eligible['bout_id']) & completed_ids)} | "
        f"remaining={len(remaining)} | excluded existing audit ids={len(excluded)}"
    )
    print(
        f"paths={args.paths} | checkpoint every {args.checkpoint_every} fights | "
        f"BASELINE stored prefight FSR | no trajectory | AGE ON | "
        f"POWER AGE={'ON' if args.age_power else 'OFF'} | saves after every fight"
    )
    print(
        "age config: config/fsr_age_modifiers.yaml | enabled+calibrated: "
        + (", ".join(enabled_age_traits) if enabled_age_traits else "NONE")
    )
    if args.age_power:
        print(
            f"power age override: modifier={POWER_AGE_INTERCEPT:+.8f} "
            f"{POWER_AGE_LINEAR:+.8f}*(age-{POWER_AGE_CENTER:.0f}); "
            f"clamp={FSR_MIN:.0f}-{FSR_MAX:.0f}"
        )
    print(f"output mode: {out_dir}")

    new_rows: list[dict[str, object]] = []
    completed_this_run = 0

    for _, bout in remaining.iterrows():
        bid = str(bout["bout_id"])
        date = pd.Timestamp(bout["_event_date"]).normalize()
        red, blue = pairs[bid]
        red_name = base._display_name(red)
        blue_name = base._display_name(blue)
        red_age = _age_value(bout, "r_age")
        blue_age = _age_value(bout, "b_age")
        _, red_power_mod = _apply_optional_power_age(red, red_age, enabled=args.age_power)
        _, blue_power_mod = _apply_optional_power_age(blue, blue_age, enabled=args.age_power)

        probs = _run_exact_age(
            red,
            blue,
            allbut._per_bout_seeds(args.seed, bid, args.paths),
            red_age=red_age,
            blue_age=blue_age,
            age_power=args.age_power,
        )

        master_row = master_by_id.get(bid)
        if master_row is None:
            raise RuntimeError(f"bout {bid} not found in master")
        actual_winner = allbut._actual_winner(master_row, red_name, blue_name)
        actual_method_raw = str(master_row["method"])
        actual_method = allbut._norm_method(actual_method_raw)

        red_p_win = probs["red_ko"] + probs["red_sub"] + probs["red_dec"]
        blue_p_win = probs["blue_ko"] + probs["blue_sub"] + probs["blue_dec"]
        predicted_winner = red_name if red_p_win >= blue_p_win else blue_name

        six = {
            f"{red_name} KO/TKO": probs["red_ko"],
            f"{red_name} SUB": probs["red_sub"],
            f"{red_name} DEC": probs["red_dec"],
            f"{blue_name} KO/TKO": probs["blue_ko"],
            f"{blue_name} SUB": probs["blue_sub"],
            f"{blue_name} DEC": probs["blue_dec"],
        }
        predicted_exact = max(six, key=six.get)
        actual_exact = f"{actual_winner} {actual_method}"
        aggregate = {
            "KO/TKO": probs["red_ko"] + probs["blue_ko"],
            "SUB": probs["red_sub"] + probs["blue_sub"],
            "DEC": probs["red_dec"] + probs["blue_dec"],
        }
        predicted_method = max(aggregate, key=aggregate.get)

        row = {
            "event_date": date,
            "bout_id": bid,
            "red": red_name,
            "blue": blue_name,
            "red_age": red_age,
            "blue_age": blue_age,
            "red_power_age_modifier": red_power_mod,
            "blue_power_age_modifier": blue_power_mod,
            "power_age_enabled": int(args.age_power),
            "actual_winner": actual_winner,
            "actual_method_raw": actual_method_raw,
            "actual_method": actual_method,
            "red_p_ko": probs["red_ko"],
            "red_p_sub": probs["red_sub"],
            "red_p_dec": probs["red_dec"],
            "blue_p_ko": probs["blue_ko"],
            "blue_p_sub": probs["blue_sub"],
            "blue_p_dec": probs["blue_dec"],
            "red_p_win": red_p_win,
            "blue_p_win": blue_p_win,
            "p_ko": aggregate["KO/TKO"],
            "p_sub": aggregate["SUB"],
            "p_dec": aggregate["DEC"],
            "predicted_winner": predicted_winner,
            "winner_correct": int(predicted_winner == actual_winner),
            "predicted_exact": predicted_exact,
            "predicted_exact_p": six[predicted_exact],
            "actual_exact": actual_exact,
            "exact_correct": int(actual_method != "OTHER" and predicted_exact == actual_exact),
            "predicted_method": predicted_method,
            "predicted_method_p": aggregate[predicted_method],
            "method_correct": int(actual_method != "OTHER" and predicted_method == actual_method),
            "age_enabled_traits": "|".join(enabled_age_traits),
            "paths": args.paths,
            "seed_base": args.seed,
        }
        new_rows.append(row)
        completed_this_run += 1

        all_done = pd.concat([completed, pd.DataFrame(new_rows)], ignore_index=True)
        all_done = (
            all_done.drop_duplicates(subset=["bout_id"], keep="last")
            .sort_values(["event_date", "bout_id"])
        )
        allbut._atomic_csv(all_done, progress_path)

        overall_done = len(
            set(all_done["bout_id"].astype(str)) & set(eligible["bout_id"].astype(str))
        )
        if completed_this_run % args.checkpoint_every == 0:
            ck = out_dir / f"checkpoint_{overall_done:04d}.csv"
            allbut._atomic_csv(all_done, ck)
            print(f"  checkpoint -> {ck}", flush=True)

        mark = "✓" if row["winner_correct"] else "✗"
        age_text = (
            f"ages={red_age:.1f}/{blue_age:.1f}"
            if red_age is not None and blue_age is not None
            else f"ages={red_age}/{blue_age}"
        )
        print(
            f"[{overall_done:03d}/{len(eligible):03d}] {date.date()} | {red_name} vs {blue_name} | "
            f"{age_text} | actual={actual_winner} {actual_method} | "
            f"pick={predicted_winner} {max(red_p_win, blue_p_win):.1%} {mark} | "
            f"KO/SUB/DEC={aggregate['KO/TKO']:.1%}/{aggregate['SUB']:.1%}/{aggregate['DEC']:.1%}",
            flush=True,
        )

    final = pd.read_csv(progress_path, dtype={"bout_id": str})
    final = final[final["bout_id"].isin(set(eligible["bout_id"].astype(str)))].copy()
    final["event_date"] = pd.to_datetime(final["event_date"], errors="raise")
    final = final.sort_values(["event_date", "bout_id"]).reset_index(drop=True)
    allbut._atomic_csv(final, final_path)

    summary = allbut._build_summary(final)
    summary["age_enabled_traits"] = "|".join(enabled_age_traits)
    summary["power_age_enabled"] = int(args.age_power)
    summary["power_age_linear"] = POWER_AGE_LINEAR if args.age_power else 0.0
    summary["age_config"] = str(age_modifiers.DEFAULT_CONFIG_PATH)
    summary["missing_red_age"] = int(final["red_age"].isna().sum())
    summary["missing_blue_age"] = int(final["blue_age"].isna().sum())
    allbut._atomic_csv(summary, summary_path)

    s = summary.iloc[0]
    print("\nSUMMARY")
    print(f"winner: {int(s.winner_correct)}/{int(s.fights)} = {s.winner_accuracy:.1%}")
    if int(s.method_scored_fights) > 0:
        print(
            f"method only: {int(s.method_only_correct)}/{int(s.method_scored_fights)} = "
            f"{s.method_only_accuracy:.1%}"
        )
        print(
            f"exact fighter+method: {int(s.exact_fighter_method_correct)}/{int(s.method_scored_fights)} = "
            f"{s.exact_fighter_method_accuracy:.1%}"
        )
        print(
            f"actual mix: KO={int(s.actual_ko_tko)} SUB={int(s.actual_sub)} DEC={int(s.actual_dec)} | "
            f"mean sim P: KO={s.mean_p_ko:.1%} SUB={s.mean_p_sub:.1%} DEC={s.mean_p_dec:.1%}"
        )
    print(
        f"age missing: red={int(s.missing_red_age)} blue={int(s.missing_blue_age)} | "
        f"configured traits={s.age_enabled_traits or 'NONE'} | "
        f"power age={'ON' if bool(s.power_age_enabled) else 'OFF'}"
    )
    print(f"wrote: {final_path}")
    print(f"wrote: {summary_path}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
