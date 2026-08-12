"""Run stored-prefight-FSR BASELINE MC with configured fight-night age modifiers.

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

Difference from ``run_2026_baseline_full_cohort.py``:
- no trajectory adjustment is applied
- stored prefight FSR is passed unchanged into the simulator
- r_age / b_age are supplied to the simulator so the externally configured
  fight-night age modifiers in config/fsr_age_modifiers.yaml are applied

Outputs
-------
data/experimental/2026_baseline_age_full_cohort/
  progress.csv
  checkpoint_0020.csv, checkpoint_0040.csv, ...
  baseline_age_2026_results.csv
  baseline_age_2026_summary.csv
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
OUT_DIR = Path("data/experimental/2026_baseline_age_full_cohort")
PROGRESS_PATH = OUT_DIR / "progress.csv"
FINAL_PATH = OUT_DIR / "baseline_age_2026_results.csv"
SUMMARY_PATH = OUT_DIR / "baseline_age_2026_summary.csv"


def _age_value(row: pd.Series, col: str) -> float | None:
    if col not in row.index or pd.isna(row[col]):
        return None
    value = float(row[col])
    return value if np.isfinite(value) else None


def _run_exact_age(
    red: pd.Series,
    blue: pd.Series,
    seeds: np.ndarray,
    *,
    red_age: float | None,
    blue_age: float | None,
) -> dict[str, float]:
    counts = {
        "red_ko": 0, "red_sub": 0, "red_dec": 0,
        "blue_ko": 0, "blue_sub": 0, "blue_dec": 0,
    }
    for seed in seeds:
        path = full.StaticFSRMCFullFightV1(
            red,
            blue,
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
    args = ap.parse_args()

    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

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
    if PROGRESS_PATH.exists():
        completed = pd.read_csv(PROGRESS_PATH, dtype={"bout_id": str})
        if "bout_id" not in completed.columns:
            raise RuntimeError(f"{PROGRESS_PATH} missing bout_id")
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
        "BASELINE stored prefight FSR | no trajectory | AGE ON | saves after every fight"
    )
    print(
        "age config: config/fsr_age_modifiers.yaml | enabled+calibrated: "
        + (", ".join(enabled_age_traits) if enabled_age_traits else "NONE")
    )

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

        # BASELINE + AGE: stored prefight profiles are untouched here. The
        # simulator creates separate fight-night effective copies from YAML.
        probs = _run_exact_age(
            red,
            blue,
            allbut._per_bout_seeds(args.seed, bid, args.paths),
            red_age=red_age,
            blue_age=blue_age,
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
        allbut._atomic_csv(all_done, PROGRESS_PATH)

        overall_done = len(
            set(all_done["bout_id"].astype(str)) & set(eligible["bout_id"].astype(str))
        )
        if completed_this_run % args.checkpoint_every == 0:
            ck = OUT_DIR / f"checkpoint_{overall_done:04d}.csv"
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

    final = pd.read_csv(PROGRESS_PATH, dtype={"bout_id": str})
    final = final[final["bout_id"].isin(set(eligible["bout_id"].astype(str)))].copy()
    final["event_date"] = pd.to_datetime(final["event_date"], errors="raise")
    final = final.sort_values(["event_date", "bout_id"]).reset_index(drop=True)
    allbut._atomic_csv(final, FINAL_PATH)

    summary = allbut._build_summary(final)
    summary["age_enabled_traits"] = "|".join(enabled_age_traits)
    summary["age_config"] = str(age_modifiers.DEFAULT_CONFIG_PATH)
    summary["missing_red_age"] = int(final["red_age"].isna().sum())
    summary["missing_blue_age"] = int(final["blue_age"].isna().sum())
    allbut._atomic_csv(summary, SUMMARY_PATH)

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
        f"enabled traits={s.age_enabled_traits or 'NONE'}"
    )
    print(f"wrote: {FINAL_PATH}")
    print(f"wrote: {SUMMARY_PATH}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
