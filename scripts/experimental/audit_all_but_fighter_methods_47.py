"""Audit per-fighter ALL-BUT win-method probabilities across the current 47-fight test set.

Cohort
------
- frozen 34-fight ALL-BUT validation set
- 2026-06-06 mature replay card (8 fights)
- 2026-05-30 mature replay card (5 fights)

For every fight this reruns ONLY the ALL-BUT profile and records the six exact outcomes:
red KO/TKO, red SUB, red DEC, blue KO/TKO, blue SUB, blue DEC.

Scoring
-------
1. exact outcome: highest of the six fighter+method probabilities vs actual winner+method
2. method only: highest aggregate KO/TKO/SUB/DEC probability vs actual method
3. actual-winner method: highest method probability for the actual winner vs actual method

Outputs
-------
data/experimental/all_but_fighter_methods_47/
  all_but_fighter_methods_47.csv
  all_but_fighter_methods_47_misses.csv
  all_but_fighter_methods_47_summary.csv
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
from scripts.experimental import run_next_event_base_all_allbut_poly2 as replay
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern

FROZEN_34 = Path(
    "data/experimental/validation_poly2_all_but_initial_fsr_mc/"
    "fsr_mc_card_validation_all_but_initial_poly2_v1.csv"
)
EVENT_8 = Path(
    "data/experimental/next_event_base_all_allbut_poly2/event_2026-06-06_comparison.csv"
)
EVENT_5 = Path(
    "data/experimental/next_event_base_all_allbut_poly2/event_2026-05-30_comparison.csv"
)
OUT_DIR = Path("data/experimental/all_but_fighter_methods_47")


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _norm_method(value: object) -> str:
    text = str(value).strip().lower()
    if "sub" in text:
        return "SUB"
    if "ko" in text or "tko" in text:
        return "KO/TKO"
    if "dec" in text:
        return "DEC"
    return "OTHER"


def _load_test_set() -> pd.DataFrame:
    for p in (FROZEN_34, EVENT_8, EVENT_5):
        _require(p)

    a = pd.read_csv(FROZEN_34)
    b = pd.read_csv(EVENT_8)
    c = pd.read_csv(EVENT_5)

    def shape(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
        required = {"bout_id", "red", "blue", "actual_winner", "event_date"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f"{cohort} missing columns: {missing}")
        return pd.DataFrame({
            "cohort": cohort,
            "bout_id": df["bout_id"].astype(str),
            "event_date": pd.to_datetime(df["event_date"], errors="raise").dt.normalize(),
            "red": df["red"].astype(str),
            "blue": df["blue"].astype(str),
            "actual_winner": df["actual_winner"].astype(str),
        })

    out = pd.concat([
        shape(a, "frozen_34"),
        shape(b, "event_2026-06-06"),
        shape(c, "event_2026-05-30"),
    ], ignore_index=True)
    if len(out) != 47:
        raise RuntimeError(f"expected 47 fights, found {len(out)}")
    if out["bout_id"].duplicated().any():
        dupes = out.loc[out["bout_id"].duplicated(keep=False), "bout_id"].tolist()
        raise RuntimeError(f"duplicate bout_ids: {dupes}")
    return out


def _actual_methods() -> dict[str, tuple[str, str]]:
    master = modern._load_master(modern.MASTER_PATH)
    result: dict[str, tuple[str, str]] = {}
    for _, row in master.iterrows():
        bid = str(row["fight_id"])
        raw = str(row["method"])
        result[bid] = (raw, _norm_method(raw))
    return result


def _run_exact(red: pd.Series, blue: pd.Series, seeds: np.ndarray) -> dict[str, float]:
    counts = {
        "red_ko": 0, "red_sub": 0, "red_dec": 0,
        "blue_ko": 0, "blue_sub": 0, "blue_dec": 0,
    }
    for seed in seeds:
        path = full.StaticFSRMCFullFightV1(red, blue, rounds=3, seed=int(seed)).run()
        side = "red" if int(path.winner) == 0 else "blue"
        method = {"KO/TKO": "ko", "SUB": "sub", "DEC": "dec"}[path.method]
        counts[f"{side}_{method}"] += 1
    n = float(len(seeds))
    return {k: v / n for k, v in counts.items()}


def _best_method(prefix: str, probs: dict[str, float]) -> tuple[str, float]:
    choices = {
        "KO/TKO": probs[f"{prefix}_ko"],
        "SUB": probs[f"{prefix}_sub"],
        "DEC": probs[f"{prefix}_dec"],
    }
    method = max(choices, key=choices.get)
    return method, float(choices[method])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    started = time.perf_counter()
    tests = _load_test_set()
    actual_method_map = _actual_methods()
    fsr = replay.poly._prepare_fsr_history()
    cohort, pairs = cohort32.build_aligned_cohort()
    pair_ids = set(map(str, pairs.keys()))

    missing_pairs = sorted(set(tests["bout_id"]) - pair_ids)
    if missing_pairs:
        raise RuntimeError(f"47-fight set has bouts missing from aligned cohort: {missing_pairs}")

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    rows: list[dict[str, object]] = []
    print("=" * 132)
    print("ALL-BUT PER-FIGHTER METHOD AUDIT — 47 FIGHTS")
    print("=" * 132)
    print(f"paths={args.paths} | ALL-BUT only | target prefight included | degree=2 | N+1 | no age")

    for i, test in tests.iterrows():
        bid = str(test["bout_id"])
        date = pd.Timestamp(test["event_date"]).normalize()
        red, blue = pairs[bid]
        red_name = base._display_name(red)
        blue_name = base._display_name(blue)

        red_ab, _ = replay._forecast(red, fsr, bid, date, True)
        blue_ab, _ = replay._forecast(blue, fsr, bid, date, True)
        probs = _run_exact(red_ab, blue_ab, seeds)

        actual_raw, actual_method = actual_method_map.get(bid, ("MISSING", "OTHER"))
        actual_winner = str(test["actual_winner"])
        actual_side = "red" if actual_winner == red_name else "blue" if actual_winner == blue_name else "unknown"

        six = {
            f"{red_name} KO/TKO": probs["red_ko"],
            f"{red_name} SUB": probs["red_sub"],
            f"{red_name} DEC": probs["red_dec"],
            f"{blue_name} KO/TKO": probs["blue_ko"],
            f"{blue_name} SUB": probs["blue_sub"],
            f"{blue_name} DEC": probs["blue_dec"],
        }
        predicted_exact = max(six, key=six.get)
        actual_exact = f"{actual_winner} {actual_method}" if actual_method != "OTHER" else f"{actual_winner} OTHER"

        aggregate = {
            "KO/TKO": probs["red_ko"] + probs["blue_ko"],
            "SUB": probs["red_sub"] + probs["blue_sub"],
            "DEC": probs["red_dec"] + probs["blue_dec"],
        }
        predicted_method = max(aggregate, key=aggregate.get)

        if actual_side in {"red", "blue"}:
            winner_method, winner_method_p = _best_method(actual_side, probs)
            actual_winner_win_p = sum(probs[f"{actual_side}_{m}"] for m in ("ko", "sub", "dec"))
        else:
            winner_method, winner_method_p, actual_winner_win_p = "UNKNOWN", np.nan, np.nan

        exact_correct = int(actual_method != "OTHER" and predicted_exact == actual_exact)
        method_correct = int(actual_method != "OTHER" and predicted_method == actual_method)
        winner_method_correct = int(actual_method != "OTHER" and winner_method == actual_method)

        row = {
            "cohort": test["cohort"], "event_date": date, "bout_id": bid,
            "red": red_name, "blue": blue_name, "actual_winner": actual_winner,
            "actual_method_raw": actual_raw, "actual_method": actual_method,
            "red_p_ko": probs["red_ko"], "red_p_sub": probs["red_sub"], "red_p_dec": probs["red_dec"],
            "blue_p_ko": probs["blue_ko"], "blue_p_sub": probs["blue_sub"], "blue_p_dec": probs["blue_dec"],
            "red_p_win": probs["red_ko"] + probs["red_sub"] + probs["red_dec"],
            "blue_p_win": probs["blue_ko"] + probs["blue_sub"] + probs["blue_dec"],
            "p_ko": aggregate["KO/TKO"], "p_sub": aggregate["SUB"], "p_dec": aggregate["DEC"],
            "predicted_exact": predicted_exact, "predicted_exact_p": six[predicted_exact],
            "actual_exact": actual_exact, "exact_correct": exact_correct,
            "predicted_method": predicted_method, "predicted_method_p": aggregate[predicted_method],
            "method_correct": method_correct,
            "actual_winner_top_method": winner_method,
            "actual_winner_top_method_joint_p": winner_method_p,
            "actual_winner_win_p": actual_winner_win_p,
            "actual_winner_top_method_conditional_p": winner_method_p / actual_winner_win_p if actual_winner_win_p else np.nan,
            "actual_winner_method_correct": winner_method_correct,
        }
        rows.append(row)

        mark = "✓" if exact_correct else "✗"
        print(
            f"[{i+1:02d}/47] {red_name} vs {blue_name} | actual={actual_exact} | "
            f"top6={predicted_exact} {six[predicted_exact]:.1%} {mark} | "
            f"R(KO/SUB/DEC)={probs['red_ko']:.1%}/{probs['red_sub']:.1%}/{probs['red_dec']:.1%} | "
            f"B={probs['blue_ko']:.1%}/{probs['blue_sub']:.1%}/{probs['blue_dec']:.1%}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    supported = out.loc[out["actual_method"].isin(["KO/TKO", "SUB", "DEC"])].copy()
    misses = supported.loc[supported["exact_correct"].eq(0)].copy()

    summary = pd.DataFrame([{
        "fights": len(out),
        "method_scored_fights": len(supported),
        "exact_fighter_method_correct": int(supported["exact_correct"].sum()),
        "exact_fighter_method_accuracy": float(supported["exact_correct"].mean()) if len(supported) else np.nan,
        "exact_fighter_method_misses": int((supported["exact_correct"] == 0).sum()),
        "method_only_correct": int(supported["method_correct"].sum()),
        "method_only_accuracy": float(supported["method_correct"].mean()) if len(supported) else np.nan,
        "actual_winner_method_correct": int(supported["actual_winner_method_correct"].sum()),
        "actual_winner_method_accuracy": float(supported["actual_winner_method_correct"].mean()) if len(supported) else np.nan,
        "actual_ko_tko": int((supported["actual_method"] == "KO/TKO").sum()),
        "actual_sub": int((supported["actual_method"] == "SUB").sum()),
        "actual_dec": int((supported["actual_method"] == "DEC").sum()),
        "predicted_method_ko_tko": int((supported["predicted_method"] == "KO/TKO").sum()),
        "predicted_method_sub": int((supported["predicted_method"] == "SUB").sum()),
        "predicted_method_dec": int((supported["predicted_method"] == "DEC").sum()),
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "all_but_fighter_methods_47.csv"
    miss_path = OUT_DIR / "all_but_fighter_methods_47_misses.csv"
    summary_path = OUT_DIR / "all_but_fighter_methods_47_summary.csv"
    out.to_csv(out_path, index=False)
    misses.to_csv(miss_path, index=False)
    summary.to_csv(summary_path, index=False)

    s = summary.iloc[0]
    print("\nSUMMARY")
    print(f"method-scored fights: {int(s.method_scored_fights)}/{len(out)}")
    print(f"EXACT fighter+method: {int(s.exact_fighter_method_correct)}/{int(s.method_scored_fights)} = {s.exact_fighter_method_accuracy:.1%} | misses={int(s.exact_fighter_method_misses)}")
    print(f"METHOD only:          {int(s.method_only_correct)}/{int(s.method_scored_fights)} = {s.method_only_accuracy:.1%}")
    print(f"ACTUAL WINNER method: {int(s.actual_winner_method_correct)}/{int(s.method_scored_fights)} = {s.actual_winner_method_accuracy:.1%}")
    print(f"actual mix: KO={int(s.actual_ko_tko)} SUB={int(s.actual_sub)} DEC={int(s.actual_dec)}")
    print(f"pred method mix: KO={int(s.predicted_method_ko_tko)} SUB={int(s.predicted_method_sub)} DEC={int(s.predicted_method_dec)}")
    print(f"wrote: {out_path}")
    print(f"wrote: {miss_path}")
    print(f"wrote: {summary_path}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
