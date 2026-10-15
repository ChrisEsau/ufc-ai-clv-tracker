"""Run ALL-BUT poly2 MC across available mature 2026 fights, excluding the current 47-fight audit set.

Contract
--------
- 2026 fights only.
- Only bouts present in the aligned leakage-safe FSR-32 mature cohort are eligible.
- ALL-BUT only: include target-fight prefight row, drop first chronological prefight FSR row,
  degree-2 fit, extrapolate N+1, fallback to stored prefight when <3 usable values remain.
- No age modifier.
- Same 3-round StaticFSRMCFullFightV1 engine used by the existing card/audit runners.
- Six exact fighter-method outcomes are recorded per bout.
- The already-audited 47 bout_ids are excluded by default.

Resumability
------------
- progress.csv is atomically rewritten after every completed fight.
- checkpoint_XXXX.csv is written every --checkpoint-every fights (default 20).
- Restarting the script skips bout_ids already present in progress.csv.
- Deterministic seeds are derived per bout from --seed + bout_id, so resume order does not
  change a bout's Monte Carlo result.

Outputs
-------
data/experimental/2026_all_but_full_cohort/
  progress.csv
  checkpoint_0020.csv, checkpoint_0040.csv, ...
  all_but_2026_results.csv
  all_but_2026_summary.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import audit_all_but_fighter_methods_47 as audit47
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_next_event_base_all_allbut_poly2 as replay
from scripts.experimental import run_previous_event_base_all_allbut_poly2 as previous

YEAR = 2026
OUT_DIR = Path("data/experimental/2026_all_but_full_cohort")
PROGRESS_PATH = OUT_DIR / "progress.csv"
FINAL_PATH = OUT_DIR / "all_but_2026_results.csv"
SUMMARY_PATH = OUT_DIR / "all_but_2026_summary.csv"
EXCLUDE_47_PATH = Path(
    "data/experimental/all_but_fighter_methods_47/all_but_fighter_methods_47.csv"
)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _norm_method(value: object) -> str:
    return audit47._norm_method(value)


def _load_excluded_bout_ids() -> set[str]:
    """Load the 47 already-audited bout_ids, without assuming dates."""
    if EXCLUDE_47_PATH.exists():
        df = pd.read_csv(EXCLUDE_47_PATH, dtype={"bout_id": str})
        if "bout_id" not in df.columns:
            raise RuntimeError(f"{EXCLUDE_47_PATH} missing bout_id")
        ids = set(df["bout_id"].astype(str))
        if len(ids) != 47:
            print(
                f"WARNING: exclusion file contains {len(ids)} unique bout_ids, expected 47; excluding what is present.",
                flush=True,
            )
        return ids

    # Fallback to the three source cohorts used to build the 47-fight audit.
    ids: set[str] = set()
    for path in (audit47.FROZEN_34, audit47.EVENT_8, audit47.EVENT_5):
        if path.exists():
            df = pd.read_csv(path, dtype={"bout_id": str})
            if "bout_id" in df.columns:
                ids.update(df["bout_id"].astype(str))
    if not ids:
        raise FileNotFoundError(
            "Could not resolve the 47-fight exclusion set. Run/retain the 47-fight audit outputs first."
        )
    return ids


def _master_map() -> dict[str, pd.Series]:
    master = replay.modern._load_master(replay.modern.MASTER_PATH) if hasattr(replay, "modern") else audit47.modern._load_master(audit47.modern.MASTER_PATH)
    return {str(row["fight_id"]): row for _, row in master.iterrows()}


def _actual_winner(row: pd.Series, red_name: str, blue_name: str) -> str:
    # Reuse the broader resolver already used by the previous-event runner.
    return previous._resolve_actual(row, red_name, blue_name)


def _per_bout_seeds(base_seed: int, bout_id: str, paths: int) -> np.ndarray:
    digest = hashlib.blake2b(
        f"{base_seed}|{bout_id}".encode("utf-8"), digest_size=8
    ).digest()
    seed64 = int.from_bytes(digest, "little", signed=False)
    rng = np.random.default_rng(seed64)
    return rng.integers(1, np.iinfo(np.int32).max, size=paths, dtype=np.int64)


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


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"fights": 0}])
    supported = df[df["actual_method"].isin(["KO/TKO", "SUB", "DEC"])].copy()
    row = {
        "fights": len(df),
        "method_scored_fights": len(supported),
        "winner_correct": int(df["winner_correct"].sum()),
        "winner_accuracy": float(df["winner_correct"].mean()),
        "exact_fighter_method_correct": int(supported["exact_correct"].sum()),
        "exact_fighter_method_accuracy": float(supported["exact_correct"].mean()) if len(supported) else np.nan,
        "method_only_correct": int(supported["method_correct"].sum()),
        "method_only_accuracy": float(supported["method_correct"].mean()) if len(supported) else np.nan,
        "actual_ko_tko": int((supported["actual_method"] == "KO/TKO").sum()),
        "actual_sub": int((supported["actual_method"] == "SUB").sum()),
        "actual_dec": int((supported["actual_method"] == "DEC").sum()),
        "mean_p_ko": float(df["p_ko"].mean()),
        "mean_p_sub": float(df["p_sub"].mean()),
        "mean_p_dec": float(df["p_dec"].mean()),
        "predicted_method_ko_tko": int((supported["predicted_method"] == "KO/TKO").sum()),
        "predicted_method_sub": int((supported["predicted_method"] == "SUB").sum()),
        "predicted_method_dec": int((supported["predicted_method"] == "DEC").sum()),
    }
    return pd.DataFrame([row])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--include-audited-47", action="store_true", help="Do not exclude the existing 47-fight audit set")
    ap.add_argument("--fresh", action="store_true", help="Ignore/delete existing progress and start this batch over")
    args = ap.parse_args()

    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    fsr = replay.poly._prepare_fsr_history()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    date_col = replay._event_date_col(cohort)
    cohort["_event_date"] = pd.to_datetime(cohort[date_col], errors="raise").dt.normalize()

    eligible = cohort[cohort["_event_date"].dt.year.eq(YEAR)].copy()
    excluded: set[str] = set()
    if not args.include_audited_47:
        excluded = _load_excluded_bout_ids()
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
    master_by_id = _master_map()

    print("=" * 136)
    print("2026 ALL-BUT FULL-COHORT RUNNER — EXCLUDING CURRENT 47")
    print("=" * 136)
    print(
        f"eligible={len(eligible)} | already complete={len(set(eligible['bout_id']) & completed_ids)} | "
        f"remaining={len(remaining)} | excluded existing audit ids={len(excluded)}"
    )
    print(
        f"paths={args.paths} | checkpoint every {args.checkpoint_every} fights | "
        "ALL-BUT degree=2 N+1 | target prefight included | no age | saves after every fight"
    )

    new_rows: list[dict[str, object]] = []
    completed_this_run = 0

    for j, bout in remaining.iterrows():
        bid = str(bout["bout_id"])
        date = pd.Timestamp(bout["_event_date"]).normalize()
        red, blue = pairs[bid]
        red_name = base._display_name(red)
        blue_name = base._display_name(blue)

        red_ab, _ = replay._forecast(red, fsr, bid, date, True)
        blue_ab, _ = replay._forecast(blue, fsr, bid, date, True)
        probs = _run_exact(red_ab, blue_ab, _per_bout_seeds(args.seed, bid, args.paths))

        master_row = master_by_id.get(bid)
        if master_row is None:
            raise RuntimeError(f"bout {bid} not found in master")
        actual_winner = _actual_winner(master_row, red_name, blue_name)
        actual_method_raw = str(master_row["method"])
        actual_method = _norm_method(actual_method_raw)

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
            "paths": args.paths,
            "seed_base": args.seed,
        }
        new_rows.append(row)
        completed_this_run += 1

        all_done = pd.concat([completed, pd.DataFrame(new_rows)], ignore_index=True)
        all_done = all_done.drop_duplicates(subset=["bout_id"], keep="last").sort_values(["event_date", "bout_id"])
        _atomic_csv(all_done, PROGRESS_PATH)

        overall_done = len(set(all_done["bout_id"].astype(str)) & set(eligible["bout_id"].astype(str)))
        if completed_this_run % args.checkpoint_every == 0:
            ck = OUT_DIR / f"checkpoint_{overall_done:04d}.csv"
            _atomic_csv(all_done, ck)
            print(f"  checkpoint -> {ck}", flush=True)

        mark = "✓" if row["winner_correct"] else "✗"
        print(
            f"[{overall_done:03d}/{len(eligible):03d}] {date.date()} | {red_name} vs {blue_name} | "
            f"actual={actual_winner} {actual_method} | pick={predicted_winner} {max(red_p_win, blue_p_win):.1%} {mark} | "
            f"KO/SUB/DEC={aggregate['KO/TKO']:.1%}/{aggregate['SUB']:.1%}/{aggregate['DEC']:.1%}",
            flush=True,
        )

    final = pd.read_csv(PROGRESS_PATH, dtype={"bout_id": str})
    final = final[final["bout_id"].isin(set(eligible["bout_id"].astype(str)))].copy()
    final["event_date"] = pd.to_datetime(final["event_date"], errors="raise")
    final = final.sort_values(["event_date", "bout_id"]).reset_index(drop=True)
    _atomic_csv(final, FINAL_PATH)

    summary = _build_summary(final)
    _atomic_csv(summary, SUMMARY_PATH)

    s = summary.iloc[0]
    print("\nSUMMARY")
    print(f"fights: {int(s['fights'])}")
    if int(s["fights"]):
        print(f"winner: {int(s['winner_correct'])}/{int(s['fights'])} = {s['winner_accuracy']:.1%}")
        if int(s.get("method_scored_fights", 0)):
            print(
                f"exact fighter+method: {int(s['exact_fighter_method_correct'])}/{int(s['method_scored_fights'])} "
                f"= {s['exact_fighter_method_accuracy']:.1%}"
            )
            print(
                f"method only: {int(s['method_only_correct'])}/{int(s['method_scored_fights'])} "
                f"= {s['method_only_accuracy']:.1%}"
            )
            print(
                f"actual mix: KO={int(s['actual_ko_tko'])} SUB={int(s['actual_sub'])} DEC={int(s['actual_dec'])}"
            )
            print(
                f"mean simulated mass: KO={s['mean_p_ko']:.1%} SUB={s['mean_p_sub']:.1%} DEC={s['mean_p_dec']:.1%}"
            )
    print(f"wrote: {FINAL_PATH}")
    print(f"wrote: {SUMMARY_PATH}")
    print(f"resume source: {PROGRESS_PATH}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
