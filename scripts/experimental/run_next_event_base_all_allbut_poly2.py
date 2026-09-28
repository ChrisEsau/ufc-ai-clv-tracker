"""Replay the next mature event after a cutoff with BASE FSR vs two Kline poly2 variants.

Variants
--------
BASE
    Stored target-fight prefight FSR, unchanged.
ALL
    All prefight FSR observations through the target-fight prefight row, degree-2,
    extrapolate one fight-sequence point (N+1).
ALL_BUT
    Same as ALL, but drop the fighter's first chronological prefight FSR point
    before fitting. Fewer than 3 usable observations falls back to stored FSR.

The target-fight prefight row is included for both trajectory variants by project
contract. No age modifier is used. All three variants use the same deterministic
seed stream per bout.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base
from scripts.experimental import run_34fight_poly2_fsr_mc_test as poly

DEFAULT_CUTOFF = "2026-07-18"
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
OUTPUT_DIR = Path("data/experimental/next_event_base_all_allbut_poly2")


def _first_present(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def _event_date_col(df: pd.DataFrame) -> str:
    col = _first_present(df.columns, ("event_date", "date", "fight_date"))
    if col is None:
        raise RuntimeError(f"could not find event-date column in cohort: {list(df.columns)}")
    return col


def _event_name(row: pd.Series) -> str:
    for col in ("event_name", "event", "event_title"):
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return "UNKNOWN EVENT"


def _actual_winner_name(row: pd.Series, red_name: str, blue_name: str) -> str:
    for col in ("actual_winner", "winner_name", "winner"):
        if col in row.index and pd.notna(row[col]):
            value = str(row[col])
            if value == red_name:
                return red_name
            if value == blue_name:
                return blue_name
            low = value.strip().lower()
            if low in {"red", "r"}:
                return red_name
            if low in {"blue", "b"}:
                return blue_name
    for col in ("winner_side", "winner_corner"):
        if col in row.index and pd.notna(row[col]):
            low = str(row[col]).strip().lower()
            if low in {"red", "r"}:
                return red_name
            if low in {"blue", "b"}:
                return blue_name
    for col in ("y", "target", "red_win", "label"):
        if col in row.index and pd.notna(row[col]):
            try:
                return red_name if int(float(row[col])) == 1 else blue_name
            except (TypeError, ValueError):
                pass
    raise RuntimeError(f"could not resolve actual winner for {red_name} vs {blue_name}")


def _forecast(profile: pd.Series, fsr: pd.DataFrame, bout_id: str, target_date: pd.Timestamp, drop_initial: bool):
    fighter_id = poly._fighter_id(profile)
    hist = fsr.loc[
        fsr["fighter_id"].eq(fighter_id)
        & (fsr["date"].lt(target_date) | fsr["fight_id"].eq(str(bout_id)))
    ].sort_values(["date", "fight_id"]).reset_index(drop=True)
    target_rows = hist.loc[hist["fight_id"].eq(str(bout_id))]
    if len(target_rows) != 1:
        raise RuntimeError(f"{fighter_id}: expected one target prefight row for {bout_id}, found {len(target_rows)}")

    predicted = profile.copy(deep=True)
    audit = []
    for trait in canonical.CANONICAL_RATINGS:
        current = float(pd.to_numeric(pd.Series([profile[trait]]), errors="raise").iloc[0])
        vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
        if drop_initial:
            vals = vals[1:]
        raw = current
        mc_value = current
        method = "latest"
        clipped = 0
        if len(vals) >= 3 and np.isfinite(vals).all():
            raw = poly._fit_next_poly2(vals)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, poly.FSR_MIN, poly.FSR_MAX))
                clipped = int(mc_value != raw)
                method = "poly2"
        predicted[trait] = mc_value
        audit.append({
            "bout_id": str(bout_id), "fighter_id": fighter_id, "trait": trait,
            "drop_initial": int(drop_initial), "fit_n": int(len(vals)),
            "method": method, "stored_fsr": current, "raw_forecast": float(raw),
            "mc_fsr": mc_value, "mc_delta": mc_value - current, "clipped": clipped,
        })
    return predicted, audit


def _run_pair(red, blue, seeds):
    wins = [0, 0]
    methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
    for seed in seeds:
        path = full.StaticFSRMCFullFightV1(red, blue, rounds=3, seed=int(seed)).run()
        wins[int(path.winner)] += 1
        methods[path.method] += 1
    n = float(len(seeds))
    return {
        "p_red_win": wins[0] / n, "p_blue_win": wins[1] / n,
        "p_ko_tko": methods["KO/TKO"] / n, "p_sub": methods["SUB"] / n,
        "p_dec": methods["DEC"] / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default=DEFAULT_CUTOFF, help="Find first mature event strictly after YYYY-MM-DD")
    ap.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    started = time.perf_counter()
    fsr = poly._prepare_fsr_history()
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    dcol = _event_date_col(cohort)
    cohort["_event_date"] = pd.to_datetime(cohort[dcol], errors="raise").dt.normalize()
    cutoff = pd.Timestamp(args.after).normalize()
    future = cohort.loc[cohort["_event_date"].gt(cutoff)].copy()
    if future.empty:
        raise RuntimeError(f"no mature cohort event found after {cutoff.date()}")
    next_date = future["_event_date"].min()
    card = future.loc[future["_event_date"].eq(next_date)].sort_values("bout_id").reset_index(drop=True)

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    print("\n" + "=" * 132)
    print("NEXT-EVENT REPLAY — BASE FSR vs ALL POLY2 vs ALL-BUT-INITIAL POLY2")
    print("=" * 132)
    print(f"cutoff: {cutoff.date()} | selected event date: {next_date.date()} | mature bouts: {len(card)}")
    print("target prefight included for ALL/ALL-BUT | degree=2 | N+1 | no age | same seeds")

    rows = []
    audits = []
    for i, bout in card.iterrows():
        bout_id = str(bout["bout_id"])
        red, blue = pairs[bout_id]
        red_name = base._display_name(red)
        blue_name = base._display_name(blue)
        actual = _actual_winner_name(bout, red_name, blue_name)

        red_all, a1 = _forecast(red, fsr, bout_id, next_date, False)
        blue_all, a2 = _forecast(blue, fsr, bout_id, next_date, False)
        red_ab, a3 = _forecast(red, fsr, bout_id, next_date, True)
        blue_ab, a4 = _forecast(blue, fsr, bout_id, next_date, True)
        for r in a1 + a2:
            r["variant"] = "ALL"
        for r in a3 + a4:
            r["variant"] = "ALL_BUT"
        audits.extend(a1 + a2 + a3 + a4)

        res_base = _run_pair(red, blue, seeds)
        res_all = _run_pair(red_all, blue_all, seeds)
        res_ab = _run_pair(red_ab, blue_ab, seeds)

        def correct(res):
            fav = red_name if res["p_red_win"] >= res["p_blue_win"] else blue_name
            return fav, int(fav == actual)
        fav_b, cor_b = correct(res_base)
        fav_a, cor_a = correct(res_all)
        fav_x, cor_x = correct(res_ab)

        rows.append({
            "event_date": next_date, "event_name": _event_name(bout), "bout_id": bout_id,
            "red": red_name, "blue": blue_name, "actual_winner": actual,
            "base_p_red": res_base["p_red_win"], "all_p_red": res_all["p_red_win"], "all_but_p_red": res_ab["p_red_win"],
            "base_favorite": fav_b, "all_favorite": fav_a, "all_but_favorite": fav_x,
            "base_correct": cor_b, "all_correct": cor_a, "all_but_correct": cor_x,
            "base_ko": res_base["p_ko_tko"], "all_ko": res_all["p_ko_tko"], "all_but_ko": res_ab["p_ko_tko"],
            "base_sub": res_base["p_sub"], "all_sub": res_all["p_sub"], "all_but_sub": res_ab["p_sub"],
            "base_dec": res_base["p_dec"], "all_dec": res_all["p_dec"], "all_but_dec": res_ab["p_dec"],
        })
        print(
            f"[{i+1:02d}/{len(card):02d}] {red_name} vs {blue_name} | actual={actual} | "
            f"BASE {res_base['p_red_win']:.1%}/{res_base['p_blue_win']:.1%} {'✓' if cor_b else '✗'} | "
            f"ALL {res_all['p_red_win']:.1%}/{res_all['p_blue_win']:.1%} {'✓' if cor_a else '✗'} | "
            f"ALL-BUT {res_ab['p_red_win']:.1%}/{res_ab['p_blue_win']:.1%} {'✓' if cor_x else '✗'}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"event_{next_date.date()}_comparison.csv"
    audit_path = OUTPUT_DIR / f"event_{next_date.date()}_trait_forecasts.csv"
    out.to_csv(out_path, index=False)
    audit.to_csv(audit_path, index=False)

    print("\nSUMMARY")
    print(f"BASE:    {int(out.base_correct.sum())}/{len(out)} = {out.base_correct.mean():.1%}")
    print(f"ALL:     {int(out.all_correct.sum())}/{len(out)} = {out.all_correct.mean():.1%}")
    print(f"ALL-BUT: {int(out.all_but_correct.sum())}/{len(out)} = {out.all_but_correct.mean():.1%}")
    print(f"wrote: {out_path}")
    print(f"wrote: {audit_path}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
