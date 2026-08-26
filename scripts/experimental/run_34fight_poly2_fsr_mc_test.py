"""Run the frozen 34-fight card with raw degree-2 fighter-specific FSR forecasts.

Diagnostic only. This intentionally tests the aggressive Kline-style idea:
for every fighter/trait, fit a quadratic to all FSR snapshots strictly before
the target fight, extrapolate exactly one UFC-fight sequence step, and use that
as the target-fight FSR input to the Monte Carlo.

Locks
-----
- exact frozen 34 bout IDs from the V1 validation baseline
- all 25 canonical learned FSR traits
- degree-2 np.polyfit on fight sequence x = 1..N
- target fight EXCLUDED from the fit (leakage safe for the target)
- raw forecast, no shrinkage
- raw forecast retained in audit; MC input clipped only to valid FSR [10, 90]
- if fewer than 3 prior snapshots exist, keep the aligned latest prefight FSR
- no age modifier: red_age/blue_age are not supplied
- same 1,000-path default and deterministic seed stream as frozen validation

Outputs
-------
data/experimental/validation_poly2_fsr_mc/
  fighter_trait_forecasts.csv
  fsr_mc_card_validation_poly2_v1.csv
  fsr_mc_card_validation_poly2_v1_summary.csv

Research only. Stored FSR and simulator configuration are not modified.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import fsr_32_historical_cohort as cohort32
from scripts.experimental import fsr_static_mc_ko_sub_decision_v1 as full
from scripts.experimental import fsr_static_mc_v0 as base

BASELINE_PATH = Path(
    "data/experimental/validation_baselines/fsr_mc_card_validation_prechange_v1.csv"
)
OUTPUT_DIR = Path("data/experimental/validation_poly2_fsr_mc")
FORECAST_AUDIT_PATH = OUTPUT_DIR / "fighter_trait_forecasts.csv"
OUTPUT_PATH = OUTPUT_DIR / "fsr_mc_card_validation_poly2_v1.csv"
SUMMARY_PATH = OUTPUT_DIR / "fsr_mc_card_validation_poly2_v1_summary.csv"
DEFAULT_PATHS = 1000
DEFAULT_SEED = 20260811
FSR_MIN = 10.0
FSR_MAX = 90.0
DEGREE = 2


def _fighter_id(profile: pd.Series) -> str:
    for key in ("fighter_id", "id"):
        if key in profile.index and pd.notna(profile[key]):
            return str(profile[key])
    raise RuntimeError("aligned fighter profile has no fighter_id")


def _fit_next_poly2(values: np.ndarray) -> float:
    """Exact Kline-style degree-2 fit over UFC observation sequence."""
    if len(values) < 3:
        raise ValueError("degree-2 fit requires at least 3 observations")
    x = np.arange(1, len(values) + 1, dtype=float)
    coeff = np.polyfit(x, values.astype(float), deg=DEGREE)
    return float(np.poly1d(coeff)(float(len(values) + 1)))


def _prepare_fsr_history() -> pd.DataFrame:
    path = fsr32.OUTPUT_PATH
    print(f"[poly2-34] loading FSR history: {path}", flush=True)
    fsr = pd.read_parquet(path).copy()
    required = {"fight_id", "fighter_id", "date", *canonical.CANONICAL_RATINGS}
    missing = sorted(required - set(fsr.columns))
    if missing:
        raise RuntimeError(f"FSR artifact missing required columns: {missing}")
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="raise")
    fsr = fsr.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    return fsr


def _forecast_profile(
    profile: pd.Series,
    fsr: pd.DataFrame,
    target_fight_id: str,
    target_date: pd.Timestamp,
    fighter_name: str,
) -> tuple[pd.Series, list[dict[str, object]]]:
    fighter_id = _fighter_id(profile)
    hist = fsr.loc[
        fsr["fighter_id"].eq(fighter_id)
        & fsr["date"].lt(target_date)
        & ~fsr["fight_id"].eq(target_fight_id)
    ].copy()
    hist = hist.sort_values(["date", "fight_id"]).reset_index(drop=True)

    predicted = profile.copy(deep=True)
    audit: list[dict[str, object]] = []

    for trait in canonical.CANONICAL_RATINGS:
        current = pd.to_numeric(pd.Series([profile.get(trait)]), errors="coerce").iloc[0]
        if pd.isna(current):
            raise RuntimeError(f"{fighter_name} profile missing numeric {trait}")
        current = float(current)

        vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
        method = "latest"
        raw = current
        mc_value = current
        clipped = 0

        if len(vals) >= 3 and np.isfinite(vals).all():
            raw = _fit_next_poly2(vals)
            if np.isfinite(raw):
                mc_value = float(np.clip(raw, FSR_MIN, FSR_MAX))
                clipped = int(mc_value != raw)
                method = "poly2"
            else:
                raw = current

        predicted[trait] = mc_value
        audit.append({
            "target_fight_id": target_fight_id,
            "target_date": target_date,
            "fighter_id": fighter_id,
            "fighter_name": fighter_name,
            "trait": trait,
            "history_n": int(len(vals)),
            "method": method,
            "aligned_latest_fsr": current,
            "raw_poly2_forecast": float(raw),
            "mc_fsr": float(mc_value),
            "raw_delta": float(raw - current),
            "mc_delta": float(mc_value - current),
            "clipped_to_fsr_range": clipped,
        })

    return predicted, audit


def _actual_probability(
    row: pd.Series,
    red_name: str,
    blue_name: str,
    p_red: float,
    p_blue: float,
) -> float:
    actual = str(row["actual_winner"])
    if actual == red_name:
        return float(p_red)
    if actual == blue_name:
        return float(p_blue)
    raise RuntimeError(f"actual winner {actual!r} does not match {red_name!r}/{blue_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.paths <= 0:
        raise ValueError("--paths must be positive")
    if not args.baseline.exists():
        raise FileNotFoundError(args.baseline)

    started = time.perf_counter()
    baseline = pd.read_csv(args.baseline)
    if len(baseline) != 34:
        raise RuntimeError(f"frozen baseline must contain 34 fights, found {len(baseline)}")
    baseline["bout_id"] = baseline["bout_id"].astype(str)
    baseline["event_date"] = pd.to_datetime(baseline["event_date"], errors="raise")

    fsr = _prepare_fsr_history()
    print("[poly2-34] building aligned historical cohort once...", flush=True)
    cohort, pairs = cohort32.build_aligned_cohort()
    cohort = cohort.copy()
    cohort["bout_id"] = cohort["bout_id"].astype(str)
    cohort_by_id = cohort.set_index("bout_id", drop=False)

    missing = [bid for bid in baseline["bout_id"] if bid not in pairs or bid not in cohort_by_id.index]
    if missing:
        raise RuntimeError(f"baseline bouts missing from aligned cohort: {missing}")

    seed_rng = np.random.default_rng(args.seed)
    seeds = seed_rng.integers(1, np.iinfo(np.int32).max, size=args.paths, dtype=np.int64)

    rows: list[dict[str, object]] = []
    forecast_rows: list[dict[str, object]] = []
    total = len(baseline)

    for index, old in baseline.iterrows():
        bout_started = time.perf_counter()
        bout_id = str(old["bout_id"])
        target_date = pd.Timestamp(old["event_date"])
        red, blue = pairs[bout_id]
        red_name = base._display_name(red)
        blue_name = base._display_name(blue)

        if red_name != str(old["red"]) or blue_name != str(old["blue"]):
            raise RuntimeError(
                f"baseline/aligned name mismatch {bout_id}: "
                f"{old['red']} vs {old['blue']} | {red_name} vs {blue_name}"
            )

        red_pred, red_audit = _forecast_profile(red, fsr, bout_id, target_date, red_name)
        blue_pred, blue_audit = _forecast_profile(blue, fsr, bout_id, target_date, blue_name)
        forecast_rows.extend(red_audit)
        forecast_rows.extend(blue_audit)

        wins = [0, 0]
        methods = {"KO/TKO": 0, "SUB": 0, "DEC": 0}
        for seed in seeds:
            # Deliberately omit red_age/blue_age: this test is trajectory only.
            sim = full.StaticFSRMCFullFightV1(
                red_pred,
                blue_pred,
                rounds=3,
                seed=int(seed),
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

        bout_forecasts = red_audit + blue_audit
        poly2_count = sum(r["method"] == "poly2" for r in bout_forecasts)
        clip_count = sum(int(r["clipped_to_fsr_range"]) for r in bout_forecasts)
        mean_abs_delta = float(np.mean([abs(float(r["mc_delta"])) for r in bout_forecasts]))
        max_abs_delta = float(np.max([abs(float(r["mc_delta"])) for r in bout_forecasts]))

        rows.append({
            "baseline_version": old["baseline_version"],
            "card_no": old["card_no"],
            "event_date": target_date,
            "event_name": old["event_name"],
            "bout_id": bout_id,
            "red": red_name,
            "blue": blue_name,
            "paths": args.paths,
            "poly2_trait_predictions": poly2_count,
            "clipped_trait_predictions": clip_count,
            "mean_abs_trait_delta": mean_abs_delta,
            "max_abs_trait_delta": max_abs_delta,
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
            f"[poly2-34] {index + 1:02d}/{total} {red_name} vs {blue_name} | "
            f"old {float(old['p_red_win']):.1%}/{float(old['p_blue_win']):.1%} -> "
            f"poly2 {p_red:.1%}/{p_blue:.1%} | actual Δ {new_actual_p - old_actual_p:+.1%} | "
            f"traits={poly2_count}/50 clips={clip_count} mean|ΔFSR|={mean_abs_delta:.2f} | "
            f"{elapsed:.1f}s bout | {total_elapsed:.1f}s total",
            flush=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    audit = pd.DataFrame(forecast_rows)
    out.to_csv(args.output, index=False)
    audit.to_csv(FORECAST_AUDIT_PATH, index=False)

    old_correct = int(out["old_mc_correct"].sum())
    new_correct = int(out["new_mc_correct"].sum())
    flips = out.loc[out["favorite_flipped"].eq(1)].copy()
    corrected = int(((out["old_mc_correct"] == 0) & (out["new_mc_correct"] == 1)).sum())
    broken = int(((out["old_mc_correct"] == 1) & (out["new_mc_correct"] == 0)).sum())

    summary = pd.DataFrame([{
        "bouts": len(out),
        "paths_per_bout": args.paths,
        "old_correct": old_correct,
        "poly2_correct": new_correct,
        "old_accuracy": old_correct / len(out),
        "poly2_accuracy": new_correct / len(out),
        "accuracy_delta": (new_correct - old_correct) / len(out),
        "favorite_flips": int(len(flips)),
        "baseline_misses_corrected": corrected,
        "baseline_correct_fights_broken": broken,
        "mean_delta_actual_winner_probability": float(out["delta_actual_winner_probability"].mean()),
        "median_delta_actual_winner_probability": float(out["delta_actual_winner_probability"].median()),
        "fights_moving_toward_actual_winner": int((out["delta_actual_winner_probability"] > 0).sum()),
        "fights_moving_away_from_actual_winner": int((out["delta_actual_winner_probability"] < 0).sum()),
        "mean_abs_trait_delta": float(audit["mc_delta"].abs().mean()),
        "median_abs_trait_delta": float(audit["mc_delta"].abs().median()),
        "max_abs_trait_delta": float(audit["mc_delta"].abs().max()),
        "poly2_trait_predictions": int(audit["method"].eq("poly2").sum()),
        "latest_fallback_trait_predictions": int(audit["method"].eq("latest").sum()),
        "clipped_trait_predictions": int(audit["clipped_to_fsr_range"].sum()),
        "old_mean_p_ko_tko": float(out["old_p_ko_tko"].mean()),
        "poly2_mean_p_ko_tko": float(out["new_p_ko_tko"].mean()),
        "old_mean_p_sub": float(out["old_p_sub"].mean()),
        "poly2_mean_p_sub": float(out["new_p_sub"].mean()),
        "old_mean_p_dec": float(out["old_p_dec"].mean()),
        "poly2_mean_p_dec": float(out["new_p_dec"].mean()),
    }])
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 120)
    print("34-FIGHT RAW POLY2 FSR -> MC DIAGNOSTIC")
    print("=" * 120)
    print(f"old winner accuracy:   {old_correct}/34 = {old_correct/34:.1%}")
    print(f"poly2 winner accuracy: {new_correct}/34 = {new_correct/34:.1%}")
    print(f"baseline misses corrected: {corrected}")
    print(f"baseline correct fights broken: {broken}")
    print(f"favorite flips: {len(flips)}")
    print(
        f"actual-winner probability mean Δ={out['delta_actual_winner_probability'].mean():+.2%} | "
        f"median Δ={out['delta_actual_winner_probability'].median():+.2%}"
    )
    print(
        f"direction: toward actual={(out['delta_actual_winner_probability'] > 0).sum()} | "
        f"away={(out['delta_actual_winner_probability'] < 0).sum()} | "
        f"unchanged={(out['delta_actual_winner_probability'] == 0).sum()}"
    )
    print(
        f"trait forecasts: poly2={audit['method'].eq('poly2').sum()} | "
        f"fallback latest={audit['method'].eq('latest').sum()} | "
        f"clipped={int(audit['clipped_to_fsr_range'].sum())}"
    )
    print(
        f"trait |ΔFSR| mean={audit['mc_delta'].abs().mean():.2f} | "
        f"median={audit['mc_delta'].abs().median():.2f} | "
        f"max={audit['mc_delta'].abs().max():.2f}"
    )
    print(
        f"methods: KO {out['old_p_ko_tko'].mean():.1%}->{out['new_p_ko_tko'].mean():.1%} | "
        f"SUB {out['old_p_sub'].mean():.1%}->{out['new_p_sub'].mean():.1%} | "
        f"DEC {out['old_p_dec'].mean():.1%}->{out['new_p_dec'].mean():.1%}"
    )

    if not flips.empty:
        print("\nFAVORITE FLIPS")
        print(flips[[
            "red", "blue", "old_mc_favorite", "new_mc_favorite", "actual_winner",
            "old_mc_correct", "new_mc_correct", "delta_actual_winner_probability",
            "mean_abs_trait_delta", "clipped_trait_predictions",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    largest = out.reindex(out["delta_actual_winner_probability"].abs().sort_values(ascending=False).index).head(10)
    print("\nLARGEST MC MOVES")
    print(largest[[
        "red", "blue", "actual_winner", "old_p_red_win", "new_p_red_win",
        "delta_actual_winner_probability", "old_mc_correct", "new_mc_correct",
        "mean_abs_trait_delta", "max_abs_trait_delta", "clipped_trait_predictions",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nwrote: {args.output}")
    print(f"wrote: {FORECAST_AUDIT_PATH}")
    print(f"wrote: {SUMMARY_PATH}")
    print(f"elapsed: {time.perf_counter() - started:.1f}s")
    print("Research only. No stored FSR, age modifier, or simulator configuration was changed.")


if __name__ == "__main__":
    main()
