"""Compare Kline global-poly2 fight-night FSR forecasts with last-3 linear forecasts.

No Monte Carlo is run.

For each fighter/trait in the completed frozen 34-fight Kline diagnostic:
- stored target-fight prefight FSR is the anchor/current state
- global forecast is read from the existing include-target-prefight poly2 audit
- last-3 forecast fits a simple line to the final three prefight FSR observations,
  including the target-fight prefight row, then extrapolates one sequence step
- both forecasts are clipped only to the same valid FSR range [10, 90]

Outputs are research-only and do not modify stored FSR or simulator settings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32


GLOBAL_AUDIT_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "fighter_trait_forecasts.csv"
)
OUTPUT_DIR = Path(
    "data/experimental/trajectory_compare_global_poly2_vs_last3"
)
DETAIL_PATH = OUTPUT_DIR / "global_poly2_vs_last3_fsr.csv"
TRAIT_SUMMARY_PATH = OUTPUT_DIR / "trait_summary.csv"
FIGHTER_FIGHT_SUMMARY_PATH = OUTPUT_DIR / "fighter_fight_summary.csv"

FSR_MIN = 10.0
FSR_MAX = 90.0


def _fit_last3_next(values: np.ndarray) -> float:
    """Fit a line to the final 3 sequence points and extrapolate one step."""
    if len(values) < 3:
        raise ValueError("last-3 linear fit requires at least 3 observations")
    recent = values[-3:].astype(float)
    x = np.array([1.0, 2.0, 3.0])
    coeff = np.polyfit(x, recent, deg=1)
    return float(np.poly1d(coeff)(4.0))


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def main() -> None:
    if not GLOBAL_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing completed global-poly2 audit: {GLOBAL_AUDIT_PATH}"
        )

    print(f"[compare] loading global Kline audit: {GLOBAL_AUDIT_PATH}", flush=True)
    audit = pd.read_csv(GLOBAL_AUDIT_PATH).copy()

    required = {
        "target_fight_id",
        "target_date",
        "fighter_id",
        "fighter_name",
        "trait",
        "aligned_latest_fsr",
        "mc_fsr",
    }
    missing = sorted(required - set(audit.columns))
    if missing:
        raise RuntimeError(f"global audit missing required columns: {missing}")

    audit["target_fight_id"] = audit["target_fight_id"].astype(str)
    audit["fighter_id"] = audit["fighter_id"].astype(str)
    audit["target_date"] = pd.to_datetime(audit["target_date"], errors="raise")

    fsr_path = fsr32.OUTPUT_PATH
    print(f"[compare] loading prefight FSR history: {fsr_path}", flush=True)
    fsr = pd.read_parquet(fsr_path).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="raise")
    fsr = fsr.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    grouped = audit.groupby(
        ["target_fight_id", "fighter_id", "fighter_name"],
        sort=False,
    )
    total = grouped.ngroups

    for i, ((fight_id, fighter_id, fighter_name), grp) in enumerate(grouped, start=1):
        target_date = pd.Timestamp(grp["target_date"].iloc[0])

        hist = fsr.loc[
            fsr["fighter_id"].eq(fighter_id)
            & (
                fsr["date"].lt(target_date)
                | fsr["fight_id"].eq(fight_id)
            )
        ].copy()
        hist = hist.sort_values(["date", "fight_id"]).reset_index(drop=True)

        target_rows = hist.loc[hist["fight_id"].eq(fight_id)]
        if len(target_rows) != 1:
            raise RuntimeError(
                f"{fighter_name}: expected one target prefight row for {fight_id}, "
                f"found {len(target_rows)}"
            )

        for _, r in grp.iterrows():
            trait = str(r["trait"])
            stored = float(r["aligned_latest_fsr"])
            global_mc = float(r["mc_fsr"])
            global_delta = global_mc - stored

            vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)

            if len(vals) >= 3 and np.isfinite(vals).all():
                last3_raw = _fit_last3_next(vals)
                last3_mc = float(np.clip(last3_raw, FSR_MIN, FSR_MAX))
                method = "last3_linear"
                clipped = int(last3_mc != last3_raw)
            else:
                last3_raw = stored
                last3_mc = stored
                method = "latest_fallback"
                clipped = 0

            last3_delta = last3_mc - stored
            global_sign = _sign(global_delta)
            last3_sign = _sign(last3_delta)
            direction_agrees = int(global_sign == last3_sign)
            direction_opposes = int(
                global_sign != 0 and last3_sign != 0 and global_sign != last3_sign
            )

            rows.append({
                "target_fight_id": fight_id,
                "target_date": target_date,
                "fighter_id": fighter_id,
                "fighter_name": fighter_name,
                "trait": trait,
                "history_n": int(len(vals)),
                "stored_prefight_fsr": stored,
                "global_poly2_fsr": global_mc,
                "global_poly2_delta": global_delta,
                "last3_raw_fsr": float(last3_raw),
                "last3_fsr": last3_mc,
                "last3_delta": last3_delta,
                "global_minus_last3_fsr": global_mc - last3_mc,
                "abs_global_minus_last3_fsr": abs(global_mc - last3_mc),
                "global_direction": global_sign,
                "last3_direction": last3_sign,
                "direction_agrees": direction_agrees,
                "direction_opposes": direction_opposes,
                "last3_method": method,
                "last3_clipped": clipped,
            })

        if i % 10 == 0 or i == total:
            print(f"[compare] processed {i}/{total} fighter-fights", flush=True)

    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DETAIL_PATH, index=False)

    trait_summary = (
        out.groupby("trait", as_index=False)
        .agg(
            n=("trait", "size"),
            mean_abs_global_delta=("global_poly2_delta", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_last3_delta=("last3_delta", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_global_vs_last3=("abs_global_minus_last3_fsr", "mean"),
            median_abs_global_vs_last3=("abs_global_minus_last3_fsr", "median"),
            opposite_direction_rate=("direction_opposes", "mean"),
            same_direction_rate=("direction_agrees", "mean"),
        )
        .sort_values("mean_abs_global_vs_last3", ascending=False)
    )
    trait_summary.to_csv(TRAIT_SUMMARY_PATH, index=False)

    fighter_fight_summary = (
        out.groupby(
            ["target_fight_id", "target_date", "fighter_id", "fighter_name"],
            as_index=False,
        )
        .agg(
            traits=("trait", "size"),
            mean_abs_global_delta=("global_poly2_delta", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_last3_delta=("last3_delta", lambda s: float(np.mean(np.abs(s)))),
            mean_abs_global_vs_last3=("abs_global_minus_last3_fsr", "mean"),
            max_abs_global_vs_last3=("abs_global_minus_last3_fsr", "max"),
            opposite_direction_traits=("direction_opposes", "sum"),
        )
        .sort_values("mean_abs_global_vs_last3", ascending=False)
    )
    fighter_fight_summary.to_csv(FIGHTER_FIGHT_SUMMARY_PATH, index=False)

    n = len(out)
    opposite = int(out["direction_opposes"].sum())
    same = int(out["direction_agrees"].sum())
    last3_clips = int(out["last3_clipped"].sum())

    print()
    print("=" * 120)
    print("GLOBAL POLY2 KLINE vs LAST-3 LINEAR FSR COMPARISON — NO MC")
    print("=" * 120)
    print(f"trait forecasts compared: {n}")
    print(
        f"global mean |delta from stored|: {out['global_poly2_delta'].abs().mean():.3f} | "
        f"median: {out['global_poly2_delta'].abs().median():.3f}"
    )
    print(
        f"last3  mean |delta from stored|: {out['last3_delta'].abs().mean():.3f} | "
        f"median: {out['last3_delta'].abs().median():.3f}"
    )
    print(
        f"global vs last3 mean |FSR difference|: "
        f"{out['abs_global_minus_last3_fsr'].mean():.3f} | "
        f"median: {out['abs_global_minus_last3_fsr'].median():.3f} | "
        f"max: {out['abs_global_minus_last3_fsr'].max():.3f}"
    )
    print(
        f"direction same: {same}/{n} = {same/n:.1%} | "
        f"opposite: {opposite}/{n} = {opposite/n:.1%}"
    )
    print(f"last3 clips to [10,90]: {last3_clips}")

    print("\nLARGEST GLOBAL vs LAST-3 DISAGREEMENTS")
    cols = [
        "fighter_name",
        "trait",
        "stored_prefight_fsr",
        "global_poly2_fsr",
        "global_poly2_delta",
        "last3_fsr",
        "last3_delta",
        "global_minus_last3_fsr",
        "direction_opposes",
    ]
    print(
        out.nlargest(30, "abs_global_minus_last3_fsr")[cols]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    print("\nTRAITS WITH MOST GLOBAL/LAST-3 DIVERGENCE")
    print(
        trait_summary.head(25).to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )

    print(f"\nwrote: {DETAIL_PATH}")
    print(f"wrote: {TRAIT_SUMMARY_PATH}")
    print(f"wrote: {FIGHTER_FIGHT_SUMMARY_PATH}")
    print("No Monte Carlo was run.")


if __name__ == "__main__":
    main()
