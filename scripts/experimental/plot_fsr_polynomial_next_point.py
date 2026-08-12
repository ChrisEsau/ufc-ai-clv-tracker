"""Plot a fighter FSR trait with polynomial fit and one-step-ahead prediction.

Example:
    PYTHONPATH=. python scripts/experimental/plot_fsr_polynomial_next_point.py \
      --fighter-id 745fa7b605f8e2da \
      --target-fight-id 52ddf20a10890b41 \
      --trait striking_power \
      --degree 2

The target fight row is excluded from the fit. The script uses all strictly
prior prefight snapshots for the fighter, indexed by fight sequence, fits a
polynomial, and predicts the next sequence point. The target row is displayed
only as an optional holdout reference marker, never used in the fit.

Shadow/research only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/experimental/fsr_polynomial_plots")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fighter-id", required=True)
    parser.add_argument("--target-fight-id", required=True)
    parser.add_argument("--trait", default="striking_power")
    parser.add_argument("--degree", type=int, default=2, choices=[2, 3])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print("[poly-plot] loading FSR-32...", flush=True)
    fsr = pd.read_parquet(FSR_PATH).copy()
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="raise")

    fighter = fsr.loc[fsr["fighter_id"].eq(str(args.fighter_id))].copy()
    if fighter.empty:
        raise RuntimeError(f"fighter_id {args.fighter_id} not found")
    if args.trait not in fighter.columns:
        raise RuntimeError(f"trait {args.trait!r} not found in FSR-32")

    fighter = fighter.sort_values(["date", "fight_id"]).reset_index(drop=True)
    target_rows = fighter.loc[fighter["fight_id"].eq(str(args.target_fight_id))]
    if len(target_rows) != 1:
        raise RuntimeError(
            f"expected exactly one target row for {args.target_fight_id}, found {len(target_rows)}"
        )
    target = target_rows.iloc[0]
    target_date = pd.Timestamp(target["date"])

    # Strictly prior snapshots only; target is a holdout reference.
    hist = fighter.loc[fighter["date"] < target_date, ["fight_id", "date", args.trait]].copy()
    hist[args.trait] = pd.to_numeric(hist[args.trait], errors="coerce")
    hist = hist.dropna(subset=[args.trait]).sort_values(["date", "fight_id"]).reset_index(drop=True)

    min_points = args.degree + 1
    if len(hist) < min_points:
        raise RuntimeError(
            f"degree {args.degree} requires at least {min_points} prior points; found {len(hist)}"
        )

    x = np.arange(1, len(hist) + 1, dtype=float)
    y = hist[args.trait].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, deg=args.degree)
    poly = np.poly1d(coeff)
    next_x = float(len(hist) + 1)
    predicted = float(poly(next_x))

    dense_x = np.linspace(1.0, next_x, 300)
    dense_y = poly(dense_x)
    target_actual = float(pd.to_numeric(pd.Series([target[args.trait]]), errors="coerce").iloc[0])

    fighter_name = (
        str(target.get("fighter_name"))
        if pd.notna(target.get("fighter_name"))
        else str(args.fighter_id)
    )

    print(f"[poly-plot] fighter: {fighter_name}", flush=True)
    print(f"[poly-plot] trait: {args.trait}", flush=True)
    print(f"[poly-plot] degree: {args.degree}", flush=True)
    print(f"[poly-plot] prior points used: {len(hist)}", flush=True)
    print("\nDATA USED IN FIT")
    display = hist.copy()
    display.insert(0, "fight_index", np.arange(1, len(hist) + 1))
    print(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPREDICTED NEXT POINT (fight index {int(next_x)}): {predicted:.3f}")
    print(f"ACTUAL TARGET PREFIGHT FSR (holdout, not fit): {target_actual:.3f}")
    print(f"ERROR vs holdout: {predicted - target_actual:+.3f}")
    print(f"POLYNOMIAL COEFFICIENTS: {coeff}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=70, label="Prior FSR points used in fit", zorder=3)
    ax.plot(dense_x, dense_y, linewidth=2, label=f"Degree-{args.degree} polynomial fit")
    ax.scatter([next_x], [predicted], s=130, marker="X", label=f"Predicted next point: {predicted:.2f}", zorder=4)
    ax.scatter([next_x], [target_actual], s=90, marker="D", label=f"Actual target holdout: {target_actual:.2f}", zorder=4)

    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(
        f"forecast {predicted:.2f}",
        (next_x, predicted),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )
    ax.annotate(
        f"actual {target_actual:.2f}",
        (next_x, target_actual),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_title(f"{fighter_name} — {args.trait} — degree-{args.degree} next-point forecast")
    ax.set_xlabel("Prefight FSR sequence index")
    ax.set_ylabel("FSR rating")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (
        f"{args.fighter_id}_{args.target_fight_id}_{args.trait}_poly{args.degree}.png"
    )
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"\n[poly-plot] saved: {out}", flush=True)


if __name__ == "__main__":
    main()
