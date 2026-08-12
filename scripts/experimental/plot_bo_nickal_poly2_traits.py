"""Plot Bo Nickal's FSR traits with the exact include-target-prefight poly2 forecast.

This reads the completed 34-fight include-target-prefight audit to discover the
Nickal/Daukaus target fight and Bo Nickal fighter ID, then recreates the degree-2
fit used by the MC diagnostic for every canonical FSR trait.

Dots include the target-fight PREFIGHT FSR snapshot. The X marker is the N+1
poly2 extrapolation used as estimated fight-night FSR (before 10-90 clipping).
Research only.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import run_34fight_poly2_fsr_mc_test as experiment

AUDIT_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "fighter_trait_forecasts.csv"
)
OUT_DIR = Path("data/experimental/fsr_polynomial_plots/bo_nickal_daukaus")
TRAITS_PER_PAGE = 8
DEGREE = 2


def main() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"missing {AUDIT_PATH}; run run_34fight_poly2_include_target_prefight_mc_test.py first"
        )

    audit = pd.read_csv(AUDIT_PATH)
    bo = audit.loc[audit["fighter_name"].eq("Bo Nickal")].copy()
    if bo.empty:
        raise RuntimeError("Bo Nickal not found in include-target-prefight forecast audit")

    # The frozen card contains one Bo Nickal bout, so this resolves the exact target.
    target_ids = bo["target_fight_id"].astype(str).unique()
    fighter_ids = bo["fighter_id"].astype(str).unique()
    if len(target_ids) != 1 or len(fighter_ids) != 1:
        raise RuntimeError(
            f"expected one Bo Nickal target/fighter ID, got targets={target_ids}, fighters={fighter_ids}"
        )
    target_fight_id = str(target_ids[0])
    fighter_id = str(fighter_ids[0])
    target_date = pd.to_datetime(bo["target_date"], errors="raise").iloc[0]

    fsr = pd.read_parquet(fsr32.OUTPUT_PATH).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="raise")

    hist = fsr.loc[
        fsr["fighter_id"].eq(fighter_id)
        & (fsr["date"].lt(target_date) | fsr["fight_id"].eq(target_fight_id))
    ].copy()
    hist = hist.sort_values(["date", "fight_id"]).reset_index(drop=True)
    if hist.loc[hist["fight_id"].eq(target_fight_id)].shape[0] != 1:
        raise RuntimeError("target prefight FSR row not found exactly once")

    results = []
    for trait in canonical.CANONICAL_RATINGS:
        vals = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 3:
            continue
        x = np.arange(1, len(vals) + 1, dtype=float)
        coeff = np.polyfit(x, vals, DEGREE)
        poly = np.poly1d(coeff)
        next_x = float(len(vals) + 1)
        raw_next = float(poly(next_x))
        mc_next = float(np.clip(raw_next, experiment.FSR_MIN, experiment.FSR_MAX))
        dense_x = np.linspace(1.0, next_x, 300)
        results.append({
            "trait": trait,
            "x": x,
            "y": vals,
            "poly": poly,
            "dense_x": dense_x,
            "dense_y": poly(dense_x),
            "next_x": next_x,
            "raw_next": raw_next,
            "mc_next": mc_next,
            "latest": float(vals[-1]),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_pages = math.ceil(len(results) / TRAITS_PER_PAGE)
    outputs = []

    for page_idx in range(total_pages):
        page = results[page_idx * TRAITS_PER_PAGE : (page_idx + 1) * TRAITS_PER_PAGE]
        ncols = 2
        nrows = math.ceil(len(page) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.7 * nrows), squeeze=False)
        axes = axes.ravel()

        for ax, r in zip(axes, page):
            ax.scatter(r["x"], r["y"], s=38, zorder=3, label="Prefight FSR")
            ax.plot(r["dense_x"], r["dense_y"], linewidth=2, label="Degree-2 fit")
            ax.scatter([r["next_x"]], [r["raw_next"]], marker="X", s=85, zorder=4, label="Predicted N+1")
            ax.axvline(r["next_x"], linestyle="--", linewidth=0.8, alpha=0.35)
            ax.set_title(r["trait"].replace("_", " "))
            ax.set_xlabel("Prefight FSR sequence index")
            ax.set_ylabel("FSR")
            ax.grid(alpha=0.2)
            txt = (
                f"target prefight {r['latest']:.2f}\n"
                f"raw N+1 {r['raw_next']:.2f}\n"
                f"MC input {r['mc_next']:.2f}\n"
                f"delta {r['mc_next'] - r['latest']:+.2f}"
            )
            ax.text(
                0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "alpha": 0.15},
            )

        for ax in axes[len(page):]:
            ax.axis("off")

        fig.suptitle(
            f"Bo Nickal vs Kyle Daukaus — poly2 fight-night FSR forecast — page {page_idx+1}/{total_pages}\n"
            "dots include target-fight PREFIGHT state | X = extrapolated N+1 used by MC",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = OUT_DIR / f"bo_nickal_vs_kyle_daukaus_poly2_traits_page{page_idx+1}.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)

    summary = pd.DataFrame([
        {
            "trait": r["trait"],
            "target_prefight_fsr": r["latest"],
            "raw_poly2_nplus1": r["raw_next"],
            "mc_input": r["mc_next"],
            "delta": r["mc_next"] - r["latest"],
        }
        for r in results
    ]).sort_values("delta", key=lambda s: s.abs(), ascending=False)

    print("\nBO NICKAL — POLY2 FSR FORECAST SUMMARY")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSaved charts:")
    for out in outputs:
        print(f"  {out}")
        print(f"  code {out}")


if __name__ == "__main__":
    main()
