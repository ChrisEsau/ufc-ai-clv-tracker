"""Plot fighter FSR traits with polynomial fits and one-step-ahead predictions.

Examples:
    # One trait
    PYTHONPATH=. python scripts/experimental/plot_fsr_polynomial_next_point.py \
      --fighter-id 745fa7b605f8e2da \
      --target-fight-id 52ddf20a10890b41 \
      --trait striking_power \
      --degree 2

    # All FSR traits in smaller paged figures
    PYTHONPATH=. python scripts/experimental/plot_fsr_polynomial_next_point.py \
      --fighter-id 745fa7b605f8e2da \
      --target-fight-id 52ddf20a10890b41 \
      --all-traits \
      --degree 2

The target fight row IS INCLUDED in every fit. The script uses all prefight
snapshots for the fighter up to and including the selected target fight,
indexed by fight sequence, fits a polynomial, and predicts one additional
sequence point beyond the target row.

For --all-traits, plots are split into pages of 8 traits so Codespaces/VS Code
can display them reliably.

Shadow/research only.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FSR_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/experimental/fsr_polynomial_plots")
TRAITS_PER_PAGE = 8

META_COLUMNS = {
    "fight_id",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "date",
    "event_id",
    "weight_class",
    "winner_id",
    "prior_ufc_fights",
}


def fsr_trait_columns(df: pd.DataFrame) -> list[str]:
    traits: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col.endswith("_updates") or col.endswith("_evidence_score"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            traits.append(col)
    return sorted(traits)


def fit_trait(
    fighter: pd.DataFrame,
    target_date: pd.Timestamp,
    trait: str,
    degree: int,
) -> dict[str, object] | None:
    hist = fighter.loc[fighter["date"] <= target_date, ["fight_id", "date", trait]].copy()
    hist[trait] = pd.to_numeric(hist[trait], errors="coerce")
    hist = hist.dropna(subset=[trait]).sort_values(["date", "fight_id"]).reset_index(drop=True)

    min_points = degree + 1
    if len(hist) < min_points:
        return None

    x = np.arange(1, len(hist) + 1, dtype=float)
    y = hist[trait].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, deg=degree)
    poly = np.poly1d(coeff)

    next_x = float(len(hist) + 1)
    predicted = float(poly(next_x))
    latest_actual = float(y[-1])
    predicted_change = predicted - latest_actual

    dense_x = np.linspace(1.0, next_x, 300)
    dense_y = poly(dense_x)

    return {
        "trait": trait,
        "hist": hist,
        "x": x,
        "y": y,
        "coeff": coeff,
        "poly": poly,
        "next_x": next_x,
        "predicted": predicted,
        "latest_actual": latest_actual,
        "predicted_change": predicted_change,
        "dense_x": dense_x,
        "dense_y": dense_y,
    }


def plot_single(result: dict[str, object], fighter_name: str, degree: int, out: Path) -> None:
    x = result["x"]
    y = result["y"]
    next_x = result["next_x"]
    predicted = result["predicted"]
    trait = str(result["trait"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=70, label="FSR points used in fit", zorder=3)
    ax.plot(result["dense_x"], result["dense_y"], linewidth=2, label=f"Degree-{degree} polynomial fit")
    ax.scatter([next_x], [predicted], s=130, marker="X", label=f"Predicted next: {predicted:.2f}", zorder=4)

    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.annotate(
        f"forecast {predicted:.2f}",
        (next_x, predicted),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=10,
    )

    ax.set_title(f"{fighter_name} — {trait} — degree-{degree} next-point forecast")
    ax.set_xlabel("Prefight FSR sequence index")
    ax.set_ylabel("FSR rating")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_page(
    results: list[dict[str, object]],
    fighter_name: str,
    degree: int,
    out: Path,
    page_number: int,
    total_pages: int,
) -> None:
    ncols = 2
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, result in zip(axes_flat, results):
        x = result["x"]
        y = result["y"]
        next_x = result["next_x"]
        predicted = result["predicted"]
        trait = str(result["trait"])

        ax.scatter(x, y, s=34, zorder=3)
        ax.plot(result["dense_x"], result["dense_y"], linewidth=2.0)
        ax.scatter([next_x], [predicted], s=75, marker="X", zorder=4)
        ax.axvline(next_x, linestyle="--", linewidth=0.8, alpha=0.35)

        ax.set_title(trait.replace("_", " "), fontsize=11)
        ax.set_xlabel("fight index", fontsize=9)
        ax.set_ylabel("FSR", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.2)

        text = (
            f"latest {result['latest_actual']:.2f}\n"
            f"next {predicted:.2f}\n"
            f"change {result['predicted_change']:+.2f}"
        )
        ax.text(
            0.03,
            0.97,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "alpha": 0.15},
        )

    for ax in axes_flat[len(results):]:
        ax.axis("off")

    fig.suptitle(
        f"{fighter_name} — FSR polynomial forecasts — page {page_number}/{total_pages}\n"
        f"degree {degree} | dots = points through target | line = fit | X = predicted next",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fighter-id", required=True)
    parser.add_argument("--target-fight-id", required=True)
    parser.add_argument("--trait", default="striking_power")
    parser.add_argument("--all-traits", action="store_true")
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

    fighter = fighter.sort_values(["date", "fight_id"]).reset_index(drop=True)
    target_rows = fighter.loc[fighter["fight_id"].eq(str(args.target_fight_id))]
    if len(target_rows) != 1:
        raise RuntimeError(
            f"expected exactly one target row for {args.target_fight_id}, found {len(target_rows)}"
        )
    target = target_rows.iloc[0]
    target_date = pd.Timestamp(target["date"])
    fighter_name = str(target.get("fighter_name")) if pd.notna(target.get("fighter_name")) else str(args.fighter_id)

    traits = fsr_trait_columns(fighter) if args.all_traits else [args.trait]
    missing = [trait for trait in traits if trait not in fighter.columns]
    if missing:
        raise RuntimeError(f"traits not found in FSR-32: {missing}")

    print(f"[poly-plot] fighter: {fighter_name}", flush=True)
    print(f"[poly-plot] target fight INCLUDED in fit: {args.target_fight_id}", flush=True)
    print(f"[poly-plot] degree: {args.degree}", flush=True)
    print(f"[poly-plot] traits requested: {len(traits)}", flush=True)

    results: list[dict[str, object]] = []
    skipped: list[str] = []
    for idx, trait in enumerate(traits, start=1):
        print(f"[poly-plot] {idx}/{len(traits)} fitting {trait}...", flush=True)
        result = fit_trait(fighter, target_date, trait, args.degree)
        if result is None:
            skipped.append(trait)
            continue
        results.append(result)

    if not results:
        raise RuntimeError(
            f"no traits had enough points through target for degree {args.degree}; need at least {args.degree + 1}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_traits:
        total_pages = math.ceil(len(results) / TRAITS_PER_PAGE)
        output_files: list[Path] = []
        for page_idx in range(total_pages):
            start = page_idx * TRAITS_PER_PAGE
            end = start + TRAITS_PER_PAGE
            page_results = results[start:end]
            out = args.output_dir / (
                f"{args.fighter_id}_{args.target_fight_id}_all_traits_poly{args.degree}_"
                f"include_target_page{page_idx + 1}.png"
            )
            plot_page(
                page_results,
                fighter_name,
                args.degree,
                out,
                page_idx + 1,
                total_pages,
            )
            output_files.append(out)
            print(f"[poly-plot] saved page {page_idx + 1}/{total_pages}: {out}", flush=True)

        rows = []
        for result in results:
            rows.append(
                {
                    "trait": result["trait"],
                    "points_used": len(result["hist"]),
                    "latest_fsr": result["latest_actual"],
                    "predicted_next": result["predicted"],
                    "predicted_change": result["predicted_change"],
                }
            )
        summary = pd.DataFrame(rows).sort_values("trait")
        print("\nALL-TRAIT NEXT-POINT SUMMARY")
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        if skipped:
            print(f"\n[poly-plot] skipped for insufficient history: {', '.join(skipped)}", flush=True)

        print("\n[poly-plot] view pages with:", flush=True)
        for out in output_files:
            print(f"code {out}", flush=True)
    else:
        result = results[0]
        hist = result["hist"].copy()
        hist.insert(0, "fight_index", np.arange(1, len(hist) + 1))
        print("\nDATA USED IN FIT (TARGET INCLUDED)")
        print(hist.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nLATEST FSR POINT: {result['latest_actual']:.3f}")
        print(f"PREDICTED NEXT POINT: {result['predicted']:.3f}")
        print(f"PREDICTED CHANGE: {result['predicted_change']:+.3f}")
        print(f"POLYNOMIAL COEFFICIENTS: {result['coeff']}", flush=True)
        out = args.output_dir / (
            f"{args.fighter_id}_{args.target_fight_id}_{args.trait}_poly{args.degree}_include_target.png"
        )
        plot_single(result, fighter_name, args.degree, out)
        print(f"\n[poly-plot] saved: {out}", flush=True)
        print(f"[poly-plot] view with: code {out}", flush=True)


if __name__ == "__main__":
    main()
