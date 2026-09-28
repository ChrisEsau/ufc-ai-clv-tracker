from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FSR_HISTORY_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)
FORECASTS_PATH = Path(
    "data/experimental/validation_poly2_include_target_prefight_mc/"
    "fighter_trait_forecasts.csv"
)
OUT_DIR = Path(
    "data/experimental/fsr_polynomial_plots/bo_cannonier_kline_charts"
)

TARGET_FIGHTS = [
    ("Bo Nickal", "Kyle Daukaus"),
    ("Jared Cannonier", "Christian Leroy Duncan"),
]
SIG_DELTA_THRESHOLD = 0.25
TRAITS_PER_PAGE = 8
N_COLS = 2


def _safe_name(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("'", "")
        .replace(".", "")
    )


def _fit_poly2(values: np.ndarray):
    x = np.arange(1, len(values) + 1, dtype=float)
    coeff = np.polyfit(x, values.astype(float), deg=2)
    poly = np.poly1d(coeff)
    next_x = float(len(values) + 1)
    predicted = float(poly(next_x))
    dense_x = np.linspace(1.0, next_x, 300)
    dense_y = poly(dense_x)
    return x, next_x, predicted, dense_x, dense_y


def _plot_page(results, fighter_name, opponent_name, page_number, total_pages, output_path):
    nrows = math.ceil(len(results) / N_COLS)
    fig, axes = plt.subplots(nrows, N_COLS, figsize=(14, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, result in zip(axes, results):
        x = result["x"]
        y = result["y"]
        next_x = result["next_x"]
        predicted = result["predicted"]
        trait = result["trait"]

        ax.scatter(x, y, s=40, zorder=3, label="Prefight FSR")
        ax.plot(result["dense_x"], result["dense_y"], linewidth=2, label="Poly2 fit")
        ax.scatter([x[-1]], [y[-1]], marker="s", s=80, zorder=4, label="Target prefight FSR")
        ax.scatter([next_x], [predicted], marker="X", s=110, zorder=5, label="Poly2 N+1 MC FSR")
        ax.axvline(next_x, linestyle="--", linewidth=0.8, alpha=0.4)

        ax.set_title(trait.replace("_", " "), fontsize=11)
        ax.set_xlabel("Prefight FSR sequence index")
        ax.set_ylabel("FSR")
        ax.grid(alpha=0.2)

        ax.text(
            0.03,
            0.97,
            (
                f"target prefight: {result['target_prefight']:.2f}\n"
                f"N+1 prediction: {predicted:.2f}\n"
                f"delta: {result['delta']:+.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "alpha": 0.15},
        )

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle(
        f"{fighter_name} vs {opponent_name}\n"
        f"Kline charts — |poly2 delta| >= {SIG_DELTA_THRESHOLD:.2f} — "
        f"page {page_number}/{total_pages}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("[kline-chart] loading FSR history...", flush=True)
    fsr = pd.read_parquet(FSR_HISTORY_PATH).copy()
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="raise")

    print("[kline-chart] loading poly2 forecast audit...", flush=True)
    forecasts = pd.read_csv(FORECASTS_PATH).copy()
    forecasts["target_fight_id"] = forecasts["target_fight_id"].astype(str)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for red_name, blue_name in TARGET_FIGHTS:
        matchup = forecasts.loc[
            forecasts["fighter_name"].isin([red_name, blue_name])
        ].copy()
        target_counts = matchup.groupby("target_fight_id")["fighter_name"].nunique()
        shared = target_counts.loc[target_counts >= 2].index.tolist()
        if not shared:
            raise RuntimeError(f"No shared target fight found for {red_name} vs {blue_name}")
        target_fight_id = str(shared[0])
        matchup = matchup.loc[matchup["target_fight_id"].eq(target_fight_id)].copy()

        print("\n" + "=" * 120)
        print(f"{red_name} vs {blue_name} | target_fight_id={target_fight_id}")
        print("=" * 120)

        fight_dir = OUT_DIR / f"{_safe_name(red_name)}_vs_{_safe_name(blue_name)}"

        for fighter_name, opponent_name in [(red_name, blue_name), (blue_name, red_name)]:
            fighter_fc = matchup.loc[matchup["fighter_name"].eq(fighter_name)].copy()
            fighter_fc = fighter_fc.loc[fighter_fc["mc_delta"].abs() >= SIG_DELTA_THRESHOLD].copy()
            fighter_fc = fighter_fc.sort_values(
                "mc_delta", key=lambda s: s.abs(), ascending=False
            )
            if fighter_fc.empty:
                print(f"[kline-chart] no significant traits for {fighter_name}")
                continue

            fighter_id = str(fighter_fc["fighter_id"].iloc[0])
            hist = fsr.loc[fsr["fighter_id"].eq(fighter_id)].copy()
            hist = hist.sort_values(["date", "fight_id"]).reset_index(drop=True)
            target_rows = hist.loc[hist["fight_id"].eq(target_fight_id)]
            if len(target_rows) != 1:
                raise RuntimeError(
                    f"{fighter_name}: expected one target prefight row for {target_fight_id}, found {len(target_rows)}"
                )
            target_date = pd.Timestamp(target_rows.iloc[0]["date"])
            hist = hist.loc[
                (hist["date"] < target_date) | hist["fight_id"].eq(target_fight_id)
            ].sort_values(["date", "fight_id"]).reset_index(drop=True)

            results = []
            for _, fc_row in fighter_fc.iterrows():
                trait = str(fc_row["trait"])
                values = pd.to_numeric(hist[trait], errors="coerce").dropna().to_numpy(dtype=float)
                if len(values) < 3:
                    continue
                x, next_x, raw_predicted, dense_x, dense_y = _fit_poly2(values)
                mc_prediction = float(fc_row["mc_fsr"])
                results.append(
                    {
                        "trait": trait,
                        "x": x,
                        "y": values,
                        "next_x": next_x,
                        "predicted": mc_prediction,
                        "raw_predicted": raw_predicted,
                        "dense_x": dense_x,
                        "dense_y": dense_y,
                        "target_prefight": float(values[-1]),
                        "delta": mc_prediction - float(values[-1]),
                    }
                )

            if not results:
                continue

            print(f"\n{fighter_name} — plotted traits={len(results)}")
            summary = pd.DataFrame(
                [
                    {
                        "trait": r["trait"],
                        "target_prefight_fsr": r["target_prefight"],
                        "poly2_n_plus_1_fsr": r["predicted"],
                        "delta": r["delta"],
                    }
                    for r in results
                ]
            ).sort_values("delta", key=lambda s: s.abs(), ascending=False)
            print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

            total_pages = math.ceil(len(results) / TRAITS_PER_PAGE)
            for page_idx in range(total_pages):
                start = page_idx * TRAITS_PER_PAGE
                page_results = results[start : start + TRAITS_PER_PAGE]
                output_path = fight_dir / f"{_safe_name(fighter_name)}_kline_page_{page_idx + 1}.png"
                _plot_page(
                    page_results,
                    fighter_name,
                    opponent_name,
                    page_idx + 1,
                    total_pages,
                    output_path,
                )
                print(f"[kline-chart] saved: {output_path}", flush=True)

    print(f"\nDone. Charts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
