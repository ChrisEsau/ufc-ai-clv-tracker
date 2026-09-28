"""Plot one FSR trait versus fighter age across the historical population.

Exploratory shadow diagnostic only. No FSR rebuilding occurs.

Default trait: knockdown_resistance

Outputs:
- one PNG with population age-bin means/medians and linear/quadratic/cubic curves
- one CSV with age-bin summaries
- one CSV with chronological-holdout fit metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as age_study


DEFAULT_TRAIT = "knockdown_resistance"
OUTPUT_DIR = Path("data/experimental/population_fsr_vs_age")
MASTER_PATH = age_study.MASTER_PATH


def _load_population(trait: str) -> pd.DataFrame:
    print(f"[population-age] loading FSR-32: {fsr32.OUTPUT_PATH}", flush=True)
    fsr = pd.read_parquet(fsr32.OUTPUT_PATH).copy()
    if trait not in fsr.columns:
        raise RuntimeError(f"trait not found in FSR-32: {trait}")
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="coerce")

    print(f"[population-age] loading master: {MASTER_PATH}", flush=True)
    master = pd.read_parquet(MASTER_PATH).copy()
    date_col = age_study.modern._resolve_date_column(master)
    master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
    master = master.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    master["fight_id"] = master["fight_id"].astype(str)
    master["r_id"] = master["r_id"].astype(str)
    master["b_id"] = master["b_id"].astype(str)
    master["r_age_calc"] = age_study._resolve_corner_age(master, "r")
    master["b_age_calc"] = age_study._resolve_corner_age(master, "b")

    red = master[["fight_id", "event_date", "r_id", "r_age_calc"]].rename(
        columns={"r_id": "fighter_id", "r_age_calc": "age"}
    )
    blue = master[["fight_id", "event_date", "b_id", "b_age_calc"]].rename(
        columns={"b_id": "fighter_id", "b_age_calc": "age"}
    )
    ages = pd.concat([red, blue], ignore_index=True)
    ages = ages.dropna(subset=["age"]).drop_duplicates(["fight_id", "fighter_id"], keep="last")

    work = fsr[["fight_id", "fighter_id", "date", "fighter_name", "prior_ufc_fights", trait]].merge(
        ages[["fight_id", "fighter_id", "age"]],
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    work[trait] = pd.to_numeric(work[trait], errors="coerce")
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work = work.dropna(subset=[trait, "age", "date"]).copy()

    # Keep plausible UFC adult ages and avoid tiny extreme-age tails dominating fits.
    work = work.loc[work["age"].between(18.0, 45.0)].copy()
    work["age_year"] = np.floor(work["age"]).astype(int)
    print(
        f"[population-age] fighter-fight rows={len(work):,} | fighters={work['fighter_id'].nunique():,} | "
        f"age={work['age'].min():.1f}-{work['age'].max():.1f}",
        flush=True,
    )
    return work.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _fit_metrics(work: pd.DataFrame, trait: str) -> pd.DataFrame:
    # Chronological 80/20 split. Degree selection is based on future-date holdout RMSE.
    dates = np.sort(work["date"].dropna().unique())
    split_date = pd.Timestamp(dates[max(1, int(len(dates) * 0.80)) - 1])
    train = work.loc[work["date"] <= split_date].copy()
    test = work.loc[work["date"] > split_date].copy()
    if test.empty:
        raise RuntimeError("chronological holdout is empty")

    x_train = train["age"].to_numpy(float)
    y_train = train[trait].to_numpy(float)
    x_test = test["age"].to_numpy(float)
    y_test = test[trait].to_numpy(float)

    rows = []
    for degree in (1, 2, 3):
        coeff = np.polyfit(x_train, y_train, degree)
        model = np.poly1d(coeff)
        pred_train = model(x_train)
        pred_test = model(x_test)
        rows.append({
            "degree": degree,
            "label": {1: "linear", 2: "quadratic", 3: "cubic"}[degree],
            "split_date": split_date,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_rmse": float(np.sqrt(np.mean((y_train - pred_train) ** 2))),
            "test_rmse": float(np.sqrt(np.mean((y_test - pred_test) ** 2))),
            "test_mae": float(np.mean(np.abs(y_test - pred_test))),
        })
    return pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)


def _age_summary(work: pd.DataFrame, trait: str) -> pd.DataFrame:
    out = (
        work.groupby("age_year", as_index=False)
        .agg(
            n=(trait, "size"),
            fighters=("fighter_id", "nunique"),
            mean_fsr=(trait, "mean"),
            median_fsr=(trait, "median"),
            std_fsr=(trait, "std"),
        )
    )
    # Plot only bins with reasonable population support.
    return out.loc[out["n"] >= 25].reset_index(drop=True)


def _plot(work: pd.DataFrame, summary: pd.DataFrame, metrics: pd.DataFrame, trait: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(summary["age_year"], summary["mean_fsr"], marker="o", linewidth=2, label="population mean")
    ax.plot(summary["age_year"], summary["median_fsr"], marker="s", linewidth=1.5, linestyle="--", label="population median")

    x = work["age"].to_numpy(float)
    y = work[trait].to_numpy(float)
    x_grid = np.linspace(summary["age_year"].min(), summary["age_year"].max(), 300)
    for degree, label in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
        model = np.poly1d(np.polyfit(x, y, degree))
        rmse = float(metrics.loc[metrics["degree"].eq(degree), "test_rmse"].iloc[0])
        ax.plot(x_grid, model(x_grid), linewidth=1.6, label=f"{label} | holdout RMSE {rmse:.2f}")

    best = metrics.iloc[0]
    ax.set_title(
        f"Population FSR vs Age — {trait.replace('_', ' ').title()}\n"
        f"Best chronological holdout fit: {best['label']} (RMSE {best['test_rmse']:.2f})"
    )
    ax.set_xlabel("Fighter age at fight")
    ax.set_ylabel("Pre-fight FSR")
    ax.grid(alpha=0.25)
    ax.legend()

    # Sample-size labels under each empirical mean point.
    y_min, y_max = ax.get_ylim()
    label_y = y_min + 0.03 * (y_max - y_min)
    for row in summary.itertuples(index=False):
        ax.text(row.age_year, label_y, f"n={row.n}", ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trait", default=DEFAULT_TRAIT)
    args = p.parse_args()
    trait = str(args.trait)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work = _load_population(trait)
    summary = _age_summary(work, trait)
    metrics = _fit_metrics(work, trait)

    safe = trait.replace("/", "_")
    summary_path = OUTPUT_DIR / f"{safe}_age_summary.csv"
    metrics_path = OUTPUT_DIR / f"{safe}_age_fit_metrics.csv"
    plot_path = OUTPUT_DIR / f"{safe}_population_vs_age.png"

    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    _plot(work, summary, metrics, trait, plot_path)

    print("\nFIT COMPARISON — chronological holdout", flush=True)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print("\nAGE-BIN POPULATION SUMMARY", flush=True)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    print(f"\nwrote: {plot_path}", flush=True)
    print(f"wrote: {summary_path}", flush=True)
    print(f"wrote: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
