"""Population next-fight FSR drift versus age for all canonical learned traits.

Exploratory shadow diagnostic only. No FSR rebuilding occurs.

For each canonical trait, construct consecutive fighter transitions from the
existing FSR-32 prefight snapshot artifact:

    delta = next_prefight_FSR - current_prefight_FSR

The x-axis is fighter age at the current fight. Each trait compares linear,
quadratic, and cubic polynomial models using a chronological 80/20 holdout.
The selected degree is then refit on all available transitions for descriptive
population-curve summaries and plotting.

Outputs:
- all-trait summary CSV
- all-model fit metrics CSV
- long transition parquet
- paged PNG plots
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as age_study

OUTPUT_DIR = Path("data/experimental/population_fsr_delta_vs_age_all_traits")
MASTER_PATH = age_study.MASTER_PATH
MIN_AGE = 18.0
MAX_AGE = 45.0
MIN_AGE_BIN_N = 25
PLOTS_PER_PAGE = 5
AGE_PROBES = (25.0, 30.0, 35.0, 38.0)


def _load_base() -> pd.DataFrame:
    print(f"[all-trait age drift] loading FSR-32: {fsr32.OUTPUT_PATH}", flush=True)
    fsr = pd.read_parquet(fsr32.OUTPUT_PATH).copy()
    required = {
        "fight_id", "fighter_id", "fighter_name", "date", "prior_ufc_fights",
        *canonical.CANONICAL_RATINGS,
    }
    missing = sorted(required - set(fsr.columns))
    if missing:
        raise RuntimeError(f"FSR-32 missing required columns: {missing}")

    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="coerce")
    fsr = fsr.dropna(subset=["date"]).copy()

    print(f"[all-trait age drift] loading master: {MASTER_PATH}", flush=True)
    master = pd.read_parquet(MASTER_PATH).copy()
    date_col = age_study.modern._resolve_date_column(master)
    master[date_col] = pd.to_datetime(master[date_col], errors="coerce")
    master = master.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    master["fight_id"] = master["fight_id"].astype(str)
    master["r_id"] = master["r_id"].astype(str)
    master["b_id"] = master["b_id"].astype(str)
    master["r_age_calc"] = age_study._resolve_corner_age(master, "r")
    master["b_age_calc"] = age_study._resolve_corner_age(master, "b")

    red = master[["fight_id", "r_id", "r_age_calc"]].rename(
        columns={"r_id": "fighter_id", "r_age_calc": "age"}
    )
    blue = master[["fight_id", "b_id", "b_age_calc"]].rename(
        columns={"b_id": "fighter_id", "b_age_calc": "age"}
    )
    ages = pd.concat([red, blue], ignore_index=True)
    ages["age"] = pd.to_numeric(ages["age"], errors="coerce")
    ages = ages.dropna(subset=["age"]).drop_duplicates(["fight_id", "fighter_id"], keep="last")

    base = fsr.merge(
        ages,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    base = base.loc[base["age"].between(MIN_AGE, MAX_AGE)].copy()
    base = base.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)

    print(
        f"[all-trait age drift] fighter-fight rows={len(base):,} | "
        f"fighters={base['fighter_id'].nunique():,} | age={base['age'].min():.1f}-{base['age'].max():.1f}",
        flush=True,
    )
    return base


def _build_transitions(base: pd.DataFrame) -> pd.DataFrame:
    print("[all-trait age drift] constructing consecutive-fight transitions...", flush=True)
    work = base.copy()
    grouped = work.groupby("fighter_id", sort=False)
    work["next_fight_id"] = grouped["fight_id"].shift(-1)
    work["next_date"] = grouped["date"].shift(-1)
    work["days_to_next_fight"] = (work["next_date"] - work["date"]).dt.days

    rows: list[pd.DataFrame] = []
    for i, trait in enumerate(canonical.CANONICAL_RATINGS, start=1):
        current = pd.to_numeric(work[trait], errors="coerce")
        nxt = grouped[trait].shift(-1)
        temp = pd.DataFrame({
            "fighter_id": work["fighter_id"],
            "fighter_name": work["fighter_name"],
            "fight_id": work["fight_id"],
            "next_fight_id": work["next_fight_id"],
            "date": work["date"],
            "next_date": work["next_date"],
            "age": pd.to_numeric(work["age"], errors="coerce"),
            "prior_ufc_fights": pd.to_numeric(work["prior_ufc_fights"], errors="coerce"),
            "days_to_next_fight": work["days_to_next_fight"],
            "trait": trait,
            "current_fsr": current,
            "next_fsr": pd.to_numeric(nxt, errors="coerce"),
        })
        temp["delta"] = temp["next_fsr"] - temp["current_fsr"]
        temp = temp.dropna(subset=["next_fight_id", "age", "current_fsr", "next_fsr", "delta"]).copy()
        rows.append(temp)
        print(
            f"[all-trait age drift] trait {i:02d}/{len(canonical.CANONICAL_RATINGS)} "
            f"{trait}: transitions={len(temp):,}",
            flush=True,
        )

    out = pd.concat(rows, ignore_index=True)
    out["age_year"] = np.floor(out["age"]).astype(int)
    return out


def _chronological_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    dates = np.sort(frame["date"].dropna().unique())
    if len(dates) < 5:
        raise RuntimeError("not enough distinct dates for chronological holdout")
    split_index = max(1, int(len(dates) * 0.80)) - 1
    split_date = pd.Timestamp(dates[split_index])
    train = frame.loc[frame["date"] <= split_date].copy()
    test = frame.loc[frame["date"] > split_date].copy()
    if train.empty or test.empty:
        raise RuntimeError("chronological train/test split is empty")

    x_train = train["age"].to_numpy(float)
    y_train = train["delta"].to_numpy(float)
    x_test = test["age"].to_numpy(float)
    y_test = test["delta"].to_numpy(float)

    rows = []
    for degree in (1, 2, 3):
        model = np.poly1d(np.polyfit(x_train, y_train, degree))
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
    return pd.DataFrame(rows).sort_values(["test_rmse", "degree"]).reset_index(drop=True)


def _zero_crossing(model: np.poly1d, low: float = 20.0, high: float = 40.0) -> float:
    roots = np.roots(model)
    candidates = sorted(
        float(r.real)
        for r in roots
        if abs(float(r.imag)) < 1e-7 and low <= float(r.real) <= high
    )
    if not candidates:
        return float("nan")
    # Prefer the crossing closest to the center of the well-supported UFC age range.
    return min(candidates, key=lambda value: abs(value - 30.0))


def _empirical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("age_year", as_index=False)
        .agg(
            n=("delta", "size"),
            fighters=("fighter_id", "nunique"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            std_delta=("delta", "std"),
        )
    )
    return summary.loc[summary["n"] >= MIN_AGE_BIN_N].reset_index(drop=True)


def _analyze(transitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, np.poly1d]]:
    summary_rows = []
    metric_rows = []
    empirical: dict[str, pd.DataFrame] = {}
    models: dict[str, np.poly1d] = {}

    total = len(canonical.CANONICAL_RATINGS)
    for i, trait in enumerate(canonical.CANONICAL_RATINGS, start=1):
        frame = transitions.loc[transitions["trait"].eq(trait)].copy()
        metrics = _chronological_metrics(frame)
        metrics.insert(0, "trait", trait)
        metric_rows.append(metrics)

        best = metrics.iloc[0]
        degree = int(best["degree"])
        model = np.poly1d(np.polyfit(frame["age"].to_numpy(float), frame["delta"].to_numpy(float), degree))
        models[trait] = model
        empirical[trait] = _empirical_summary(frame)

        row = {
            "trait": trait,
            "transitions": len(frame),
            "fighters": frame["fighter_id"].nunique(),
            "best_degree": degree,
            "best_fit": str(best["label"]),
            "holdout_rmse": float(best["test_rmse"]),
            "holdout_mae": float(best["test_mae"]),
            "zero_crossing_age_20_40": _zero_crossing(model),
            "empirical_mean_delta": float(frame["delta"].mean()),
            "empirical_median_delta": float(frame["delta"].median()),
        }
        for age in AGE_PROBES:
            row[f"fitted_delta_age_{int(age)}"] = float(model(age))
        summary_rows.append(row)
        print(
            f"[all-trait age drift] fit {i:02d}/{total} {trait}: "
            f"best={best['label']} rmse={best['test_rmse']:.3f} "
            f"zero={row['zero_crossing_age_20_40']:.2f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    metrics_all = pd.concat(metric_rows, ignore_index=True)
    return summary, metrics_all, empirical, models


def _plot_pages(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    empirical: dict[str, pd.DataFrame],
    models: dict[str, np.poly1d],
) -> list[Path]:
    paths: list[Path] = []
    traits = list(canonical.CANONICAL_RATINGS)
    pages = math.ceil(len(traits) / PLOTS_PER_PAGE)

    for page in range(pages):
        chunk = traits[page * PLOTS_PER_PAGE:(page + 1) * PLOTS_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(12, 3.4 * len(chunk)), squeeze=False)

        for ax, trait in zip(axes.ravel(), chunk):
            emp = empirical[trait]
            model = models[trait]
            row = summary.loc[summary["trait"].eq(trait)].iloc[0]
            trait_metrics = metrics.loc[metrics["trait"].eq(trait)]

            ax.plot(emp["age_year"], emp["mean_delta"], marker="o", linewidth=1.8, label="mean")
            ax.plot(emp["age_year"], emp["median_delta"], marker="s", linestyle="--", linewidth=1.3, label="median")
            if not emp.empty:
                x_grid = np.linspace(float(emp["age_year"].min()), float(emp["age_year"].max()), 250)
                ax.plot(
                    x_grid,
                    model(x_grid),
                    linewidth=2.0,
                    label=f"selected {row['best_fit']} | holdout RMSE {row['holdout_rmse']:.2f}",
                )
            ax.axhline(0.0, linestyle=":", linewidth=1.0, label="no change")
            ax.set_title(trait.replace("_", " ").title())
            ax.set_ylabel("Next FSR − current FSR")
            ax.grid(alpha=0.25)

            if not emp.empty:
                y_min, y_max = ax.get_ylim()
                label_y = y_min + 0.03 * (y_max - y_min)
                for sample in emp.itertuples(index=False):
                    ax.text(sample.age_year, label_y, f"n={sample.n}", ha="center", va="bottom", fontsize=6, rotation=90)
            ax.legend(fontsize=7, loc="best")

        axes[-1, 0].set_xlabel("Fighter age at current fight")
        fig.suptitle(
            f"Population Next-Fight FSR Change vs Age — Canonical Traits — Page {page + 1}/{pages}\n"
            "Selected polynomial degree chosen by chronological holdout RMSE",
            fontsize=14,
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = OUTPUT_DIR / f"population_fsr_delta_vs_age_page_{page + 1}.png"
        fig.savefig(path, dpi=165, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        print(f"[all-trait age drift] wrote plot page {page + 1}/{pages}: {path}", flush=True)

    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_base()
    transitions = _build_transitions(base)

    transition_path = OUTPUT_DIR / "population_fsr_age_transitions.parquet"
    print(f"[all-trait age drift] writing {len(transitions):,} transition rows...", flush=True)
    transitions.to_parquet(transition_path, index=False)

    summary, metrics, empirical, models = _analyze(transitions)
    summary_path = OUTPUT_DIR / "population_fsr_delta_vs_age_summary.csv"
    metrics_path = OUTPUT_DIR / "population_fsr_delta_vs_age_fit_metrics.csv"
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_paths = _plot_pages(summary, metrics, empirical, models)

    display_cols = [
        "trait", "best_fit", "holdout_rmse", "zero_crossing_age_20_40",
        "fitted_delta_age_25", "fitted_delta_age_30", "fitted_delta_age_35",
        "fitted_delta_age_38", "transitions",
    ]
    print("\n" + "=" * 150)
    print("ALL CANONICAL FSR TRAITS — POPULATION NEXT-FIGHT DRIFT VS AGE")
    print("=" * 150)
    print(summary[display_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    print(f"\nwrote: {summary_path}", flush=True)
    print(f"wrote: {metrics_path}", flush=True)
    print(f"wrote: {transition_path}", flush=True)
    for path in plot_paths:
        print(f"wrote: {path}", flush=True)


if __name__ == "__main__":
    main()
