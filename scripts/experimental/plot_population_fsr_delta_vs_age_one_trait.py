"""Plot one FSR trait's next-snapshot change versus fighter age.

Exploratory shadow diagnostic only. No FSR rebuilding occurs.

For each fighter, consecutive existing FSR-32 prefight snapshots are paired:

    delta = next_prefight_FSR - current_prefight_FSR

Because stored FSR changes only when fight evidence is applied, the next prefight
snapshot contains the state after the current fight (assuming no intervening
state mutation). Thus this measures the observed one-fight rating update as a
function of fighter age at the current fight.

Default trait: knockdown_resistance

Outputs:
- one PNG with population age-bin mean/median delta and linear/quadratic/cubic fits
- one CSV with age-bin summaries
- one CSV with chronological-holdout fit metrics
- one CSV with the fighter-fight transition rows used in the audit
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
OUTPUT_DIR = Path("data/experimental/population_fsr_delta_vs_age")
MASTER_PATH = age_study.MASTER_PATH


def _load_population(trait: str) -> pd.DataFrame:
    print(f"[population-delta-age] loading FSR-32: {fsr32.OUTPUT_PATH}", flush=True)
    fsr = pd.read_parquet(fsr32.OUTPUT_PATH).copy()
    if trait not in fsr.columns:
        raise RuntimeError(f"trait not found in FSR-32: {trait}")

    required = {"fight_id", "fighter_id", "fighter_name", "date", "prior_ufc_fights", trait}
    missing = sorted(required - set(fsr.columns))
    if missing:
        raise RuntimeError(f"FSR-32 missing required columns: {missing}")

    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["date"] = pd.to_datetime(fsr["date"], errors="coerce")
    fsr[trait] = pd.to_numeric(fsr[trait], errors="coerce")

    print(f"[population-delta-age] loading master ages: {MASTER_PATH}", flush=True)
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
    ages = ages.dropna(subset=["age"]).drop_duplicates(["fight_id", "fighter_id"], keep="last")

    work = fsr[["fight_id", "fighter_id", "fighter_name", "date", "prior_ufc_fights", trait]].merge(
        ages,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    work = work.dropna(subset=[trait, "age", "date"]).copy()
    work = work.loc[work["age"].between(18.0, 45.0)].copy()

    # Pair each fighter snapshot with that fighter's next stored snapshot.
    work = work.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    grouped = work.groupby("fighter_id", sort=False)
    work["next_fight_id"] = grouped["fight_id"].shift(-1)
    work["next_date"] = grouped["date"].shift(-1)
    work["next_trait"] = grouped[trait].shift(-1)
    work["next_prior_ufc_fights"] = grouped["prior_ufc_fights"].shift(-1)

    transitions = work.dropna(subset=["next_fight_id", "next_date", "next_trait"]).copy()
    transitions = transitions.loc[transitions["next_date"] > transitions["date"]].copy()
    transitions["delta_fsr"] = transitions["next_trait"].astype(float) - transitions[trait].astype(float)
    transitions["days_to_next_fight"] = (
        transitions["next_date"] - transitions["date"]
    ).dt.total_seconds() / 86400.0
    transitions["age_year"] = np.floor(transitions["age"]).astype(int)

    print(
        f"[population-delta-age] transitions={len(transitions):,} | "
        f"fighters={transitions['fighter_id'].nunique():,} | "
        f"age={transitions['age'].min():.1f}-{transitions['age'].max():.1f} | "
        f"mean delta={transitions['delta_fsr'].mean():+.3f}",
        flush=True,
    )
    return transitions.sort_values(["date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _fit_metrics(work: pd.DataFrame) -> pd.DataFrame:
    # Chronological 80/20 split. Degree selection is based on future-date holdout RMSE.
    dates = np.sort(work["date"].dropna().unique())
    split_index = max(1, int(len(dates) * 0.80)) - 1
    split_date = pd.Timestamp(dates[split_index])
    train = work.loc[work["date"] <= split_date].copy()
    test = work.loc[work["date"] > split_date].copy()
    if test.empty:
        raise RuntimeError("chronological holdout is empty")

    x_train = train["age"].to_numpy(float)
    y_train = train["delta_fsr"].to_numpy(float)
    x_test = test["age"].to_numpy(float)
    y_test = test["delta_fsr"].to_numpy(float)

    rows: list[dict[str, object]] = []
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
            "test_bias": float(np.mean(pred_test - y_test)),
        })
    return pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)


def _age_summary(work: pd.DataFrame) -> pd.DataFrame:
    out = (
        work.groupby("age_year", as_index=False)
        .agg(
            n=("delta_fsr", "size"),
            fighters=("fighter_id", "nunique"),
            mean_delta=("delta_fsr", "mean"),
            median_delta=("delta_fsr", "median"),
            std_delta=("delta_fsr", "std"),
            mean_current_fsr=(DEFAULT_TRAIT, "mean") if DEFAULT_TRAIT in work.columns else ("delta_fsr", "size"),
        )
    )
    # Keep the same support threshold used in the level plot.
    return out.loc[out["n"] >= 25].reset_index(drop=True)


def _plot(work: pd.DataFrame, summary: pd.DataFrame, metrics: pd.DataFrame, trait: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        summary["age_year"],
        summary["mean_delta"],
        marker="o",
        linewidth=2,
        label="population mean change",
    )
    ax.plot(
        summary["age_year"],
        summary["median_delta"],
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="population median change",
    )

    x = work["age"].to_numpy(float)
    y = work["delta_fsr"].to_numpy(float)
    x_grid = np.linspace(summary["age_year"].min(), summary["age_year"].max(), 300)
    for degree, label in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
        model = np.poly1d(np.polyfit(x, y, degree))
        rmse = float(metrics.loc[metrics["degree"].eq(degree), "test_rmse"].iloc[0])
        ax.plot(x_grid, model(x_grid), linewidth=1.6, label=f"{label} | holdout RMSE {rmse:.2f}")

    ax.axhline(0.0, linewidth=1.3, linestyle=":", label="no change")

    best = metrics.iloc[0]
    ax.set_title(
        f"Population Next-Fight FSR Change vs Age — {trait.replace('_', ' ').title()}\n"
        f"Best chronological holdout fit: {best['label']} (RMSE {best['test_rmse']:.2f})"
    )
    ax.set_xlabel("Fighter age at current fight")
    ax.set_ylabel("Next prefight FSR − current prefight FSR")
    ax.grid(alpha=0.25)
    ax.legend()

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

    # Build the summary explicitly here so it works for any requested trait.
    summary = (
        work.groupby("age_year", as_index=False)
        .agg(
            n=("delta_fsr", "size"),
            fighters=("fighter_id", "nunique"),
            mean_delta=("delta_fsr", "mean"),
            median_delta=("delta_fsr", "median"),
            std_delta=("delta_fsr", "std"),
            mean_current_fsr=(trait, "mean"),
        )
    )
    summary = summary.loc[summary["n"] >= 25].reset_index(drop=True)
    metrics = _fit_metrics(work)

    safe = trait.replace("/", "_")
    transition_path = OUTPUT_DIR / f"{safe}_transitions.csv"
    summary_path = OUTPUT_DIR / f"{safe}_delta_age_summary.csv"
    metrics_path = OUTPUT_DIR / f"{safe}_delta_age_fit_metrics.csv"
    plot_path = OUTPUT_DIR / f"{safe}_population_delta_vs_age.png"

    keep_transition = [
        "fighter_id", "fighter_name", "fight_id", "date", "next_fight_id", "next_date",
        "age", "age_year", "prior_ufc_fights", trait, "next_trait", "delta_fsr",
        "days_to_next_fight",
    ]
    work[keep_transition].to_csv(transition_path, index=False)
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    _plot(work, summary, metrics, trait, plot_path)

    print("\nFIT COMPARISON — chronological holdout", flush=True)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print("\nAGE-BIN NEXT-FIGHT FSR CHANGE", flush=True)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:+.3f}"), flush=True)
    print(f"\nwrote: {plot_path}", flush=True)
    print(f"wrote: {summary_path}", flush=True)
    print(f"wrote: {metrics_path}", flush=True)
    print(f"wrote: {transition_path}", flush=True)
    print("No FSR replay was performed.", flush=True)


if __name__ == "__main__":
    main()
