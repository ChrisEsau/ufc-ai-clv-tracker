"""Population FSR level/drift curve audit for all 25 canonical ratings.

Purpose
-------
Use the existing leakage-safe FSR-32 snapshot database to study population-level
career trajectories without rebuilding any FSR ratings.

For every canonical trait this script builds two complementary views:

1. LEVEL curves
   - y = current stored pre-fight FSR
   - x = fighter age OR prior UFC fights

2. DRIFT curves
   - y = next stored state - current stored state
   - x = fighter age OR prior UFC fights at the current fight

The next stored pre-fight snapshot is the fighter's post-current-fight stored
state because stored FSR changes only after observed UFC fight evidence and does
not drift between fights. Consecutive rows are required to increase
``prior_ufc_fights`` by exactly one.

For each trait/view/x-variable, linear (degree 1), quadratic (degree 2), and
cubic (degree 3) polynomial regressions are compared using a chronological
80/20 holdout split. Model selection is based on held-out RMSE, not in-sample
fit. The script also saves binned empirical summaries and paginated plots.

This is a research/shadow diagnostic. It does not modify FSR or simulator code.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as age_study
from scripts.experimental import fsr_static_mc_ko_tko_v2_fsr_joint_signal_cv_2020plus_mature as modern


FSR_PATH = fsr32.OUTPUT_PATH
MASTER_PATH = modern.MASTER_PATH
OUTPUT_DIR = Path("data/experimental/fsr_population_trajectory_curves")
TRANSITION_PATH = OUTPUT_DIR / "fsr_population_transitions.parquet"
FIT_PATH = OUTPUT_DIR / "fsr_population_curve_fit_summary.csv"
BIN_PATH = OUTPUT_DIR / "fsr_population_binned_curves.csv"

DEGREES = (1, 2, 3)
DEGREE_NAMES = {1: "linear", 2: "quadratic", 3: "cubic"}
HOLDOUT_FRACTION = 0.20
MIN_TRAIN_ROWS = 200
MIN_TEST_ROWS = 50
MIN_BIN_N = 20
TRAITS_PER_PAGE = 6


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def _date_col(df: pd.DataFrame) -> str:
    for col in ("date", "event_date", "fight_date"):
        if col in df.columns:
            return col
    raise RuntimeError("frame has no date/event_date/fight_date column")


def _load_age_map() -> pd.DataFrame:
    raw = pd.read_parquet(MASTER_PATH).copy()
    date_col = modern._resolve_date_column(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    raw["fight_id"] = raw["fight_id"].astype(str)
    raw["r_id"] = raw["r_id"].astype(str)
    raw["b_id"] = raw["b_id"].astype(str)
    raw = raw.sort_values(["event_date", "fight_id"]).drop_duplicates("fight_id", keep="last")
    raw["r_age"] = age_study._resolve_corner_age(raw, "r")
    raw["b_age"] = age_study._resolve_corner_age(raw, "b")

    red = raw[["fight_id", "event_date", "r_id", "r_age"]].rename(
        columns={"r_id": "fighter_id", "r_age": "age"}
    )
    blue = raw[["fight_id", "event_date", "b_id", "b_age"]].rename(
        columns={"b_id": "fighter_id", "b_age": "age"}
    )
    out = pd.concat([red, blue], ignore_index=True)
    out["fighter_id"] = out["fighter_id"].astype(str)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    if out.duplicated(["fight_id", "fighter_id"]).any():
        raise RuntimeError("master age map violates fighter-fight grain")
    return out


def _prepare_snapshots() -> pd.DataFrame:
    if not FSR_PATH.exists():
        raise RuntimeError(f"FSR-32 artifact not found: {FSR_PATH}")
    if not MASTER_PATH.exists():
        raise RuntimeError(f"master artifact not found: {MASTER_PATH}")

    fsr = pd.read_parquet(FSR_PATH).copy()
    required = {"fight_id", "fighter_id", "prior_ufc_fights", *canonical.CANONICAL_RATINGS}
    missing = sorted(required - set(fsr.columns))
    if missing:
        raise RuntimeError(f"FSR-32 artifact missing population-audit columns: {missing}")

    dcol = _date_col(fsr)
    fsr["date"] = pd.to_datetime(fsr[dcol], errors="coerce")
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    fsr["prior_ufc_fights"] = pd.to_numeric(fsr["prior_ufc_fights"], errors="coerce")
    fsr = fsr.dropna(subset=["date", "prior_ufc_fights"]).copy()
    fsr["prior_ufc_fights"] = fsr["prior_ufc_fights"].astype(int)

    ages = _load_age_map()[["fight_id", "fighter_id", "age"]]
    fsr = fsr.merge(ages, on=["fight_id", "fighter_id"], how="left", validate="one_to_one")
    fsr = fsr.sort_values(["fighter_id", "date", "fight_id"]).reset_index(drop=True)
    fsr["days_since_prev_fight"] = fsr.groupby("fighter_id")["date"].diff().dt.days
    return fsr


def _build_transition_table(fsr: pd.DataFrame) -> pd.DataFrame:
    work = fsr.copy()
    grouped = work.groupby("fighter_id", sort=False)
    work["next_fight_id"] = grouped["fight_id"].shift(-1)
    work["next_date"] = grouped["date"].shift(-1)
    work["next_prior_ufc_fights"] = grouped["prior_ufc_fights"].shift(-1)

    for trait in canonical.CANONICAL_RATINGS:
        work[f"next__{trait}"] = grouped[trait].shift(-1)
        work[f"delta__{trait}"] = pd.to_numeric(work[f"next__{trait}"], errors="coerce") - pd.to_numeric(work[trait], errors="coerce")

    # Require the next observed UFC snapshot to be exactly one fight later.
    mask = work["next_fight_id"].notna() & (
        pd.to_numeric(work["next_prior_ufc_fights"], errors="coerce")
        == work["prior_ufc_fights"] + 1
    )
    out = work.loc[mask].copy()
    out["days_to_next_fight"] = (out["next_date"] - out["date"]).dt.days

    keep = [
        "fight_id", "next_fight_id", "date", "next_date", "fighter_id",
        "fighter_name" if "fighter_name" in out.columns else "fighter_id",
        "age", "prior_ufc_fights", "days_since_prev_fight", "days_to_next_fight",
    ]
    # avoid duplicate fighter_id if no fighter_name column exists
    dedup_keep: list[str] = []
    for col in keep:
        if col not in dedup_keep:
            dedup_keep.append(col)
    for trait in canonical.CANONICAL_RATINGS:
        dedup_keep.extend([trait, f"next__{trait}", f"delta__{trait}"])
    return out[dedup_keep].reset_index(drop=True)


def _chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.array(sorted(pd.to_datetime(frame["date"].dropna().unique())))
    if len(dates) < 5:
        raise RuntimeError("not enough unique dates for chronological holdout")
    cut_index = max(1, min(len(dates) - 1, int(math.floor(len(dates) * (1.0 - HOLDOUT_FRACTION)))))
    cutoff = pd.Timestamp(dates[cut_index])
    train = frame.loc[frame["date"] < cutoff].copy()
    test = frame.loc[frame["date"] >= cutoff].copy()
    return train, test, cutoff


def _fit_poly(train_x: np.ndarray, train_y: np.ndarray, degree: int) -> np.poly1d:
    coeff = np.polyfit(train_x.astype(float), train_y.astype(float), deg=degree)
    return np.poly1d(coeff)


def _fit_one(
    frame: pd.DataFrame,
    *,
    trait: str,
    view: str,
    x_name: str,
    y_name: str,
) -> list[dict[str, object]]:
    work = frame[["date", x_name, y_name]].copy()
    work[x_name] = pd.to_numeric(work[x_name], errors="coerce")
    work[y_name] = pd.to_numeric(work[y_name], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if x_name == "age":
        work = work.loc[work[x_name].between(18.0, 50.0)]
    elif x_name == "prior_ufc_fights":
        work = work.loc[work[x_name].between(0, 40)]

    if len(work) < MIN_TRAIN_ROWS + MIN_TEST_ROWS:
        return []
    train, test, cutoff = _chronological_split(work)
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        return []

    rows: list[dict[str, object]] = []
    for degree in DEGREES:
        model = _fit_poly(train[x_name].to_numpy(), train[y_name].to_numpy(), degree)
        train_pred = model(train[x_name].to_numpy(dtype=float))
        test_pred = model(test[x_name].to_numpy(dtype=float))
        test_rmse = float(np.sqrt(mean_squared_error(test[y_name], test_pred)))
        rows.append({
            "trait": trait,
            "view": view,
            "x": x_name,
            "degree": degree,
            "model": DEGREE_NAMES[degree],
            "n_total": len(work),
            "n_train": len(train),
            "n_test": len(test),
            "cutoff_date": cutoff,
            "train_rmse": float(np.sqrt(mean_squared_error(train[y_name], train_pred))),
            "test_rmse": test_rmse,
            "test_mae": float(mean_absolute_error(test[y_name], test_pred)),
            "test_r2": float(r2_score(test[y_name], test_pred)),
            "coef": ";".join(f"{v:.12g}" for v in model.c),
        })
    return rows


def _bin_stats(frame: pd.DataFrame, x_name: str, y_name: str) -> pd.DataFrame:
    work = frame[[x_name, y_name]].copy()
    work[x_name] = pd.to_numeric(work[x_name], errors="coerce")
    work[y_name] = pd.to_numeric(work[y_name], errors="coerce")
    work = work.dropna()
    if x_name == "age":
        work = work.loc[work[x_name].between(18.0, 50.0)].copy()
        work["bin"] = np.floor(work[x_name]).astype(int)
    else:
        work = work.loc[work[x_name].between(0, 40)].copy()
        work["bin"] = work[x_name].round().astype(int)
    stats = work.groupby("bin", as_index=False)[y_name].agg(["count", "mean", "median", "std"]).reset_index()
    stats = stats.rename(columns={"count": "n"})
    stats["sem"] = stats["std"] / np.sqrt(stats["n"].clip(lower=1))
    return stats


def _parse_poly(coeff_text: str) -> np.poly1d:
    return np.poly1d([float(v) for v in str(coeff_text).split(";")])


def _plot_pages(
    frame: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    view: str,
    x_name: str,
    y_lookup: dict[str, str],
    output_dir: Path,
) -> None:
    traits = list(canonical.CANONICAL_RATINGS)
    pages = math.ceil(len(traits) / TRAITS_PER_PAGE)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page in range(pages):
        chunk = traits[page * TRAITS_PER_PAGE:(page + 1) * TRAITS_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(11, 3.2 * len(chunk)), squeeze=False)
        for ax, trait in zip(axes.ravel(), chunk):
            y_name = y_lookup[trait]
            stats = _bin_stats(frame, x_name, y_name)
            shown = stats.loc[stats["n"] >= MIN_BIN_N].copy()
            if not shown.empty:
                ax.plot(shown["bin"], shown["mean"], marker="o", linewidth=1.6, label="empirical mean")
                ax.plot(shown["bin"], shown["median"], marker=".", linestyle="--", linewidth=1.0, label="empirical median")

            trait_fits = fits.loc[(fits["trait"] == trait) & (fits["view"] == view) & (fits["x"] == x_name)].copy()
            if not trait_fits.empty:
                best_idx = trait_fits["test_rmse"].idxmin()
                best_degree = int(trait_fits.loc[best_idx, "degree"])
                xmin = float(shown["bin"].min()) if not shown.empty else float(frame[x_name].quantile(0.02))
                xmax = float(shown["bin"].max()) if not shown.empty else float(frame[x_name].quantile(0.98))
                xs = np.linspace(xmin, xmax, 200)
                for row in trait_fits.sort_values("degree").itertuples(index=False):
                    model = _parse_poly(row.coef)
                    label = f"{row.model} RMSE={row.test_rmse:.3f}"
                    if int(row.degree) == best_degree:
                        label += " BEST"
                        ax.plot(xs, model(xs), linewidth=2.2, label=label)
                    else:
                        ax.plot(xs, model(xs), linestyle="--", linewidth=1.0, alpha=0.7, label=label)

            if view == "drift":
                ax.axhline(0.0, linewidth=0.8)
            ax.set_title(trait.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Age" if x_name == "age" else "Prior UFC fights")
            ax.set_ylabel("FSR" if view == "level" else "Next-state ΔFSR")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="best")

        fig.suptitle(
            f"FSR Population {view.title()} Curves — {('Age' if x_name == 'age' else 'UFC Experience')}",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        path = output_dir / f"{view}_{x_name}_page{page + 1}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[population curves] loading existing FSR-32: {FSR_PATH}", flush=True)
    fsr = _prepare_snapshots()
    print(
        f"[population curves] snapshots={len(fsr):,} fighters={fsr['fighter_id'].nunique():,} "
        f"age coverage={fsr['age'].notna().mean():.1%} | elapsed={_elapsed(start)}",
        flush=True,
    )

    print("[population curves] building consecutive-fight transition table...", flush=True)
    transitions = _build_transition_table(fsr)
    transition_path = args.output_dir / TRANSITION_PATH.name
    transitions.to_parquet(transition_path, index=False)
    print(
        f"[population curves] transitions={len(transitions):,} fighters={transitions['fighter_id'].nunique():,} "
        f"-> {transition_path} | elapsed={_elapsed(start)}",
        flush=True,
    )

    fit_rows: list[dict[str, object]] = []
    bin_rows: list[pd.DataFrame] = []
    total_jobs = len(canonical.CANONICAL_RATINGS) * 4
    job = 0

    for trait in canonical.CANONICAL_RATINGS:
        for view, source, y_name in (
            ("level", fsr, trait),
            ("drift", transitions, f"delta__{trait}"),
        ):
            for x_name in ("age", "prior_ufc_fights"):
                job += 1
                if job == 1 or job % 10 == 0 or job == total_jobs:
                    print(
                        f"[population curves] fits {job}/{total_jobs} ({100*job/total_jobs:.0f}%) | "
                        f"{trait} | {view}/{x_name} | elapsed={_elapsed(start)}",
                        flush=True,
                    )
                fit_rows.extend(_fit_one(source, trait=trait, view=view, x_name=x_name, y_name=y_name))
                stats = _bin_stats(source, x_name, y_name)
                if not stats.empty:
                    stats.insert(0, "trait", trait)
                    stats.insert(1, "view", view)
                    stats.insert(2, "x", x_name)
                    bin_rows.append(stats)

    fits = pd.DataFrame(fit_rows)
    if fits.empty:
        raise RuntimeError("population curve audit produced no fits")
    fits["best_for_trait_view_x"] = False
    for _, group in fits.groupby(["trait", "view", "x"], sort=False):
        fits.loc[group["test_rmse"].idxmin(), "best_for_trait_view_x"] = True

    fit_path = args.output_dir / FIT_PATH.name
    fits.to_csv(fit_path, index=False)
    bins = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()
    bin_path = args.output_dir / BIN_PATH.name
    bins.to_csv(bin_path, index=False)

    print("\nBEST POPULATION CURVE BY HELD-OUT RMSE", flush=True)
    best = fits.loc[fits["best_for_trait_view_x"]].copy()
    pivot = best.pivot(index="trait", columns=["view", "x"], values="model")
    print(pivot.to_string(), flush=True)

    print("\nBEST MODEL COUNTS", flush=True)
    counts = (
        best.groupby(["view", "x", "model"]).size().rename("traits").reset_index()
        .sort_values(["view", "x", "traits"], ascending=[True, True, False])
    )
    print(counts.to_string(index=False), flush=True)

    if not args.no_plots:
        print(f"\n[population curves] writing paginated plots | elapsed={_elapsed(start)}", flush=True)
        plot_dir = args.output_dir / "plots"
        _plot_pages(
            fsr,
            fits,
            view="level",
            x_name="age",
            y_lookup={trait: trait for trait in canonical.CANONICAL_RATINGS},
            output_dir=plot_dir,
        )
        _plot_pages(
            fsr,
            fits,
            view="level",
            x_name="prior_ufc_fights",
            y_lookup={trait: trait for trait in canonical.CANONICAL_RATINGS},
            output_dir=plot_dir,
        )
        _plot_pages(
            transitions,
            fits,
            view="drift",
            x_name="age",
            y_lookup={trait: f"delta__{trait}" for trait in canonical.CANONICAL_RATINGS},
            output_dir=plot_dir,
        )
        _plot_pages(
            transitions,
            fits,
            view="drift",
            x_name="prior_ufc_fights",
            y_lookup={trait: f"delta__{trait}" for trait in canonical.CANONICAL_RATINGS},
            output_dir=plot_dir,
        )
        print(f"[population curves] plots -> {plot_dir}", flush=True)

    print(f"\nfit summary: {fit_path}", flush=True)
    print(f"binned curves: {bin_path}", flush=True)
    print(f"transitions: {transition_path}", flush=True)
    print(f"complete | elapsed={_elapsed(start)}", flush=True)
    print("No FSR replay was performed.", flush=True)


if __name__ == "__main__":
    main()
