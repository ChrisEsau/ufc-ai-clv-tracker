"""Explore age-related striking-power evidence effects hidden by non-degrading stored power.

Research/shadow only. No FSR replay and no simulator changes.

Outputs a four-panel diagnostic by integer fighter age:
1) raw positive power-event rate;
2) raw Round-1 knockdown rate;
3) raw Round-1 KO/TKO-win rate;
4) residual positive power-event rate after controlling for prefight stored striking_power,
   Round-1 significant-strike opportunity, and prior UFC fight count.

The residual panel is the key aging diagnostic: negative values mean fighters at that age
produce fewer positive power events than expected from their stored power/opportunity/experience.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experimental import build_fsr_32_database as fsr32
from scripts.experimental import fsr_age_adjustment_kd_durability_controlled_2020plus_mature as age_study
from scripts.experimental import fsr_striking_power_evidence_v8_hierarchical_ko_kd_sweep as power_v8


OUTPUT_DIR = Path("data/experimental/striking_power_age_effect")
PLOT_PATH = OUTPUT_DIR / "striking_power_age_effect.png"
SUMMARY_PATH = OUTPUT_DIR / "striking_power_age_effect_by_age.csv"
ROW_PATH = OUTPUT_DIR / "striking_power_age_effect_rows.csv"


def _load_age_rows(master: pd.DataFrame) -> pd.DataFrame:
    date_col = age_study.modern._resolve_date_column(master)
    m = master.copy()
    m[date_col] = pd.to_datetime(m[date_col], errors="coerce")
    m = m.dropna(subset=[date_col]).copy().rename(columns={date_col: "event_date"})
    m["fight_id"] = m["fight_id"].astype(str)
    m["r_id"] = m["r_id"].astype(str)
    m["b_id"] = m["b_id"].astype(str)
    m["r_age_calc"] = age_study._resolve_corner_age(m, "r")
    m["b_age_calc"] = age_study._resolve_corner_age(m, "b")
    red = m[["fight_id", "event_date", "r_id", "r_age_calc"]].rename(
        columns={"r_id": "fighter_id", "r_age_calc": "age"}
    )
    blue = m[["fight_id", "event_date", "b_id", "b_age_calc"]].rename(
        columns={"b_id": "fighter_id", "b_age_calc": "age"}
    )
    out = pd.concat([red, blue], ignore_index=True)
    out["fighter_id"] = out["fighter_id"].astype(str)
    out = out.dropna(subset=["age"]).drop_duplicates(["fight_id", "fighter_id"], keep="last")
    return out


def _build_frame() -> pd.DataFrame:
    print("[power-age] loading master/round stats...", flush=True)
    master = pd.read_parquet(power_v8.v2.MASTER_PATH)
    rounds = pd.read_parquet(power_v8.v2.ROUND_STATS_PATH)

    print("[power-age] building V8 fighter-fight power evidence...", flush=True)
    _, scored = power_v8.build_v8_rankings(master, rounds)
    scored = scored.copy()
    scored["fight_id"] = scored["fight_id"].astype(str)
    scored["fighter_id"] = scored["fighter_id"].astype(str)

    ages = _load_age_rows(master)

    print(f"[power-age] loading prefight FSR-32: {fsr32.OUTPUT_PATH}", flush=True)
    fsr = pd.read_parquet(
        fsr32.OUTPUT_PATH,
        columns=["fight_id", "fighter_id", "prior_ufc_fights", "striking_power"],
    ).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)

    work = scored.merge(
        ages[["fight_id", "fighter_id", "event_date", "age"]],
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        fsr,
        on=["fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )

    for col in ("age", "striking_power", "prior_ufc_fights", "sig_str_landed", "kd", "fight_power_evidence_v8"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["age", "striking_power", "prior_ufc_fights", "sig_str_landed", "kd"])
    work = work.loc[work["age"].between(18.0, 45.0)].copy()
    work["age_year"] = np.floor(work["age"]).astype(int)
    work["power_event_int"] = work["power_event"].astype(bool).astype(int)
    work["r1_ko_event_int"] = work["r1_ko_event"].astype(bool).astype(int)
    work["r1_kd_event_int"] = work["kd"].gt(0).astype(int)

    # Expected power-event probability without age. This controls for the exact
    # non-degrading stored power signal, current R1 striking opportunity, and experience.
    features = ["striking_power", "sig_str_landed", "prior_ufc_fights"]
    X = work[features].astype(float)
    y = work["power_event_int"].astype(int)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
    ])
    model.fit(X, y)
    work["expected_power_event_no_age"] = model.predict_proba(X)[:, 1]
    work["power_event_residual"] = work["power_event_int"] - work["expected_power_event_no_age"]

    print(
        f"[power-age] rows={len(work):,} | fighters={work['fighter_id'].nunique():,} | "
        f"age={work['age'].min():.1f}-{work['age'].max():.1f}",
        flush=True,
    )
    return work.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _summarize(work: pd.DataFrame) -> pd.DataFrame:
    summary = (
        work.groupby("age_year", as_index=False)
        .agg(
            n=("fight_id", "size"),
            fighters=("fighter_id", "nunique"),
            mean_prefight_power=("striking_power", "mean"),
            power_event_rate=("power_event_int", "mean"),
            r1_kd_rate=("r1_kd_event_int", "mean"),
            r1_ko_win_rate=("r1_ko_event_int", "mean"),
            mean_power_evidence=("fight_power_evidence_v8", "mean"),
            adjusted_power_event_residual=("power_event_residual", "mean"),
        )
    )
    return summary.loc[summary["n"] >= 25].reset_index(drop=True)


def _add_sample_labels(ax, summary: pd.DataFrame) -> None:
    y0, y1 = ax.get_ylim()
    y = y0 + 0.03 * (y1 - y0)
    for row in summary.itertuples(index=False):
        ax.text(row.age_year, y, f"n={row.n}", ha="center", va="bottom", fontsize=7, rotation=90)


def _plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    specs = [
        ("power_event_rate", "Positive V8 power-event rate", "Rate"),
        ("r1_kd_rate", "Round-1 knockdown event rate", "Rate"),
        ("r1_ko_win_rate", "Round-1 KO/TKO win rate", "Rate"),
        (
            "adjusted_power_event_residual",
            "Power-event residual after controlling stored power + R1 opportunity + prior UFC fights",
            "Observed - expected event probability",
        ),
    ]
    for ax, (col, title, ylabel) in zip(axes, specs):
        ax.plot(summary["age_year"], summary[col], marker="o", linewidth=2)
        if col == "adjusted_power_event_residual":
            ax.axhline(0.0, linestyle="--", linewidth=1)
            # Smooth linear/quadratic/cubic overlays for visual comparison only.
            x = summary["age_year"].to_numpy(float)
            y = summary[col].to_numpy(float)
            grid = np.linspace(x.min(), x.max(), 300)
            for degree, label in ((1, "linear"), (2, "quadratic"), (3, "cubic")):
                model = np.poly1d(np.polyfit(x, y, degree))
                ax.plot(grid, model(grid), linewidth=1.4, label=label)
            ax.legend()
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        _add_sample_labels(ax, summary)

    axes[-1].set_xlabel("Fighter age at fight")
    fig.suptitle(
        "Striking Power vs Age — Fight-Level Evidence Diagnostic\n"
        "Stored striking_power is non-degrading; bottom panel looks for age decline hidden by that contract",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(PLOT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work = _build_frame()
    summary = _summarize(work)
    work.to_csv(ROW_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    _plot(summary)

    print("\nSTRIKING POWER AGE SUMMARY", flush=True)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    print(f"\nwrote: {PLOT_PATH}", flush=True)
    print(f"wrote: {SUMMARY_PATH}", flush=True)
    print(f"wrote: {ROW_PATH}", flush=True)


if __name__ == "__main__":
    main()
