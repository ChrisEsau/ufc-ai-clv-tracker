"""Stage 1 flow discrimination attribution.

Measures where fight-level flow signal is preserved or lost:

    raw FSR traits
        ->
    exact EVENT MC matchup transforms
        ->
    realized Stage 1 Monte Carlo flow
        ->
    historical fight flow

No winner metrics.
No population calibration.
No global-parameter tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_mc_v1.components.fsr_v2_mechanics import (
    matchup_probability,
    escape_rate,
)


TRAITS = [
    "takedown_tendency",
    "takedown_suppression",
    "takedown_offense",
    "takedown_defense",
    "takedown_completion_baseline",
    "escape_offense",
    "escape_defense",
    "escape_population_mean_seconds",
]


def corr(a, b, method):
    z = pd.DataFrame({"a": a, "b": b}).dropna()

    if len(z) < 3 or z.a.nunique() < 2 or z.b.nunique() < 2:
        return np.nan

    return z.a.corr(z.b, method=method)


def pair_score(hist_r, hist_b, pred_r, pred_b):
    hist_r = pd.Series(hist_r, dtype=float)
    hist_b = pd.Series(hist_b, dtype=float)
    pred_r = pd.Series(pred_r, dtype=float)
    pred_b = pd.Series(pred_b, dtype=float)

    hist_edge = hist_r - hist_b
    pred_edge = pred_r - pred_b

    valid = (
        hist_edge.notna()
        & pred_edge.notna()
    )

    hist_edge = hist_edge[valid]
    pred_edge = pred_edge[valid]

    hist_non_tie = ~np.isclose(hist_edge, 0.0)
    pred_non_tie = ~np.isclose(pred_edge, 0.0)

    resolved = hist_non_tie & pred_non_tie

    correct = (
        hist_non_tie
        & (
            np.sign(hist_edge)
            == np.sign(pred_edge)
        )
    )

    hist_side = pd.concat(
        [
            hist_r[valid].reset_index(drop=True),
            hist_b[valid].reset_index(drop=True),
        ],
        ignore_index=True,
    )

    pred_side = pd.concat(
        [
            pred_r[valid].reset_index(drop=True),
            pred_b[valid].reset_index(drop=True),
        ],
        ignore_index=True,
    )

    return {
        "n": int(valid.sum()),

        "direction_accuracy": (
            correct.sum() / hist_non_tie.sum()
            if hist_non_tie.sum()
            else np.nan
        ),

        "resolved_direction_accuracy": (
            correct.sum() / resolved.sum()
            if resolved.sum()
            else np.nan
        ),

        "resolution": (
            resolved.sum() / hist_non_tie.sum()
            if hist_non_tie.sum()
            else np.nan
        ),

        "edge_pearson":
            corr(
                hist_edge,
                pred_edge,
                "pearson",
            ),

        "edge_spearman":
            corr(
                hist_edge,
                pred_edge,
                "spearman",
            ),

        "side_pearson":
            corr(
                hist_side,
                pred_side,
                "pearson",
            ),

        "side_spearman":
            corr(
                hist_side,
                pred_side,
                "spearman",
            ),
    }


def single_score(actual, predicted):
    actual = pd.Series(actual, dtype=float)
    predicted = pd.Series(predicted, dtype=float)

    mask = (
        actual.notna()
        & predicted.notna()
    )

    return {
        "n": int(mask.sum()),
        "pearson": corr(
            actual[mask],
            predicted[mask],
            "pearson",
        ),
        "spearman": corr(
            actual[mask],
            predicted[mask],
            "spearman",
        ),
    }


def fmt(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.3f}"


def attach_fsr(x):
    fsr = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    ).copy()

    fsr["fight_id"] = (
        fsr["fight_id"].astype(str)
    )

    x["bout_id"] = (
        x["bout_id"].astype(str)
    )

    fsr = fsr[
        fsr["fight_id"].isin(
            set(x["bout_id"])
        )
    ].copy()

    required = [
        "fight_id",
        "fighter_name",
        *TRAITS,
    ]

    missing = (
        set(required)
        - set(fsr.columns)
    )

    if missing:
        raise RuntimeError(
            f"FSR snapshot missing: {sorted(missing)}"
        )

    for side in ("red", "blue"):
        name_col = f"{side}_fighter"

        f = fsr[required].copy()

        f = f.rename(
            columns={
                "fight_id": "bout_id",
                "fighter_name": name_col,
                **{
                    c: f"{side}_{c}"
                    for c in TRAITS
                },
            }
        )

        x = x.merge(
            f,
            on=[
                "bout_id",
                name_col,
            ],
            how="left",
            validate="one_to_one",
        )

    fsr_cols = [
        f"{side}_{trait}"
        for side in ("red", "blue")
        for trait in TRAITS
    ]

    if x[fsr_cols].isna().any().any():
        bad = x.loc[
            x[fsr_cols].isna().any(axis=1),
            [
                "bout_id",
                "red_fighter",
                "blue_fighter",
            ],
        ]

        raise RuntimeError(
            "Failed to resolve FSR rows:\n"
            + bad.head(20).to_string(index=False)
        )

    return x


def build_signals(x):
    # ---------------------------------------------------------
    # Historical fight-level rates
    # ---------------------------------------------------------
    exposure = x[
        "actual_elapsed_seconds"
    ].astype(float)

    for side in ("red", "blue"):
        x[f"hist_{side}_td_attempt_rate"] = (
            x[f"historical_{side}_td_attempts"]
            * 900.0
            / exposure
        )

        x[f"hist_{side}_td_landed_rate"] = (
            x[f"historical_{side}_td_landed"]
            * 900.0
            / exposure
        )

        x[f"hist_{side}_ground_entry_rate"] = (
            x[f"historical_{side}_ground_entries"]
            * 900.0
            / exposure
        )

        x[f"hist_{side}_control_share"] = (
            x[
                f"historical_{side}_ground_control_seconds"
            ]
            / exposure
        )

        # MC rates.
        x[f"mc_{side}_td_attempt_rate"] = (
            x[f"simulated_mean_{side}_td_attempts"]
            * 900.0
            / exposure
        )

        x[f"mc_{side}_td_landed_rate"] = (
            x[f"simulated_mean_{side}_td_landed"]
            * 900.0
            / exposure
        )

        x[f"mc_{side}_ground_entry_rate"] = (
            x[f"simulated_mean_{side}_ground_entries"]
            * 900.0
            / exposure
        )

        x[f"mc_{side}_control_share"] = (
            x[
                f"simulated_mean_{side}_ground_control_seconds"
            ]
            / exposure
        )

    # ---------------------------------------------------------
    # Stage A: raw takedown tendency only.
    # ---------------------------------------------------------
    x["fsr_red_tendency_only"] = (
        x["red_takedown_tendency"]
    )
    x["fsr_blue_tendency_only"] = (
        x["blue_takedown_tendency"]
    )

    # ---------------------------------------------------------
    # Stage B: tendency - opponent suppression,
    # BEFORE clipping.
    #
    # This isolates the effect of suppression.
    # ---------------------------------------------------------
    x["fsr_red_unclipped_td_rate"] = (
        x["red_takedown_tendency"]
        - x["blue_takedown_suppression"]
    )

    x["fsr_blue_unclipped_td_rate"] = (
        x["blue_takedown_tendency"]
        - x["red_takedown_suppression"]
    )

    # ---------------------------------------------------------
    # Stage C: exact EVENT MC effective_rate transform.
    #
    # Global common multiplier intentionally omitted because
    # it cannot improve discrimination.
    # ---------------------------------------------------------
    x["fsr_red_effective_td_rate"] = np.maximum(
        x["fsr_red_unclipped_td_rate"],
        0.0,
    )

    x["fsr_blue_effective_td_rate"] = np.maximum(
        x["fsr_blue_unclipped_td_rate"],
        0.0,
    )

    # ---------------------------------------------------------
    # Exact EVENT MC takedown completion probabilities.
    # ---------------------------------------------------------
    x["fsr_red_td_completion"] = x.apply(
        lambda r: matchup_probability(
            r["red_takedown_completion_baseline"],
            r["red_takedown_offense"],
            r["blue_takedown_defense"],
        ),
        axis=1,
    )

    x["fsr_blue_td_completion"] = x.apply(
        lambda r: matchup_probability(
            r["blue_takedown_completion_baseline"],
            r["blue_takedown_offense"],
            r["red_takedown_defense"],
        ),
        axis=1,
    )

    # Expected landed-TD hazard before stochastic simulation.
    x["fsr_red_landed_td_hazard"] = (
        x["fsr_red_effective_td_rate"]
        * x["fsr_red_td_completion"]
    )

    x["fsr_blue_landed_td_hazard"] = (
        x["fsr_blue_effective_td_rate"]
        * x["fsr_blue_td_completion"]
    )

    # ---------------------------------------------------------
    # Exact expected ground residence after each side gets top.
    #
    # RED on top -> BLUE must escape.
    # BLUE on top -> RED must escape.
    # ---------------------------------------------------------
    x["fsr_red_top_mean_seconds"] = x.apply(
        lambda r: 1.0 / escape_rate(
            r["blue_escape_offense"],
            r["red_escape_defense"],
            r["blue_escape_population_mean_seconds"],
        ),
        axis=1,
    )

    x["fsr_blue_top_mean_seconds"] = x.apply(
        lambda r: 1.0 / escape_rate(
            r["red_escape_offense"],
            r["blue_escape_defense"],
            r["red_escape_population_mean_seconds"],
        ),
        axis=1,
    )

    # Structural control-pressure proxy:
    #
    # entry hazard * expected top residence.
    #
    # This is NOT claimed to be the simulator's exact stationary
    # distribution. It is a discrimination attribution diagnostic.
    x["fsr_red_control_pressure"] = (
        x["fsr_red_landed_td_hazard"]
        * x["fsr_red_top_mean_seconds"]
    )

    x["fsr_blue_control_pressure"] = (
        x["fsr_blue_landed_td_hazard"]
        * x["fsr_blue_top_mean_seconds"]
    )

    return x


def print_table(title, rows):
    print("\n" + title)
    print(
        f"{'signal':31s}"
        f"{'dir':>8s}"
        f"{'resolved':>10s}"
        f"{'edge r':>10s}"
        f"{'edge rho':>11s}"
        f"{'side rho':>11s}"
    )

    for label, result in rows:
        print(
            f"{label:31s}"
            f"{fmt(result['direction_accuracy']):>8s}"
            f"{fmt(result['resolved_direction_accuracy']):>10s}"
            f"{fmt(result['edge_pearson']):>10s}"
            f"{fmt(result['edge_spearman']):>11s}"
            f"{fmt(result['side_spearman']):>11s}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    x = pd.read_csv(args.csv)
    x = attach_fsr(x)
    x = build_signals(x)

    # =========================================================
    # TD ATTEMPT DISCRIMINATION
    # =========================================================
    attempt_rows = []

    hist_r = x["hist_red_td_attempt_rate"]
    hist_b = x["hist_blue_td_attempt_rate"]

    for label, red, blue in [
        (
            "FSR tendency only",
            x["fsr_red_tendency_only"],
            x["fsr_blue_tendency_only"],
        ),
        (
            "FSR + opponent suppression",
            x["fsr_red_unclipped_td_rate"],
            x["fsr_blue_unclipped_td_rate"],
        ),
        (
            "EVENT MC effective TD rate",
            x["fsr_red_effective_td_rate"],
            x["fsr_blue_effective_td_rate"],
        ),
        (
            "Stage 1 MC realized attempts",
            x["mc_red_td_attempt_rate"],
            x["mc_blue_td_attempt_rate"],
        ),
    ]:
        attempt_rows.append(
            (
                label,
                pair_score(
                    hist_r,
                    hist_b,
                    red,
                    blue,
                ),
            )
        )

    # =========================================================
    # LANDED TD DISCRIMINATION
    # =========================================================
    landed_rows = []

    hist_r = x["hist_red_td_landed_rate"]
    hist_b = x["hist_blue_td_landed_rate"]

    for label, red, blue in [
        (
            "FSR attempt rate only",
            x["fsr_red_effective_td_rate"],
            x["fsr_blue_effective_td_rate"],
        ),
        (
            "FSR expected landed hazard",
            x["fsr_red_landed_td_hazard"],
            x["fsr_blue_landed_td_hazard"],
        ),
        (
            "Stage 1 MC realized landed",
            x["mc_red_td_landed_rate"],
            x["mc_blue_td_landed_rate"],
        ),
    ]:
        landed_rows.append(
            (
                label,
                pair_score(
                    hist_r,
                    hist_b,
                    red,
                    blue,
                ),
            )
        )

    # =========================================================
    # GROUND ENTRY DISCRIMINATION
    # =========================================================
    entry_rows = []

    hist_r = x["hist_red_ground_entry_rate"]
    hist_b = x["hist_blue_ground_entry_rate"]

    for label, red, blue in [
        (
            "FSR expected landed hazard",
            x["fsr_red_landed_td_hazard"],
            x["fsr_blue_landed_td_hazard"],
        ),
        (
            "Stage 1 MC realized entries",
            x["mc_red_ground_entry_rate"],
            x["mc_blue_ground_entry_rate"],
        ),
    ]:
        entry_rows.append(
            (
                label,
                pair_score(
                    hist_r,
                    hist_b,
                    red,
                    blue,
                ),
            )
        )

    # =========================================================
    # CONTROL DISCRIMINATION
    # =========================================================
    control_rows = []

    hist_r = x["hist_red_control_share"]
    hist_b = x["hist_blue_control_share"]

    for label, red, blue in [
        (
            "FSR landed hazard only",
            x["fsr_red_landed_td_hazard"],
            x["fsr_blue_landed_td_hazard"],
        ),
        (
            "FSR + expected residence",
            x["fsr_red_control_pressure"],
            x["fsr_blue_control_pressure"],
        ),
        (
            "Stage 1 MC realized control",
            x["mc_red_control_share"],
            x["mc_blue_control_share"],
        ),
    ]:
        control_rows.append(
            (
                label,
                pair_score(
                    hist_r,
                    hist_b,
                    red,
                    blue,
                ),
            )
        )

    # =========================================================
    # PURE TD COMPLETION DISCRIMINATION
    # =========================================================
    hist_success = []
    fsr_success = []
    mc_success = []

    for side in ("red", "blue"):
        h_att = x[
            f"historical_{side}_td_attempts"
        ].astype(float)

        h_land = x[
            f"historical_{side}_td_landed"
        ].astype(float)

        m_att = x[
            f"simulated_mean_{side}_td_attempts"
        ].astype(float)

        m_land = x[
            f"simulated_mean_{side}_td_landed"
        ].astype(float)

        h = h_land / h_att
        m = m_land / m_att

        hist_success.append(
            h.where(h_att > 0)
        )

        fsr_success.append(
            x[f"fsr_{side}_td_completion"]
            .where(h_att > 0)
        )

        mc_success.append(
            m.where(
                (h_att > 0)
                & (m_att > 0)
            )
        )

    hist_success = pd.concat(
        hist_success,
        ignore_index=True,
    )

    fsr_success = pd.concat(
        fsr_success,
        ignore_index=True,
    )

    mc_success = pd.concat(
        mc_success,
        ignore_index=True,
    )

    fsr_completion = single_score(
        hist_success,
        fsr_success,
    )

    mc_completion = single_score(
        hist_success,
        mc_success,
    )

    # =========================================================
    # PURE RESIDENCE DISCRIMINATION
    # =========================================================
    hist_residence = []
    fsr_residence = []
    mc_residence = []

    for side in ("red", "blue"):
        h_entries = x[
            f"historical_{side}_ground_entries"
        ].astype(float)

        h_control = x[
            f"historical_{side}_ground_control_seconds"
        ].astype(float)

        m_entries = x[
            f"simulated_mean_{side}_ground_entries"
        ].astype(float)

        m_control = x[
            f"simulated_mean_{side}_ground_control_seconds"
        ].astype(float)

        hist_residence.append(
            (h_control / h_entries)
            .where(h_entries > 0)
        )

        fsr_residence.append(
            x[
                f"fsr_{side}_top_mean_seconds"
            ].where(h_entries > 0)
        )

        mc_residence.append(
            (m_control / m_entries)
            .where(
                (h_entries > 0)
                & (m_entries > 0)
            )
        )

    hist_residence = pd.concat(
        hist_residence,
        ignore_index=True,
    )

    fsr_residence = pd.concat(
        fsr_residence,
        ignore_index=True,
    )

    mc_residence = pd.concat(
        mc_residence,
        ignore_index=True,
    )

    fsr_residence_score = single_score(
        hist_residence,
        fsr_residence,
    )

    mc_residence_score = single_score(
        hist_residence,
        mc_residence,
    )

    print("=" * 110)
    print(
        "EVENT MC V1 — STAGE 1 DISCRIMINATION ATTRIBUTION"
    )
    print("=" * 110)
    print(
        f"fights={len(x)} | "
        "NO GLOBAL CALIBRATION METRICS"
    )

    print_table(
        "TD ATTEMPT DISCRIMINATION",
        attempt_rows,
    )

    print_table(
        "TD LANDED DISCRIMINATION",
        landed_rows,
    )

    print_table(
        "GROUND ENTRY DISCRIMINATION",
        entry_rows,
    )

    print_table(
        "GROUND CONTROL DISCRIMINATION",
        control_rows,
    )

    print(
        "\nTD COMPLETION — FIGHTER-SIDE DISCRIMINATION"
    )
    print(
        f"{'signal':31s}"
        f"{'N':>8s}"
        f"{'Pearson':>10s}"
        f"{'Spearman':>11s}"
    )

    print(
        f"{'Exact FSR matchup probability':31s}"
        f"{fsr_completion['n']:>8d}"
        f"{fmt(fsr_completion['pearson']):>10s}"
        f"{fmt(fsr_completion['spearman']):>11s}"
    )

    print(
        f"{'Stage 1 MC empirical success':31s}"
        f"{mc_completion['n']:>8d}"
        f"{fmt(mc_completion['pearson']):>10s}"
        f"{fmt(mc_completion['spearman']):>11s}"
    )

    print(
        "\nGROUND RESIDENCE — FIGHTER-SIDE DISCRIMINATION"
    )
    print(
        f"{'signal':31s}"
        f"{'N':>8s}"
        f"{'Pearson':>10s}"
        f"{'Spearman':>11s}"
    )

    print(
        f"{'Exact FSR expected residence':31s}"
        f"{fsr_residence_score['n']:>8d}"
        f"{fmt(fsr_residence_score['pearson']):>10s}"
        f"{fmt(fsr_residence_score['spearman']):>11s}"
    )

    print(
        f"{'Stage 1 MC empirical residence':31s}"
        f"{mc_residence_score['n']:>8d}"
        f"{fmt(mc_residence_score['pearson']):>10s}"
        f"{fmt(mc_residence_score['spearman']):>11s}"
    )

    output = (
        args.output
        or args.csv.with_name(
            args.csv.stem
            + "_discrimination_attribution.csv"
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x.to_csv(
        output,
        index=False,
    )

    print(f"\nwrote joined attribution: {output}")


if __name__ == "__main__":
    main()
