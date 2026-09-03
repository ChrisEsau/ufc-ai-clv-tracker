"""Stage 1 fight-flow scorer.

Scores only upstream fight flow:
- fighter-specific TD attempts
- fighter-specific TD landed
- fighter-specific ground entries
- fighter-specific ground control
- fight-level ground phase share
- TD completion

No winner, method, market, damage, KO, SUB-conversion, or judging metrics.

Because the flow replay is conditioned on actual historical fight duration,
duration-normalized rates/shares are the PRIMARY discrimination metrics.
Raw totals are retained as secondary population-calibration diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v2.replay.engine import aggregate_fights
from pipeline.fsr_v2.sources.round_stats import (
    build_paired_rounds,
    load_round_stats,
)


SPECS = {
    "td_attempts": {
        "hist_red": "historical_red_td_attempts",
        "hist_blue": "historical_blue_td_attempts",
        "mc_red": "simulated_mean_red_td_attempts",
        "mc_blue": "simulated_mean_blue_td_attempts",
        "normalization": "per15",
    },
    "td_landed": {
        "hist_red": "historical_red_td_landed",
        "hist_blue": "historical_blue_td_landed",
        "mc_red": "simulated_mean_red_td_landed",
        "mc_blue": "simulated_mean_blue_td_landed",
        "normalization": "per15",
    },
    "ground_entries": {
        "hist_red": "historical_red_ground_entries",
        "hist_blue": "historical_blue_ground_entries",
        "mc_red": "simulated_mean_red_ground_entries",
        "mc_blue": "simulated_mean_blue_ground_entries",
        "normalization": "per15",
    },
    "ground_control": {
        "hist_red": "historical_red_ground_control_seconds",
        "hist_blue": "historical_blue_ground_control_seconds",
        "mc_red": "simulated_mean_red_ground_control_seconds",
        "mc_blue": "simulated_mean_blue_ground_control_seconds",
        "normalization": "share",
    },
}


def corr(a, b, method):
    z = pd.DataFrame(
        {"a": a, "b": b}
    ).dropna()

    if (
        len(z) < 3
        or z.a.nunique() < 2
        or z.b.nunique() < 2
    ):
        return None

    return float(
        z.a.corr(
            z.b,
            method=method,
        )
    )


def fmt(v, digits=3):
    if v is None or pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def normalized(values, exposure, kind):
    values = values.astype(float)
    exposure = exposure.astype(float)

    if kind == "per15":
        return values * 900.0 / exposure

    if kind == "share":
        return values / exposure

    raise ValueError(kind)


def load_historical():
    fights = aggregate_fights(
        build_paired_rounds(
            rounds=load_round_stats()
        )
    ).copy()

    fights["fight_id"] = (
        fights["fight_id"].astype(str)
    )

    side = fights[
        [
            "fight_id",
            "fighter_name",
            "td_attempted",
            "td_landed",
            "ground_entries",
            "qualified_control_inflicted_seconds",
        ]
    ].copy()

    # modeled_ground_exposure_seconds is shared fight opportunity.
    # MAX prevents reciprocal fighter rows from double counting it.
    fight = (
        fights.groupby(
            "fight_id",
            as_index=False,
        )
        .agg(
            historical_ground_seconds=(
                "modeled_ground_exposure_seconds",
                "max",
            ),
            historical_elapsed_seconds=(
                "fight_elapsed_seconds",
                "max",
            ),
        )
    )

    return side, fight


def attach_historical(replay):
    side, fight = load_historical()

    replay = replay.copy()
    replay["bout_id"] = (
        replay["bout_id"].astype(str)
    )

    red = side.rename(
        columns={
            "fight_id": "bout_id",
            "fighter_name": "red_fighter",
            "td_attempted":
                "historical_red_td_attempts",
            "td_landed":
                "historical_red_td_landed",
            "ground_entries":
                "historical_red_ground_entries",
            "qualified_control_inflicted_seconds":
                "historical_red_ground_control_seconds",
        }
    )

    blue = side.rename(
        columns={
            "fight_id": "bout_id",
            "fighter_name": "blue_fighter",
            "td_attempted":
                "historical_blue_td_attempts",
            "td_landed":
                "historical_blue_td_landed",
            "ground_entries":
                "historical_blue_ground_entries",
            "qualified_control_inflicted_seconds":
                "historical_blue_ground_control_seconds",
        }
    )

    x = replay.merge(
        red,
        on=[
            "bout_id",
            "red_fighter",
        ],
        how="left",
        validate="one_to_one",
    )

    x = x.merge(
        blue,
        on=[
            "bout_id",
            "blue_fighter",
        ],
        how="left",
        validate="one_to_one",
    )

    x = x.merge(
        fight.rename(
            columns={
                "fight_id": "bout_id"
            }
        ),
        on="bout_id",
        how="left",
        validate="one_to_one",
    )

    required = [
        "historical_red_td_attempts",
        "historical_blue_td_attempts",
        "historical_ground_seconds",
        "historical_elapsed_seconds",
    ]

    if x[required].isna().any().any():
        bad = x.loc[
            x[required].isna().any(axis=1),
            [
                "bout_id",
                "red_fighter",
                "blue_fighter",
            ],
        ]

        raise RuntimeError(
            "Missing historical observations:\n"
            + bad.to_string(index=False)
        )

    exposure_error = (
        x["actual_elapsed_seconds"]
        - x["historical_elapsed_seconds"]
    ).abs()

    if exposure_error.max() > 1e-6:
        raise RuntimeError(
            "Replay/historical exposure mismatch: "
            f"max error={exposure_error.max()}"
        )

    return x


def score_side_metric(x, name, spec):
    exposure = x[
        "actual_elapsed_seconds"
    ]

    hr_raw = x[spec["hist_red"]].astype(float)
    hb_raw = x[spec["hist_blue"]].astype(float)
    mr_raw = x[spec["mc_red"]].astype(float)
    mb_raw = x[spec["mc_blue"]].astype(float)

    hr = normalized(
        hr_raw,
        exposure,
        spec["normalization"],
    )
    hb = normalized(
        hb_raw,
        exposure,
        spec["normalization"],
    )
    mr = normalized(
        mr_raw,
        exposure,
        spec["normalization"],
    )
    mb = normalized(
        mb_raw,
        exposure,
        spec["normalization"],
    )

    hist_edge = hr - hb
    mc_edge = mr - mb

    hist_sign = np.sign(hist_edge)
    mc_sign = np.sign(mc_edge)

    hist_non_tie = ~np.isclose(
        hist_edge,
        0.0,
        atol=1e-12,
    )
    mc_non_tie = ~np.isclose(
        mc_edge,
        0.0,
        atol=1e-12,
    )

    resolved = (
        hist_non_tie
        & mc_non_tie
    )

    correct = (
        hist_non_tie
        & (hist_sign == mc_sign)
    )

    hist_side = pd.concat(
        [hr, hb],
        ignore_index=True,
    )
    mc_side = pd.concat(
        [mr, mb],
        ignore_index=True,
    )

    hist_raw_total = (
        hr_raw + hb_raw
    )
    mc_raw_total = (
        mr_raw + mb_raw
    )

    x[
        f"historical_{name}_edge_normalized"
    ] = hist_edge

    x[
        f"simulated_{name}_edge_normalized"
    ] = mc_edge

    return {
        "historical_non_tie_fights":
            int(hist_non_tie.sum()),

        "model_resolution_rate": (
            float(
                resolved.sum()
                / hist_non_tie.sum()
            )
            if hist_non_tie.sum()
            else None
        ),

        # MC tie counts as incorrect.
        "direction_accuracy": (
            float(
                correct.sum()
                / hist_non_tie.sum()
            )
            if hist_non_tie.sum()
            else None
        ),

        "resolved_direction_accuracy": (
            float(
                correct.sum()
                / resolved.sum()
            )
            if resolved.sum()
            else None
        ),

        # PRIMARY discrimination.
        "edge_pearson":
            corr(
                hist_edge,
                mc_edge,
                "pearson",
            ),

        "edge_spearman":
            corr(
                hist_edge,
                mc_edge,
                "spearman",
            ),

        "fighter_side_pearson":
            corr(
                hist_side,
                mc_side,
                "pearson",
            ),

        "fighter_side_spearman":
            corr(
                hist_side,
                mc_side,
                "spearman",
            ),

        "normalized_side_mae":
            float(
                np.mean(
                    np.abs(
                        mc_side
                        - hist_side
                    )
                )
            ),

        "normalized_edge_mae":
            float(
                np.mean(
                    np.abs(
                        mc_edge
                        - hist_edge
                    )
                )
            ),

        "historical_mean_abs_edge":
            float(
                hist_edge.abs().mean()
            ),

        "simulated_mean_abs_edge":
            float(
                mc_edge.abs().mean()
            ),

        # SECONDARY population calibration.
        "historical_raw_total_per_fight":
            float(
                hist_raw_total.mean()
            ),

        "simulated_raw_total_per_fight":
            float(
                mc_raw_total.mean()
            ),

        "population_ratio": (
            float(
                mc_raw_total.sum()
                / hist_raw_total.sum()
            )
            if hist_raw_total.sum()
            else None
        ),
    }


def score_phase(x):
    exposure = x[
        "actual_elapsed_seconds"
    ].astype(float)

    hist_seconds = x[
        "historical_ground_seconds"
    ].astype(float)

    mc_seconds = x[
        "simulated_mean_ground_seconds"
    ].astype(float)

    hist_share = (
        hist_seconds / exposure
    )
    mc_share = (
        mc_seconds / exposure
    )

    x[
        "historical_ground_share"
    ] = hist_share

    x[
        "simulated_ground_share"
    ] = mc_share

    return {
        "historical_mean_ground_seconds":
            float(hist_seconds.mean()),

        "simulated_mean_ground_seconds":
            float(mc_seconds.mean()),

        "ground_seconds_population_ratio": (
            float(
                mc_seconds.sum()
                / hist_seconds.sum()
            )
            if hist_seconds.sum()
            else None
        ),

        "ground_seconds_mae":
            float(
                np.abs(
                    mc_seconds
                    - hist_seconds
                ).mean()
            ),

        # PRIMARY phase discrimination.
        "historical_mean_ground_share":
            float(hist_share.mean()),

        "simulated_mean_ground_share":
            float(mc_share.mean()),

        "ground_share_mae":
            float(
                np.abs(
                    mc_share
                    - hist_share
                ).mean()
            ),

        "ground_share_pearson":
            corr(
                hist_share,
                mc_share,
                "pearson",
            ),

        "ground_share_spearman":
            corr(
                hist_share,
                mc_share,
                "spearman",
            ),
    }


def score_td_success(x):
    hist_attempts = pd.concat(
        [
            x[
                "historical_red_td_attempts"
            ],
            x[
                "historical_blue_td_attempts"
            ],
        ],
        ignore_index=True,
    ).astype(float)

    hist_landed = pd.concat(
        [
            x[
                "historical_red_td_landed"
            ],
            x[
                "historical_blue_td_landed"
            ],
        ],
        ignore_index=True,
    ).astype(float)

    mc_attempts = pd.concat(
        [
            x[
                "simulated_mean_red_td_attempts"
            ],
            x[
                "simulated_mean_blue_td_attempts"
            ],
        ],
        ignore_index=True,
    ).astype(float)

    mc_landed = pd.concat(
        [
            x[
                "simulated_mean_red_td_landed"
            ],
            x[
                "simulated_mean_blue_td_landed"
            ],
        ],
        ignore_index=True,
    ).astype(float)

    pooled_hist = (
        hist_landed.sum()
        / hist_attempts.sum()
    )

    pooled_mc = (
        mc_landed.sum()
        / mc_attempts.sum()
    )

    mask = (
        (hist_attempts > 0)
        & (mc_attempts > 0)
    )

    hist_success = (
        hist_landed[mask]
        / hist_attempts[mask]
    )

    mc_success = (
        mc_landed[mask]
        / mc_attempts[mask]
    )

    return {
        "historical_pooled_success":
            float(pooled_hist),

        "simulated_pooled_success":
            float(pooled_mc),

        "fighter_observations_with_attempts":
            int(mask.sum()),

        "fighter_success_pearson":
            corr(
                hist_success,
                mc_success,
                "pearson",
            ),

        "fighter_success_spearman":
            corr(
                hist_success,
                mc_success,
                "spearman",
            ),

        "fighter_success_mae":
            float(
                np.abs(
                    mc_success
                    - hist_success
                ).mean()
            )
            if mask.sum()
            else None,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--joined",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    replay = pd.read_csv(args.csv)

    required = {
        "bout_id",
        "red_fighter",
        "blue_fighter",
        "actual_elapsed_seconds",
        "simulated_mean_ground_seconds",
        "simulated_mean_standing_seconds",
        "simulated_mean_red_td_attempts",
        "simulated_mean_blue_td_attempts",
        "simulated_mean_red_td_landed",
        "simulated_mean_blue_td_landed",
        "simulated_mean_red_ground_entries",
        "simulated_mean_blue_ground_entries",
        "simulated_mean_red_ground_control_seconds",
        "simulated_mean_blue_ground_control_seconds",
    }

    missing = required - set(
        replay.columns
    )

    if missing:
        raise RuntimeError(
            "Stage 1 replay missing columns: "
            f"{sorted(missing)}"
        )

    x = attach_historical(
        replay
    )

    flow = {
        name: score_side_metric(
            x,
            name,
            spec,
        )
        for name, spec
        in SPECS.items()
    }

    phase = score_phase(x)
    td_success = score_td_success(x)

    report = {
        "replay_csv": str(args.csv),
        "fights": int(len(x)),
        "flow_metrics": flow,
        "phase_metrics": phase,
        "td_success": td_success,
        "historical_semantics": {
            "ground_entries":
                "TD landed plus inferred zero-TD entries with ground evidence",
            "ground_control":
                "qualified UFCStats control proxy in rounds with explicit ground evidence",
            "ground_seconds":
                "modeled ground-exposure proxy; not literal observed phase time",
        },
    }

    print("=" * 118)
    print(
        "EVENT MC V1 — STAGE 1 FIGHT-FLOW SCORE"
    )
    print("=" * 118)
    print(
        f"fights={len(x)} | "
        "PRIMARY METRICS ARE DURATION-NORMALIZED"
    )

    print(
        "\nFIGHT-LEVEL DISCRIMINATION"
    )

    print(
        f"{'metric':18s}"
        f"{'dir acc':>10s}"
        f"{'edge r':>10s}"
        f"{'edge rho':>11s}"
        f"{'side r':>10s}"
        f"{'side rho':>11s}"
        f"{'edge MAE':>11s}"
    )

    for name, m in flow.items():
        print(
            f"{name:18s}"
            f"{fmt(m['direction_accuracy']):>10s}"
            f"{fmt(m['edge_pearson']):>10s}"
            f"{fmt(m['edge_spearman']):>11s}"
            f"{fmt(m['fighter_side_pearson']):>10s}"
            f"{fmt(m['fighter_side_spearman']):>11s}"
            f"{fmt(m['normalized_edge_mae']):>11s}"
        )

    print(
        "\nEDGE MAGNITUDE — NORMALIZED"
    )

    print(
        f"{'metric':18s}"
        f"{'hist |edge|':>14s}"
        f"{'MC |edge|':>14s}"
        f"{'resolution':>12s}"
    )

    for name, m in flow.items():
        print(
            f"{name:18s}"
            f"{fmt(m['historical_mean_abs_edge']):>14s}"
            f"{fmt(m['simulated_mean_abs_edge']):>14s}"
            f"{fmt(m['model_resolution_rate']):>12s}"
        )

    print(
        "\nPOPULATION CALIBRATION — SECONDARY"
    )

    print(
        f"{'metric':18s}"
        f"{'historical':>14s}"
        f"{'MC':>14s}"
        f"{'ratio':>10s}"
    )

    for name, m in flow.items():
        print(
            f"{name:18s}"
            f"{fmt(m['historical_raw_total_per_fight']):>14s}"
            f"{fmt(m['simulated_raw_total_per_fight']):>14s}"
            f"{fmt(m['population_ratio']):>10s}"
        )

    print(
        "\nGROUND PHASE"
    )

    print(
        "ground seconds/fight  "
        f"hist={phase['historical_mean_ground_seconds']:.1f}  "
        f"MC={phase['simulated_mean_ground_seconds']:.1f}  "
        f"ratio={phase['ground_seconds_population_ratio']:.3f}"
    )

    print(
        "ground share          "
        f"hist={phase['historical_mean_ground_share']:.3f}  "
        f"MC={phase['simulated_mean_ground_share']:.3f}  "
        f"MAE={phase['ground_share_mae']:.3f}"
    )

    print(
        "ground-share discrimination  "
        f"Pearson={fmt(phase['ground_share_pearson'])}  "
        f"Spearman={fmt(phase['ground_share_spearman'])}"
    )

    print(
        "\nTAKEDOWN COMPLETION"
    )

    print(
        "pooled success  "
        f"hist={td_success['historical_pooled_success']:.3f}  "
        f"MC={td_success['simulated_pooled_success']:.3f}"
    )

    print(
        "fighter success discrimination  "
        f"Pearson={fmt(td_success['fighter_success_pearson'])}  "
        f"Spearman={fmt(td_success['fighter_success_spearman'])}  "
        f"MAE={fmt(td_success['fighter_success_mae'])}"
    )

    summary = (
        args.summary
        or args.csv.with_name(
            args.csv.stem
            + "_stage1_score.json"
        )
    )

    joined = (
        args.joined
        or args.csv.with_name(
            args.csv.stem
            + "_stage1_joined.csv"
        )
    )

    summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joined.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    x.to_csv(
        joined,
        index=False,
    )

    print(
        f"\nwrote summary: {summary}"
    )
    print(
        f"wrote joined : {joined}"
    )


if __name__ == "__main__":
    main()
