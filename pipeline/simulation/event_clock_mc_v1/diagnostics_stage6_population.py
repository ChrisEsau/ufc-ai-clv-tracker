from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    metrics,
    within_bout_direction,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import (
    direct_feature_columns,
    estimate_nb_alpha,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    build_pair_frame,
    fit_control_models,
    fit_count_hurdle,
)

from pipeline.simulation.event_clock_mc_v1.stage6_time_competition import (
    fit_ownership_concentration,
    simulate_path,
)


FIGHTS = 500
PATHS = 20
SEED = 20260817

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage6_population_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage6_population_paths_500x20.csv"
)


def distribution_line(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            values.mean()
        ),
        "std": float(
            values.std(ddof=1)
        ),
        "zero": float(
            np.mean(
                values == 0
            )
        ),
        "p10": float(
            np.quantile(
                values,
                .10,
            )
        ),
        "p50": float(
            np.quantile(
                values,
                .50,
            )
        ),
        "p90": float(
            np.quantile(
                values,
                .90,
            )
        ),
        "p99": float(
            np.quantile(
                values,
                .99,
            )
        ),
    }


def print_distribution(
    label,
    historical,
    simulated,
):
    h = distribution_line(
        historical
    )

    s = distribution_line(
        simulated
    )

    print()
    print(label)
    print("-" * 140)

    print(
        "HIST | "
        f"mean={h['mean']:.3f} | "
        f"std={h['std']:.3f} | "
        f"zero={h['zero']:.2%} | "
        f"p10={h['p10']:.2f} | "
        f"p50={h['p50']:.2f} | "
        f"p90={h['p90']:.2f} | "
        f"p99={h['p99']:.2f}"
    )

    print(
        "SIM  | "
        f"mean={s['mean']:.3f} | "
        f"std={s['std']:.3f} | "
        f"zero={s['zero']:.2%} | "
        f"p10={s['p10']:.2f} | "
        f"p50={s['p50']:.2f} | "
        f"p90={s['p90']:.2f} | "
        f"p99={s['p99']:.2f}"
    )


def build_path_pair(
    paths,
):
    rows = []

    for (
        fight_id,
        path,
    ), group in paths.groupby(
        [
            "fight_id",
            "path",
        ],
        sort=False,
    ):
        red = (
            group[
                group["side"]
                == "red"
            ]
            .iloc[0]
        )

        blue = (
            group[
                group["side"]
                == "blue"
            ]
            .iloc[0]
        )

        duration = float(
            red["duration"]
        )

        red_control = float(
            red[
                "sim_qualified_control_inflicted_seconds"
            ]
        )

        blue_control = float(
            blue[
                "sim_qualified_control_inflicted_seconds"
            ]
        )

        total_control = (
            red_control
            + blue_control
        )

        red_standing = float(
            red[
                "sim_standing_attempted"
            ]
        )

        blue_standing = float(
            blue[
                "sim_standing_attempted"
            ]
        )

        total_standing = (
            red_standing
            + blue_standing
        )

        red_td_landed = float(
            red[
                "sim_td_landed"
            ]
        )

        blue_td_landed = float(
            blue[
                "sim_td_landed"
            ]
        )

        rows.append(
            {
                "fight_id":
                    str(fight_id),

                "path":
                    int(path),

                "duration":
                    duration,

                "red_control":
                    red_control,

                "blue_control":
                    blue_control,

                "total_control":
                    total_control,

                "control_share":
                    (
                        total_control
                        / duration
                    ),

                "red_standing":
                    red_standing,

                "blue_standing":
                    blue_standing,

                "total_standing":
                    total_standing,

                "standing_per_15m":
                    (
                        total_standing
                        / duration
                        * 900.0
                    ),

                "red_td_landed":
                    red_td_landed,

                "blue_td_landed":
                    blue_td_landed,

                "td_landed_diff":
                    (
                        red_td_landed
                        - blue_td_landed
                    ),

                "red_control_share":
                    (
                        red_control
                        / total_control
                        if total_control > 0
                        else np.nan
                    ),

                "target_control":
                    float(
                        red[
                            "target_control"
                        ]
                    ),

                "realized_control":
                    float(
                        red[
                            "realized_control"
                        ]
                    ),

                "free_seconds":
                    float(
                        red[
                            "free_seconds"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def historical_fight_frame(
    test,
):
    rows = []

    for fight_id, group in test.groupby(
        "fight_id",
        sort=False,
    ):
        if len(group) != 2:
            continue

        red = (
            group[
                group["side"]
                == "red"
            ]
            .iloc[0]
        )

        blue = (
            group[
                group["side"]
                == "blue"
            ]
            .iloc[0]
        )

        duration = float(
            red["duration"]
        )

        red_control = float(
            red[
                "qualified_control_inflicted_seconds"
            ]
        )

        blue_control = float(
            blue[
                "qualified_control_inflicted_seconds"
            ]
        )

        total_control = (
            red_control
            + blue_control
        )

        red_standing = float(
            red[
                "standing_attempted"
            ]
        )

        blue_standing = float(
            blue[
                "standing_attempted"
            ]
        )

        total_standing = (
            red_standing
            + blue_standing
        )

        red_td_landed = float(
            red[
                "td_landed"
            ]
        )

        blue_td_landed = float(
            blue[
                "td_landed"
            ]
        )

        rows.append(
            {
                "fight_id":
                    str(fight_id),

                "duration":
                    duration,

                "red_control":
                    red_control,

                "blue_control":
                    blue_control,

                "total_control":
                    total_control,

                "control_share":
                    (
                        total_control
                        / duration
                    ),

                "red_standing":
                    red_standing,

                "blue_standing":
                    blue_standing,

                "total_standing":
                    total_standing,

                "standing_per_15m":
                    (
                        total_standing
                        / duration
                        * 900.0
                    ),

                "red_td_landed":
                    red_td_landed,

                "blue_td_landed":
                    blue_td_landed,

                "td_landed_diff":
                    (
                        red_td_landed
                        - blue_td_landed
                    ),

                "red_control_share":
                    (
                        red_control
                        / total_control
                        if total_control > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def spearman(
    frame,
    a,
    b,
):
    subset = (
        frame[
            [
                a,
                b,
            ]
        ]
        .dropna()
    )

    if len(subset) < 2:
        return np.nan

    return float(
        subset.corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )


def same_td_control_side(
    frame,
):
    subset = (
        frame[
            frame[
                "total_control"
            ] > 0
        ]
        .copy()
    )

    control_diff = (
        subset[
            "red_control"
        ]
        - subset[
            "blue_control"
        ]
    )

    keep = (
        (subset[
            "td_landed_diff"
        ] != 0)
        &
        (control_diff != 0)
    )

    subset = subset[
        keep
    ].copy()

    if len(subset) == 0:
        return np.nan, 0

    correct = (
        np.sign(
            subset[
                "td_landed_diff"
            ]
        )
        ==
        np.sign(
            subset[
                "red_control"
            ]
            - subset[
                "blue_control"
            ]
        )
    )

    return (
        float(
            correct.mean()
        ),
        len(subset),
    )


def quartile_interaction(
    frame,
    label,
):
    working = (
        frame[
            [
                "control_share",
                "standing_per_15m",
            ]
        ]
        .dropna()
        .copy()
    )

    working[
        "quartile"
    ] = pd.qcut(
        working[
            "control_share"
        ],
        q=4,
        duplicates="drop",
    )

    grouped = (
        working.groupby(
            "quartile",
            observed=False,
        )
        .agg(
            rows=(
                "standing_per_15m",
                "size",
            ),
            mean_control_share=(
                "control_share",
                "mean",
            ),
            mean_standing_per_15m=(
                "standing_per_15m",
                "mean",
            ),
        )
    )

    print()
    print(label)
    print(
        grouped.to_string(
            float_format=lambda x:
                f"{x:.3f}"
        )
    )


def main():

    print("=" * 140)
    print(
        "EVENT CLOCK MC — "
        "STAGE 6 POPULATION AUDIT"
    )
    print("=" * 140)

    train, test = (
        prepare_direct_predictions()
    )

    if (
        test[
            "fight_id"
        ].nunique()
        != FIGHTS
    ):
        raise RuntimeError(
            "Expected exactly "
            f"{FIGHTS} fresh fights"
        )

    # ---------------------------------------------------------
    # Derived standing fields.
    # ---------------------------------------------------------

    for frame in (
        train,
        test,
    ):
        frame[
            "standing_attempted"
        ] = (
            frame[
                "distance_attempted"
            ]
            + frame[
                "clinch_attempted"
            ]
        )

        frame[
            "standing_landed"
        ] = (
            frame[
                "distance_landed"
            ]
            + frame[
                "clinch_landed"
            ]
        )

        frame[
            "pred_standing_attempted"
        ] = (
            frame[
                "pred_distance_attempted"
            ]
            + frame[
                "pred_clinch_attempted"
            ]
        )

        frame[
            "pred_standing_landed"
        ] = (
            frame[
                "pred_distance_landed"
            ]
            + frame[
                "pred_clinch_landed"
            ]
        )

    feature_cols = (
        direct_feature_columns()
    )

    # ---------------------------------------------------------
    # Hurdle calibration.
    # ---------------------------------------------------------

    hurdle_extra_alpha = {}

    for family in (
        "td",
        "ground",
    ):
        hurdle_extra_alpha[
            family
        ] = fit_count_hurdle(
            train,
            test,
            family,
            feature_cols,
        )

    # ---------------------------------------------------------
    # Distance / clinch path variance.
    # ---------------------------------------------------------

    always_alpha = {}

    for family in (
        "distance",
        "clinch",
    ):
        always_alpha[
            family
        ] = estimate_nb_alpha(
            train[
                f"{family}_attempted"
            ],
            train[
                f"pred_{family}_attempted"
            ],
        )

    # ---------------------------------------------------------
    # Shared competitive control.
    # ---------------------------------------------------------

    train_pair = (
        build_pair_frame(
            train
        )
    )

    test_pair = (
        build_pair_frame(
            test
        )
    )

    (
        td_control_beta,
        control_alpha,
    ) = fit_control_models(
        train_pair,
        test_pair,
    )

    ownership_kappa = (
        fit_ownership_concentration(
            train_pair
        )
    )

    pair_lookup = {
        str(
            row[
                "fight_id"
            ]
        ): row
        for _, row
        in test_pair.iterrows()
    }

    # ---------------------------------------------------------
    # Run 500 x 20.
    # ---------------------------------------------------------

    print()
    print("=" * 140)
    print(
        f"RUNNING STAGE 6 — "
        f"{FIGHTS} fights x "
        f"{PATHS} paths"
    )
    print("=" * 140)

    path_rows = []

    groups = list(
        test.groupby(
            "fight_id",
            sort=False,
        )
    )

    for fight_index, (
        fight_id,
        pair,
    ) in enumerate(
        groups
    ):
        pair_info = (
            pair_lookup[
                str(fight_id)
            ]
        )

        for path_index in range(
            PATHS
        ):
            rng = (
                np.random.default_rng(
                    SEED
                    + fight_index
                    * 100000
                    + path_index
                )
            )

            output = simulate_path(
                pair,
                pair_info,
                rng,
                always_alpha,
                hurdle_extra_alpha,
                control_alpha,
                td_control_beta,
                ownership_kappa,
            )

            for _, fighter in (
                pair.iterrows()
            ):
                side = (
                    fighter[
                        "side"
                    ]
                )

                row = {
                    "fight_id":
                        str(
                            fight_id
                        ),

                    "path":
                        path_index,

                    "side":
                        side,

                    "fighter_name":
                        fighter[
                            "fighter_name"
                        ],

                    "duration":
                        float(
                            fighter[
                                "duration"
                            ]
                        ),

                    "target_control":
                        float(
                            output[
                                "target_control_seconds"
                            ]
                        ),

                    "realized_control":
                        float(
                            output[
                                "realized_control_seconds"
                            ]
                        ),

                    "free_seconds":
                        float(
                            output[
                                "free_seconds"
                            ]
                        ),
                }

                for family in (
                    "distance",
                    "clinch",
                    "ground",
                    "td",
                ):
                    row[
                        f"sim_{family}_attempted"
                    ] = float(
                        output[
                            f"{side}_{family}_attempted"
                        ]
                    )

                    row[
                        f"sim_{family}_landed"
                    ] = float(
                        output[
                            f"{side}_{family}_landed"
                        ]
                    )

                row[
                    "sim_standing_attempted"
                ] = (
                    row[
                        "sim_distance_attempted"
                    ]
                    + row[
                        "sim_clinch_attempted"
                    ]
                )

                row[
                    "sim_standing_landed"
                ] = (
                    row[
                        "sim_distance_landed"
                    ]
                    + row[
                        "sim_clinch_landed"
                    ]
                )

                row[
                    "sim_qualified_control_inflicted_seconds"
                ] = float(
                    output[
                        f"{side}_qualified_control_inflicted_seconds"
                    ]
                )

                path_rows.append(
                    row
                )

        if (
            (fight_index + 1)
            % 50
            == 0
        ):
            print(
                f"completed "
                f"{fight_index + 1}/"
                f"{FIGHTS}"
            )

    paths = pd.DataFrame(
        path_rows
    )

    # ---------------------------------------------------------
    # Fighter-level MC means.
    # ---------------------------------------------------------

    aggregation = {
        "sim_standing_attempted":
            "mean",
        "sim_standing_landed":
            "mean",
        "sim_distance_attempted":
            "mean",
        "sim_distance_landed":
            "mean",
        "sim_clinch_attempted":
            "mean",
        "sim_clinch_landed":
            "mean",
        "sim_td_attempted":
            "mean",
        "sim_td_landed":
            "mean",
        "sim_ground_attempted":
            "mean",
        "sim_ground_landed":
            "mean",
        "sim_qualified_control_inflicted_seconds":
            "mean",
    }

    sim_mean = (
        paths.groupby(
            [
                "fight_id",
                "side",
                "fighter_name",
            ],
            as_index=False,
        )
        .agg(
            aggregation
        )
    )

    result = test.merge(
        sim_mean,
        on=[
            "fight_id",
            "side",
            "fighter_name",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Fighter discrimination.
    # ---------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "FIGHTER-LEVEL "
        "DISCRIMINATION"
    )
    print("=" * 140)

    for target, label in (
        (
            "standing_attempted",
            "STANDING ATTEMPTS",
        ),
        (
            "standing_landed",
            "STANDING LANDED",
        ),
        (
            "td_attempted",
            "TD ATTEMPTS",
        ),
        (
            "td_landed",
            "TD LANDED",
        ),
        (
            "ground_attempted",
            "GROUND ATTEMPTS",
        ),
        (
            "ground_landed",
            "GROUND LANDED",
        ),
        (
            "qualified_control_inflicted_seconds",
            "CONTROL SEC",
        ),
    ):

        direct_col = (
            f"pred_{target}"
        )

        sim_col = (
            f"sim_{target}"
        )

        (
            _,
            direct_rho,
            direct_mae,
        ) = metrics(
            result[
                target
            ],
            result[
                direct_col
            ],
        )

        (
            _,
            sim_rho,
            sim_mae,
        ) = metrics(
            result[
                target
            ],
            result[
                sim_col
            ],
        )

        (
            direct_side,
            direct_n,
        ) = within_bout_direction(
            result,
            target,
            direct_col,
        )

        (
            sim_side,
            sim_n,
        ) = within_bout_direction(
            result,
            target,
            sim_col,
        )

        print()
        print(label)
        print("-" * 140)

        print(
            f"mean | "
            f"HIST={result[target].mean():.3f} | "
            f"DIRECT={result[direct_col].mean():.3f} | "
            f"STAGE6={result[sim_col].mean():.3f}"
        )

        print(
            f"DIRECT | "
            f"rho={direct_rho:+.4f} | "
            f"MAE={direct_mae:.3f} | "
            f"correct-side={direct_side:.2%} "
            f"(N={direct_n})"
        )

        print(
            f"STAGE6 | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f} | "
            f"correct-side={sim_side:.2%} "
            f"(N={sim_n})"
        )

    # ---------------------------------------------------------
    # Path distributions.
    # ---------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "PATH DISTRIBUTIONS"
    )
    print("=" * 140)

    for target, sim_col, label in (
        (
            "standing_attempted",
            "sim_standing_attempted",
            "STANDING ATTEMPTS",
        ),
        (
            "td_attempted",
            "sim_td_attempted",
            "TD ATTEMPTS",
        ),
        (
            "td_landed",
            "sim_td_landed",
            "TD LANDED",
        ),
        (
            "ground_attempted",
            "sim_ground_attempted",
            "GROUND ATTEMPTS",
        ),
        (
            "ground_landed",
            "sim_ground_landed",
            "GROUND LANDED",
        ),
        (
            "qualified_control_inflicted_seconds",
            "sim_qualified_control_inflicted_seconds",
            "CONTROL SEC",
        ),
    ):
        print_distribution(
            label,
            result[
                target
            ],
            paths[
                sim_col
            ],
        )

    # ---------------------------------------------------------
    # Fight-level time interaction.
    # ---------------------------------------------------------

    historical_fights = (
        historical_fight_frame(
            result
        )
    )

    simulated_paths = (
        build_path_pair(
            paths
        )
    )

    print()
    print("=" * 140)
    print(
        "CONTROL ↔ STANDING "
        "TIME-COMPETITION AUDIT"
    )
    print("=" * 140)

    historical_raw = spearman(
        historical_fights,
        "total_control",
        "total_standing",
    )

    simulated_raw = spearman(
        simulated_paths,
        "total_control",
        "total_standing",
    )

    historical_normalized = spearman(
        historical_fights,
        "control_share",
        "standing_per_15m",
    )

    simulated_normalized = spearman(
        simulated_paths,
        "control_share",
        "standing_per_15m",
    )

    print(
        f"Raw total-control vs "
        f"standing-attempt Spearman:"
    )

    print(
        f"  HIST   = "
        f"{historical_raw:+.4f}"
    )

    print(
        f"  STAGE6 = "
        f"{simulated_raw:+.4f}"
    )

    print()

    print(
        "Duration-normalized:"
    )

    print(
        f"control share vs "
        f"standing attempts/15m"
    )

    print(
        f"  HIST   = "
        f"{historical_normalized:+.4f}"
    )

    print(
        f"  STAGE6 = "
        f"{simulated_normalized:+.4f}"
    )

    quartile_interaction(
        historical_fights,
        "HISTORICAL CONTROL-SHARE QUARTILES",
    )

    quartile_interaction(
        simulated_paths,
        "STAGE6 CONTROL-SHARE QUARTILES",
    )

    # ---------------------------------------------------------
    # TD -> control interaction.
    # ---------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "TD ↔ CONTROL OWNERSHIP "
        "INTERACTION"
    )
    print("=" * 140)

    hist_td_rho = spearman(
        historical_fights[
            historical_fights[
                "total_control"
            ] > 0
        ],
        "td_landed_diff",
        "red_control_share",
    )

    sim_td_rho = spearman(
        simulated_paths[
            simulated_paths[
                "total_control"
            ] > 0
        ],
        "td_landed_diff",
        "red_control_share",
    )

    (
        hist_same,
        hist_n,
    ) = same_td_control_side(
        historical_fights
    )

    (
        sim_same,
        sim_n,
    ) = same_td_control_side(
        simulated_paths
    )

    print(
        f"TD landed differential vs "
        f"Red control share Spearman:"
    )

    print(
        f"  HIST   = "
        f"{hist_td_rho:+.4f}"
    )

    print(
        f"  STAGE6 = "
        f"{sim_td_rho:+.4f}"
    )

    print()

    print(
        "When TD landed totals differ, "
        "same fighter owns more control:"
    )

    print(
        f"  HIST   = "
        f"{hist_same:.2%} "
        f"(N={hist_n})"
    )

    print(
        f"  STAGE6 = "
        f"{sim_same:.2%} "
        f"(N={sim_n})"
    )

    # ---------------------------------------------------------
    # Control timing sanity.
    # ---------------------------------------------------------

    unique_path = (
        simulated_paths.copy()
    )

    target_error = (
        unique_path[
            "realized_control"
        ]
        - unique_path[
            "target_control"
        ]
    )

    print()
    print("=" * 140)
    print(
        "FINITE-TIMELINE SANITY"
    )
    print("=" * 140)

    print(
        f"Mean target control/path:   "
        f"{unique_path['target_control'].mean():.2f}s"
    )

    print(
        f"Mean realized control/path: "
        f"{unique_path['realized_control'].mean():.2f}s"
    )

    print(
        f"Mean target-realized error: "
        f"{target_error.mean():+.4f}s"
    )

    print(
        f"Max abs target-realized error: "
        f"{target_error.abs().max():.4f}s"
    )

    print(
        f"Mean free fight time/path:  "
        f"{unique_path['free_seconds'].mean():.2f}s"
    )

    violations = int(
        (
            unique_path[
                "realized_control"
            ]
            >
            unique_path[
                "duration"
            ]
            + 1e-9
        ).sum()
    )

    print(
        f"Control > fight duration:   "
        f"{violations}"
    )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT,
        index=False,
    )

    paths.to_csv(
        PATH_OUT,
        index=False,
    )

    print()
    print(
        f"wrote: {OUT}"
    )

    print(
        f"wrote: {PATH_OUT}"
    )


if __name__ == "__main__":
    main()
