from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

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


DEFAULT_PATHS = 5000
DEFAULT_SEED = 20260817

DISPLAY_STATS = (
    ("standing_attempted", "Standing attempts"),
    ("standing_landed", "Standing landed"),
    ("distance_attempted", "Distance attempts"),
    ("distance_landed", "Distance landed"),
    ("clinch_attempted", "Clinch attempts"),
    ("clinch_landed", "Clinch landed"),
    ("td_attempted", "TD attempts"),
    ("td_landed", "TD landed"),
    ("qualified_control_inflicted_seconds", "Control seconds"),
    ("ground_attempted", "Ground attempts"),
    ("ground_landed", "Ground landed"),
)

LEAD_STATS = (
    ("standing_attempted", "standing attempts"),
    ("standing_landed", "standing landed"),
    ("td_attempted", "TD attempts"),
    ("td_landed", "TD landed"),
    ("qualified_control_inflicted_seconds", "control"),
    ("ground_attempted", "ground attempts"),
    ("ground_landed", "ground landed"),
)


def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().casefold(),
    )


def find_fight(
    test: pd.DataFrame,
    fighter1: str | None,
    fighter2: str | None,
    fight_id: str | None,
):
    if fight_id:
        matched = test[
            test["fight_id"].astype(str)
            == str(fight_id)
        ]

        if matched["fight_id"].nunique() != 1:
            raise SystemExit(
                f"No unique fight found for fight_id={fight_id}"
            )

        if len(matched) != 2:
            raise SystemExit(
                f"{fight_id}: expected exactly two fighter rows"
            )

        return matched.copy()

    if not fighter1 or not fighter2:
        raise SystemExit(
            "Supply either --fight-id or both "
            "--fighter1 and --fighter2."
        )

    q1 = normalize(fighter1)
    q2 = normalize(fighter2)

    matches = []

    for fid, group in test.groupby(
        "fight_id",
        sort=False,
    ):
        names = (
            group["fighter_name"]
            .astype(str)
            .tolist()
        )

        hit1 = [
            name
            for name in names
            if q1 in normalize(name)
        ]

        hit2 = [
            name
            for name in names
            if q2 in normalize(name)
        ]

        if (
            hit1
            and hit2
            and hit1[0] != hit2[0]
        ):
            matches.append(
                group.copy()
            )

    if not matches:
        raise SystemExit(
            f"No fresh-500 matchup found matching "
            f"{fighter1!r} vs {fighter2!r}"
        )

    if len(matches) > 1:
        print()
        print("Multiple matching fights:")
        print()

        for group in matches:
            date = (
                pd.Timestamp(
                    group["event_date"].iloc[0]
                )
                .date()
            )

            names = (
                group["fighter_name"]
                .astype(str)
                .tolist()
            )

            print(
                f"{date} | "
                f"{names[0]} vs {names[1]} | "
                f"fight_id="
                f"{group['fight_id'].iloc[0]}"
            )

        raise SystemExit(
            "\nUse --fight-id to select one."
        )

    return matches[0]


def prepare_engine():

    print("=" * 120)
    print("BUILDING EVENT-CLOCK MATCHUP CONTEXT")
    print("=" * 120)

    train, test = (
        prepare_direct_predictions()
    )

    for frame in (
        train,
        test,
    ):
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

    return {
        "test": test,
        "test_pair": test_pair,
        "always_alpha": always_alpha,
        "hurdle_extra_alpha":
            hurdle_extra_alpha,
        "control_alpha":
            control_alpha,
        "td_control_beta":
            td_control_beta,
        "ownership_kappa":
            ownership_kappa,
    }


def simulate_matchup(
    pair,
    pair_info,
    paths,
    seed,
    context,
):

    rows = []

    for path_index in range(
        paths
    ):
        rng = np.random.default_rng(
            seed + path_index
        )

        output = simulate_path(
            pair,
            pair_info,
            rng,
            context[
                "always_alpha"
            ],
            context[
                "hurdle_extra_alpha"
            ],
            context[
                "control_alpha"
            ],
            context[
                "td_control_beta"
            ],
            context[
                "ownership_kappa"
            ],
        )

        for _, fighter in (
            pair.iterrows()
        ):
            side = fighter[
                "side"
            ]

            row = {
                "path":
                    path_index,
                "side":
                    side,
                "fighter_name":
                    fighter[
                        "fighter_name"
                    ],
            }

            for family in (
                "distance",
                "clinch",
                "td",
                "ground",
            ):
                row[
                    f"{family}_attempted"
                ] = output[
                    f"{side}_{family}_attempted"
                ]

                row[
                    f"{family}_landed"
                ] = output[
                    f"{side}_{family}_landed"
                ]

            row[
                "qualified_control_inflicted_seconds"
            ] = output[
                f"{side}_qualified_control_inflicted_seconds"
            ]

            row[
                "standing_attempted"
            ] = (
                row[
                    "distance_attempted"
                ]
                + row[
                    "clinch_attempted"
                ]
            )

            row[
                "standing_landed"
            ] = (
                row[
                    "distance_landed"
                ]
                + row[
                    "clinch_landed"
                ]
            )

            rows.append(
                row
            )

        if (
            paths >= 1000
            and (
                (path_index + 1)
                % max(
                    paths // 10,
                    1,
                )
                == 0
            )
        ):
            print(
                f"simulated "
                f"{path_index + 1:,}/"
                f"{paths:,} paths"
            )

    return pd.DataFrame(
        rows
    )


def print_prefight(
    pair,
    pair_info,
):

    print()
    print("=" * 120)
    print("PREFIGHT DIRECT EXPECTATIONS")
    print("=" * 120)

    for side in (
        "red",
        "blue",
    ):
        fighter = (
            pair[
                pair["side"]
                == side
            ]
            .iloc[0]
        )

        print()
        print(
            f"{side.upper():4} — "
            f"{fighter['fighter_name']}"
        )

        print(
            f"Standing attempts: "
            f"{fighter['pred_standing_attempted']:.2f}"
        )

        print(
            f"Standing landed:   "
            f"{fighter['pred_standing_landed']:.2f}"
        )

        print(
            f"TD attempts:       "
            f"{fighter['pred_td_attempted']:.2f}"
        )

        print(
            f"TD landed:         "
            f"{fighter['pred_td_landed']:.2f}"
        )

        print(
            f"TD active path:    "
            f"{fighter['pred_td_positive_probability']:.1%}"
        )

        print(
            f"Ground attempts:   "
            f"{fighter['pred_ground_attempted']:.2f}"
        )

        print(
            f"Ground active path:"
            f" {fighter['pred_ground_positive_probability']:.1%}"
        )

        print(
            f"Control:           "
            f"{fighter['pred_qualified_control_inflicted_seconds']:.1f}s"
        )

    print()
    print(
        f"Any control in fight: "
        f"{pair_info['pred_control_any_probability']:.1%}"
    )

    print(
        f"Prefight Red control ownership: "
        f"{pair_info['pred_red_control_share']:.1%}"
    )


def print_distribution_table(
    pair,
    paths_df,
):

    print()
    print("=" * 150)
    print("SIMULATED FIGHTER OUTPUT DISTRIBUTIONS")
    print("=" * 150)

    for side in (
        "red",
        "blue",
    ):
        fighter = (
            pair[
                pair["side"]
                == side
            ]
            .iloc[0]
        )

        subset = paths_df[
            paths_df["side"]
            == side
        ]

        print()
        print(
            f"{side.upper()} — "
            f"{fighter['fighter_name']}"
        )

        print(
            f"{'STAT':28}"
            f"{'MEAN':>10}"
            f"{'P10':>10}"
            f"{'P50':>10}"
            f"{'P90':>10}"
            f"{'ZERO':>10}"
        )

        print("-" * 78)

        for stat, label in (
            DISPLAY_STATS
        ):
            values = (
                subset[
                    stat
                ]
                .to_numpy(float)
            )

            zero = (
                np.mean(
                    values == 0
                )
            )

            print(
                f"{label:28}"
                f"{values.mean():>10.2f}"
                f"{np.quantile(values, .10):>10.2f}"
                f"{np.quantile(values, .50):>10.2f}"
                f"{np.quantile(values, .90):>10.2f}"
                f"{zero:>9.1%}"
            )


def print_lead_probabilities(
    pair,
    paths_df,
):

    red_name = (
        pair[
            pair["side"] == "red"
        ][
            "fighter_name"
        ]
        .iloc[0]
    )

    blue_name = (
        pair[
            pair["side"] == "blue"
        ][
            "fighter_name"
        ]
        .iloc[0]
    )

    red = (
        paths_df[
            paths_df["side"]
            == "red"
        ]
        .sort_values("path")
        .reset_index(drop=True)
    )

    blue = (
        paths_df[
            paths_df["side"]
            == "blue"
        ]
        .sort_values("path")
        .reset_index(drop=True)
    )

    print()
    print("=" * 120)
    print("MATCHUP STAT-LEAD PROBABILITIES")
    print("=" * 120)

    print()
    print(
        f"{'STAT':24}"
        f"{red_name[:24]:>26}"
        f"{'TIE':>12}"
        f"{blue_name[:24]:>26}"
    )

    print("-" * 88)

    for stat, label in (
        LEAD_STATS
    ):
        rv = (
            red[stat]
            .to_numpy(float)
        )

        bv = (
            blue[stat]
            .to_numpy(float)
        )

        red_lead = np.mean(
            rv > bv
        )

        blue_lead = np.mean(
            bv > rv
        )

        tie = np.mean(
            rv == bv
        )

        print(
            f"{label:24}"
            f"{red_lead:>25.1%}"
            f"{tie:>12.1%}"
            f"{blue_lead:>25.1%}"
        )



def print_joint_coherence(
    pair,
    paths_df,
):
    red = (
        paths_df[
            paths_df["side"]
            == "red"
        ]
        .sort_values("path")
        .reset_index(drop=True)
    )

    blue = (
        paths_df[
            paths_df["side"]
            == "blue"
        ]
        .sort_values("path")
        .reset_index(drop=True)
    )

    total_control = (
        red[
            "qualified_control_inflicted_seconds"
        ].to_numpy(float)
        + blue[
            "qualified_control_inflicted_seconds"
        ].to_numpy(float)
    )

    total_standing = (
        red[
            "standing_attempted"
        ].to_numpy(float)
        + blue[
            "standing_attempted"
        ].to_numpy(float)
    )

    frame = pd.DataFrame(
        {
            "control":
                total_control,
            "standing":
                total_standing,
        }
    )

    rho = (
        frame.corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )

    actual_control = float(
        pair[
            "qualified_control_inflicted_seconds"
        ].sum()
    )

    actual_standing = float(
        pair[
            "distance_attempted"
        ].sum()
        + pair[
            "clinch_attempted"
        ].sum()
    )

    print()
    print("=" * 120)
    print(
        "JOINT FIGHT COHERENCE"
    )
    print("=" * 120)

    print(
        f"Total-control vs total-standing "
        f"Spearman: {rho:+.4f}"
    )

    print()
    print(
        f"Historical total control: "
        f"{actual_control:.0f}s"
    )

    print(
        f"Historical total standing attempts: "
        f"{actual_standing:.0f}"
    )

    print()
    print(
        "Standing attempts by simulated "
        "total-control quartile:"
    )

    bins = pd.qcut(
        frame["control"],
        q=4,
        duplicates="drop",
    )

    grouped = (
        frame.assign(
            control_bin=bins
        )
        .groupby(
            "control_bin",
            observed=False,
        )
        .agg(
            paths=(
                "standing",
                "size",
            ),
            mean_control=(
                "control",
                "mean",
            ),
            mean_standing=(
                "standing",
                "mean",
            ),
        )
    )

    print(
        grouped.to_string(
            float_format=lambda x:
                f"{x:.1f}"
        )
    )

    extreme = (
        frame[
            frame["control"]
            >= actual_control
        ]
    )

    print()

    if len(extreme):

        print(
            f"Paths with >= historical "
            f"control ({actual_control:.0f}s): "
            f"{len(extreme):,} "
            f"({len(extreme) / len(frame):.2%})"
        )

        print(
            f"Mean standing attempts in those paths: "
            f"{extreme['standing'].mean():.1f}"
        )

        print(
            f"P(standing <= historical {actual_standing:.0f} "
            f"| control >= historical): "
            f"{(extreme['standing'] <= actual_standing).mean():.2%}"
        )

    else:

        print(
            "No simulated path reached the "
            "historical control total."
        )


def print_historical(
    pair,
):

    print()
    print("=" * 120)
    print(
        "HISTORICAL OBSERVED STATS "
        "(NOT USED TO GENERATE THESE PATHS)"
    )
    print("=" * 120)

    for side in (
        "red",
        "blue",
    ):
        fighter = (
            pair[
                pair["side"]
                == side
            ]
            .iloc[0]
        )

        print()
        print(
            f"{side.upper()} — "
            f"{fighter['fighter_name']}"
        )

        standing_att = (
            float(
                fighter[
                    "distance_attempted"
                ]
            )
            + float(
                fighter[
                    "clinch_attempted"
                ]
            )
        )

        standing_lnd = (
            float(
                fighter[
                    "distance_landed"
                ]
            )
            + float(
                fighter[
                    "clinch_landed"
                ]
            )
        )

        print(
            f"Standing: "
            f"{standing_lnd:.0f}/"
            f"{standing_att:.0f}"
        )

        print(
            f"TD:       "
            f"{fighter['td_landed']:.0f}/"
            f"{fighter['td_attempted']:.0f}"
        )

        print(
            f"Control:  "
            f"{fighter['qualified_control_inflicted_seconds']:.0f}s"
        )

        print(
            f"Ground:   "
            f"{fighter['ground_landed']:.0f}/"
            f"{fighter['ground_attempted']:.0f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fighter1",
        type=str,
    )

    parser.add_argument(
        "--fighter2",
        type=str,
    )

    parser.add_argument(
        "--fight-id",
        type=str,
    )

    parser.add_argument(
        "--paths",
        type=int,
        default=DEFAULT_PATHS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()

    context = (
        prepare_engine()
    )

    pair = find_fight(
        context["test"],
        args.fighter1,
        args.fighter2,
        args.fight_id,
    )

    fight_id = str(
        pair[
            "fight_id"
        ].iloc[0]
    )

    date = (
        pd.Timestamp(
            pair[
                "event_date"
            ].iloc[0]
        )
        .date()
    )

    red = (
        pair[
            pair["side"]
            == "red"
        ]
        .iloc[0]
    )

    blue = (
        pair[
            pair["side"]
            == "blue"
        ]
        .iloc[0]
    )

    pair_info = (
        context[
            "test_pair"
        ][
            context[
                "test_pair"
            ][
                "fight_id"
            ].astype(str)
            == fight_id
        ]
        .iloc[0]
    )

    print()
    print("=" * 120)
    print("EVENT CLOCK MC — SINGLE FIGHT")
    print("=" * 120)

    print(
        f"fight_id: {fight_id}"
    )

    print(
        f"date:     {date}"
    )

    print(
        f"RED:      "
        f"{red['fighter_name']}"
    )

    print(
        f"BLUE:     "
        f"{blue['fighter_name']}"
    )

    print(
        f"paths:    "
        f"{args.paths:,}"
    )

    print_prefight(
        pair,
        pair_info,
    )

    print()
    print("=" * 120)
    print("RUNNING PATHS")
    print("=" * 120)

    paths_df = simulate_matchup(
        pair,
        pair_info,
        args.paths,
        args.seed,
        context,
    )

    print_distribution_table(
        pair,
        paths_df,
    )

    print_lead_probabilities(
        pair,
        paths_df,
    )

    print_joint_coherence(
        pair,
        paths_df,
    )

    print_historical(
        pair,
    )

    output = Path(
        "data/diagnostics/"
        "event_clock_mc_v1/"
        f"single_fight_{fight_id}_"
        f"{args.paths}paths.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths_df.to_csv(
        output,
        index=False,
    )

    print()
    print(
        f"wrote: {output}"
    )


if __name__ == "__main__":
    main()
