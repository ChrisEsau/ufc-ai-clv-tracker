from pathlib import Path

import numpy as np
import pandas as pd


STAGE9 = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage9_final_flow_500x20.csv"
)

STAGE10 = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10_decision_judge_fresh.csv"
)

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10_decision_miss_audit.csv"
)


def winner_edge(
    red_value,
    blue_value,
    red_win,
):
    """
    Positive = the ACTUAL winner led this statistic.
    Negative = the actual loser led it.
    """
    diff = (
        float(red_value)
        - float(blue_value)
    )

    return (
        diff
        if int(red_win) == 1
        else -diff
    )


def build_fight_rows(
    stage9,
    stage10,
):
    rows = []

    for _, fight in (
        stage10.iterrows()
    ):
        fight_id = str(
            fight["fight_id"]
        )

        pair = stage9[
            stage9["fight_id"]
            == fight_id
        ]

        if len(pair) != 2:
            continue

        red = pair[
            pair["side"] == "red"
        ].iloc[0]

        blue = pair[
            pair["side"] == "blue"
        ].iloc[0]

        red_win = int(
            fight["red_win"]
        )

        oracle_pick_red = (
            float(
                fight[
                    "oracle_p_red"
                ]
            )
            >= 0.5
        )

        expected_pick_red = (
            float(
                fight[
                    "expected_p_red"
                ]
            )
            >= 0.5
        )

        mc_pick_red = (
            float(
                fight[
                    "mc_p_red"
                ]
            )
            >= 0.5
        )

        oracle_correct = (
            int(
                oracle_pick_red
            )
            == red_win
        )

        expected_correct = (
            int(
                expected_pick_red
            )
            == red_win
        )

        mc_correct = (
            int(
                mc_pick_red
            )
            == red_win
        )

        # ---------------------------------------------------------
        # Stage-9 expected standing landed.
        # ---------------------------------------------------------

        red_standing_land_rate = (
            float(
                red[
                    "pred_standing_landed"
                ]
            )
            / max(
                float(
                    red[
                        "pred_standing_attempted"
                    ]
                ),
                1e-9,
            )
        )

        blue_standing_land_rate = (
            float(
                blue[
                    "pred_standing_landed"
                ]
            )
            / max(
                float(
                    blue[
                        "pred_standing_attempted"
                    ]
                ),
                1e-9,
            )
        )

        red_expected_standing_landed = (
            float(
                red[
                    "pred_stage9_standing_attempted"
                ]
            )
            * red_standing_land_rate
        )

        blue_expected_standing_landed = (
            float(
                blue[
                    "pred_stage9_standing_attempted"
                ]
            )
            * blue_standing_land_rate
        )

        # ---------------------------------------------------------
        # Actual winner-relative edges.
        # ---------------------------------------------------------

        actual_standing = winner_edge(
            red["standing_landed"],
            blue["standing_landed"],
            red_win,
        )

        actual_ground = winner_edge(
            red["ground_landed"],
            blue["ground_landed"],
            red_win,
        )

        actual_sig = (
            actual_standing
            + actual_ground
        )

        actual_td = winner_edge(
            red["td_landed"],
            blue["td_landed"],
            red_win,
        )

        actual_control = winner_edge(
            red[
                "qualified_control_inflicted_seconds"
            ],
            blue[
                "qualified_control_inflicted_seconds"
            ],
            red_win,
        )

        # ---------------------------------------------------------
        # Prefight expected winner-relative edges.
        # ---------------------------------------------------------

        expected_standing = winner_edge(
            red_expected_standing_landed,
            blue_expected_standing_landed,
            red_win,
        )

        expected_ground = winner_edge(
            red["pred_ground_landed"],
            blue["pred_ground_landed"],
            red_win,
        )

        expected_sig = (
            expected_standing
            + expected_ground
        )

        expected_td = winner_edge(
            red["pred_td_landed"],
            blue["pred_td_landed"],
            red_win,
        )

        expected_control = winner_edge(
            red[
                "pred_qualified_control_inflicted_seconds"
            ],
            blue[
                "pred_qualified_control_inflicted_seconds"
            ],
            red_win,
        )

        # ---------------------------------------------------------
        # Stage-9 MC mean winner-relative edges.
        # ---------------------------------------------------------

        sim_standing = winner_edge(
            red["sim_standing_landed"],
            blue["sim_standing_landed"],
            red_win,
        )

        sim_ground = winner_edge(
            red["sim_ground_landed"],
            blue["sim_ground_landed"],
            red_win,
        )

        sim_sig = (
            sim_standing
            + sim_ground
        )

        sim_td = winner_edge(
            red["sim_td_landed"],
            blue["sim_td_landed"],
            red_win,
        )

        sim_control = winner_edge(
            red["sim_control"],
            blue["sim_control"],
            red_win,
        )

        if not mc_correct:
            if oracle_correct:
                miss_type = (
                    "SIMULATION_MISS"
                )
            else:
                miss_type = (
                    "JUDGE_OR_ROUND_MISS"
                )
        else:
            miss_type = (
                "MC_CORRECT"
            )

        rows.append(
            {
                "fight_id":
                    fight_id,

                "red_name":
                    fight[
                        "red_name"
                    ],

                "blue_name":
                    fight[
                        "blue_name"
                    ],

                "winner_name":
                    fight[
                        "winner_name"
                    ],

                "method":
                    fight[
                        "method"
                    ],

                "miss_type":
                    miss_type,

                "oracle_correct":
                    oracle_correct,

                "expected_correct":
                    expected_correct,

                "mc_correct":
                    mc_correct,

                "oracle_p_red":
                    float(
                        fight[
                            "oracle_p_red"
                        ]
                    ),

                "expected_p_red":
                    float(
                        fight[
                            "expected_p_red"
                        ]
                    ),

                "mc_p_red":
                    float(
                        fight[
                            "mc_p_red"
                        ]
                    ),

                "mc_confidence":
                    max(
                        float(
                            fight[
                                "mc_p_red"
                            ]
                        ),
                        1.0
                        - float(
                            fight[
                                "mc_p_red"
                            ]
                        ),
                    ),

                # Actual winner-relative fight stats.
                "actual_winner_sig_edge":
                    actual_sig,

                "actual_winner_standing_edge":
                    actual_standing,

                "actual_winner_ground_edge":
                    actual_ground,

                "actual_winner_td_edge":
                    actual_td,

                "actual_winner_control_edge":
                    actual_control,

                # Expected winner-relative fight stats.
                "expected_winner_sig_edge":
                    expected_sig,

                "expected_winner_standing_edge":
                    expected_standing,

                "expected_winner_ground_edge":
                    expected_ground,

                "expected_winner_td_edge":
                    expected_td,

                "expected_winner_control_edge":
                    expected_control,

                # Simulated winner-relative fight stats.
                "sim_winner_sig_edge":
                    sim_sig,

                "sim_winner_standing_edge":
                    sim_standing,

                "sim_winner_ground_edge":
                    sim_ground,

                "sim_winner_td_edge":
                    sim_td,

                "sim_winner_control_edge":
                    sim_control,
            }
        )

    return pd.DataFrame(
        rows
    )


def print_edge_summary(
    frame,
    label,
):
    print()
    print("=" * 140)
    print(label)
    print("=" * 140)

    print(
        f"fights: {len(frame)}"
    )

    if len(frame) == 0:
        return

    for family in (
        "sig",
        "standing",
        "ground",
        "td",
        "control",
    ):
        actual = (
            frame[
                f"actual_winner_{family}_edge"
            ]
        )

        expected = (
            frame[
                f"expected_winner_{family}_edge"
            ]
        )

        simulated = (
            frame[
                f"sim_winner_{family}_edge"
            ]
        )

        print()
        print(
            family.upper()
        )

        print(
            f"actual winner led:    "
            f"{(actual > 0).mean():.2%}"
        )

        print(
            f"expected winner led:  "
            f"{(expected > 0).mean():.2%}"
        )

        print(
            f"sim winner led:       "
            f"{(simulated > 0).mean():.2%}"
        )

        print(
            f"mean actual edge:      "
            f"{actual.mean():+.2f}"
        )

        print(
            f"mean expected edge:    "
            f"{expected.mean():+.2f}"
        )

        print(
            f"mean simulated edge:   "
            f"{simulated.mean():+.2f}"
        )


def main():

    print("=" * 140)
    print(
        "STAGE 10 — DECISION MISS AUDIT"
    )
    print("=" * 140)

    stage9 = pd.read_csv(
        STAGE9,
        low_memory=False,
    )

    stage10 = pd.read_csv(
        STAGE10,
        low_memory=False,
    )

    stage9[
        "fight_id"
    ] = (
        stage9[
            "fight_id"
        ].astype(str)
    )

    stage10[
        "fight_id"
    ] = (
        stage10[
            "fight_id"
        ].astype(str)
    )

    audit = build_fight_rows(
        stage9,
        stage10,
    )

    misses = audit[
        ~audit[
            "mc_correct"
        ]
    ].copy()

    simulation_misses = misses[
        misses[
            "miss_type"
        ]
        == "SIMULATION_MISS"
    ]

    judge_misses = misses[
        misses[
            "miss_type"
        ]
        == "JUDGE_OR_ROUND_MISS"
    ]

    print()
    print(
        f"All fresh decisions: "
        f"{len(audit)}"
    )

    print(
        f"MC misses: "
        f"{len(misses)} "
        f"({len(misses) / len(audit):.2%})"
    )

    print(
        f"Simulation misses "
        f"(oracle correct): "
        f"{len(simulation_misses)}"
    )

    print(
        f"Judge/round misses "
        f"(oracle also wrong): "
        f"{len(judge_misses)}"
    )

    print_edge_summary(
        simulation_misses,
        "SIMULATION MISSES — ORACLE GOT THE WINNER RIGHT",
    )

    print_edge_summary(
        judge_misses,
        "JUDGE / ROUND-REPRESENTATION MISSES",
    )

    print()
    print("=" * 180)
    print(
        "MOST CONFIDENT SIMULATION MISSES"
    )
    print("=" * 180)

    columns = [
        "fight_id",
        "red_name",
        "blue_name",
        "winner_name",
        "method",
        "oracle_p_red",
        "expected_p_red",
        "mc_p_red",
        "actual_winner_sig_edge",
        "expected_winner_sig_edge",
        "sim_winner_sig_edge",
        "actual_winner_td_edge",
        "sim_winner_td_edge",
        "actual_winner_control_edge",
        "sim_winner_control_edge",
    ]

    print(
        simulation_misses
        .sort_values(
            "mc_confidence",
            ascending=False,
        )[
            columns
        ]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.2f}",
        )
    )

    print()
    print("=" * 180)
    print(
        "JUDGE / ROUND-REPRESENTATION MISSES"
    )
    print("=" * 180)

    print(
        judge_misses
        .sort_values(
            "mc_confidence",
            ascending=False,
        )[
            columns
        ]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.2f}",
        )
    )

    print()
    print("=" * 140)
    print(
        "MISS TYPE BY DECISION METHOD"
    )
    print("=" * 140)

    print(
        pd.crosstab(
            misses[
                "method"
            ],
            misses[
                "miss_type"
            ],
        ).to_string()
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUT,
        index=False,
    )

    print()
    print(
        f"wrote: {OUT}"
    )


if __name__ == "__main__":
    main()
