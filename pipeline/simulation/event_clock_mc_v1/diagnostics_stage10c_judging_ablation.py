from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROUND_STATS = Path("data/fight_details/ufc_round_stats.parquet")
STAGE10 = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10_decision_judge_fresh.csv"
)

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10c_judging_ablation.csv"
)

# Old EVENT MC calibrated weights
KD_WEIGHT = 10.080282
TD_WEIGHT = 2.021731
SUB_WEIGHT = 2.854417
CTRL_WEIGHT = 0.048904


def norm_col(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(frame, candidates):
    normalized = {
        norm_col(c): c
        for c in frame.columns
    }

    for candidate in candidates:
        key = norm_col(candidate)
        if key in normalized:
            return normalized[key]

    return None


def score_variant(
    variant,
    sig,
    kd,
    td,
    sub,
    ctrl,
):
    if variant == "SIG_ONLY":
        return sig

    if variant == "SIG_KD":
        return (
            sig
            + KD_WEIGHT * kd
        )

    if variant == "SIG_KD_TD":
        return (
            sig
            + KD_WEIGHT * kd
            + TD_WEIGHT * td
        )

    if variant == "SIG_KD_TD_CTRL":
        return (
            sig
            + KD_WEIGHT * kd
            + TD_WEIGHT * td
            + CTRL_WEIGHT * ctrl
        )

    if variant == "OLD_FULL":
        return (
            sig
            + KD_WEIGHT * kd
            + TD_WEIGHT * td
            + SUB_WEIGHT * sub
            + CTRL_WEIGHT * ctrl
        )

    raise ValueError(
        f"Unknown variant: {variant}"
    )


def damage_first_winner(
    red,
    blue,
):
    """
    Simple hierarchy diagnostic:

    1. More knockdowns wins the round.
    2. If KD tied, more sig strikes wins.
    3. If sig tied, more TD wins.
    4. If TD tied, more SUB attempts wins.
    5. If still tied, more control wins.
    6. Otherwise tie.

    This is intentionally NOT a proposed final judge.
    It tests whether prioritizing observable damage helps.
    """

    if red["kd"] != blue["kd"]:
        return (
            "red"
            if red["kd"] > blue["kd"]
            else "blue"
        )

    if red["sig"] != blue["sig"]:
        return (
            "red"
            if red["sig"] > blue["sig"]
            else "blue"
        )

    if red["td"] != blue["td"]:
        return (
            "red"
            if red["td"] > blue["td"]
            else "blue"
        )

    if red["sub"] != blue["sub"]:
        return (
            "red"
            if red["sub"] > blue["sub"]
            else "blue"
        )

    if red["ctrl"] != blue["ctrl"]:
        return (
            "red"
            if red["ctrl"] > blue["ctrl"]
            else "blue"
        )

    return "tie"


def main():
    print("=" * 145)
    print("STAGE 10C — HISTORICAL ROUND-JUDGING ABLATION")
    print("=" * 145)

    rounds = pd.read_parquet(
        ROUND_STATS
    )

    decisions = pd.read_csv(
        STAGE10,
        low_memory=False,
    )

    decisions["fight_id"] = (
        decisions["fight_id"]
        .astype(str)
    )

    fight_col = find_column(
        rounds,
        [
            "fight_id",
            "bout_id",
        ],
    )

    round_col = find_column(
        rounds,
        [
            "round",
            "round_number",
        ],
    )

    side_col = find_column(
        rounds,
        [
            "corner",
            "side",
            "fighter_side",
        ],
    )

    fighter_col = find_column(
        rounds,
        [
            "fighter_name",
            "fighter",
            "name",
        ],
    )

    sig_col = find_column(
        rounds,
        [
            "sig_str_landed",
            "sig_strikes_landed",
            "significant_strikes_landed",
        ],
    )

    kd_col = find_column(
        rounds,
        [
            "kd",
            "knockdowns",
        ],
    )

    td_col = find_column(
        rounds,
        [
            "td_landed",
            "takedowns_landed",
        ],
    )

    sub_col = find_column(
        rounds,
        [
            "sub_att",
            "submission_attempts",
        ],
    )

    ctrl_col = find_column(
        rounds,
        [
            "ctrl_sec",
            "ctrl_seconds",
            "control_seconds",
        ],
    )

    required = [
        fight_col,
        round_col,
        sig_col,
        kd_col,
        td_col,
        sub_col,
        ctrl_col,
    ]

    if any(x is None for x in required):
        raise RuntimeError(
            "Could not resolve required columns."
        )

    rounds[fight_col] = (
        rounds[fight_col]
        .astype(str)
    )

    rounds = rounds[
        rounds[fight_col].isin(
            set(
                decisions["fight_id"]
            )
        )
    ].copy()

    variants = [
        "SIG_ONLY",
        "SIG_KD",
        "SIG_KD_TD",
        "SIG_KD_TD_CTRL",
        "OLD_FULL",
        "DAMAGE_FIRST",
    ]

    result_rows = []

    for _, fight in decisions.iterrows():

        fight_id = str(
            fight["fight_id"]
        )

        fight_rounds = rounds[
            rounds[fight_col]
            == fight_id
        ]

        if fight_rounds.empty:
            continue

        red_name = str(
            fight["red_name"]
        )

        blue_name = str(
            fight["blue_name"]
        )

        actual_red_win = int(
            fight["red_win"]
        )

        for variant in variants:

            red_rounds = 0
            blue_rounds = 0
            tied_rounds = 0

            for round_number, group in (
                fight_rounds.groupby(
                    round_col,
                    sort=True,
                )
            ):

                red_row = None
                blue_row = None

                if side_col is not None:

                    red_candidates = group[
                        group[side_col]
                        .astype(str)
                        .str.lower()
                        .isin(
                            ["red", "r"]
                        )
                    ]

                    blue_candidates = group[
                        group[side_col]
                        .astype(str)
                        .str.lower()
                        .isin(
                            ["blue", "b"]
                        )
                    ]

                    if len(red_candidates) == 1:
                        red_row = (
                            red_candidates.iloc[0]
                        )

                    if len(blue_candidates) == 1:
                        blue_row = (
                            blue_candidates.iloc[0]
                        )

                if (
                    red_row is None
                    or blue_row is None
                ) and fighter_col is not None:

                    red_candidates = group[
                        group[fighter_col]
                        .astype(str)
                        == red_name
                    ]

                    blue_candidates = group[
                        group[fighter_col]
                        .astype(str)
                        == blue_name
                    ]

                    if len(red_candidates) == 1:
                        red_row = (
                            red_candidates.iloc[0]
                        )

                    if len(blue_candidates) == 1:
                        blue_row = (
                            blue_candidates.iloc[0]
                        )

                if (
                    red_row is None
                    or blue_row is None
                ):
                    continue

                red_stats = {
                    "sig":
                        float(
                            red_row[sig_col]
                        ),
                    "kd":
                        float(
                            red_row[kd_col]
                        ),
                    "td":
                        float(
                            red_row[td_col]
                        ),
                    "sub":
                        float(
                            red_row[sub_col]
                        ),
                    "ctrl":
                        float(
                            red_row[ctrl_col]
                        ),
                }

                blue_stats = {
                    "sig":
                        float(
                            blue_row[sig_col]
                        ),
                    "kd":
                        float(
                            blue_row[kd_col]
                        ),
                    "td":
                        float(
                            blue_row[td_col]
                        ),
                    "sub":
                        float(
                            blue_row[sub_col]
                        ),
                    "ctrl":
                        float(
                            blue_row[ctrl_col]
                        ),
                }

                if variant == "DAMAGE_FIRST":

                    winner = damage_first_winner(
                        red_stats,
                        blue_stats,
                    )

                else:

                    red_score = score_variant(
                        variant,
                        red_stats["sig"],
                        red_stats["kd"],
                        red_stats["td"],
                        red_stats["sub"],
                        red_stats["ctrl"],
                    )

                    blue_score = score_variant(
                        variant,
                        blue_stats["sig"],
                        blue_stats["kd"],
                        blue_stats["td"],
                        blue_stats["sub"],
                        blue_stats["ctrl"],
                    )

                    if red_score > blue_score:
                        winner = "red"

                    elif blue_score > red_score:
                        winner = "blue"

                    else:
                        winner = "tie"

                if winner == "red":
                    red_rounds += 1

                elif winner == "blue":
                    blue_rounds += 1

                else:
                    tied_rounds += 1

            if red_rounds > blue_rounds:
                predicted_red_win = 1

            elif blue_rounds > red_rounds:
                predicted_red_win = 0

            else:
                predicted_red_win = np.nan

            result_rows.append(
                {
                    "fight_id":
                        fight_id,

                    "red_name":
                        red_name,

                    "blue_name":
                        blue_name,

                    "winner_name":
                        fight["winner_name"],

                    "method":
                        fight["method"],

                    "variant":
                        variant,

                    "actual_red_win":
                        actual_red_win,

                    "predicted_red_win":
                        predicted_red_win,

                    "red_rounds":
                        red_rounds,

                    "blue_rounds":
                        blue_rounds,

                    "tied_rounds":
                        tied_rounds,
                }
            )

    result = pd.DataFrame(
        result_rows
    )

    result["correct"] = np.where(
        result[
            "predicted_red_win"
        ].notna(),
        result[
            "predicted_red_win"
        ].astype("Int64")
        == result[
            "actual_red_win"
        ].astype("Int64"),
        np.nan,
    )

    print()
    print("=" * 145)
    print("OVERALL")
    print("=" * 145)

    summary_rows = []

    for variant, group in (
        result.groupby(
            "variant",
            sort=False,
        )
    ):

        scored = group[
            group[
                "predicted_red_win"
            ].notna()
        ]

        accuracy = (
            scored["correct"].mean()
            if len(scored)
            else np.nan
        )

        summary_rows.append(
            {
                "variant":
                    variant,

                "scored":
                    len(scored),

                "ties":
                    len(group)
                    - len(scored),

                "accuracy":
                    accuracy,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "accuracy":
                    lambda x:
                        f"{x:.2%}",
            },
        )
    )

    print()
    print("=" * 145)
    print("BY DECISION TYPE")
    print("=" * 145)

    method_rows = []

    for (
        variant,
        method
    ), group in result.groupby(
        [
            "variant",
            "method",
        ],
        sort=False,
    ):

        scored = group[
            group[
                "predicted_red_win"
            ].notna()
        ]

        method_rows.append(
            {
                "variant":
                    variant,

                "method":
                    method,

                "fights":
                    len(scored),

                "accuracy":
                    scored[
                        "correct"
                    ].mean(),
            }
        )

    method_summary = pd.DataFrame(
        method_rows
    )

    pivot = (
        method_summary.pivot(
            index="variant",
            columns="method",
            values="accuracy",
        )
    )

    print(
        pivot.to_string(
            float_format=lambda x:
                f"{x:.2%}"
        )
    )

    print()
    print("=" * 145)
    print("CHANGE VS OLD FULL JUDGE")
    print("=" * 145)

    old = result[
        result["variant"]
        == "OLD_FULL"
    ][
        [
            "fight_id",
            "correct",
        ]
    ].rename(
        columns={
            "correct":
                "old_correct"
        }
    )

    for variant in variants:

        if variant == "OLD_FULL":
            continue

        current = result[
            result["variant"]
            == variant
        ][
            [
                "fight_id",
                "correct",
                "red_name",
                "blue_name",
                "winner_name",
                "method",
            ]
        ].merge(
            old,
            on="fight_id",
            how="left",
        )

        fixed = current[
            (current["correct"] == True)
            & (
                current[
                    "old_correct"
                ] == False
            )
        ]

        broken = current[
            (current["correct"] == False)
            & (
                current[
                    "old_correct"
                ] == True
            )
        ]

        print()
        print(variant)
        print(
            f"old misses fixed: "
            f"{len(fixed)}"
        )
        print(
            f"old correct fights broken: "
            f"{len(broken)}"
        )

    result.to_csv(
        OUT,
        index=False,
    )

    print()
    print(
        f"wrote: {OUT}"
    )


if __name__ == "__main__":
    main()
