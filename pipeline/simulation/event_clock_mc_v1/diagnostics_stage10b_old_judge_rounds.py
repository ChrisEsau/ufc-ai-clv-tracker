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
    "stage10b_old_judge_rounds.csv"
)

# Exact old EVENT MC judging calibration
LANDED_STRIKE_WEIGHT = 1.0
KNOCKDOWN_WEIGHT = 10.080282
TAKEDOWN_WEIGHT = 2.021731
SUB_ATTEMPT_WEIGHT = 2.854417
CONTROL_WEIGHT_PER_SECOND = 0.048904


def norm_col(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(frame, candidates):
    by_norm = {
        norm_col(col): col
        for col in frame.columns
    }

    for candidate in candidates:
        key = norm_col(candidate)
        if key in by_norm:
            return by_norm[key]

    return None


def score_round(
    sig_landed,
    knockdowns,
    td_landed,
    sub_attempts,
    control_seconds,
):
    return (
        LANDED_STRIKE_WEIGHT * sig_landed
        + KNOCKDOWN_WEIGHT * knockdowns
        + TAKEDOWN_WEIGHT * td_landed
        + SUB_ATTEMPT_WEIGHT * sub_attempts
        + CONTROL_WEIGHT_PER_SECOND * control_seconds
    )


def main():
    print("=" * 140)
    print("STAGE 10B — OLD EVENT MC ROUND JUDGE ON HISTORICAL DECISIONS")
    print("=" * 140)

    rounds = pd.read_parquet(ROUND_STATS)
    decisions = pd.read_csv(STAGE10, low_memory=False)

    decisions["fight_id"] = decisions["fight_id"].astype(str)

    print(f"round rows: {len(rounds)}")
    print(f"fresh decisions: {len(decisions)}")

    print()
    print("ROUND-STATS COLUMNS")
    print("-" * 140)

    likely = [
        c for c in rounds.columns
        if any(
            key in norm_col(c)
            for key in (
                "fight",
                "round",
                "fighter",
                "side",
                "sig",
                "kd",
                "td",
                "sub",
                "ctrl",
                "control",
            )
        )
    ]

    for c in likely:
        print(c)

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
            "side",
            "corner",
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

    sig_landed_col = find_column(
        rounds,
        [
            "sig_str_landed",
            "sig_strikes_landed",
            "significant_strikes_landed",
            "sig_landed",
        ],
    )

    kd_col = find_column(
        rounds,
        [
            "kd",
            "knockdowns",
            "knockdown",
        ],
    )

    td_landed_col = find_column(
        rounds,
        [
            "td_landed",
            "takedowns_landed",
            "takedown_landed",
        ],
    )

    sub_attempt_col = find_column(
        rounds,
        [
            "sub_att",
            "sub_attempts",
            "submission_attempts",
        ],
    )

    control_col = find_column(
        rounds,
        [
            "ctrl_sec",
            "ctrl_seconds",
            "control_seconds",
            "ctrl",
            "control",
        ],
    )

    resolved = {
        "fight_id": fight_col,
        "round": round_col,
        "side": side_col,
        "fighter": fighter_col,
        "sig_landed": sig_landed_col,
        "kd": kd_col,
        "td_landed": td_landed_col,
        "sub_attempts": sub_attempt_col,
        "control": control_col,
    }

    print()
    print("RESOLVED COLUMNS")
    print("-" * 140)

    for key, value in resolved.items():
        print(f"{key:15s}: {value}")

    required = [
        fight_col,
        round_col,
        sig_landed_col,
        kd_col,
        td_landed_col,
        sub_attempt_col,
        control_col,
    ]

    if any(x is None for x in required):
        raise RuntimeError(
            "Could not resolve all required round-stat columns."
        )

    rounds[fight_col] = rounds[fight_col].astype(str)

    rounds = rounds[
        rounds[fight_col].isin(
            set(decisions["fight_id"])
        )
    ].copy()

    rows = []

    for fight_id, fight_decision in decisions.set_index("fight_id").iterrows():

        fight_rounds = rounds[
            rounds[fight_col] == fight_id
        ].copy()

        if fight_rounds.empty:
            continue

        red_name = str(
            fight_decision["red_name"]
        )
        blue_name = str(
            fight_decision["blue_name"]
        )

        red_win_actual = int(
            fight_decision["red_win"]
        )

        card_red_rounds = 0
        card_blue_rounds = 0

        round_details = []

        for round_number, group in fight_rounds.groupby(
            round_col,
            sort=True,
        ):
            red_row = None
            blue_row = None

            if side_col is not None:
                red_candidates = group[
                    group[side_col]
                    .astype(str)
                    .str.lower()
                    .isin(["red", "r"])
                ]

                blue_candidates = group[
                    group[side_col]
                    .astype(str)
                    .str.lower()
                    .isin(["blue", "b"])
                ]

                if len(red_candidates) == 1:
                    red_row = red_candidates.iloc[0]

                if len(blue_candidates) == 1:
                    blue_row = blue_candidates.iloc[0]

            if (
                (red_row is None or blue_row is None)
                and fighter_col is not None
            ):
                red_candidates = group[
                    group[fighter_col].astype(str)
                    == red_name
                ]

                blue_candidates = group[
                    group[fighter_col].astype(str)
                    == blue_name
                ]

                if len(red_candidates) == 1:
                    red_row = red_candidates.iloc[0]

                if len(blue_candidates) == 1:
                    blue_row = blue_candidates.iloc[0]

            if red_row is None or blue_row is None:
                continue

            red_score = score_round(
                float(red_row[sig_landed_col]),
                float(red_row[kd_col]),
                float(red_row[td_landed_col]),
                float(red_row[sub_attempt_col]),
                float(red_row[control_col]),
            )

            blue_score = score_round(
                float(blue_row[sig_landed_col]),
                float(blue_row[kd_col]),
                float(blue_row[td_landed_col]),
                float(blue_row[sub_attempt_col]),
                float(blue_row[control_col]),
            )

            if red_score > blue_score:
                round_winner = "red"
                card_red_rounds += 1

            elif blue_score > red_score:
                round_winner = "blue"
                card_blue_rounds += 1

            else:
                round_winner = "tie"

            round_details.append(
                {
                    "round":
                        int(round_number),

                    "red_score_value":
                        red_score,

                    "blue_score_value":
                        blue_score,

                    "round_winner":
                        round_winner,

                    "red_sig":
                        float(red_row[sig_landed_col]),

                    "blue_sig":
                        float(blue_row[sig_landed_col]),

                    "red_kd":
                        float(red_row[kd_col]),

                    "blue_kd":
                        float(blue_row[kd_col]),

                    "red_td":
                        float(red_row[td_landed_col]),

                    "blue_td":
                        float(blue_row[td_landed_col]),

                    "red_sub":
                        float(red_row[sub_attempt_col]),

                    "blue_sub":
                        float(blue_row[sub_attempt_col]),

                    "red_ctrl":
                        float(red_row[control_col]),

                    "blue_ctrl":
                        float(blue_row[control_col]),
                }
            )

        if card_red_rounds > card_blue_rounds:
            predicted_red_win = 1

        elif card_blue_rounds > card_red_rounds:
            predicted_red_win = 0

        else:
            predicted_red_win = np.nan

        rows.append(
            {
                "fight_id":
                    fight_id,

                "red_name":
                    red_name,

                "blue_name":
                    blue_name,

                "winner_name":
                    fight_decision["winner_name"],

                "method":
                    fight_decision["method"],

                "actual_red_win":
                    red_win_actual,

                "predicted_red_win":
                    predicted_red_win,

                "red_rounds":
                    card_red_rounds,

                "blue_rounds":
                    card_blue_rounds,

                "rounds_scored":
                    len(round_details),

                "round_details":
                    str(round_details),
            }
        )

    result = pd.DataFrame(rows)

    scored = result[
        result["predicted_red_win"].notna()
    ].copy()

    scored["correct"] = (
        scored["predicted_red_win"].astype(int)
        == scored["actual_red_win"].astype(int)
    )

    print()
    print("=" * 140)
    print("RESULT")
    print("=" * 140)

    print(
        f"decisions available: "
        f"{len(result)}"
    )

    print(
        f"non-tied cards scored: "
        f"{len(scored)}"
    )

    print(
        f"tied cards: "
        f"{len(result) - len(scored)}"
    )

    print(
        f"accuracy: "
        f"{scored['correct'].mean():.2%}"
    )

    print()
    print("BY DECISION TYPE")
    print("-" * 140)

    summary = (
        scored.groupby(
            "method"
        )
        .agg(
            fights=(
                "fight_id",
                "count",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
        )
    )

    print(
        summary.to_string(
            float_format=lambda x:
                f"{x:.2%}",
        )
    )

    misses = scored[
        ~scored["correct"]
    ].copy()

    print()
    print("=" * 140)
    print("MISSES")
    print("=" * 140)

    print(
        misses[
            [
                "fight_id",
                "red_name",
                "blue_name",
                "winner_name",
                "method",
                "red_rounds",
                "blue_rounds",
            ]
        ]
        .to_string(
            index=False
        )
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
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
