from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)


STAGE9_RESULT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage9_final_flow_500x20.csv"
)

STAGE9_PATHS = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage9_final_flow_paths_500x20.csv"
)

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10_decision_judge_fresh.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage10_decision_judge_paths.csv"
)


# =============================================================================
# COLUMN / OUTCOME HELPERS
# =============================================================================

def norm_name(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower(),
    )


def norm_col(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def find_column(
    frame,
    candidates,
):
    by_norm = {
        norm_col(col): col
        for col in frame.columns
    }

    for candidate in candidates:
        key = norm_col(
            candidate
        )

        if key in by_norm:
            return by_norm[
                key
            ]

    return None


def first_non_null(
    series,
):
    x = series.dropna()

    if len(x):
        return x.iloc[0]

    return np.nan


def is_win_value(value):
    text = str(
        value
    ).strip().lower()

    return text in {
        "w",
        "win",
        "winner",
        "won",
        "1",
        "true",
    }


def is_loss_value(value):
    text = str(
        value
    ).strip().lower()

    return text in {
        "l",
        "loss",
        "loser",
        "lost",
        "0",
        "false",
    }


def extract_outcomes_from_fighter_frame(
    frame,
):
    """
    Try to extract method + winner directly from the fighter-level
    frame returned by the current prototype data builder.

    Returns one row per fight or None.
    """

    method_col = find_column(
        frame,
        [
            "method",
            "method_norm",
            "method_clean",
            "finish_method",
            "win_method",
            "result_method",
            "method_of_victory",
            "actual_method",
        ],
    )

    if method_col is None:
        return None

    result_col = find_column(
        frame,
        [
            "result",
            "fighter_result",
            "outcome",
            "wl",
            "w_l",
        ],
    )

    winner_side_col = find_column(
        frame,
        [
            "winner_side",
            "winning_side",
            "winner_corner",
            "actual_winner_side",
        ],
    )

    winner_name_col = find_column(
        frame,
        [
            "winner_name",
            "actual_winner",
            "winner_fighter",
            "winner_fighter_name",
            "winner",
        ],
    )

    rows = []

    for fight_id, group in frame.groupby(
        "fight_id",
        sort=False,
    ):
        if len(group) != 2:
            continue

        red_rows = group[
            group["side"]
            == "red"
        ]

        blue_rows = group[
            group["side"]
            == "blue"
        ]

        if (
            len(red_rows) != 1
            or len(blue_rows) != 1
        ):
            continue

        red = red_rows.iloc[0]
        blue = blue_rows.iloc[0]

        red_name = str(
            red[
                "fighter_name"
            ]
        )

        blue_name = str(
            blue[
                "fighter_name"
            ]
        )

        method = first_non_null(
            group[
                method_col
            ]
        )

        red_win = None
        winner_name = None

        # ---------------------------------------------------------
        # Row-level W/L is the cleanest inference.
        # ---------------------------------------------------------

        if result_col is not None:

            red_result = red[
                result_col
            ]

            blue_result = blue[
                result_col
            ]

            if (
                is_win_value(
                    red_result
                )
                and is_loss_value(
                    blue_result
                )
            ):
                red_win = 1
                winner_name = red_name

            elif (
                is_win_value(
                    blue_result
                )
                and is_loss_value(
                    red_result
                )
            ):
                red_win = 0
                winner_name = blue_name

        # ---------------------------------------------------------
        # Explicit winner side.
        # ---------------------------------------------------------

        if (
            red_win is None
            and winner_side_col
            is not None
        ):
            winner_side = str(
                first_non_null(
                    group[
                        winner_side_col
                    ]
                )
            ).strip().lower()

            if winner_side in {
                "red",
                "r",
            }:
                red_win = 1
                winner_name = red_name

            elif winner_side in {
                "blue",
                "b",
            }:
                red_win = 0
                winner_name = blue_name

        # ---------------------------------------------------------
        # Explicit winner name.
        # ---------------------------------------------------------

        if (
            red_win is None
            and winner_name_col
            is not None
        ):
            winner = first_non_null(
                group[
                    winner_name_col
                ]
            )

            winner_norm = norm_name(
                winner
            )

            if winner_norm == norm_name(
                red_name
            ):
                red_win = 1
                winner_name = red_name

            elif winner_norm == norm_name(
                blue_name
            ):
                red_win = 0
                winner_name = blue_name

            elif winner_norm in {
                "red",
                "r",
            }:
                red_win = 1
                winner_name = red_name

            elif winner_norm in {
                "blue",
                "b",
            }:
                red_win = 0
                winner_name = blue_name

        if red_win is None:
            continue

        rows.append(
            {
                "fight_id":
                    str(
                        fight_id
                    ),

                "method":
                    str(
                        method
                    ),

                "red_win":
                    int(
                        red_win
                    ),

                "winner_name":
                    winner_name,

                "red_name":
                    red_name,

                "blue_name":
                    blue_name,
            }
        )

    if not rows:
        return None

    return pd.DataFrame(
        rows
    )


def candidate_master_files():
    exact = [
        Path(
            "data/ufc_master.parquet"
        ),
        Path(
            "data/ufc_master.csv"
        ),
        Path(
            "data/master/ufc_master.parquet"
        ),
        Path(
            "data/master/ufc_master.csv"
        ),
        Path(
            "data/processed/ufc_master.parquet"
        ),
        Path(
            "data/processed/ufc_master.csv"
        ),
    ]

    found = []

    for path in exact:
        if path.exists():
            found.append(
                path
            )

    data_root = Path(
        "data"
    )

    if data_root.exists():
        for pattern in (
            "**/*ufc*master*.parquet",
            "**/*ufc*master*.csv",
        ):
            for path in (
                data_root.glob(
                    pattern
                )
            ):
                if (
                    path not in found
                ):
                    found.append(
                        path
                    )

    return found


def read_table(
    path,
):
    if (
        path.suffix.lower()
        == ".parquet"
    ):
        return pd.read_parquet(
            path
        )

    return pd.read_csv(
        path,
        low_memory=False,
    )


def extract_outcomes_from_master(
    master,
    reference,
):
    """
    Fallback if the prototype fighter frame does not retain result fields.
    Requires a fight/bout id shared with the prototype rows.
    """

    id_col = find_column(
        master,
        [
            "fight_id",
            "bout_id",
            "fightid",
            "boutid",
        ],
    )

    method_col = find_column(
        master,
        [
            "method",
            "method_norm",
            "method_clean",
            "finish_method",
            "win_method",
            "method_of_victory",
        ],
    )

    winner_col = find_column(
        master,
        [
            "winner_name",
            "winner",
            "actual_winner",
            "winner_fighter",
            "winner_fighter_name",
        ],
    )

    winner_side_col = find_column(
        master,
        [
            "winner_side",
            "winning_side",
            "winner_corner",
        ],
    )

    red_name_col = find_column(
        master,
        [
            "red_fighter",
            "red_fighter_name",
            "r_fighter",
            "fighter_red",
        ],
    )

    blue_name_col = find_column(
        master,
        [
            "blue_fighter",
            "blue_fighter_name",
            "b_fighter",
            "fighter_blue",
        ],
    )

    if (
        id_col is None
        or method_col is None
    ):
        return None

    reference_names = {}

    for fight_id, group in (
        reference.groupby(
            "fight_id",
            sort=False,
        )
    ):
        if len(group) != 2:
            continue

        red = group[
            group["side"]
            == "red"
        ]

        blue = group[
            group["side"]
            == "blue"
        ]

        if (
            len(red) == 1
            and len(blue) == 1
        ):
            reference_names[
                str(
                    fight_id
                )
            ] = (
                str(
                    red.iloc[0][
                        "fighter_name"
                    ]
                ),
                str(
                    blue.iloc[0][
                        "fighter_name"
                    ]
                ),
            )

    rows = []

    for _, row in (
        master.iterrows()
    ):
        fight_id = str(
            row[
                id_col
            ]
        )

        if (
            fight_id
            not in reference_names
        ):
            continue

        red_name, blue_name = (
            reference_names[
                fight_id
            ]
        )

        method = row[
            method_col
        ]

        red_win = None
        winner_name = None

        if winner_side_col is not None:
            side = str(
                row[
                    winner_side_col
                ]
            ).strip().lower()

            if side in {
                "red",
                "r",
            }:
                red_win = 1
                winner_name = red_name

            elif side in {
                "blue",
                "b",
            }:
                red_win = 0
                winner_name = blue_name

        if (
            red_win is None
            and winner_col
            is not None
        ):
            winner = row[
                winner_col
            ]

            winner_norm = norm_name(
                winner
            )

            if winner_norm == norm_name(
                red_name
            ):
                red_win = 1
                winner_name = red_name

            elif winner_norm == norm_name(
                blue_name
            ):
                red_win = 0
                winner_name = blue_name

            elif winner_norm in {
                "red",
                "r",
            }:
                red_win = 1
                winner_name = red_name

            elif winner_norm in {
                "blue",
                "b",
            }:
                red_win = 0
                winner_name = blue_name

        if (
            red_win is None
            and red_name_col
            is not None
            and blue_name_col
            is not None
            and winner_col
            is not None
        ):
            winner = norm_name(
                row[
                    winner_col
                ]
            )

            red_external = norm_name(
                row[
                    red_name_col
                ]
            )

            blue_external = norm_name(
                row[
                    blue_name_col
                ]
            )

            if winner == red_external:
                red_win = 1
                winner_name = red_name

            elif winner == blue_external:
                red_win = 0
                winner_name = blue_name

        if red_win is None:
            continue

        rows.append(
            {
                "fight_id":
                    fight_id,

                "method":
                    str(
                        method
                    ),

                "red_win":
                    int(
                        red_win
                    ),

                "winner_name":
                    winner_name,

                "red_name":
                    red_name,

                "blue_name":
                    blue_name,
            }
        )

    if not rows:
        return None

    return (
        pd.DataFrame(
            rows
        )
        .drop_duplicates(
            "fight_id",
            keep="last",
        )
    )


def resolve_outcomes(
    train,
    test,
):
    combined = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
        sort=False,
    )

    direct = (
        extract_outcomes_from_fighter_frame(
            combined
        )
    )

    if (
        direct is not None
        and len(direct)
        >= 100
    ):
        print(
            "Outcome source: "
            "prototype fighter rows"
        )

        return direct

    for path in (
        candidate_master_files()
    ):
        try:
            master = read_table(
                path
            )
        except Exception:
            continue

        extracted = (
            extract_outcomes_from_master(
                master,
                combined,
            )
        )

        if (
            extracted is not None
            and len(extracted)
            >= 100
        ):
            print(
                f"Outcome source: "
                f"{path}"
            )

            return extracted

    print()
    print(
        "Could not automatically resolve "
        "winner/method fields."
    )

    print()
    print(
        "Prototype columns containing likely "
        "outcome terms:"
    )

    likely = [
        col
        for col in combined.columns
        if any(
            key in norm_col(
                col
            )
            for key in (
                "winner",
                "result",
                "outcome",
                "method",
                "finish",
            )
        )
    ]

    for col in likely:
        print(
            f"  {col}"
        )

    raise RuntimeError(
        "Outcome source could not be resolved."
    )


# =============================================================================
# PAIR FEATURE BUILDERS
# =============================================================================

def add_common_stats(
    frame,
):
    frame = frame.copy()

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

    return frame


def make_pair_features(
    frame,
    standing_landed_col,
    ground_landed_col,
    td_landed_col,
    control_col,
    path_col=None,
):
    group_cols = [
        "fight_id",
    ]

    if path_col is not None:
        group_cols.append(
            path_col
        )

    rows = []

    for keys, group in (
        frame.groupby(
            group_cols,
            sort=False,
        )
    ):
        if len(group) != 2:
            continue

        red_rows = group[
            group["side"]
            == "red"
        ]

        blue_rows = group[
            group["side"]
            == "blue"
        ]

        if (
            len(red_rows) != 1
            or len(blue_rows) != 1
        ):
            continue

        red = red_rows.iloc[0]
        blue = blue_rows.iloc[0]

        duration = float(
            red[
                "duration"
            ]
        )

        duration = max(
            duration,
            1.0,
        )

        red_standing = float(
            red[
                standing_landed_col
            ]
        )

        blue_standing = float(
            blue[
                standing_landed_col
            ]
        )

        red_ground = float(
            red[
                ground_landed_col
            ]
        )

        blue_ground = float(
            blue[
                ground_landed_col
            ]
        )

        red_td = float(
            red[
                td_landed_col
            ]
        )

        blue_td = float(
            blue[
                td_landed_col
            ]
        )

        red_control = float(
            red[
                control_col
            ]
        )

        blue_control = float(
            blue[
                control_col
            ]
        )

        red_sig = (
            red_standing
            + red_ground
        )

        blue_sig = (
            blue_standing
            + blue_ground
        )

        total_sig = (
            red_sig
            + blue_sig
        )

        total_control = (
            red_control
            + blue_control
        )

        row = {
            "fight_id":
                str(
                    red[
                        "fight_id"
                    ]
                ),

            "duration":
                duration,

            "red_name":
                str(
                    red[
                        "fighter_name"
                    ]
                ),

            "blue_name":
                str(
                    blue[
                        "fighter_name"
                    ]
                ),

            # Primary effective-striking signal.
            "sig_landed_diff_15m":
                (
                    red_sig
                    - blue_sig
                )
                / duration
                * 900.0,

            "standing_landed_diff_15m":
                (
                    red_standing
                    - blue_standing
                )
                / duration
                * 900.0,

            "ground_landed_diff_15m":
                (
                    red_ground
                    - blue_ground
                )
                / duration
                * 900.0,

            # Grappling effectiveness.
            "td_landed_diff_15m":
                (
                    red_td
                    - blue_td
                )
                / duration
                * 900.0,

            "control_diff_15m":
                (
                    red_control
                    - blue_control
                )
                / duration
                * 900.0,

            # Relative shares provide scale-independent dominance.
            "sig_share_edge":
                (
                    red_sig
                    / total_sig
                    - 0.5
                    if total_sig > 0
                    else 0.0
                ),

            "control_share_edge":
                (
                    red_control
                    / total_control
                    - 0.5
                    if total_control > 0
                    else 0.0
                ),
        }

        if path_col is not None:
            if isinstance(
                keys,
                tuple,
            ):
                row[
                    "path"
                ] = int(
                    keys[
                        1
                    ]
                )
            else:
                row[
                    "path"
                ] = int(
                    red[
                        path_col
                    ]
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


FEATURES = [
    "sig_landed_diff_15m",
    "standing_landed_diff_15m",
    "ground_landed_diff_15m",
    "td_landed_diff_15m",
    "control_diff_15m",
    "sig_share_edge",
    "control_share_edge",
]


# =============================================================================
# METRICS
# =============================================================================

def probability_metrics(
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.clip(
        np.asarray(
            p,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    return {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),

        "auc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),

        "logloss":
            float(
                log_loss(
                    y,
                    p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),
    }


def print_metrics(
    label,
    y,
    p,
):
    m = probability_metrics(
        y,
        p,
    )

    print()
    print(label)
    print("-" * 100)

    print(
        f"Accuracy: "
        f"{m['accuracy']:.2%}"
    )

    print(
        f"AUC:      "
        f"{m['auc']:.4f}"
    )

    print(
        f"Brier:    "
        f"{m['brier']:.4f}"
    )

    print(
        f"Log loss: "
        f"{m['logloss']:.4f}"
    )

    return m


def print_confidence(
    label,
    y,
    p,
):
    y = np.asarray(
        y,
        dtype=int,
    )

    p = np.asarray(
        p,
        dtype=float,
    )

    confidence = np.maximum(
        p,
        1.0 - p,
    )

    prediction = (
        p >= 0.5
    ).astype(int)

    print()
    print(label)
    print("-" * 100)

    print(
        f"{'THRESHOLD':>12}"
        f"{'N':>8}"
        f"{'COVERAGE':>12}"
        f"{'ACCURACY':>12}"
    )

    for threshold in (
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ):
        keep = (
            confidence
            >= threshold
        )

        n = int(
            keep.sum()
        )

        if n == 0:
            accuracy = np.nan
        else:
            accuracy = float(
                np.mean(
                    prediction[
                        keep
                    ]
                    == y[
                        keep
                    ]
                )
            )

        print(
            f"{threshold:>11.0%}"
            f"{n:>8}"
            f"{n / len(y):>11.1%}"
            f"{accuracy:>11.1%}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 130)
    print(
        "EVENT CLOCK MC — "
        "STAGE 10 CONDITIONAL DECISION JUDGE"
    )
    print("=" * 130)

    if not STAGE9_RESULT.exists():
        raise FileNotFoundError(
            f"Missing Stage-9 result: "
            f"{STAGE9_RESULT}"
        )

    if not STAGE9_PATHS.exists():
        raise FileNotFoundError(
            f"Missing Stage-9 paths: "
            f"{STAGE9_PATHS}"
        )

    # -----------------------------------------------------------------
    # Rebuild the exact historical train/test cohort used by the flow
    # prototypes.
    # -----------------------------------------------------------------

    train, test = (
        prepare_direct_predictions()
    )

    train = add_common_stats(
        train
    )

    test = add_common_stats(
        test
    )

    outcomes = resolve_outcomes(
        train,
        test,
    )

    outcomes[
        "fight_id"
    ] = (
        outcomes[
            "fight_id"
        ].astype(str)
    )

    # Decisions only. Draws/no-contests are already absent because we
    # require an identifiable winner.
    outcomes[
        "is_decision"
    ] = (
        outcomes[
            "method"
        ]
        .astype(str)
        .str.contains(
            "decision",
            case=False,
            na=False,
        )
    )

    train_ids = set(
        train[
            "fight_id"
        ].astype(str)
    )

    test_ids = set(
        test[
            "fight_id"
        ].astype(str)
    )

    train_decisions = (
        outcomes[
            outcomes[
                "fight_id"
            ].isin(
                train_ids
            )
            &
            outcomes[
                "is_decision"
            ]
        ]
        .copy()
    )

    test_decisions = (
        outcomes[
            outcomes[
                "fight_id"
            ].isin(
                test_ids
            )
            &
            outcomes[
                "is_decision"
            ]
        ]
        .copy()
    )

    print()
    print("=" * 130)
    print(
        "DECISION COHORT"
    )
    print("=" * 130)

    print(
        f"Training decisions: "
        f"{len(train_decisions)}"
    )

    print(
        f"Fresh test decisions: "
        f"{len(test_decisions)}"
    )

    if (
        len(train_decisions)
        < 100
        or len(test_decisions)
        < 30
    ):
        raise RuntimeError(
            "Decision cohort unexpectedly small; "
            "check outcome/method mapping."
        )

    print()
    print(
        "Fresh decision methods:"
    )

    print(
        test_decisions[
            "method"
        ]
        .value_counts()
        .head(10)
        .to_string()
    )

    # -----------------------------------------------------------------
    # Historical-stat pair features.
    # -----------------------------------------------------------------

    train_actual = (
        make_pair_features(
            train,
            standing_landed_col=
                "standing_landed",
            ground_landed_col=
                "ground_landed",
            td_landed_col=
                "td_landed",
            control_col=
                "qualified_control_inflicted_seconds",
        )
    )

    test_actual = (
        make_pair_features(
            test,
            standing_landed_col=
                "standing_landed",
            ground_landed_col=
                "ground_landed",
            td_landed_col=
                "td_landed",
            control_col=
                "qualified_control_inflicted_seconds",
        )
    )

    train_actual = (
        train_actual.merge(
            train_decisions[
                [
                    "fight_id",
                    "red_win",
                    "winner_name",
                    "method",
                ]
            ],
            on="fight_id",
            how="inner",
            validate="one_to_one",
        )
    )

    test_actual = (
        test_actual.merge(
            test_decisions[
                [
                    "fight_id",
                    "red_win",
                    "winner_name",
                    "method",
                ]
            ],
            on="fight_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # -----------------------------------------------------------------
    # Fit judge ONLY on historical training decisions.
    # No fighter identity, odds, FSR, or actual outcome features.
    # -----------------------------------------------------------------

    judge = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                ),
            ),
        ]
    )

    judge.fit(
        train_actual[
            FEATURES
        ],
        train_actual[
            "red_win"
        ],
    )

    print()
    print("=" * 130)
    print(
        "JUDGE COEFFICIENTS"
    )
    print("=" * 130)

    coefficients = pd.DataFrame(
        {
            "feature":
                FEATURES,

            "coefficient":
                judge.named_steps[
                    "logistic"
                ].coef_[
                    0
                ],
        }
    )

    print(
        coefficients
        .assign(
            abs_coefficient=lambda x:
                x[
                    "coefficient"
                ].abs()
        )
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )[
            [
                "feature",
                "coefficient",
            ]
        ]
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:+.4f}",
        )
    )

    # -----------------------------------------------------------------
    # 1. ORACLE: actual historical stats.
    # -----------------------------------------------------------------

    oracle_p = (
        judge.predict_proba(
            test_actual[
                FEATURES
            ]
        )[
            :,
            1,
        ]
    )

    # -----------------------------------------------------------------
    # 2. PREFIGHT EXPECTED STAGE-9 STATS.
    # -----------------------------------------------------------------

    stage9_result = pd.read_csv(
        STAGE9_RESULT,
        low_memory=False,
    )

    stage9_result[
        "fight_id"
    ] = (
        stage9_result[
            "fight_id"
        ].astype(str)
    )

    # Stage 9 directly predicts standing attempts after expected control
    # is accounted for. Preserve the direct standing landing efficiency
    # and apply it to that Stage-9 expected attempt count.
    landing_rate = np.divide(
        stage9_result[
            "pred_standing_landed"
        ].astype(float),
        np.maximum(
            stage9_result[
                "pred_standing_attempted"
            ].astype(float),
            1e-9,
        ),
    )

    stage9_result[
        "pred_stage9_standing_landed"
    ] = (
        stage9_result[
            "pred_stage9_standing_attempted"
        ].astype(float)
        * landing_rate
    )

    expected_pairs = (
        make_pair_features(
            stage9_result,
            standing_landed_col=
                "pred_stage9_standing_landed",
            ground_landed_col=
                "pred_ground_landed",
            td_landed_col=
                "pred_td_landed",
            control_col=
                "pred_qualified_control_inflicted_seconds",
        )
    )

    expected_pairs = (
        expected_pairs.merge(
            test_decisions[
                [
                    "fight_id",
                    "red_win",
                ]
            ],
            on="fight_id",
            how="inner",
            validate="one_to_one",
        )
    )

    expected_p = (
        judge.predict_proba(
            expected_pairs[
                FEATURES
            ]
        )[
            :,
            1,
        ]
    )

    # -----------------------------------------------------------------
    # 3. MONTE CARLO PATH JUDGING.
    # -----------------------------------------------------------------

    paths = pd.read_csv(
        STAGE9_PATHS,
        low_memory=False,
    )

    paths[
        "fight_id"
    ] = (
        paths[
            "fight_id"
        ].astype(str)
    )

    paths = paths[
        paths[
            "fight_id"
        ].isin(
            set(
                test_decisions[
                    "fight_id"
                ]
            )
        )
    ].copy()

    path_pairs = (
        make_pair_features(
            paths,
            standing_landed_col=
                "sim_standing_landed",
            ground_landed_col=
                "sim_ground_landed",
            td_landed_col=
                "sim_td_landed",
            control_col=
                "sim_control",
            path_col="path",
        )
    )

    path_pairs[
        "judge_p_red"
    ] = (
        judge.predict_proba(
            path_pairs[
                FEATURES
            ]
        )[
            :,
            1,
        ]
    )

    path_pairs[
        "judge_red_wins_path"
    ] = (
        path_pairs[
            "judge_p_red"
        ]
        >= 0.5
    ).astype(int)

    mc = (
        path_pairs.groupby(
            "fight_id",
            as_index=False,
        )
        .agg(
            mc_p_red=(
                "judge_p_red",
                "mean",
            ),

            mc_red_win_path_share=(
                "judge_red_wins_path",
                "mean",
            ),

            paths=(
                "path",
                "nunique",
            ),
        )
    )

    mc = (
        mc.merge(
            test_decisions[
                [
                    "fight_id",
                    "red_win",
                ]
            ],
            on="fight_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # -----------------------------------------------------------------
    # Align all three evaluations by fight id.
    # -----------------------------------------------------------------

    oracle_eval = (
        test_actual[
            [
                "fight_id",
                "red_name",
                "blue_name",
                "duration",
                "red_win",
                "winner_name",
                "method",
            ]
        ]
        .copy()
    )

    oracle_eval[
        "oracle_p_red"
    ] = oracle_p

    expected_eval = (
        expected_pairs[
            [
                "fight_id",
            ]
        ]
        .copy()
    )

    expected_eval[
        "expected_p_red"
    ] = expected_p

    final = (
        oracle_eval
        .merge(
            expected_eval,
            on="fight_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            mc[
                [
                    "fight_id",
                    "mc_p_red",
                    "mc_red_win_path_share",
                    "paths",
                ]
            ],
            on="fight_id",
            how="inner",
            validate="one_to_one",
        )
    )

    y = (
        final[
            "red_win"
        ].to_numpy(int)
    )

    # -----------------------------------------------------------------
    # Reports.
    # -----------------------------------------------------------------

    print()
    print("=" * 130)
    print(
        "FRESH DECISION-WINNER RESULTS"
    )
    print("=" * 130)

    print_metrics(
        "HISTORICAL-STATS JUDGE "
        "(ORACLE INPUT)",
        y,
        final[
            "oracle_p_red"
        ],
    )

    print_metrics(
        "PREFIGHT EXPECTED-STATS JUDGE",
        y,
        final[
            "expected_p_red"
        ],
    )

    print_metrics(
        "STAGE-9 MONTE CARLO DECISION JUDGE",
        y,
        final[
            "mc_p_red"
        ],
    )

    print_confidence(
        "MC DECISION CONFIDENCE",
        y,
        final[
            "mc_p_red"
        ],
    )

    # -----------------------------------------------------------------
    # Simple baseline: fighter with more actual significant strikes.
    # Useful for interpreting whether the learned judge adds value.
    # -----------------------------------------------------------------

    oracle_feature_lookup = (
        test_actual.set_index(
            "fight_id"
        )
    )

    sig_baseline = []

    for fight_id in final[
        "fight_id"
    ]:
        diff = float(
            oracle_feature_lookup.loc[
                fight_id,
                "sig_landed_diff_15m",
            ]
        )

        sig_baseline.append(
            int(
                diff >= 0
            )
        )

    sig_baseline = np.asarray(
        sig_baseline,
        dtype=int,
    )

    print()
    print("=" * 130)
    print(
        "REFERENCE BASELINE"
    )
    print("=" * 130)

    print(
        "Actual-stat winner by total "
        "significant strikes landed:"
    )

    print(
        f"Accuracy: "
        f"{np.mean(sig_baseline == y):.2%}"
    )

    # -----------------------------------------------------------------
    # Misses.
    # -----------------------------------------------------------------

    final[
        "oracle_pick"
    ] = np.where(
        final[
            "oracle_p_red"
        ] >= 0.5,
        final[
            "red_name"
        ],
        final[
            "blue_name"
        ],
    )

    final[
        "expected_pick"
    ] = np.where(
        final[
            "expected_p_red"
        ] >= 0.5,
        final[
            "red_name"
        ],
        final[
            "blue_name"
        ],
    )

    final[
        "mc_pick"
    ] = np.where(
        final[
            "mc_p_red"
        ] >= 0.5,
        final[
            "red_name"
        ],
        final[
            "blue_name"
        ],
    )

    final[
        "mc_correct"
    ] = (
        (
            final[
                "mc_p_red"
            ] >= 0.5
        ).astype(int)
        ==
        final[
            "red_win"
        ]
    )

    final[
        "mc_confidence"
    ] = np.maximum(
        final[
            "mc_p_red"
        ],
        1.0
        - final[
            "mc_p_red"
        ],
    )

    misses = (
        final[
            ~final[
                "mc_correct"
            ]
        ]
        .sort_values(
            "mc_confidence",
            ascending=False,
        )
    )

    print()
    print("=" * 130)
    print(
        "MOST CONFIDENT MC DECISION MISSES"
    )
    print("=" * 130)

    show = misses[
        [
            "fight_id",
            "red_name",
            "blue_name",
            "winner_name",
            "method",
            "oracle_p_red",
            "expected_p_red",
            "mc_p_red",
            "mc_confidence",
        ]
    ].head(
        20
    )

    print(
        show.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    # -----------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUT,
        index=False,
    )

    path_pairs.to_csv(
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
