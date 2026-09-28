from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    metrics,
    within_bout_direction,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import (
    direct_feature_columns,
    draw_frailty,
    estimate_nb_alpha,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    build_pair_frame,
    fit_control_models,
    fit_count_hurdle,
    logit,
    sigmoid,
)


FIGHTS = 500
PATHS = 20
SEED = 20260817
CONTROL_CHUNK_SECONDS = 10.0

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage7_budget_timeline_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage7_budget_timeline_paths_500x20.csv"
)


# =====================================================================
# BASIC HELPERS
# =====================================================================

def dist(values):
    x = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "zero": float(np.mean(x == 0)),
        "p10": float(np.quantile(x, .10)),
        "p50": float(np.quantile(x, .50)),
        "p90": float(np.quantile(x, .90)),
        "p99": float(np.quantile(x, .99)),
    }


def print_dist(
    label,
    historical,
    simulated,
):
    h = dist(historical)
    s = dist(simulated)

    print()
    print(label)
    print("-" * 135)

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


def spearman(
    frame,
    a,
    b,
):
    x = (
        frame[
            [a, b]
        ]
        .dropna()
    )

    if len(x) < 2:
        return np.nan

    return float(
        x.corr(
            method="spearman"
        ).iloc[0, 1]
    )


# =====================================================================
# PAIR / FREE-TIME DATA
# =====================================================================

def add_historical_free_time(
    frame,
):
    frame = frame.copy()

    pair_control = (
        frame.groupby(
            "fight_id"
        )[
            "qualified_control_inflicted_seconds"
        ]
        .transform("sum")
    )

    frame[
        "historical_total_control"
    ] = pair_control

    frame[
        "historical_free_seconds"
    ] = np.maximum(
        frame["duration"].astype(float)
        - pair_control.astype(float),
        1.0,
    )

    return frame


def pair_lookup_from_frame(
    frame,
):
    rows = {}

    for fight_id, group in frame.groupby(
        "fight_id",
        sort=False,
    ):
        if len(group) != 2:
            continue

        red = (
            group[
                group["side"] == "red"
            ].iloc[0]
        )

        blue = (
            group[
                group["side"] == "blue"
            ].iloc[0]
        )

        rows[
            str(fight_id)
        ] = {
            "red_td_landed":
                float(
                    red["td_landed"]
                ),

            "blue_td_landed":
                float(
                    blue["td_landed"]
                ),

            "red_control":
                float(
                    red[
                        "qualified_control_inflicted_seconds"
                    ]
                ),

            "blue_control":
                float(
                    blue[
                        "qualified_control_inflicted_seconds"
                    ]
                ),
        }

    return rows


# =====================================================================
# STANDING FREE-TIME RATE MODEL
# =====================================================================

def fit_standing_free_time_model(
    train,
    test,
    feature_cols,
):
    """
    Model standing-attempt intensity per 15 minutes of NON-CONTROL time.

    Weighted Poisson regression:
        target = attempts / free_seconds * 900
        weight = free_seconds / 900

    This approximates an exposure-offset Poisson model while remaining
    simple and leakage-safe inside the existing prototype framework.
    """

    train = train.copy()
    test = test.copy()

    train_rate = (
        train[
            "standing_attempted"
        ].astype(float)
        / train[
            "historical_free_seconds"
        ].astype(float)
        * 900.0
    )

    exposure_weight = (
        train[
            "historical_free_seconds"
        ].astype(float)
        / 900.0
    )

    model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "poisson",
                PoissonRegressor(
                    alpha=0.10,
                    max_iter=2000,
                ),
            ),
        ]
    )

    model.fit(
        train[
            feature_cols
        ],
        train_rate,
        poisson__sample_weight=
            exposure_weight,
    )

    train[
        "pred_standing_rate_free_15m"
    ] = np.maximum(
        model.predict(
            train[
                feature_cols
            ]
        ),
        0.0,
    )

    test[
        "pred_standing_rate_free_15m"
    ] = np.maximum(
        model.predict(
            test[
                feature_cols
            ]
        ),
        0.0,
    )

    train_expected_on_actual_free = (
        train[
            "pred_standing_rate_free_15m"
        ]
        * train[
            "historical_free_seconds"
        ]
        / 900.0
    )

    standing_alpha = estimate_nb_alpha(
        train[
            "standing_attempted"
        ],
        train_expected_on_actual_free,
    )

    print()
    print("=" * 115)
    print(
        "STANDING FREE-TIME RATE MODEL"
    )
    print("=" * 115)

    print(
        f"historical training standing mean: "
        f"{train['standing_attempted'].mean():.3f}"
    )

    print(
        f"rate-model training mean using "
        f"actual free time: "
        f"{train_expected_on_actual_free.mean():.3f}"
    )

    print(
        f"standing free-time NB alpha: "
        f"{standing_alpha:.4f}"
    )

    return (
        train,
        test,
        model,
        standing_alpha,
    )


# =====================================================================
# VALIDATED PATH-BUDGET DRAWS
# =====================================================================

def draw_hurdle_attempt_budget(
    row,
    family,
    alpha,
    rng,
):
    p_positive = float(
        np.clip(
            row[
                f"pred_{family}_positive_probability"
            ],
            0.0,
            1.0,
        )
    )

    if rng.random() >= p_positive:
        return 0

    conditional_mean = max(
        1.0,
        float(
            row[
                f"pred_{family}_conditional_attempts"
            ]
        ),
    )

    extra_mean = max(
        0.0,
        conditional_mean - 1.0,
    )

    frailty = draw_frailty(
        rng,
        alpha,
    )

    extras = int(
        rng.poisson(
            extra_mean
            * frailty
        )
    )

    return 1 + extras


def draw_landed_budget(
    attempts,
    row,
    family,
    rng,
):
    if attempts <= 0:
        return 0

    predicted_attempts = max(
        float(
            row[
                f"pred_{family}_attempted"
            ]
        ),
        1e-9,
    )

    predicted_landed = max(
        float(
            row[
                f"pred_{family}_landed"
            ]
        ),
        0.0,
    )

    p_land = float(
        np.clip(
            predicted_landed
            / predicted_attempts,
            0.0,
            1.0,
        )
    )

    return int(
        rng.binomial(
            attempts,
            p_land,
        )
    )


def draw_control_total(
    pair_info,
    control_alpha,
    duration,
    rng,
):
    p_any = float(
        np.clip(
            pair_info[
                "pred_control_any_probability"
            ],
            0.0,
            1.0,
        )
    )

    if rng.random() >= p_any:
        return 0.0

    conditional_mean = float(
        np.clip(
            pair_info[
                "pred_control_conditional_total"
            ],
            0.0,
            duration,
        )
    )

    if conditional_mean <= 0:
        return 0.0

    first = min(
        CONTROL_CHUNK_SECONDS,
        conditional_mean,
    )

    expected_extra = max(
        0.0,
        (
            conditional_mean
            - first
        )
        / CONTROL_CHUNK_SECONDS,
    )

    frailty = draw_frailty(
        rng,
        control_alpha,
    )

    extra_chunks = int(
        rng.poisson(
            expected_extra
            * frailty
        )
    )

    total = (
        first
        + extra_chunks
        * CONTROL_CHUNK_SECONDS
    )

    return float(
        np.clip(
            total,
            0.0,
            duration,
        )
    )


# =====================================================================
# CONTROL OWNERSHIP AFTER TD DIFFERENTIAL
# =====================================================================

def fit_td_adjusted_ownership_kappa(
    train,
    train_pair,
    td_control_beta,
):
    """
    Stage 6 fitted ownership variance around prefight share alone.

    Stage 7 first shifts the expected ownership by ACTUAL TD differential
    in training, then fits residual Beta dispersion around that adjusted
    expectation.

    This prevents random ownership variance from drowning out TD signal.
    """

    actual = (
        pair_lookup_from_frame(
            train
        )
    )

    errors = []

    for _, pair_row in (
        train_pair.iterrows()
    ):
        fight_id = str(
            pair_row[
                "fight_id"
            ]
        )

        if fight_id not in actual:
            continue

        a = actual[
            fight_id
        ]

        total_control = (
            a["red_control"]
            + a["blue_control"]
        )

        if total_control <= 0:
            continue

        td_diff = (
            a["red_td_landed"]
            - a["blue_td_landed"]
        )

        base_share = float(
            np.clip(
                pair_row[
                    "pred_red_control_share"
                ],
                1e-5,
                1.0 - 1e-5,
            )
        )

        adjusted_share = float(
            sigmoid(
                logit(
                    base_share
                )
                + td_control_beta
                * td_diff
            )
        )

        actual_share = (
            a["red_control"]
            / total_control
        )

        denom = max(
            adjusted_share
            * (
                1.0
                - adjusted_share
            ),
            1e-6,
        )

        errors.append(
            (
                actual_share
                - adjusted_share
            ) ** 2
            / denom
        )

    ratio = float(
        np.mean(
            errors
        )
    )

    if ratio <= 1e-9:
        kappa = 100.0
    else:
        kappa = (
            1.0 / ratio
            - 1.0
        )

    kappa = float(
        np.clip(
            kappa,
            0.25,
            100.0,
        )
    )

    print()
    print("=" * 115)
    print(
        "TD-ADJUSTED CONTROL OWNERSHIP"
    )
    print("=" * 115)

    print(
        f"TD ownership beta: "
        f"{td_control_beta:+.4f}"
    )

    print(
        f"residual ownership beta "
        f"concentration kappa: "
        f"{kappa:.4f}"
    )

    return kappa


def draw_control_split(
    total_control,
    pair_info,
    red_td_landed,
    blue_td_landed,
    td_control_beta,
    ownership_kappa,
    rng,
):
    if total_control <= 0:
        return 0.0, 0.0

    base_share = float(
        np.clip(
            pair_info[
                "pred_red_control_share"
            ],
            1e-5,
            1.0 - 1e-5,
        )
    )

    td_diff = (
        red_td_landed
        - blue_td_landed
    )

    adjusted_share = float(
        sigmoid(
            logit(
                base_share
            )
            + td_control_beta
            * td_diff
        )
    )

    a = max(
        adjusted_share
        * ownership_kappa,
        1e-5,
    )

    b = max(
        (
            1.0
            - adjusted_share
        )
        * ownership_kappa,
        1e-5,
    )

    path_share = float(
        rng.beta(
            a,
            b,
        )
    )

    red_control = (
        round(
            total_control
            * path_share
            / CONTROL_CHUNK_SECONDS
        )
        * CONTROL_CHUNK_SECONDS
    )

    red_control = float(
        np.clip(
            red_control,
            0.0,
            total_control,
        )
    )

    blue_control = (
        total_control
        - red_control
    )

    return (
        red_control,
        blue_control,
    )


# =====================================================================
# ONE STAGE-7 PATH
# =====================================================================

def simulate_path(
    pair,
    pair_info,
    hurdle_alpha,
    control_alpha,
    ownership_kappa,
    td_control_beta,
    standing_alpha,
    rng,
):
    red = (
        pair[
            pair["side"] == "red"
        ].iloc[0]
    )

    blue = (
        pair[
            pair["side"] == "blue"
        ].iloc[0]
    )

    duration = float(
        red["duration"]
    )

    # --------------------------------------------------------------
    # 1. PRE-DRAW TD / GROUND PATH BUDGETS.
    # --------------------------------------------------------------

    result = {}

    for side, fighter in (
        ("red", red),
        ("blue", blue),
    ):
        for family in (
            "td",
            "ground",
        ):
            attempts = (
                draw_hurdle_attempt_budget(
                    fighter,
                    family,
                    hurdle_alpha[
                        family
                    ],
                    rng,
                )
            )

            landed = (
                draw_landed_budget(
                    attempts,
                    fighter,
                    family,
                    rng,
                )
            )

            result[
                f"{side}_{family}_attempted"
            ] = attempts

            result[
                f"{side}_{family}_landed"
            ] = landed

    # --------------------------------------------------------------
    # 2. DRAW ONE SHARED CONTROL TOTAL.
    # --------------------------------------------------------------

    total_control = (
        draw_control_total(
            pair_info,
            control_alpha,
            duration,
            rng,
        )
    )

    # --------------------------------------------------------------
    # 3. OWNERSHIP RESPONDS TO THIS PATH'S TD DIFFERENTIAL FIRST.
    # --------------------------------------------------------------

    (
        red_control,
        blue_control,
    ) = draw_control_split(
        total_control,
        pair_info,
        result[
            "red_td_landed"
        ],
        result[
            "blue_td_landed"
        ],
        td_control_beta,
        ownership_kappa,
        rng,
    )

    result[
        "red_control"
    ] = red_control

    result[
        "blue_control"
    ] = blue_control

    result[
        "total_control"
    ] = total_control

    # --------------------------------------------------------------
    # 4. CONTROL CONSUMES THE FINITE TIMELINE.
    # --------------------------------------------------------------

    free_seconds = max(
        duration
        - total_control,
        0.0,
    )

    result[
        "free_seconds"
    ] = free_seconds

    # --------------------------------------------------------------
    # 5. STANDING OUTPUT IS AN AVAILABLE-FREE-TIME RATE.
    # --------------------------------------------------------------

    for side, fighter in (
        ("red", red),
        ("blue", blue),
    ):
        rate_15m = max(
            float(
                fighter[
                    "pred_standing_rate_free_15m"
                ]
            ),
            0.0,
        )

        expected_attempts = (
            rate_15m
            * free_seconds
            / 900.0
        )

        frailty = draw_frailty(
            rng,
            standing_alpha,
        )

        attempts = int(
            rng.poisson(
                expected_attempts
                * frailty
            )
        )

        old_pred_attempts = max(
            float(
                fighter[
                    "pred_standing_attempted"
                ]
            ),
            1e-9,
        )

        old_pred_landed = max(
            float(
                fighter[
                    "pred_standing_landed"
                ]
            ),
            0.0,
        )

        p_land = float(
            np.clip(
                old_pred_landed
                / old_pred_attempts,
                0.0,
                1.0,
            )
        )

        landed = int(
            rng.binomial(
                attempts,
                p_land,
            )
        )

        result[
            f"{side}_standing_attempted"
        ] = attempts

        result[
            f"{side}_standing_landed"
        ] = landed

    return result


# =====================================================================
# FIGHT-LEVEL INTERACTION FRAMES
# =====================================================================

def historical_fight_frame(
    result,
):
    rows = []

    for fight_id, group in result.groupby(
        "fight_id",
        sort=False,
    ):
        red = (
            group[
                group["side"] == "red"
            ].iloc[0]
        )

        blue = (
            group[
                group["side"] == "blue"
            ].iloc[0]
        )

        duration = float(
            red["duration"]
        )

        rc = float(
            red[
                "qualified_control_inflicted_seconds"
            ]
        )

        bc = float(
            blue[
                "qualified_control_inflicted_seconds"
            ]
        )

        control = rc + bc

        standing = (
            float(
                red["standing_attempted"]
            )
            + float(
                blue["standing_attempted"]
            )
        )

        rows.append(
            {
                "fight_id":
                    str(fight_id),

                "duration":
                    duration,

                "red_control":
                    rc,

                "blue_control":
                    bc,

                "total_control":
                    control,

                "control_share":
                    control
                    / duration,

                "total_standing":
                    standing,

                "standing_per_15m":
                    standing
                    / duration
                    * 900.0,

                "td_diff":
                    float(
                        red["td_landed"]
                    )
                    - float(
                        blue["td_landed"]
                    ),

                "red_control_share":
                    (
                        rc / control
                        if control > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def simulated_fight_frame(
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
                group["side"] == "red"
            ].iloc[0]
        )

        blue = (
            group[
                group["side"] == "blue"
            ].iloc[0]
        )

        duration = float(
            red["duration"]
        )

        rc = float(
            red["sim_control"]
        )

        bc = float(
            blue["sim_control"]
        )

        control = rc + bc

        standing = (
            float(
                red[
                    "sim_standing_attempted"
                ]
            )
            + float(
                blue[
                    "sim_standing_attempted"
                ]
            )
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
                    rc,

                "blue_control":
                    bc,

                "total_control":
                    control,

                "control_share":
                    control
                    / duration,

                "total_standing":
                    standing,

                "standing_per_15m":
                    standing
                    / duration
                    * 900.0,

                "td_diff":
                    float(
                        red[
                            "sim_td_landed"
                        ]
                    )
                    - float(
                        blue[
                            "sim_td_landed"
                        ]
                    ),

                "red_control_share":
                    (
                        rc / control
                        if control > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def same_td_control_side(
    frame,
):
    x = (
        frame[
            frame[
                "total_control"
            ] > 0
        ]
        .copy()
    )

    control_diff = (
        x["red_control"]
        - x["blue_control"]
    )

    keep = (
        (x["td_diff"] != 0)
        &
        (control_diff != 0)
    )

    x = x[
        keep
    ]

    if len(x) == 0:
        return np.nan, 0

    correct = (
        np.sign(
            x["td_diff"]
        )
        ==
        np.sign(
            x["red_control"]
            - x["blue_control"]
        )
    )

    return (
        float(
            correct.mean()
        ),
        len(x),
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 140)
    print(
        "EVENT CLOCK MC — "
        "STAGE 7 BUDGET + FINITE TIMELINE AUDIT"
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
            f"Expected {FIGHTS} "
            "fresh evaluation fights."
        )

    # --------------------------------------------------------------
    # Standing fields.
    # --------------------------------------------------------------

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

    train = (
        add_historical_free_time(
            train
        )
    )

    test = (
        add_historical_free_time(
            test
        )
    )

    feature_cols = (
        direct_feature_columns()
    )

    # --------------------------------------------------------------
    # Existing validated hurdle calibration.
    # --------------------------------------------------------------

    hurdle_alpha = {}

    for family in (
        "td",
        "ground",
    ):
        hurdle_alpha[
            family
        ] = fit_count_hurdle(
            train,
            test,
            family,
            feature_cols,
        )

    # --------------------------------------------------------------
    # Standing free-time rate.
    # --------------------------------------------------------------

    (
        train,
        test,
        _,
        standing_alpha,
    ) = fit_standing_free_time_model(
        train,
        test,
        feature_cols,
    )

    # --------------------------------------------------------------
    # Shared control.
    # --------------------------------------------------------------

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
        fit_td_adjusted_ownership_kappa(
            train,
            train_pair,
            td_control_beta,
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

    # --------------------------------------------------------------
    # Expected standing count under PREDICTED control amount.
    # This is the prefight Stage-7 standing expectation.
    # --------------------------------------------------------------

    predicted_pair_control = {
        str(
            row["fight_id"]
        ):
        float(
            row[
                "pred_total_control"
            ]
        )
        for _, row
        in test_pair.iterrows()
    }

    test[
        "pred_stage7_free_seconds"
    ] = [
        max(
            float(duration)
            - predicted_pair_control[
                str(fid)
            ],
            0.0,
        )
        for fid, duration
        in zip(
            test["fight_id"],
            test["duration"],
        )
    ]

    test[
        "pred_stage7_standing_attempted"
    ] = (
        test[
            "pred_standing_rate_free_15m"
        ]
        * test[
            "pred_stage7_free_seconds"
        ]
        / 900.0
    )

    # --------------------------------------------------------------
    # Paths.
    # --------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        f"RUNNING STAGE 7 — "
        f"{FIGHTS} fights x "
        f"{PATHS} paths"
    )
    print("=" * 140)

    rows = []

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

        for path in range(
            PATHS
        ):
            rng = (
                np.random.default_rng(
                    SEED
                    + fight_index
                    * 100000
                    + path
                )
            )

            sim = simulate_path(
                pair,
                pair_info,
                hurdle_alpha,
                control_alpha,
                ownership_kappa,
                td_control_beta,
                standing_alpha,
                rng,
            )

            for _, fighter in (
                pair.iterrows()
            ):
                side = (
                    fighter["side"]
                )

                rows.append(
                    {
                        "fight_id":
                            str(fight_id),

                        "path":
                            path,

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

                        "sim_standing_attempted":
                            sim[
                                f"{side}_standing_attempted"
                            ],

                        "sim_standing_landed":
                            sim[
                                f"{side}_standing_landed"
                            ],

                        "sim_td_attempted":
                            sim[
                                f"{side}_td_attempted"
                            ],

                        "sim_td_landed":
                            sim[
                                f"{side}_td_landed"
                            ],

                        "sim_ground_attempted":
                            sim[
                                f"{side}_ground_attempted"
                            ],

                        "sim_ground_landed":
                            sim[
                                f"{side}_ground_landed"
                            ],

                        "sim_control":
                            sim[
                                f"{side}_control"
                            ],

                        "sim_total_control":
                            sim[
                                "total_control"
                            ],

                        "sim_free_seconds":
                            sim[
                                "free_seconds"
                            ],
                    }
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
        rows
    )

    # --------------------------------------------------------------
    # Fighter means.
    # --------------------------------------------------------------

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
            sim_standing_attempted=(
                "sim_standing_attempted",
                "mean",
            ),

            sim_standing_landed=(
                "sim_standing_landed",
                "mean",
            ),

            sim_td_attempted=(
                "sim_td_attempted",
                "mean",
            ),

            sim_td_landed=(
                "sim_td_landed",
                "mean",
            ),

            sim_ground_attempted=(
                "sim_ground_attempted",
                "mean",
            ),

            sim_ground_landed=(
                "sim_ground_landed",
                "mean",
            ),

            sim_control=(
                "sim_control",
                "mean",
            ),
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

    # --------------------------------------------------------------
    # Fighter discrimination.
    # --------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "FIGHTER-LEVEL DISCRIMINATION"
    )
    print("=" * 140)

    targets = (
        (
            "standing_attempted",
            "pred_stage7_standing_attempted",
            "sim_standing_attempted",
            "STANDING ATTEMPTS",
        ),

        (
            "td_attempted",
            "pred_td_attempted",
            "sim_td_attempted",
            "TD ATTEMPTS",
        ),

        (
            "td_landed",
            "pred_td_landed",
            "sim_td_landed",
            "TD LANDED",
        ),

        (
            "ground_attempted",
            "pred_ground_attempted",
            "sim_ground_attempted",
            "GROUND ATTEMPTS",
        ),

        (
            "ground_landed",
            "pred_ground_landed",
            "sim_ground_landed",
            "GROUND LANDED",
        ),

        (
            "qualified_control_inflicted_seconds",
            "pred_qualified_control_inflicted_seconds",
            "sim_control",
            "CONTROL SEC",
        ),
    )

    for (
        actual,
        direct,
        sim,
        label,
    ) in targets:

        (
            _,
            direct_rho,
            direct_mae,
        ) = metrics(
            result[
                actual
            ],
            result[
                direct
            ],
        )

        (
            _,
            sim_rho,
            sim_mae,
        ) = metrics(
            result[
                actual
            ],
            result[
                sim
            ],
        )

        (
            direct_side,
            direct_n,
        ) = within_bout_direction(
            result,
            actual,
            direct,
        )

        (
            sim_side,
            sim_n,
        ) = within_bout_direction(
            result,
            actual,
            sim,
        )

        print()
        print(label)
        print("-" * 140)

        print(
            f"mean | "
            f"HIST={result[actual].mean():.3f} | "
            f"EXPECT={result[direct].mean():.3f} | "
            f"STAGE7={result[sim].mean():.3f}"
        )

        print(
            f"EXPECT | "
            f"rho={direct_rho:+.4f} | "
            f"MAE={direct_mae:.3f} | "
            f"correct-side="
            f"{direct_side:.2%} "
            f"(N={direct_n})"
        )

        print(
            f"STAGE7 | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f} | "
            f"correct-side="
            f"{sim_side:.2%} "
            f"(N={sim_n})"
        )

    # --------------------------------------------------------------
    # Marginal path distributions.
    # --------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "PATH DISTRIBUTIONS"
    )
    print("=" * 140)

    for (
        actual,
        sim,
        label,
    ) in (
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
            "sim_control",
            "CONTROL SEC",
        ),
    ):
        print_dist(
            label,
            result[
                actual
            ],
            paths[
                sim
            ],
        )

    # --------------------------------------------------------------
    # Interaction audit.
    # --------------------------------------------------------------

    hist_fight = (
        historical_fight_frame(
            result
        )
    )

    sim_fight = (
        simulated_fight_frame(
            paths
        )
    )

    print()
    print("=" * 140)
    print(
        "CONTROL ↔ STANDING "
        "FINITE-TIMELINE AUDIT"
    )
    print("=" * 140)

    print(
        "control share vs "
        "standing attempts / 15m:"
    )

    print(
        f"  HIST   = "
        f"{spearman(hist_fight, 'control_share', 'standing_per_15m'):+.4f}"
    )

    print(
        f"  STAGE7 = "
        f"{spearman(sim_fight, 'control_share', 'standing_per_15m'):+.4f}"
    )

    print()
    print("=" * 140)
    print(
        "TD ↔ CONTROL OWNERSHIP"
    )
    print("=" * 140)

    hist_rho = spearman(
        hist_fight[
            hist_fight[
                "total_control"
            ] > 0
        ],
        "td_diff",
        "red_control_share",
    )

    sim_rho = spearman(
        sim_fight[
            sim_fight[
                "total_control"
            ] > 0
        ],
        "td_diff",
        "red_control_share",
    )

    (
        hist_same,
        hist_n,
    ) = same_td_control_side(
        hist_fight
    )

    (
        sim_same,
        sim_n,
    ) = same_td_control_side(
        sim_fight
    )

    print(
        f"TD differential vs "
        f"Red control-share Spearman:"
    )

    print(
        f"  HIST   = "
        f"{hist_rho:+.4f}"
    )

    print(
        f"  STAGE7 = "
        f"{sim_rho:+.4f}"
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
        f"  STAGE7 = "
        f"{sim_same:.2%} "
        f"(N={sim_n})"
    )

    # --------------------------------------------------------------
    # Timeline sanity.
    # --------------------------------------------------------------

    path_fights = (
        paths.drop_duplicates(
            [
                "fight_id",
                "path",
            ]
        )
    )

    print()
    print("=" * 140)
    print(
        "FINITE TIMELINE SANITY"
    )
    print("=" * 140)

    print(
        f"Mean control/path: "
        f"{path_fights['sim_total_control'].mean():.2f}s"
    )

    print(
        f"Mean free time/path: "
        f"{path_fights['sim_free_seconds'].mean():.2f}s"
    )

    violations = int(
        (
            path_fights[
                "sim_total_control"
            ]
            >
            path_fights[
                "duration"
            ]
            + 1e-9
        ).sum()
    )

    print(
        f"Control > duration: "
        f"{violations}"
    )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

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
