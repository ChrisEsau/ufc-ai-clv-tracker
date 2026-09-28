from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
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
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    build_pair_frame,
    fit_control_models,
    fit_count_hurdle,
    logit,
    sigmoid,
)

from pipeline.simulation.event_clock_mc_v1.diagnostics_stage7_budget_timeline import (
    add_historical_free_time,
    draw_control_total,
    draw_hurdle_attempt_budget,
    draw_landed_budget,
    fit_standing_free_time_model,
    historical_fight_frame,
    pair_lookup_from_frame,
    print_dist,
    same_td_control_side,
    simulated_fight_frame,
    spearman,
)

from pipeline.simulation.event_clock_mc_v1.diagnostics_stage8_grappling_calibration import (
    fit_directional_ownership_kappa,
)


FIGHTS = 500
PATHS = 20
SEED = 20260817
CONTROL_CHUNK_SECONDS = 10.0

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage9_final_flow_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage9_final_flow_paths_500x20.csv"
)


# =====================================================================
# GROUND POSITIVE-COUNT SHAPE CALIBRATION
# =====================================================================

def ground_shape(values):
    x = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            x.mean()
        ),
        "std": float(
            x.std(ddof=1)
        ),
        "p90": float(
            np.quantile(
                x,
                .90,
            )
        ),
        "p99": float(
            np.quantile(
                x,
                .99,
            )
        ),
    }


def fit_ground_alpha_by_shape(
    train,
):
    """
    Keep the exact Stage-7 hurdle/count architecture.

    Tune ONLY Gamma-Poisson alpha so simulated positive ground-count
    shape resembles historical positive ground-count shape.

    This deliberately does NOT use NB likelihood because Stage 8 showed
    likelihood overweights extreme heterogeneous observations and blows
    up the population tail.
    """

    positive = (
        train[
            "ground_attempted"
        ].astype(float)
        > 0
    )

    actual = (
        train.loc[
            positive,
            "ground_attempted",
        ]
        .astype(float)
        .to_numpy()
    )

    conditional_mean = np.maximum(
        train.loc[
            positive,
            "pred_ground_conditional_attempts",
        ]
        .astype(float)
        .to_numpy(),
        1.0,
    )

    expected_extra = np.maximum(
        conditional_mean
        - 1.0,
        0.0,
    )

    target = ground_shape(
        actual
    )

    candidates = np.geomspace(
        0.02,
        1.50,
        100,
    )

    records = []

    # Common deterministic seed for a fair comparison.
    for candidate_index, alpha in enumerate(
        candidates
    ):
        draws = []

        # Several replications stabilize p99 selection.
        for replicate in range(
            4
        ):
            rng = np.random.default_rng(
                730000
                + candidate_index
                + replicate * 10000
            )

            if alpha <= 1e-12:
                frailty = np.ones(
                    len(
                        expected_extra
                    )
                )
            else:
                shape = (
                    1.0 / alpha
                )

                scale = alpha

                frailty = rng.gamma(
                    shape=shape,
                    scale=scale,
                    size=len(
                        expected_extra
                    ),
                )

            extras = rng.poisson(
                expected_extra
                * frailty
            )

            draws.append(
                1.0
                + extras.astype(float)
            )

        simulated = np.concatenate(
            draws
        )

        shape = ground_shape(
            simulated
        )

        # Relative shape error.
        # Tail gets extra weight because that is the known failure.
        loss = (
            0.50
            * (
                (
                    shape["std"]
                    - target["std"]
                )
                / max(
                    target["std"],
                    1e-6,
                )
            ) ** 2
            +
            0.75
            * (
                (
                    shape["p90"]
                    - target["p90"]
                )
                / max(
                    target["p90"],
                    1e-6,
                )
            ) ** 2
            +
            1.25
            * (
                (
                    shape["p99"]
                    - target["p99"]
                )
                / max(
                    target["p99"],
                    1e-6,
                )
            ) ** 2
        )

        records.append(
            {
                "alpha":
                    float(
                        alpha
                    ),

                "loss":
                    float(
                        loss
                    ),

                "mean":
                    shape[
                        "mean"
                    ],

                "std":
                    shape[
                        "std"
                    ],

                "p90":
                    shape[
                        "p90"
                    ],

                "p99":
                    shape[
                        "p99"
                    ],
            }
        )

    search = pd.DataFrame(
        records
    )

    best = (
        search.sort_values(
            "loss"
        )
        .iloc[0]
    )

    alpha = float(
        best["alpha"]
    )

    print()
    print("=" * 120)
    print(
        "GROUND POSITIVE-PATH "
        "SHAPE CALIBRATION"
    )
    print("=" * 120)

    print(
        f"historical positive rows: "
        f"{positive.sum()}"
    )

    print(
        "Historical positive shape:"
    )

    print(
        f"  mean={target['mean']:.3f} | "
        f"std={target['std']:.3f} | "
        f"p90={target['p90']:.2f} | "
        f"p99={target['p99']:.2f}"
    )

    print()
    print(
        f"selected alpha: "
        f"{alpha:.4f}"
    )

    print(
        "Selected simulated positive shape:"
    )

    print(
        f"  mean={best['mean']:.3f} | "
        f"std={best['std']:.3f} | "
        f"p90={best['p90']:.2f} | "
        f"p99={best['p99']:.2f}"
    )

    print()
    print(
        "Top 5 alpha candidates:"
    )

    print(
        search.sort_values(
            "loss"
        )
        .head(5)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    return alpha


# =====================================================================
# CONTROL MINORITY-PARTICIPATION MODEL
# =====================================================================

def fit_control_minority_models(
    train,
    train_pair,
    td_control_beta,
):
    """
    Separate control allocation into:

      1. dominant owner — handled by Stage-8 TD-adjusted ownership logic;
      2. whether minority fighter receives ANY control;
      3. if yes, how much minority share they receive.

    Training features represent simulated path-state information:
      - realized total control;
      - TD-adjusted ownership confidence.

    No result information is required at forward simulation time.
    """

    actual = (
        pair_lookup_from_frame(
            train
        )
    )

    rows = []

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

        red_control = float(
            a[
                "red_control"
            ]
        )

        blue_control = float(
            a[
                "blue_control"
            ]
        )

        total_control = (
            red_control
            + blue_control
        )

        if total_control <= 0:
            continue

        td_diff = (
            float(
                a[
                    "red_td_landed"
                ]
            )
            - float(
                a[
                    "blue_td_landed"
                ]
            )
        )

        base_share = float(
            np.clip(
                pair_row[
                    "pred_red_control_share"
                ],
                1e-6,
                1.0 - 1e-6,
            )
        )

        adjusted_logit = (
            logit(
                base_share
            )
            + td_control_beta
            * td_diff
        )

        confidence = abs(
            float(
                adjusted_logit
            )
        )

        minority_control = min(
            red_control,
            blue_control,
        )

        minority_positive = (
            minority_control
            > 0
        )

        minority_share = (
            minority_control
            / total_control
        )

        rows.append(
            {
                "fight_id":
                    fight_id,

                "log_total_control":
                    np.log1p(
                        total_control
                    ),

                "ownership_confidence":
                    confidence,

                "minority_positive":
                    int(
                        minority_positive
                    ),

                "minority_share":
                    float(
                        minority_share
                    ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    X = frame[
        [
            "log_total_control",
            "ownership_confidence",
        ]
    ].to_numpy(float)

    y = frame[
        "minority_positive"
    ].to_numpy(int)

    classifier = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                ),
            ),
        ]
    )

    classifier.fit(
        X,
        y,
    )

    positive = (
        frame[
            "minority_positive"
        ]
        == 1
    )

    positive_frame = (
        frame[
            positive
        ]
        .copy()
    )

    X_share = (
        positive_frame[
            [
                "log_total_control",
                "ownership_confidence",
            ]
        ]
        .to_numpy(float)
    )

    # minority_share is bounded (0, .5].
    # Scale to (0,1), then model on logit scale.
    scaled_share = np.clip(
        positive_frame[
            "minority_share"
        ].to_numpy(float)
        / 0.5,
        1e-4,
        1.0 - 1e-4,
    )

    y_share = np.log(
        scaled_share
        / (
            1.0
            - scaled_share
        )
    )

    share_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=1.0
                ),
            ),
        ]
    )

    share_model.fit(
        X_share,
        y_share,
    )

    fitted_share = (
        share_model.predict(
            X_share
        )
    )

    residual_sigma = float(
        np.std(
            y_share
            - fitted_share,
            ddof=1,
        )
    )

    predicted_minor = (
        classifier.predict_proba(
            X
        )[
            :,
            1,
        ]
    )

    actual_minor_rate = float(
        y.mean()
    )

    predicted_minor_rate = float(
        predicted_minor.mean()
    )

    print()
    print("=" * 120)
    print(
        "CONTROL MINORITY-PARTICIPATION MODEL"
    )
    print("=" * 120)

    print(
        f"positive-control training fights: "
        f"{len(frame)}"
    )

    print(
        f"historical minority fighter "
        f"gets control: "
        f"{actual_minor_rate:.2%}"
    )

    print(
        f"model mean probability: "
        f"{predicted_minor_rate:.2%}"
    )

    print(
        f"conditional minority-share mean: "
        f"{positive_frame['minority_share'].mean():.2%}"
    )

    print(
        f"minority-share logit residual sigma: "
        f"{residual_sigma:.4f}"
    )

    return (
        classifier,
        share_model,
        residual_sigma,
    )


# =====================================================================
# STAGE-9 CONTROL SPLIT
# =====================================================================

def draw_stage9_control_split(
    total_control,
    pair_info,
    red_td_landed,
    blue_td_landed,
    td_control_beta,
    dominance_kappa,
    minority_classifier,
    minority_share_model,
    minority_residual_sigma,
    rng,
):
    if total_control <= 0:
        return 0.0, 0.0

    base_red_share = float(
        np.clip(
            pair_info[
                "pred_red_control_share"
            ],
            1e-6,
            1.0 - 1e-6,
        )
    )

    td_diff = (
        red_td_landed
        - blue_td_landed
    )

    adjusted_logit = (
        logit(
            base_red_share
        )
        + td_control_beta
        * td_diff
    )

    adjusted_share = float(
        sigmoid(
            adjusted_logit
        )
    )

    # -------------------------------------------------------------
    # 1. WHO IS THE DOMINANT OWNER?
    #
    # Preserve Stage-8 fitted TD-adjusted directional logic.
    # -------------------------------------------------------------

    a = max(
        adjusted_share
        * dominance_kappa,
        1e-6,
    )

    b = max(
        (
            1.0
            - adjusted_share
        )
        * dominance_kappa,
        1e-6,
    )

    latent_share = float(
        rng.beta(
            a,
            b,
        )
    )

    red_dominant = (
        latent_share
        >= 0.5
    )

    # -------------------------------------------------------------
    # 2. DOES THE MINORITY FIGHTER GET ANY CONTROL?
    # -------------------------------------------------------------

    confidence = abs(
        float(
            adjusted_logit
        )
    )

    X = np.asarray(
        [
            [
                np.log1p(
                    total_control
                ),
                confidence,
            ]
        ],
        dtype=float,
    )

    minority_probability = float(
        minority_classifier.predict_proba(
            X
        )[
            0,
            1,
        ]
    )

    minority_positive = (
        total_control
        >= (
            2.0
            * CONTROL_CHUNK_SECONDS
        )
        and rng.random()
        < minority_probability
    )

    # -------------------------------------------------------------
    # 3. IF YES, HOW MUCH?
    # -------------------------------------------------------------

    if minority_positive:

        predicted_logit_share = float(
            minority_share_model.predict(
                X
            )[
                0
            ]
        )

        noisy_logit_share = (
            predicted_logit_share
            + rng.normal(
                0.0,
                minority_residual_sigma,
            )
        )

        scaled_share = float(
            sigmoid(
                noisy_logit_share
            )
        )

        minority_share = float(
            np.clip(
                0.5
                * scaled_share,
                0.01,
                0.49,
            )
        )

        minority_control = (
            round(
                (
                    total_control
                    * minority_share
                )
                / CONTROL_CHUNK_SECONDS
            )
            * CONTROL_CHUNK_SECONDS
        )

        minority_control = float(
            np.clip(
                minority_control,
                CONTROL_CHUNK_SECONDS,
                max(
                    CONTROL_CHUNK_SECONDS,
                    total_control
                    - CONTROL_CHUNK_SECONDS,
                ),
            )
        )

        if (
            minority_control
            >= total_control
        ):
            minority_control = max(
                0.0,
                total_control
                - CONTROL_CHUNK_SECONDS,
            )

        if (
            minority_control
            > total_control / 2.0
        ):
            minority_control = (
                np.floor(
                    (
                        total_control
                        / 2.0
                    )
                    / CONTROL_CHUNK_SECONDS
                )
                * CONTROL_CHUNK_SECONDS
            )

    else:
        minority_control = 0.0

    dominant_control = (
        total_control
        - minority_control
    )

    if red_dominant:
        red_control = (
            dominant_control
        )

        blue_control = (
            minority_control
        )

    else:
        red_control = (
            minority_control
        )

        blue_control = (
            dominant_control
        )

    return (
        float(
            red_control
        ),
        float(
            blue_control
        ),
    )


# =====================================================================
# ONE STAGE-9 PATH
# =====================================================================

def simulate_stage9_path(
    pair,
    pair_info,
    hurdle_alpha,
    control_alpha,
    dominance_kappa,
    td_control_beta,
    standing_alpha,
    minority_classifier,
    minority_share_model,
    minority_residual_sigma,
    rng,
):
    red = (
        pair[
            pair[
                "side"
            ] == "red"
        ].iloc[0]
    )

    blue = (
        pair[
            pair[
                "side"
            ] == "blue"
        ].iloc[0]
    )

    duration = float(
        red[
            "duration"
        ]
    )

    result = {}

    # -------------------------------------------------------------
    # TD + GROUND PATH BUDGETS
    # -------------------------------------------------------------

    for side, fighter in (
        (
            "red",
            red,
        ),
        (
            "blue",
            blue,
        ),
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

    # -------------------------------------------------------------
    # SHARED CONTROL TOTAL
    # -------------------------------------------------------------

    total_control = (
        draw_control_total(
            pair_info,
            control_alpha,
            duration,
            rng,
        )
    )

    (
        red_control,
        blue_control,
    ) = draw_stage9_control_split(
        total_control,
        pair_info,
        result[
            "red_td_landed"
        ],
        result[
            "blue_td_landed"
        ],
        td_control_beta,
        dominance_kappa,
        minority_classifier,
        minority_share_model,
        minority_residual_sigma,
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

    # -------------------------------------------------------------
    # FINITE FREE-TIME BUDGET
    # -------------------------------------------------------------

    free_seconds = max(
        duration
        - total_control,
        0.0,
    )

    result[
        "free_seconds"
    ] = free_seconds

    # -------------------------------------------------------------
    # STANDING RATE ON FREE TIME
    # -------------------------------------------------------------

    for side, fighter in (
        (
            "red",
            red,
        ),
        (
            "blue",
            blue,
        ),
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

        old_attempts = max(
            float(
                fighter[
                    "pred_standing_attempted"
                ]
            ),
            1e-9,
        )

        old_landed = max(
            float(
                fighter[
                    "pred_standing_landed"
                ]
            ),
            0.0,
        )

        p_land = float(
            np.clip(
                old_landed
                / old_attempts,
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
# MAIN
# =====================================================================

def main():

    print("=" * 140)
    print(
        "EVENT CLOCK MC — "
        "STAGE 9 FINAL BASIC-FLOW CALIBRATION"
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
            "fresh fights."
        )

    # -------------------------------------------------------------
    # Standing fields
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # TD hurdle — frozen Stage-7/8 structure
    # -------------------------------------------------------------

    hurdle_alpha = {}

    hurdle_alpha[
        "td"
    ] = fit_count_hurdle(
        train,
        test,
        "td",
        feature_cols,
    )

    # Populate ground hurdle / conditional predictions.
    original_ground_alpha = (
        fit_count_hurdle(
            train,
            test,
            "ground",
            feature_cols,
        )
    )

    calibrated_ground_alpha = (
        fit_ground_alpha_by_shape(
            train
        )
    )

    hurdle_alpha[
        "ground"
    ] = calibrated_ground_alpha

    print()
    print(
        "GROUND DISPERSION"
    )

    print(
        f"Stage-7 alpha: "
        f"{original_ground_alpha:.4f}"
    )

    print(
        f"Stage-9 calibrated alpha: "
        f"{calibrated_ground_alpha:.4f}"
    )

    # -------------------------------------------------------------
    # Standing — frozen Stage 7
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Shared control
    # -------------------------------------------------------------

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

    dominance_kappa = (
        fit_directional_ownership_kappa(
            train,
            train_pair,
            td_control_beta,
        )
    )

    (
        minority_classifier,
        minority_share_model,
        minority_residual_sigma,
    ) = fit_control_minority_models(
        train,
        train_pair,
        td_control_beta,
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

    # -------------------------------------------------------------
    # Expected standing under predicted control
    # -------------------------------------------------------------

    predicted_pair_control = {
        str(
            row[
                "fight_id"
            ]
        ): float(
            row[
                "pred_total_control"
            ]
        )
        for _, row
        in test_pair.iterrows()
    }

    test[
        "pred_stage9_free_seconds"
    ] = [
        max(
            float(
                duration
            )
            - predicted_pair_control[
                str(
                    fight_id
                )
            ],
            0.0,
        )
        for fight_id, duration
        in zip(
            test[
                "fight_id"
            ],
            test[
                "duration"
            ],
        )
    ]

    test[
        "pred_stage9_standing_attempted"
    ] = (
        test[
            "pred_standing_rate_free_15m"
        ]
        * test[
            "pred_stage9_free_seconds"
        ]
        / 900.0
    )

    # -------------------------------------------------------------
    # RUN PATHS
    # -------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        f"RUNNING STAGE 9 — "
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
                str(
                    fight_id
                )
            ]
        )

        for path in range(
            PATHS
        ):
            rng = np.random.default_rng(
                SEED
                + fight_index * 100000
                + path
            )

            sim = simulate_stage9_path(
                pair,
                pair_info,
                hurdle_alpha,
                control_alpha,
                dominance_kappa,
                td_control_beta,
                standing_alpha,
                minority_classifier,
                minority_share_model,
                minority_residual_sigma,
                rng,
            )

            for _, fighter in (
                pair.iterrows()
            ):
                side = (
                    fighter[
                        "side"
                    ]
                )

                rows.append(
                    {
                        "fight_id":
                            str(
                                fight_id
                            ),

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

    # -------------------------------------------------------------
    # Fighter means
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Fighter discrimination
    # -------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "FIGHTER-LEVEL DISCRIMINATION"
    )
    print("=" * 140)

    targets = (
        (
            "standing_attempted",
            "pred_stage9_standing_attempted",
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
        expected,
        simulated,
        label,
    ) in targets:

        (
            _,
            expected_rho,
            expected_mae,
        ) = metrics(
            result[
                actual
            ],
            result[
                expected
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
                simulated
            ],
        )

        (
            expected_side,
            expected_n,
        ) = within_bout_direction(
            result,
            actual,
            expected,
        )

        (
            sim_side,
            sim_n,
        ) = within_bout_direction(
            result,
            actual,
            simulated,
        )

        print()
        print(label)
        print("-" * 140)

        print(
            f"mean | "
            f"HIST={result[actual].mean():.3f} | "
            f"EXPECT={result[expected].mean():.3f} | "
            f"STAGE9={result[simulated].mean():.3f}"
        )

        print(
            f"EXPECT | "
            f"rho={expected_rho:+.4f} | "
            f"MAE={expected_mae:.3f} | "
            f"correct-side={expected_side:.2%} "
            f"(N={expected_n})"
        )

        print(
            f"STAGE9 | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f} | "
            f"correct-side={sim_side:.2%} "
            f"(N={sim_n})"
        )

    # -------------------------------------------------------------
    # Path distributions
    # -------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "PATH DISTRIBUTIONS"
    )
    print("=" * 140)

    for (
        actual,
        simulated,
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
                simulated
            ],
        )

    # -------------------------------------------------------------
    # Fight interaction diagnostics
    # -------------------------------------------------------------

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
        f"  STAGE9 = "
        f"{spearman(sim_fight, 'control_share', 'standing_per_15m'):+.4f}"
    )

    print()
    print("=" * 140)
    print(
        "TD ↔ CONTROL OWNERSHIP"
    )
    print("=" * 140)

    hist_td_rho = spearman(
        hist_fight[
            hist_fight[
                "total_control"
            ] > 0
        ],
        "td_diff",
        "red_control_share",
    )

    sim_td_rho = spearman(
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
        "TD differential vs "
        "Red control-share Spearman:"
    )

    print(
        f"  HIST   = "
        f"{hist_td_rho:+.4f}"
    )

    print(
        f"  STAGE9 = "
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
        f"  STAGE9 = "
        f"{sim_same:.2%} "
        f"(N={sim_n})"
    )

    # -------------------------------------------------------------
    # Control allocation diagnostics
    # -------------------------------------------------------------

    sim_positive = (
        sim_fight[
            sim_fight[
                "total_control"
            ] > 0
        ]
        .copy()
    )

    hist_positive = (
        hist_fight[
            hist_fight[
                "total_control"
            ] > 0
        ]
        .copy()
    )

    hist_minority_zero = (
        np.minimum(
            hist_positive[
                "red_control"
            ],
            hist_positive[
                "blue_control"
            ],
        )
        == 0
    )

    sim_minority_zero = (
        np.minimum(
            sim_positive[
                "red_control"
            ],
            sim_positive[
                "blue_control"
            ],
        )
        == 0
    )

    print()
    print("=" * 140)
    print(
        "CONTROL ALLOCATION"
    )
    print("=" * 140)

    print(
        "Positive-control fights with "
        "one fighter receiving ZERO control:"
    )

    print(
        f"  HIST   = "
        f"{hist_minority_zero.mean():.2%}"
    )

    print(
        f"  STAGE9 = "
        f"{sim_minority_zero.mean():.2%}"
    )

    fighter_zero_hist = float(
        np.mean(
            result[
                "qualified_control_inflicted_seconds"
            ]
            == 0
        )
    )

    fighter_zero_sim = float(
        np.mean(
            paths[
                "sim_control"
            ]
            == 0
        )
    )

    print()

    print(
        "Fighter-level control zero:"
    )

    print(
        f"  HIST   = "
        f"{fighter_zero_hist:.2%}"
    )

    print(
        f"  STAGE9 = "
        f"{fighter_zero_sim:.2%}"
    )

    # -------------------------------------------------------------
    # Timeline sanity
    # -------------------------------------------------------------

    path_fights = (
        paths.drop_duplicates(
            [
                "fight_id",
                "path",
            ]
        )
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

    print(
        f"Control > duration: "
        f"{violations}"
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

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
