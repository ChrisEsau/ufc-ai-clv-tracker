from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist
from scipy.stats import nbinom

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    metrics,
    within_bout_direction,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import (
    direct_feature_columns,
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
    fit_standing_free_time_model,
    historical_fight_frame,
    pair_lookup_from_frame,
    print_dist,
    same_td_control_side,
    simulate_path,
    simulated_fight_frame,
    spearman,
)


FIGHTS = 500
PATHS = 20
SEED = 20260817

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage8_grappling_calibration_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "stage8_grappling_calibration_paths_500x20.csv"
)


# =====================================================================
# GROUND POSITIVE-PATH DISPERSION
# =====================================================================

def fit_ground_extra_alpha_mle(train):
    """
    Stage 7:
        positive ground attempts =
            1 + Gamma-Poisson(extra attempts)

    Keep that exact architecture.

    Only replace the rough moment estimate of alpha with the
    training-set NB maximum-likelihood estimate.

    For each historical positive ground path:
        observed extra = actual attempts - 1
        expected extra = predicted conditional attempts - 1
    """

    positive = (
        train["ground_attempted"]
        .astype(float)
        > 0
    )

    y = (
        train.loc[
            positive,
            "ground_attempted",
        ]
        .astype(float)
        .to_numpy()
        - 1.0
    )

    mu = np.maximum(
        train.loc[
            positive,
            "pred_ground_conditional_attempts",
        ]
        .astype(float)
        .to_numpy()
        - 1.0,
        1e-9,
    )

    def negative_log_likelihood(
        log_alpha,
    ):
        alpha = float(
            np.exp(
                log_alpha
            )
        )

        size = (
            1.0
            / alpha
        )

        prob = (
            size
            / (
                size
                + mu
            )
        )

        ll = nbinom.logpmf(
            y,
            size,
            prob,
        )

        if not np.all(
            np.isfinite(
                ll
            )
        ):
            return np.inf

        return float(
            -ll.sum()
        )

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(
            np.log(1e-4),
            np.log(10.0),
        ),
        method="bounded",
        options={
            "xatol": 1e-7,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Ground NB MLE failed: "
            f"{result.message}"
        )

    alpha = float(
        np.exp(
            result.x
        )
    )

    actual_extra_mean = float(
        y.mean()
    )

    predicted_extra_mean = float(
        mu.mean()
    )

    actual_positive_mean = float(
        train.loc[
            positive,
            "ground_attempted",
        ].mean()
    )

    print()
    print("=" * 120)
    print(
        "GROUND POSITIVE-PATH "
        "NEGATIVE-BINOMIAL MLE"
    )
    print("=" * 120)

    print(
        f"positive training rows: "
        f"{positive.sum()}"
    )

    print(
        f"actual positive ground mean: "
        f"{actual_positive_mean:.3f}"
    )

    print(
        f"actual mean extra attempts: "
        f"{actual_extra_mean:.3f}"
    )

    print(
        f"predicted mean extra attempts: "
        f"{predicted_extra_mean:.3f}"
    )

    print(
        f"MLE positive-extra alpha: "
        f"{alpha:.4f}"
    )

    return alpha


# =====================================================================
# CONTROL OWNERSHIP DIRECTIONAL CALIBRATION
# =====================================================================

def fit_directional_ownership_kappa(
    train,
    train_pair,
    td_control_beta,
):
    """
    Calibrate only the residual ownership noise.

    We already have:
        prefight Red ownership probability
        TD -> ownership logit coefficient

    For every training fight with:
        - positive control,
        - unequal TD landed totals,
        - unequal control ownership,

    calculate the expected probability that the fighter with more
    TDs owns >50% of control under a Beta ownership draw.

    Choose kappa so that this TRAINING expected direction rate matches
    the historical TRAINING direction rate.

    This is deliberately different from Stage 6/7's residual-variance
    fit, which allowed extreme ownership noise to swamp TD signal.
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

        control_diff = (
            a["red_control"]
            - a["blue_control"]
        )

        if (
            td_diff == 0
            or control_diff == 0
        ):
            continue

        base_share = float(
            np.clip(
                pair_row[
                    "pred_red_control_share"
                ],
                1e-6,
                1.0 - 1e-6,
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

        actual_same_side = (
            np.sign(
                td_diff
            )
            ==
            np.sign(
                control_diff
            )
        )

        rows.append(
            {
                "td_diff":
                    float(
                        td_diff
                    ),

                "adjusted_share":
                    adjusted_share,

                "actual_same_side":
                    float(
                        actual_same_side
                    ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    if len(frame) == 0:
        raise RuntimeError(
            "No ownership calibration rows."
        )

    historical_rate = float(
        frame[
            "actual_same_side"
        ].mean()
    )

    def expected_same_side_rate(
        kappa,
    ):
        mu = np.clip(
            frame[
                "adjusted_share"
            ].to_numpy(float),
            1e-6,
            1.0 - 1e-6,
        )

        a = np.maximum(
            mu * kappa,
            1e-8,
        )

        b = np.maximum(
            (
                1.0 - mu
            )
            * kappa,
            1e-8,
        )

        red_majority_prob = (
            1.0
            - beta_dist.cdf(
                0.5,
                a,
                b,
            )
        )

        td_diff = (
            frame[
                "td_diff"
            ].to_numpy(float)
        )

        probability = np.where(
            td_diff > 0,
            red_majority_prob,
            1.0
            - red_majority_prob,
        )

        return float(
            probability.mean()
        )

    candidates = np.geomspace(
        0.25,
        500.0,
        500,
    )

    expected_rates = np.asarray(
        [
            expected_same_side_rate(
                k
            )
            for k in candidates
        ],
        dtype=float,
    )

    index = int(
        np.argmin(
            np.abs(
                expected_rates
                - historical_rate
            )
        )
    )

    fitted_kappa = float(
        candidates[
            index
        ]
    )

    fitted_rate = float(
        expected_rates[
            index
        ]
    )

    infinite_rate = float(
        np.mean(
            np.where(
                frame[
                    "td_diff"
                ].to_numpy(float)
                > 0,
                frame[
                    "adjusted_share"
                ].to_numpy(float)
                > 0.5,
                frame[
                    "adjusted_share"
                ].to_numpy(float)
                < 0.5,
            )
        )
    )

    print()
    print("=" * 120)
    print(
        "TD-ADJUSTED CONTROL OWNERSHIP "
        "DIRECTION CALIBRATION"
    )
    print("=" * 120)

    print(
        f"training calibration fights: "
        f"{len(frame)}"
    )

    print(
        f"historical same-TD/control-side: "
        f"{historical_rate:.2%}"
    )

    print(
        f"deterministic adjusted-share "
        f"direction ceiling: "
        f"{infinite_rate:.2%}"
    )

    print(
        f"selected ownership kappa: "
        f"{fitted_kappa:.4f}"
    )

    print(
        f"expected training direction "
        f"at selected kappa: "
        f"{fitted_rate:.2%}"
    )

    return fitted_kappa


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 140)
    print(
        "EVENT CLOCK MC — "
        "STAGE 8 GRAPPLING CALIBRATION"
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
            f"Expected exactly "
            f"{FIGHTS} fresh fights."
        )

    # -----------------------------------------------------------------
    # Standing fields.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Hurdle models.
    #
    # Keep TD exactly as Stage 7.
    # Fit normal Stage-7 ground model first so conditional predictions
    # are populated, then override only its extra-count alpha by MLE.
    # -----------------------------------------------------------------

    hurdle_alpha = {}

    hurdle_alpha[
        "td"
    ] = fit_count_hurdle(
        train,
        test,
        "td",
        feature_cols,
    )

    original_ground_alpha = (
        fit_count_hurdle(
            train,
            test,
            "ground",
            feature_cols,
        )
    )

    calibrated_ground_alpha = (
        fit_ground_extra_alpha_mle(
            train
        )
    )

    hurdle_alpha[
        "ground"
    ] = calibrated_ground_alpha

    print()
    print(
        "GROUND ALPHA CHANGE"
    )

    print(
        f"Stage-7 moment alpha: "
        f"{original_ground_alpha:.4f}"
    )

    print(
        f"Stage-8 MLE alpha:    "
        f"{calibrated_ground_alpha:.4f}"
    )

    # -----------------------------------------------------------------
    # Standing stays exactly Stage 7.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Shared control.
    # -----------------------------------------------------------------

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
        fit_directional_ownership_kappa(
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

    # -----------------------------------------------------------------
    # Stage-7 standing expectation based on predicted total control.
    # -----------------------------------------------------------------

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
        "pred_stage8_free_seconds"
    ] = [
        max(
            float(duration)
            - predicted_pair_control[
                str(fight_id)
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
        "pred_stage8_standing_attempted"
    ] = (
        test[
            "pred_standing_rate_free_15m"
        ]
        * test[
            "pred_stage8_free_seconds"
        ]
        / 900.0
    )

    # -----------------------------------------------------------------
    # Run paths.
    # -----------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        f"RUNNING STAGE 8 — "
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

    # -----------------------------------------------------------------
    # Fighter-level means.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Fighter discrimination.
    # -----------------------------------------------------------------

    print()
    print("=" * 140)
    print(
        "FIGHTER-LEVEL DISCRIMINATION"
    )
    print("=" * 140)

    targets = (
        (
            "standing_attempted",
            "pred_stage8_standing_attempted",
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
            f"HIST="
            f"{result[actual].mean():.3f} | "
            f"EXPECT="
            f"{result[expected].mean():.3f} | "
            f"STAGE8="
            f"{result[simulated].mean():.3f}"
        )

        print(
            f"EXPECT | "
            f"rho={expected_rho:+.4f} | "
            f"MAE={expected_mae:.3f} | "
            f"correct-side="
            f"{expected_side:.2%} "
            f"(N={expected_n})"
        )

        print(
            f"STAGE8 | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f} | "
            f"correct-side="
            f"{sim_side:.2%} "
            f"(N={sim_n})"
        )

    # -----------------------------------------------------------------
    # Marginal path distributions.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Fight-level interactions.
    # -----------------------------------------------------------------

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

    hist_time_rho = spearman(
        hist_fight,
        "control_share",
        "standing_per_15m",
    )

    sim_time_rho = spearman(
        sim_fight,
        "control_share",
        "standing_per_15m",
    )

    print(
        "control share vs "
        "standing attempts / 15m:"
    )

    print(
        f"  HIST   = "
        f"{hist_time_rho:+.4f}"
    )

    print(
        f"  STAGE8 = "
        f"{sim_time_rho:+.4f}"
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
        f"  STAGE8 = "
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
        f"  STAGE8 = "
        f"{sim_same:.2%} "
        f"(N={sim_n})"
    )

    # -----------------------------------------------------------------
    # Finite timeline sanity.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------

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
