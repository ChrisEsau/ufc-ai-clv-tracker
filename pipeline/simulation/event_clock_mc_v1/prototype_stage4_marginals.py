from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    FSR_ATTRS,
    metrics,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import (
    CONTROL_AMOUNT_ALPHA,
    CONTROL_OCCURRENCE_ALPHA,
    ControlHurdle,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)


PATHS = 20
SEED = 20260817

FAMILIES = (
    "distance",
    "clinch",
    "ground",
    "td",
)

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage4_marginals_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage4_marginal_paths.csv"
)


# =============================================================================
# FEATURE COLUMNS
# =============================================================================

def direct_feature_columns():

    cols = [
        "scheduled_rounds",
        "fighter_age",
        "opponent_age",
        "effective_standing_rate",
        "effective_td_rate",
        "effective_ground_rate",
        "td_completion_matchup",
        "standing_accuracy_matchup",
        "ground_accuracy_matchup",
        "retention_mean_base",
        "successful_td_pressure",
        "control_pressure",
        "age_edge",
    ]

    for attr in FSR_ATTRS:
        cols.append(
            f"self_{attr}"
        )
        cols.append(
            f"opp_{attr}"
        )

    return cols


# =============================================================================
# NEGATIVE-BINOMIAL / GAMMA-POISSON DISPERSION
#
# Conditional on path frailty:
#
#     N | G ~ Poisson(mu * G)
#
# and:
#
#     G ~ Gamma(k, scale=1/k)
#
# so:
#
#     E[G] = 1
#
# and therefore:
#
#     E[N] = mu
#
# while variance and zero probability become much larger than Poisson.
#
# NB2:
#
#     Var(N) = mu + alpha * mu^2
#
# where k = 1 / alpha.
# =============================================================================

def estimate_nb_alpha(
    actual,
    predicted,
):

    y = np.asarray(
        actual,
        dtype=float,
    )

    mu = np.maximum(
        np.asarray(
            predicted,
            dtype=float,
        ),
        1e-9,
    )

    numerator = np.sum(
        (y - mu) ** 2
        - y
    )

    denominator = np.sum(
        mu ** 2
    )

    alpha = (
        numerator / denominator
        if denominator > 0
        else 0.0
    )

    return float(
        np.clip(
            alpha,
            1e-6,
            20.0,
        )
    )


def draw_frailty(
    rng,
    alpha,
):

    if alpha <= 1e-5:
        return 1.0

    shape = (
        1.0 / alpha
    )

    scale = alpha

    return float(
        rng.gamma(
            shape=shape,
            scale=scale,
        )
    )


# =============================================================================
# CONTROL CONDITIONAL-AMOUNT DISPERSION
#
# Control already has a hurdle model:
#
#   P(control > 0)
#   E(control | control > 0)
#
# Once the control clock fires, draw a fraction of total fight time from:
#
#   Beta(mean * concentration,
#        (1-mean) * concentration)
#
# This is bounded in [0, fight duration] and preserves the conditional mean.
# =============================================================================

def estimate_beta_concentration(
    train,
    predicted_positive_seconds,
):

    y = train[
        "qualified_control_inflicted_seconds"
    ].to_numpy(float)

    duration = (
        train["duration"]
        .to_numpy(float)
    )

    keep = (
        y > 0
    )

    actual_share = (
        y[keep]
        / duration[keep]
    )

    mean_share = (
        predicted_positive_seconds[
            keep
        ]
        / duration[keep]
    )

    actual_share = np.clip(
        actual_share,
        1e-5,
        1.0 - 1e-5,
    )

    mean_share = np.clip(
        mean_share,
        1e-5,
        1.0 - 1e-5,
    )

    scaled_sq_error = (
        (actual_share - mean_share) ** 2
        / (
            mean_share
            * (1.0 - mean_share)
        )
    )

    ratio = float(
        np.mean(
            scaled_sq_error
        )
    )

    if ratio <= 1e-9:
        concentration = 100.0
    else:
        concentration = (
            1.0 / ratio
            - 1.0
        )

    return float(
        np.clip(
            concentration,
            0.25,
            100.0,
        )
    )


# =============================================================================
# PATH SIMULATION
# =============================================================================

def simulate_path(
    pair,
    rng,
    nb_alpha,
    beta_concentration,
):

    duration = float(
        pair["duration"].iloc[0]
    )

    fighters = {
        row["side"]: row
        for _, row
        in pair.iterrows()
    }

    output = {}

    count_clocks = []

    control_hazards = {}

    for side in (
        "red",
        "blue",
    ):

        row = fighters[
            side
        ]

        for family in FAMILIES:

            output[
                f"{side}_{family}_attempted"
            ] = 0.0

            output[
                f"{side}_{family}_landed"
            ] = 0.0

            expected = max(
                0.0,
                float(
                    row[
                        f"pred_{family}_attempted"
                    ]
                ),
            )

            frailty = (
                draw_frailty(
                    rng,
                    nb_alpha[
                        family
                    ],
                )
            )

            path_mean = (
                expected
                * frailty
            )

            rate = (
                path_mean
                / duration
                if duration > 0
                else 0.0
            )

            if rate > 0:
                count_clocks.append(
                    (
                        side,
                        family,
                        rate,
                    )
                )

        output[
            f"{side}_qualified_control_inflicted_seconds"
        ] = 0.0

        output[
            f"{side}_control_occurred"
        ] = 0.0

        p_control = float(
            np.clip(
                row[
                    "pred_control_positive_probability"
                ],
                0.0,
                0.999999,
            )
        )

        # For a constant exponential clock:
        #
        #   P(event before T)
        #       = 1 - exp(-lambda*T)
        #
        # therefore:
        #
        #   lambda
        #       = -log(1-p) / T
        #
        control_hazards[
            side
        ] = (
            -np.log1p(
                -p_control
            )
            / duration
            if (
                p_control > 0
                and duration > 0
            )
            else 0.0
        )

    active_control = {
        "red": True,
        "blue": True,
    }

    time = 0.0

    while time < duration:

        clocks = list(
            count_clocks
        )

        for side in (
            "red",
            "blue",
        ):

            if (
                active_control[
                    side
                ]
                and control_hazards[
                    side
                ] > 0
            ):
                clocks.append(
                    (
                        side,
                        "control",
                        control_hazards[
                            side
                        ],
                    )
                )

        total_rate = sum(
            clock[2]
            for clock in clocks
        )

        if total_rate <= 0:
            break

        time += float(
            rng.exponential(
                1.0
                / total_rate
            )
        )

        if time >= duration:
            break

        draw = (
            rng.random()
            * total_rate
        )

        running = 0.0
        selected = clocks[-1]

        for clock in clocks:

            running += (
                clock[2]
            )

            if draw <= running:
                selected = clock
                break

        side, family, _ = (
            selected
        )

        row = fighters[
            side
        ]

        if family == "control":

            active_control[
                side
            ] = False

            output[
                f"{side}_control_occurred"
            ] = 1.0

            positive_seconds = float(
                row[
                    "pred_positive_control_seconds"
                ]
            )

            mean_share = (
                positive_seconds
                / duration
                if duration > 0
                else 0.0
            )

            mean_share = float(
                np.clip(
                    mean_share,
                    1e-5,
                    1.0 - 1e-5,
                )
            )

            a = (
                mean_share
                * beta_concentration
            )

            b = (
                (1.0 - mean_share)
                * beta_concentration
            )

            share = float(
                rng.beta(
                    a,
                    b,
                )
            )

            output[
                f"{side}_qualified_control_inflicted_seconds"
            ] = (
                share
                * duration
            )

            continue

        output[
            f"{side}_{family}_attempted"
        ] += 1.0

        expected_att = float(
            row[
                f"pred_{family}_attempted"
            ]
        )

        expected_lnd = float(
            row[
                f"pred_{family}_landed"
            ]
        )

        p_land = (
            np.clip(
                expected_lnd
                / expected_att,
                0.0,
                0.98,
            )
            if expected_att > 1e-9
            else 0.0
        )

        if (
            rng.random()
            < p_land
        ):
            output[
                f"{side}_{family}_landed"
            ] += 1.0

    # Joint physical control constraint.
    total_control = (
        output[
            "red_qualified_control_inflicted_seconds"
        ]
        + output[
            "blue_qualified_control_inflicted_seconds"
        ]
    )

    rescaled = False

    if total_control > duration:

        scale = (
            duration
            / total_control
        )

        output[
            "red_qualified_control_inflicted_seconds"
        ] *= scale

        output[
            "blue_qualified_control_inflicted_seconds"
        ] *= scale

        rescaled = True

    output[
        "control_rescaled"
    ] = float(
        rescaled
    )

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "EVENT CLOCK MC V3 — "
        "MARGINAL DISTRIBUTION CALIBRATION"
    )
    print("=" * 120)

    train, test = (
        prepare_direct_predictions()
    )

    feature_cols = (
        direct_feature_columns()
    )

    x_train = (
        train[
            feature_cols
        ].to_numpy(float)
    )

    x_test = (
        test[
            feature_cols
        ].to_numpy(float)
    )

    train_exposure = (
        train["duration"]
        .to_numpy(float)
        / 900.0
    )

    test_exposure = (
        test["duration"]
        .to_numpy(float)
        / 900.0
    )

    # =========================================================
    # Refit control hurdle so occurrence and positive-amount
    # predictions are explicitly retained.
    # =========================================================

    control_model = (
        ControlHurdle(
            occurrence_alpha=
                CONTROL_OCCURRENCE_ALPHA,
            amount_alpha=
                CONTROL_AMOUNT_ALPHA,
        )
        .fit(
            x_train,
            train[
                "qualified_control_inflicted_seconds"
            ].to_numpy(float),
            train_exposure,
        )
    )

    (
        train_control,
        train_p_control,
        train_positive_rate,
    ) = control_model.predict(
        x_train,
        train_exposure,
    )

    (
        test_control,
        test_p_control,
        test_positive_rate,
    ) = control_model.predict(
        x_test,
        test_exposure,
    )

    train[
        "pred_control_positive_probability"
    ] = train_p_control

    test[
        "pred_control_positive_probability"
    ] = test_p_control

    train[
        "pred_positive_control_seconds"
    ] = (
        train_positive_rate
        * train_exposure
    )

    test[
        "pred_positive_control_seconds"
    ] = (
        test_positive_rate
        * test_exposure
    )

    train[
        "pred_qualified_control_inflicted_seconds"
    ] = train_control

    test[
        "pred_qualified_control_inflicted_seconds"
    ] = test_control

    # =========================================================
    # Fit count overdispersion.
    # =========================================================

    print()
    print("=" * 120)
    print(
        "FITTED GAMMA-POISSON "
        "OVERDISPERSION"
    )
    print("=" * 120)

    nb_alpha = {}

    for family in FAMILIES:

        target = (
            f"{family}_attempted"
        )

        pred = (
            f"pred_{target}"
        )

        alpha = (
            estimate_nb_alpha(
                train[target],
                train[pred],
            )
        )

        nb_alpha[
            family
        ] = alpha

        k = (
            1.0 / alpha
        )

        mu = (
            train[pred]
            .to_numpy(float)
        )

        implied_zero = float(
            np.mean(
                (
                    1.0
                    + alpha * mu
                )
                ** (
                    -1.0
                    / alpha
                )
            )
        )

        actual_zero = float(
            (
                train[target]
                == 0
            ).mean()
        )

        print(
            f"{family:10} | "
            f"alpha={alpha:.4f} | "
            f"k={k:.4f} | "
            f"train zero="
            f"{actual_zero:.2%} | "
            f"NB implied zero="
            f"{implied_zero:.2%}"
        )

    # =========================================================
    # Fit bounded positive-control variance.
    # =========================================================

    beta_concentration = (
        estimate_beta_concentration(
            train,
            train[
                "pred_positive_control_seconds"
            ].to_numpy(float),
        )
    )

    print()
    print("=" * 120)
    print(
        "CONTROL DISTRIBUTION"
    )
    print("=" * 120)

    print(
        f"historical train positive share: "
        f"{(train['qualified_control_inflicted_seconds'] > 0).mean():.2%}"
    )

    print(
        f"predicted train positive share: "
        f"{train['pred_control_positive_probability'].mean():.2%}"
    )

    print(
        f"historical test positive share: "
        f"{(test['qualified_control_inflicted_seconds'] > 0).mean():.2%}"
    )

    print(
        f"predicted test positive share: "
        f"{test['pred_control_positive_probability'].mean():.2%}"
    )

    print(
        f"conditional-control beta "
        f"concentration: "
        f"{beta_concentration:.4f}"
    )

    # =========================================================
    # RUN CLOCKS
    # =========================================================

    print()
    print("=" * 120)
    print(
        f"RUNNING MARGINAL-CALIBRATED "
        f"CLOCKS — "
        f"500 fights x {PATHS} paths"
    )
    print("=" * 120)

    path_rows = []

    fight_groups = list(
        test.groupby(
            "fight_id",
            sort=False,
        )
    )

    for fight_index, (
        fight_id,
        pair,
    ) in enumerate(
        fight_groups
    ):

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

            output = (
                simulate_path(
                    pair,
                    rng,
                    nb_alpha,
                    beta_concentration,
                )
            )

            for _, row in (
                pair.iterrows()
            ):

                side = row[
                    "side"
                ]

                path_row = {
                    "fight_id":
                        fight_id,
                    "path":
                        path_index,
                    "side":
                        side,
                    "fighter_name":
                        row[
                            "fighter_name"
                        ],
                    "duration":
                        float(
                            row[
                                "duration"
                            ]
                        ),
                    "control_rescaled":
                        output[
                            "control_rescaled"
                        ],
                }

                for family in (
                    FAMILIES
                ):

                    path_row[
                        f"sim_{family}_attempted"
                    ] = output[
                        f"{side}_{family}_attempted"
                    ]

                    path_row[
                        f"sim_{family}_landed"
                    ] = output[
                        f"{side}_{family}_landed"
                    ]

                    path_row[
                        f"pred_{family}_attempted"
                    ] = float(
                        row[
                            f"pred_{family}_attempted"
                        ]
                    )

                path_row[
                    "sim_qualified_control_inflicted_seconds"
                ] = output[
                    f"{side}_qualified_control_inflicted_seconds"
                ]

                path_row[
                    "pred_qualified_control_inflicted_seconds"
                ] = float(
                    row[
                        "pred_qualified_control_inflicted_seconds"
                    ]
                )

                path_rows.append(
                    path_row
                )

        if (
            (fight_index + 1)
            % 50
            == 0
        ):
            print(
                f"completed "
                f"{fight_index + 1}/500"
            )

    paths = pd.DataFrame(
        path_rows
    )

    # =========================================================
    # PATH MEANS
    # =========================================================

    agg = {}

    for family in FAMILIES:

        agg[
            f"sim_{family}_attempted"
        ] = "mean"

        agg[
            f"sim_{family}_landed"
        ] = "mean"

    agg[
        "sim_qualified_control_inflicted_seconds"
    ] = "mean"

    mean_sim = (
        paths.groupby(
            [
                "fight_id",
                "side",
                "fighter_name",
            ],
            as_index=False,
        )
        .agg(agg)
    )

    result = test.merge(
        mean_sim,
        on=[
            "fight_id",
            "side",
            "fighter_name",
        ],
        how="left",
        validate="one_to_one",
    )

    for prefix in (
        "",
        "pred_",
        "sim_",
    ):

        result[
            f"{prefix}standing_attempted"
        ] = (
            result[
                f"{prefix}distance_attempted"
            ]
            + result[
                f"{prefix}clinch_attempted"
            ]
        )

        result[
            f"{prefix}standing_landed"
        ] = (
            result[
                f"{prefix}distance_landed"
            ]
            + result[
                f"{prefix}clinch_landed"
            ]
        )

    # =========================================================
    # MARGINAL MEANS
    # =========================================================

    print()
    print("=" * 125)
    print(
        "MARGINAL MEANS / "
        "FIGHTER DISCRIMINATION"
    )
    print("=" * 125)

    for target, label in (
        (
            "standing_attempted",
            "STANDING ATTEMPTS",
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

        direct = (
            f"pred_{target}"
        )

        sim = (
            f"sim_{target}"
        )

        _, direct_rho, direct_mae = (
            metrics(
                result[
                    target
                ],
                result[
                    direct
                ],
            )
        )

        _, sim_rho, sim_mae = (
            metrics(
                result[
                    target
                ],
                result[
                    sim
                ],
            )
        )

        print()
        print(label)
        print("-" * 125)

        print(
            f"HIST={result[target].mean():.3f} | "
            f"DIRECT={result[direct].mean():.3f} | "
            f"CLOCK={result[sim].mean():.3f}"
        )

        print(
            f"DIRECT | "
            f"rho={direct_rho:+.4f} | "
            f"MAE={direct_mae:.3f}"
        )

        print(
            f"CLOCK  | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f}"
        )

    # =========================================================
    # DISTRIBUTION SHAPE
    # =========================================================

    print()
    print("=" * 125)
    print(
        "PATH DISTRIBUTIONS — "
        "HISTORICAL VS CLOCK"
    )
    print("=" * 125)

    for target, label in (
        (
            "distance_attempted",
            "DISTANCE ATTEMPTS",
        ),
        (
            "clinch_attempted",
            "CLINCH ATTEMPTS",
        ),
        (
            "td_attempted",
            "TD ATTEMPTS",
        ),
        (
            "ground_attempted",
            "GROUND ATTEMPTS",
        ),
        (
            "qualified_control_inflicted_seconds",
            "CONTROL SEC",
        ),
    ):

        actual = (
            result[
                target
            ].to_numpy(float)
        )

        synthetic = (
            paths[
                f"sim_{target}"
            ].to_numpy(float)
        )

        print()
        print(label)
        print("-" * 125)

        print(
            f"HIST | "
            f"mean={actual.mean():.3f} | "
            f"std={actual.std(ddof=1):.3f} | "
            f"zero={(actual == 0).mean():.2%} | "
            f"p90={np.quantile(actual, .90):.2f} | "
            f"p99={np.quantile(actual, .99):.2f}"
        )

        print(
            f"SIM  | "
            f"mean={synthetic.mean():.3f} | "
            f"std={synthetic.std(ddof=1):.3f} | "
            f"zero={(synthetic == 0).mean():.2%} | "
            f"p90={np.quantile(synthetic, .90):.2f} | "
            f"p99={np.quantile(synthetic, .99):.2f}"
        )

    # =========================================================
    # CONTROL PHYSICAL SANITY
    # =========================================================

    unique_paths = (
        paths.drop_duplicates(
            [
                "fight_id",
                "path",
            ]
        )
    )

    print()
    print("=" * 125)
    print("CONTROL PHYSICAL SANITY")
    print("=" * 125)

    print(
        f"path share requiring joint "
        f"control rescale: "
        f"{unique_paths['control_rescaled'].mean():.2%}"
    )

    # =========================================================
    # SAVE
    # =========================================================

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
