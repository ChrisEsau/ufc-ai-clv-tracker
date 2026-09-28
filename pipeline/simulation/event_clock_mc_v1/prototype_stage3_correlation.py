from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    build_feature_rows,
    build_historical_targets,
    metrics,
    within_bout_direction,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import (
    BINOMIAL_ALPHA,
    CONTROL_AMOUNT_ALPHA,
    CONTROL_OCCURRENCE_ALPHA,
    POISSON_ALPHA,
    TEST_FIGHTS,
    FAMILIES,
    BinomialRidge,
    ControlHurdle,
    PoissonExposureRidge,
    build_training_master,
)

from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
)


PATHS = 20
SEED = 20260817

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage3_correlation_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage3_correlation_paths.csv"
)

GRAPPLING_TARGETS = {
    "td": "td_attempted",
    "ground": "ground_attempted",
    "control": "qualified_control_inflicted_seconds",
}


# =============================================================================
# DIRECT MODELS — SAME ARCHITECTURE AS STAGE 2
# =============================================================================

def prepare_direct_predictions():

    hist = build_historical_targets()

    train_master, train_fsr = (
        build_training_master()
    )

    print(
        f"training fights: {len(train_master)}"
    )

    train_features = build_feature_rows(
        train_master,
        train_fsr,
    )

    test_master, test_fsr, selection = (
        select_fresh_cohort(
            TEST_FIGHTS,
            offset=0,
        )
    )

    test_features = build_feature_rows(
        test_master,
        test_fsr,
    )

    train = train_features.merge(
        hist,
        on=[
            "fight_id",
            "fighter_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    test = test_features.merge(
        hist,
        on=[
            "fight_id",
            "fighter_name",
        ],
        how="inner",
        validate="one_to_one",
    )

    metadata_cols = {
        "fight_id",
        "event_date",
        "side",
        "fighter_name",
        "opponent_name",
        "duration",
    }

    feature_cols = [
        c
        for c in train_features.columns
        if c not in metadata_cols
    ]

    x_train = train[
        feature_cols
    ].to_numpy(float)

    x_test = test[
        feature_cols
    ].to_numpy(float)

    train_exposure = (
        train["duration"].to_numpy(float)
        / 900.0
    )

    test_exposure = (
        test["duration"].to_numpy(float)
        / 900.0
    )

    for family in FAMILIES:

        att = f"{family}_attempted"
        lnd = f"{family}_landed"

        attempt_model = (
            PoissonExposureRidge(
                alpha=POISSON_ALPHA
            )
            .fit(
                x_train,
                train[att].to_numpy(float),
                train_exposure,
            )
        )

        train_pred_att = (
            attempt_model.predict(
                x_train,
                train_exposure,
            )
        )

        test_pred_att = (
            attempt_model.predict(
                x_test,
                test_exposure,
            )
        )

        completion_model = (
            BinomialRidge(
                alpha=BINOMIAL_ALPHA
            )
            .fit(
                x_train,
                successes=train[
                    lnd
                ].to_numpy(float),
                trials=train[
                    att
                ].to_numpy(float),
            )
        )

        train_p = (
            completion_model
            .predict_probability(
                x_train
            )
        )

        test_p = (
            completion_model
            .predict_probability(
                x_test
            )
        )

        train[
            f"pred_{att}"
        ] = train_pred_att

        train[
            f"pred_{lnd}"
        ] = (
            train_pred_att
            * train_p
        )

        test[
            f"pred_{att}"
        ] = test_pred_att

        test[
            f"pred_{lnd}"
        ] = (
            test_pred_att
            * test_p
        )

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

    train_control, _, _ = (
        control_model.predict(
            x_train,
            train_exposure,
        )
    )

    test_control, _, _ = (
        control_model.predict(
            x_test,
            test_exposure,
        )
    )

    train[
        "pred_qualified_control_inflicted_seconds"
    ] = train_control

    test[
        "pred_qualified_control_inflicted_seconds"
    ] = test_control

    # Individual cap only.
    for frame in (
        train,
        test,
    ):
        frame[
            "pred_qualified_control_inflicted_seconds"
        ] = np.minimum(
            frame[
                "pred_qualified_control_inflicted_seconds"
            ],
            frame["duration"],
        )

    # Fight-level cap for TEST predictions.
    for fight_id, group in test.groupby(
        "fight_id"
    ):

        idx = group.index

        duration = float(
            group["duration"].iloc[0]
        )

        total = float(
            group[
                "pred_qualified_control_inflicted_seconds"
            ].sum()
        )

        if total > duration:

            test.loc[
                idx,
                "pred_qualified_control_inflicted_seconds",
            ] *= (
                duration / total
            )

    print(
        f"fresh evaluation fights: "
        f"{test['fight_id'].nunique()}"
    )

    print(
        f"dates: "
        f"{selection['first_event_date']} "
        f"through "
        f"{selection['last_event_date']}"
    )

    return train, test


# =============================================================================
# FIT A SINGLE SHARED GRAPPLING FACTOR
#
# Residual:
#
#   log(1 + actual) - log(1 + predicted)
#
# We learn how TD, ground striking and control residuals move together.
#
# Then each simulated path gets:
#
#   factor_i = exp(beta_i * Z - beta_i^2 / 2)
#
# E[factor_i] = 1 exactly.
# =============================================================================

def estimate_grappling_factor(
    train,
):

    residuals = pd.DataFrame(
        index=train.index
    )

    for name, target in (
        GRAPPLING_TARGETS.items()
    ):

        pred = f"pred_{target}"

        residuals[name] = (
            np.log1p(
                train[target]
                .to_numpy(float)
            )
            - np.log1p(
                train[pred]
                .to_numpy(float)
            )
        )

    corr = residuals.corr()

    std = residuals.std(
        ddof=1
    )

    print()
    print("=" * 100)
    print(
        "TRAINING LOG-RESIDUAL "
        "CORRELATION"
    )
    print("=" * 100)

    print(
        corr.to_string(
            float_format=lambda x:
                f"{x:+.4f}"
        )
    )

    # ---------------------------------------------------------
    # Three-variable one-factor approximation.
    #
    # If:
    #   r12 = l1*l2
    #   r13 = l1*l3
    #   r23 = l2*l3
    #
    # then:
    #   l1 = sqrt(r12*r13/r23)
    #
    # Use PCA fallback if correlations don't support the
    # positive one-factor solution.
    # ---------------------------------------------------------

    names = [
        "td",
        "ground",
        "control",
    ]

    r_tg = float(
        corr.loc[
            "td",
            "ground",
        ]
    )

    r_tc = float(
        corr.loc[
            "td",
            "control",
        ]
    )

    r_gc = float(
        corr.loc[
            "ground",
            "control",
        ]
    )

    valid_direct = (
        r_tg > 0
        and r_tc > 0
        and r_gc > 0
    )

    if valid_direct:

        loadings = {
            "td": np.sqrt(
                r_tg
                * r_tc
                / r_gc
            ),
            "ground": np.sqrt(
                r_tg
                * r_gc
                / r_tc
            ),
            "control": np.sqrt(
                r_tc
                * r_gc
                / r_tg
            ),
        }

    else:

        values = corr.loc[
            names,
            names,
        ].to_numpy(float)

        eigval, eigvec = (
            np.linalg.eigh(values)
        )

        index = int(
            np.argmax(eigval)
        )

        vector = (
            eigvec[:, index]
            * np.sqrt(
                max(
                    eigval[index],
                    0.0,
                )
            )
        )

        if vector[0] < 0:
            vector *= -1.0

        loadings = dict(
            zip(
                names,
                vector,
            )
        )

    for name in names:
        loadings[name] = float(
            np.clip(
                loadings[name],
                -0.95,
                0.95,
            )
        )

    # Raw log-space factor strength.
    beta = {
        name: float(
            np.clip(
                loadings[name]
                * std[name],
                -1.25,
                1.25,
            )
        )
        for name in names
    }

    print()
    print("=" * 100)
    print("FITTED GRAPPLING FACTOR")
    print("=" * 100)

    for name in names:
        print(
            f"{name:10} | "
            f"resid std="
            f"{std[name]:.4f} | "
            f"loading="
            f"{loadings[name]:+.4f} | "
            f"beta="
            f"{beta[name]:+.4f}"
        )

    return beta


# =============================================================================
# CORRELATED CLOCK PATH
# =============================================================================

def simulate_correlated_path(
    pair,
    rng,
    control_mark_mean,
    beta,
):

    duration = float(
        pair["duration"].iloc[0]
    )

    fighters = {
        row["side"]: row
        for _, row in pair.iterrows()
    }

    output = {}

    latent = {}

    for side in (
        "red",
        "blue",
    ):

        # Independent fighter-level grappling night.
        z = float(
            rng.normal()
        )

        latent[side] = z

        for family in FAMILIES:

            output[
                f"{side}_{family}_attempted"
            ] = 0.0

            output[
                f"{side}_{family}_landed"
            ] = 0.0

        output[
            f"{side}_qualified_control_inflicted_seconds"
        ] = 0.0

        output[
            f"{side}_control_episodes"
        ] = 0.0

        output[
            f"{side}_grappling_latent"
        ] = z

    clocks = []

    for side in (
        "red",
        "blue",
    ):

        row = fighters[side]

        z = latent[side]

        # Mean-preserving shared multipliers.
        grapple_factor = {
            family: np.exp(
                beta[family] * z
                - 0.5
                * beta[family] ** 2
            )
            for family in (
                "td",
                "ground",
                "control",
            )
        }

        for family in FAMILIES:

            expected = max(
                0.0,
                float(
                    row[
                        f"pred_{family}_attempted"
                    ]
                ),
            )

            if family in (
                "td",
                "ground",
            ):
                expected *= (
                    grapple_factor[
                        family
                    ]
                )

            rate = (
                expected / duration
                if duration > 0
                else 0.0
            )

            if rate > 0:
                clocks.append(
                    (
                        side,
                        family,
                        rate,
                    )
                )

        expected_control = max(
            0.0,
            float(
                row[
                    "pred_qualified_control_inflicted_seconds"
                ]
            ),
        )

        expected_control *= (
            grapple_factor[
                "control"
            ]
        )

        expected_episodes = (
            expected_control
            / control_mark_mean
            if control_mark_mean > 0
            else 0.0
        )

        control_rate = (
            expected_episodes
            / duration
            if duration > 0
            else 0.0
        )

        if control_rate > 0:
            clocks.append(
                (
                    side,
                    "control",
                    control_rate,
                )
            )

    total_rate = sum(
        x[2]
        for x in clocks
    )

    time = 0.0
    total_control = 0.0

    if total_rate <= 0:
        return output

    while time < duration:

        time += float(
            rng.exponential(
                1.0 / total_rate
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

            running += clock[2]

            if draw <= running:
                selected = clock
                break

        side, family, _ = (
            selected
        )

        row = fighters[side]

        if family == "control":

            remaining = max(
                0.0,
                duration
                - total_control,
            )

            if remaining <= 0:
                continue

            mark = float(
                rng.gamma(
                    shape=2.0,
                    scale=
                        control_mark_mean
                        / 2.0,
                )
            )

            mark = min(
                mark,
                remaining,
            )

            output[
                f"{side}_qualified_control_inflicted_seconds"
            ] += mark

            output[
                f"{side}_control_episodes"
            ] += 1.0

            total_control += mark

            continue

        output[
            f"{side}_{family}_attempted"
        ] += 1.0

        pred_att = float(
            row[
                f"pred_{family}_attempted"
            ]
        )

        pred_lnd = float(
            row[
                f"pred_{family}_landed"
            ]
        )

        p = (
            np.clip(
                pred_lnd
                / pred_att,
                0.0,
                0.98,
            )
            if pred_att > 1e-9
            else 0.0
        )

        if rng.random() < p:

            output[
                f"{side}_{family}_landed"
            ] += 1.0

    return output


# =============================================================================
# CORRELATION REPORT
# =============================================================================

def correlation_report(
    title,
    frame,
    cols,
):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    corr = (
        frame[list(cols.values())]
        .corr(
            method="spearman"
        )
    )

    corr.index = cols.keys()
    corr.columns = cols.keys()

    print(
        corr.to_string(
            float_format=lambda x:
                f"{x:+.4f}"
        )
    )


def residual_frame(
    actual,
    pred,
    prefix_actual="",
    prefix_pred="pred_",
):

    result = pd.DataFrame()

    for name, target in (
        GRAPPLING_TARGETS.items()
    ):

        a = (
            actual[
                f"{prefix_actual}{target}"
            ]
            .to_numpy(float)
        )

        p = (
            pred[
                f"{prefix_pred}{target}"
            ]
            .to_numpy(float)
        )

        result[name] = (
            np.log1p(a)
            - np.log1p(p)
        )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "EVENT CLOCK MC V2 — "
        "MEAN-PRESERVING GRAPPLING CORRELATION"
    )
    print("=" * 120)

    train, test = (
        prepare_direct_predictions()
    )

    beta = (
        estimate_grappling_factor(
            train
        )
    )

    # Historical control episode duration.
    total_entries = float(
        train[
            "ground_entries"
        ].sum()
    )

    total_control = float(
        train[
            "qualified_control_inflicted_seconds"
        ].sum()
    )

    control_mark_mean = float(
        np.clip(
            total_control
            / max(
                total_entries,
                1.0,
            ),
            15.0,
            120.0,
        )
    )

    print()
    print(
        f"control mark mean: "
        f"{control_mark_mean:.2f}s"
    )

    # =========================================================
    # RUN PATHS
    # =========================================================

    print()
    print("=" * 120)
    print(
        f"RUNNING CORRELATED CLOCKS — "
        f"500 fights x {PATHS} paths"
    )
    print("=" * 120)

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
    ) in enumerate(groups):

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

            result = (
                simulate_correlated_path(
                    pair,
                    rng,
                    control_mark_mean,
                    beta,
                )
            )

            for _, row in (
                pair.iterrows()
            ):

                side = row["side"]

                out = {
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
                }

                for family in FAMILIES:

                    out[
                        f"sim_{family}_attempted"
                    ] = result[
                        f"{side}_{family}_attempted"
                    ]

                    out[
                        f"sim_{family}_landed"
                    ] = result[
                        f"{side}_{family}_landed"
                    ]

                out[
                    "sim_qualified_control_inflicted_seconds"
                ] = result[
                    f"{side}_qualified_control_inflicted_seconds"
                ]

                out[
                    "grappling_latent"
                ] = result[
                    f"{side}_grappling_latent"
                ]

                # Copy direct means for residual comparison.
                for name, target in (
                    GRAPPLING_TARGETS.items()
                ):

                    out[
                        f"pred_{target}"
                    ] = float(
                        row[
                            f"pred_{target}"
                        ]
                    )

                path_rows.append(
                    out
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

    aggregation = {}

    for family in FAMILIES:

        aggregation[
            f"sim_{family}_attempted"
        ] = "mean"

        aggregation[
            f"sim_{family}_landed"
        ] = "mean"

    aggregation[
        "sim_qualified_control_inflicted_seconds"
    ] = "mean"

    sim_mean = (
        paths.groupby(
            [
                "fight_id",
                "side",
                "fighter_name",
            ],
            as_index=False,
        )
        .agg(aggregation)
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

    # Standing totals.
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
    # MARGINAL CHECK
    # =========================================================

    print()
    print("=" * 135)
    print(
        "MARGINAL CALIBRATION — "
        "SHOULD REMAIN CLOSE TO V1"
    )
    print("=" * 135)

    report = (
        (
            "standing_attempted",
            "standing attempts",
        ),
        (
            "standing_landed",
            "standing landed",
        ),
        (
            "td_attempted",
            "TD attempts",
        ),
        (
            "td_landed",
            "TD landed",
        ),
        (
            "ground_attempted",
            "ground attempts",
        ),
        (
            "ground_landed",
            "ground landed",
        ),
        (
            "qualified_control_inflicted_seconds",
            "control seconds",
        ),
    )

    for target, label in report:

        direct = (
            f"pred_{target}"
        )

        clock = (
            f"sim_{target}"
        )

        (
            dr,
            drho,
            dmae,
        ) = metrics(
            result[target],
            result[direct],
        )

        (
            cr,
            crho,
            cmae,
        ) = metrics(
            result[target],
            result[clock],
        )

        print()
        print(label.upper())
        print("-" * 135)

        print(
            f"HIST={result[target].mean():.3f} | "
            f"DIRECT={result[direct].mean():.3f} | "
            f"CLOCK={result[clock].mean():.3f}"
        )

        print(
            f"DIRECT | "
            f"rho={drho:+.4f} | "
            f"MAE={dmae:.3f}"
        )

        print(
            f"CLOCK  | "
            f"rho={crho:+.4f} | "
            f"MAE={cmae:.3f}"
        )

    # =========================================================
    # RAW OUTPUT CORRELATIONS
    # =========================================================

    historical_cols = {
        "TD":
            "td_attempted",
        "GROUND":
            "ground_attempted",
        "CONTROL":
            "qualified_control_inflicted_seconds",
    }

    direct_cols = {
        "TD":
            "pred_td_attempted",
        "GROUND":
            "pred_ground_attempted",
        "CONTROL":
            "pred_qualified_control_inflicted_seconds",
    }

    synthetic_cols = {
        "TD":
            "sim_td_attempted",
        "GROUND":
            "sim_ground_attempted",
        "CONTROL":
            "sim_qualified_control_inflicted_seconds",
    }

    correlation_report(
        "HISTORICAL RAW OUTPUT SPEARMAN",
        result,
        historical_cols,
    )

    correlation_report(
        "DIRECT-MEAN RAW OUTPUT SPEARMAN",
        result,
        direct_cols,
    )

    correlation_report(
        "SYNTHETIC PATH-POPULATION RAW SPEARMAN",
        paths,
        synthetic_cols,
    )

    # =========================================================
    # RESIDUAL CORRELATIONS
    #
    # This is the most important test.
    # Does a fighter having an unexpectedly grappling-heavy
    # night move TD/control/ground together?
    # =========================================================

    hist_resid = pd.DataFrame()

    sim_resid = pd.DataFrame()

    for name, target in (
        GRAPPLING_TARGETS.items()
    ):

        hist_resid[name] = (
            np.log1p(
                result[target]
            )
            - np.log1p(
                result[
                    f"pred_{target}"
                ]
            )
        )

        sim_resid[name] = (
            np.log1p(
                paths[
                    f"sim_{target}"
                ]
            )
            - np.log1p(
                paths[
                    f"pred_{target}"
                ]
            )
        )

    print()
    print("=" * 100)
    print(
        "HISTORICAL RESIDUAL "
        "SPEARMAN"
    )
    print("=" * 100)

    print(
        hist_resid.corr(
            method="spearman"
        ).to_string(
            float_format=lambda x:
                f"{x:+.4f}"
        )
    )

    print()
    print("=" * 100)
    print(
        "SYNTHETIC PATH RESIDUAL "
        "SPEARMAN"
    )
    print("=" * 100)

    print(
        sim_resid.corr(
            method="spearman"
        ).to_string(
            float_format=lambda x:
                f"{x:+.4f}"
        )
    )

    # =========================================================
    # DISPERSION / ZERO CHECK
    # =========================================================

    print()
    print("=" * 115)
    print(
        "DISTRIBUTION SHAPE — "
        "HISTORICAL VS SYNTHETIC PATHS"
    )
    print("=" * 115)

    for target, label in (
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
            result[target]
            .to_numpy(float)
        )

        synthetic = (
            paths[
                f"sim_{target}"
            ]
            .to_numpy(float)
        )

        print()
        print(label)
        print("-" * 115)

        print(
            f"HIST | "
            f"mean={actual.mean():.3f} | "
            f"std={actual.std(ddof=1):.3f} | "
            f"zero={(actual == 0).mean():.2%}"
        )

        print(
            f"SIM  | "
            f"mean={synthetic.mean():.3f} | "
            f"std={synthetic.std(ddof=1):.3f} | "
            f"zero={(synthetic == 0).mean():.2%}"
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
    print(f"wrote: {OUT}")
    print(f"wrote: {PATH_OUT}")


if __name__ == "__main__":
    main()
