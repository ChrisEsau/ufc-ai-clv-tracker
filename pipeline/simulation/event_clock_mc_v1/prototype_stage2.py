from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.common.paths import (
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
    MASTER_PATH,
)

from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    build_feature_rows,
    build_historical_targets,
    metrics,
    simulate_fight,
    within_bout_direction,
)


# =============================================================================
# SETTINGS
# =============================================================================

CUTOFF = pd.Timestamp("2025-03-22")

TRAIN_MAX_FIGHTS = 3000
TEST_FIGHTS = 500
PATHS = 20

SEED = 20260817

POISSON_ALPHA = 20.0
BINOMIAL_ALPHA = 20.0
CONTROL_OCCURRENCE_ALPHA = 20.0
CONTROL_AMOUNT_ALPHA = 20.0

OLD_PATH = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage1_500x20.csv"
)

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage2_500x20.csv"
)

FAMILIES = (
    "distance",
    "clinch",
    "ground",
    "td",
)


# =============================================================================
# FEATURE PREPROCESSOR
# =============================================================================

class Standardizer:
    def fit(self, x):
        x = np.asarray(x, dtype=float)

        self.mean_ = np.nanmean(
            x,
            axis=0,
        )

        self.mean_ = np.where(
            np.isfinite(self.mean_),
            self.mean_,
            0.0,
        )

        filled = np.where(
            np.isfinite(x),
            x,
            self.mean_,
        )

        self.std_ = np.std(
            filled,
            axis=0,
        )

        self.std_ = np.where(
            self.std_ > 1e-9,
            self.std_,
            1.0,
        )

        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)

        filled = np.where(
            np.isfinite(x),
            x,
            self.mean_,
        )

        return (
            filled - self.mean_
        ) / self.std_


# =============================================================================
# POISSON RIDGE
#
# Direct arithmetic conditional mean:
#
#   E[count] = exposure * exp(X beta)
#
# exposure = observed fight seconds / 900
# =============================================================================

class PoissonExposureRidge:
    def __init__(
        self,
        alpha=20.0,
    ):
        self.alpha = float(alpha)

    def fit(
        self,
        x,
        y,
        exposure,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        exposure = np.asarray(
            exposure,
            dtype=float,
        )

        self.scaler_ = Standardizer().fit(x)

        z = self.scaler_.transform(x)

        exposure = np.maximum(
            exposure,
            1e-6,
        )

        base_rate = (
            y.sum()
            / exposure.sum()
        )

        initial = np.zeros(
            z.shape[1] + 1,
            dtype=float,
        )

        initial[0] = np.log(
            max(base_rate, 1e-6)
        )

        def objective(beta):
            eta = (
                beta[0]
                + z @ beta[1:]
            )

            clipped = np.clip(
                eta,
                -15.0,
                15.0,
            )

            mu = (
                exposure
                * np.exp(clipped)
            )

            loss = np.sum(
                mu
                - y * eta
            )

            penalty = (
                0.5
                * self.alpha
                * np.sum(beta[1:] ** 2)
            )

            return loss + penalty

        def gradient(beta):
            eta = (
                beta[0]
                + z @ beta[1:]
            )

            clipped = np.clip(
                eta,
                -15.0,
                15.0,
            )

            mu = (
                exposure
                * np.exp(clipped)
            )

            residual = (
                mu - y
            )

            grad = np.empty_like(beta)

            grad[0] = residual.sum()

            grad[1:] = (
                z.T @ residual
                + self.alpha * beta[1:]
            )

            return grad

        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            options={
                "maxiter": 1000,
                "ftol": 1e-10,
            },
        )

        if not result.success:
            raise RuntimeError(
                "Poisson fit failed: "
                f"{result.message}"
            )

        self.beta_ = result.x

        return self

    def predict(
        self,
        x,
        exposure,
    ):
        z = self.scaler_.transform(x)

        eta = (
            self.beta_[0]
            + z @ self.beta_[1:]
        )

        rate = np.exp(
            np.clip(
                eta,
                -15.0,
                15.0,
            )
        )

        return (
            np.asarray(
                exposure,
                dtype=float,
            )
            * rate
        )


# =============================================================================
# BINOMIAL RIDGE
#
# Used for:
# - strike landing conditional on attempt
# - TD completion conditional on attempt
# =============================================================================

class BinomialRidge:
    def __init__(
        self,
        alpha=20.0,
    ):
        self.alpha = float(alpha)

    def fit(
        self,
        x,
        successes,
        trials,
    ):
        x = np.asarray(x, dtype=float)

        successes = np.asarray(
            successes,
            dtype=float,
        )

        trials = np.asarray(
            trials,
            dtype=float,
        )

        keep = (
            np.isfinite(successes)
            & np.isfinite(trials)
            & (trials > 0)
        )

        x = x[keep]
        successes = successes[keep]
        trials = trials[keep]

        self.scaler_ = Standardizer().fit(x)

        z = self.scaler_.transform(x)

        overall = np.clip(
            successes.sum()
            / trials.sum(),
            1e-5,
            1.0 - 1e-5,
        )

        initial = np.zeros(
            z.shape[1] + 1,
            dtype=float,
        )

        initial[0] = np.log(
            overall / (1.0 - overall)
        )

        def objective(beta):
            eta = (
                beta[0]
                + z @ beta[1:]
            )

            loss = np.sum(
                trials
                * np.logaddexp(
                    0.0,
                    eta,
                )
                - successes * eta
            )

            penalty = (
                0.5
                * self.alpha
                * np.sum(beta[1:] ** 2)
            )

            return loss + penalty

        def gradient(beta):
            eta = (
                beta[0]
                + z @ beta[1:]
            )

            p = (
                1.0
                / (
                    1.0
                    + np.exp(
                        -np.clip(
                            eta,
                            -30.0,
                            30.0,
                        )
                    )
                )
            )

            residual = (
                trials * p
                - successes
            )

            grad = np.empty_like(beta)

            grad[0] = residual.sum()

            grad[1:] = (
                z.T @ residual
                + self.alpha * beta[1:]
            )

            return grad

        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="L-BFGS-B",
            options={
                "maxiter": 1000,
                "ftol": 1e-10,
            },
        )

        if not result.success:
            raise RuntimeError(
                "Binomial fit failed: "
                f"{result.message}"
            )

        self.beta_ = result.x

        return self

    def predict_probability(self, x):
        z = self.scaler_.transform(x)

        eta = (
            self.beta_[0]
            + z @ self.beta_[1:]
        )

        return (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        eta,
                        -30.0,
                        30.0,
                    )
                )
            )
        )


# =============================================================================
# CONTROL HURDLE
#
# PART 1:
#   P(control > 0)
#
# PART 2:
#   E(control rate | control > 0)
#
# Positive amount model uses log-rate ridge + Duan smearing correction,
# which converts the log prediction back toward the arithmetic mean.
# =============================================================================

class ControlHurdle:
    def __init__(
        self,
        occurrence_alpha=20.0,
        amount_alpha=20.0,
    ):
        self.occurrence_alpha = float(
            occurrence_alpha
        )

        self.amount_alpha = float(
            amount_alpha
        )

    def fit(
        self,
        x,
        control_seconds,
        exposure,
    ):
        x = np.asarray(x, dtype=float)

        y = np.asarray(
            control_seconds,
            dtype=float,
        )

        exposure = np.maximum(
            np.asarray(
                exposure,
                dtype=float,
            ),
            1e-6,
        )

        positive = (
            y > 0
        ).astype(float)

        # -------------------------------------------------------------
        # Occurrence
        # -------------------------------------------------------------

        self.occurrence_ = BinomialRidge(
            alpha=self.occurrence_alpha
        ).fit(
            x,
            successes=positive,
            trials=np.ones_like(
                positive
            ),
        )

        # -------------------------------------------------------------
        # Positive amount
        # -------------------------------------------------------------

        keep = (
            y > 0
        )

        x_pos = x[keep]

        exposure_pos = (
            exposure[keep]
        )

        y_rate = (
            y[keep]
            / exposure_pos
        )

        self.amount_scaler_ = (
            Standardizer().fit(
                x_pos
            )
        )

        z = self.amount_scaler_.transform(
            x_pos
        )

        design = np.column_stack(
            [
                np.ones(len(z)),
                z,
            ]
        )

        target = np.log(
            np.maximum(
                y_rate,
                1e-6,
            )
        )

        # Fight duration gives more stable positive-control rate estimates.
        weights = np.clip(
            exposure_pos,
            0.10,
            5.0 / 3.0,
        )

        sqrt_w = np.sqrt(weights)

        weighted_design = (
            design
            * sqrt_w[:, None]
        )

        weighted_target = (
            target
            * sqrt_w
        )

        penalty = (
            np.eye(
                design.shape[1]
            )
            * self.amount_alpha
        )

        penalty[0, 0] = 0.0

        lhs = (
            weighted_design.T
            @ weighted_design
            + penalty
        )

        rhs = (
            weighted_design.T
            @ weighted_target
        )

        self.amount_beta_ = (
            np.linalg.solve(
                lhs,
                rhs,
            )
        )

        fitted_log = (
            design
            @ self.amount_beta_
        )

        residual = (
            target
            - fitted_log
        )

        # Duan smearing:
        # estimate E[Y] rather than exp(E[log Y]).
        self.smearing_ = float(
            np.average(
                np.exp(
                    np.clip(
                        residual,
                        -10.0,
                        10.0,
                    )
                ),
                weights=weights,
            )
        )

        self.smearing_ = float(
            np.clip(
                self.smearing_,
                1.0,
                10.0,
            )
        )

        return self

    def predict(
        self,
        x,
        exposure,
    ):
        x = np.asarray(x, dtype=float)

        exposure = np.asarray(
            exposure,
            dtype=float,
        )

        p_positive = (
            self.occurrence_
            .predict_probability(x)
        )

        z = (
            self.amount_scaler_
            .transform(x)
        )

        design = np.column_stack(
            [
                np.ones(len(z)),
                z,
            ]
        )

        positive_rate = (
            np.exp(
                np.clip(
                    design
                    @ self.amount_beta_,
                    -15.0,
                    15.0,
                )
            )
            * self.smearing_
        )

        expected = (
            p_positive
            * positive_rate
            * exposure
        )

        return (
            expected,
            p_positive,
            positive_rate,
        )


# =============================================================================
# TRAINING COHORT
#
# Unlike the old generic population harness, this does NOT impose the
# >=3-prior-UFC-fights evaluation rule.
#
# We want the prediction model training population to resemble the fresh
# event-clock evaluation population.
# =============================================================================

def build_training_master():
    master = (
        pd.read_parquet(
            MASTER_PATH
        )
        .drop_duplicates(
            "fight_id"
        )
        .copy()
    )

    master["event_date"] = (
        pd.to_datetime(
            master["date"],
            errors="raise",
        )
    )

    fsr = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    ).copy()

    fsr["fight_id"] = (
        fsr["fight_id"]
        .astype(str)
    )

    valid = (
        fsr.groupby(
            "fight_id"
        )
        .size()
    )

    valid_ids = set(
        valid[
            valid == 2
        ].index
    )

    master["fight_id"] = (
        master["fight_id"]
        .astype(str)
    )

    master = master[
        (master["event_date"] < CUTOFF)
        & master["fight_id"].isin(
            valid_ids
        )
        & master["total_rounds"].isin(
            [3, 5]
        )
        & master["match_time_sec"].notna()
    ].copy()

    master = (
        master.sort_values(
            [
                "event_date",
                "fight_id",
            ]
        )
        .tail(
            TRAIN_MAX_FIGHTS
        )
        .reset_index(drop=True)
    )

    return master, fsr


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "EVENT CLOCK MC V1 — "
        "ARITHMETIC-MEAN DIRECT OUTPUT MODELS"
    )
    print("=" * 120)

    hist = build_historical_targets()

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    train_master, train_fsr = (
        build_training_master()
    )

    print()
    print(
        f"training fights before cutoff: "
        f"{len(train_master)}"
    )

    print(
        "prior-UFC-fight restriction: NONE"
    )

    train_features = build_feature_rows(
        train_master,
        train_fsr,
    )

    # ---------------------------------------------------------------------
    # Exact fresh evaluation cohort
    # ---------------------------------------------------------------------

    test_master, test_fsr, selection = (
        select_fresh_cohort(
            TEST_FIGHTS,
            offset=0,
        )
    )

    print()
    print(
        f"fresh evaluation fights: "
        f"{len(test_master)}"
    )

    print(
        f"fresh dates: "
        f"{selection['first_event_date']} "
        f"through "
        f"{selection['last_event_date']}"
    )

    test_features = build_feature_rows(
        test_master,
        test_fsr,
    )

    # ---------------------------------------------------------------------
    # Targets
    # ---------------------------------------------------------------------

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

    print()
    print(
        f"training fighter-fight rows "
        f"with targets: {len(train)}"
    )

    print(
        f"evaluation fighter-fight rows "
        f"with targets: {len(test)}"
    )

    if (
        test["fight_id"].nunique()
        != TEST_FIGHTS
    ):
        raise RuntimeError(
            "evaluation join lost fights"
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
        col
        for col in train_features.columns
        if col not in metadata_cols
    ]

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

    exposure_train = (
        train["duration"]
        .to_numpy(float)
        / 900.0
    )

    exposure_test = (
        test["duration"]
        .to_numpy(float)
        / 900.0
    )

    # ---------------------------------------------------------------------
    # Fit attempt count + conditional completion models
    # ---------------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "DIRECT PREFIGHT COUNT / "
        "COMPLETION MODELS"
    )
    print("=" * 120)

    for family in FAMILIES:

        attempt_col = (
            f"{family}_attempted"
        )

        landed_col = (
            f"{family}_landed"
        )

        attempt_model = (
            PoissonExposureRidge(
                alpha=POISSON_ALPHA
            )
            .fit(
                x_train,
                train[
                    attempt_col
                ].to_numpy(float),
                exposure_train,
            )
        )

        pred_attempts = (
            attempt_model.predict(
                x_test,
                exposure_test,
            )
        )

        completion_model = (
            BinomialRidge(
                alpha=BINOMIAL_ALPHA
            )
            .fit(
                x_train,
                successes=train[
                    landed_col
                ].to_numpy(float),
                trials=train[
                    attempt_col
                ].to_numpy(float),
            )
        )

        p_landed = (
            completion_model
            .predict_probability(
                x_test
            )
        )

        pred_landed = (
            pred_attempts
            * p_landed
        )

        test[
            f"pred_{attempt_col}"
        ] = pred_attempts

        test[
            f"pred_{landed_col}"
        ] = pred_landed

        test[
            f"pred_{family}_completion"
        ] = p_landed

        print(
            f"{family:10} | "
            f"train att mean="
            f"{train[attempt_col].mean():.3f} | "
            f"test HIST="
            f"{test[attempt_col].mean():.3f} | "
            f"test PRED="
            f"{pred_attempts.mean():.3f}"
        )

    # ---------------------------------------------------------------------
    # Control hurdle
    # ---------------------------------------------------------------------

    print()
    print("=" * 120)
    print("CONTROL HURDLE MODEL")
    print("=" * 120)

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
            exposure_train,
        )
    )

    (
        pred_control,
        pred_control_positive,
        pred_positive_control_rate,
    ) = control_model.predict(
        x_test,
        exposure_test,
    )

    test[
        "pred_qualified_control_inflicted_seconds"
    ] = pred_control

    test[
        "pred_control_positive_probability"
    ] = pred_control_positive

    test[
        "pred_positive_control_rate_per_15m"
    ] = pred_positive_control_rate

    # Individual physical cap.
    test[
        "pred_qualified_control_inflicted_seconds"
    ] = np.minimum(
        test[
            "pred_qualified_control_inflicted_seconds"
        ],
        test["duration"],
    )

    # Joint physical cap.
    rescaled_control_fights = 0

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

            rescaled_control_fights += 1

    print(
        f"training positive-control share: "
        f"{(train['qualified_control_inflicted_seconds'] > 0).mean():.2%}"
    )

    print(
        f"test historical positive-control share: "
        f"{(test['qualified_control_inflicted_seconds'] > 0).mean():.2%}"
    )

    print(
        f"test predicted mean positive probability: "
        f"{test['pred_control_positive_probability'].mean():.2%}"
    )

    print(
        f"Duan smearing factor: "
        f"{control_model.smearing_:.4f}"
    )

    print(
        f"control predictions requiring "
        f"fight-level physical rescale: "
        f"{rescaled_control_fights}/"
        f"{TEST_FIGHTS}"
    )

    # ---------------------------------------------------------------------
    # Control mark duration remains a path-variance choice, not mean model.
    # ---------------------------------------------------------------------

    total_train_entries = float(
        train[
            "ground_entries"
        ].sum()
    )

    total_train_control = float(
        train[
            "qualified_control_inflicted_seconds"
        ].sum()
    )

    control_mark_mean = (
        total_train_control
        / total_train_entries
        if total_train_entries > 0
        else 60.0
    )

    control_mark_mean = float(
        np.clip(
            control_mark_mean,
            15.0,
            120.0,
        )
    )

    print(
        f"control episode mark mean: "
        f"{control_mark_mean:.2f}s"
    )

    # ---------------------------------------------------------------------
    # Actual competing-clock MC
    # ---------------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        f"RUNNING COMPETING CLOCKS — "
        f"{TEST_FIGHTS} fights x "
        f"{PATHS} paths"
    )
    print("=" * 120)

    sim_rows = []

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

        if len(pair) != 2:
            raise RuntimeError(
                f"{fight_id}: "
                "expected two corners"
            )

        result = simulate_fight(
            pair,
            PATHS,
            SEED
            + fight_index * 100000,
            control_mark_mean,
        )

        for _, fighter_row in (
            pair.iterrows()
        ):

            side = (
                fighter_row["side"]
            )

            out = {
                "fight_id": fight_id,
                "side": side,
                "fighter_name":
                    fighter_row[
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
                "sim_control_episodes"
            ] = result[
                f"{side}_control_episodes"
            ]

            sim_rows.append(out)

        if (
            (fight_index + 1) % 50
            == 0
            or fight_index + 1
            == TEST_FIGHTS
        ):
            print(
                f"completed "
                f"{fight_index + 1}/"
                f"{TEST_FIGHTS}"
            )

    sim = pd.DataFrame(
        sim_rows
    )

    test = test.merge(
        sim,
        on=[
            "fight_id",
            "side",
            "fighter_name",
        ],
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------------
    # Standing = distance + clinch.
    # ---------------------------------------------------------------------

    for suffix in (
        "",
        "pred_",
        "sim_",
    ):

        test[
            f"{suffix}standing_attempted"
        ] = (
            test[
                f"{suffix}distance_attempted"
            ]
            + test[
                f"{suffix}clinch_attempted"
            ]
        )

        test[
            f"{suffix}standing_landed"
        ] = (
            test[
                f"{suffix}distance_landed"
            ]
            + test[
                f"{suffix}clinch_landed"
            ]
        )

    # ---------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------

    report_targets = (
        (
            "standing_attempted",
            "standing attempts",
        ),
        (
            "standing_landed",
            "standing landed",
        ),
        (
            "distance_attempted",
            "distance attempts",
        ),
        (
            "clinch_attempted",
            "clinch attempts",
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
            "td_attempted",
            "TD attempts",
        ),
        (
            "td_landed",
            "TD landed",
        ),
        (
            "qualified_control_inflicted_seconds",
            "qualified control sec",
        ),
    )

    print()
    print("=" * 145)
    print(
        "EVENT CLOCK V1 — "
        "FRESH 500 RESULTS"
    )
    print("=" * 145)

    for target, label in (
        report_targets
    ):

        pred_col = (
            f"pred_{target}"
        )

        sim_col = (
            f"sim_{target}"
        )

        actual_mean = float(
            test[target].mean()
        )

        pred_mean = float(
            test[pred_col].mean()
        )

        sim_mean = float(
            test[sim_col].mean()
        )

        (
            pred_r,
            pred_rho,
            pred_mae,
        ) = metrics(
            test[target],
            test[pred_col],
        )

        (
            sim_r,
            sim_rho,
            sim_mae,
        ) = metrics(
            test[target],
            test[sim_col],
        )

        (
            pred_direction,
            pred_n,
        ) = within_bout_direction(
            test,
            target,
            pred_col,
        )

        (
            sim_direction,
            sim_n,
        ) = within_bout_direction(
            test,
            target,
            sim_col,
        )

        print()
        print(label.upper())
        print("-" * 145)

        print(
            f"population mean | "
            f"HIST={actual_mean:.3f} | "
            f"DIRECT={pred_mean:.3f} | "
            f"CLOCK={sim_mean:.3f}"
        )

        print(
            f"DIRECT         | "
            f"r={pred_r:+.4f} | "
            f"rho={pred_rho:+.4f} | "
            f"MAE={pred_mae:.3f} | "
            f"within-bout="
            f"{pred_direction:.2%} "
            f"(N={pred_n})"
        )

        print(
            f"CLOCK MC       | "
            f"r={sim_r:+.4f} | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f} | "
            f"within-bout="
            f"{sim_direction:.2%} "
            f"(N={sim_n})"
        )

    # ---------------------------------------------------------------------
    # V0 vs V1 comparison
    # ---------------------------------------------------------------------

    if OLD_PATH.exists():

        old = pd.read_csv(
            OLD_PATH
        )

        old["fight_id"] = (
            old["fight_id"]
            .astype(str)
        )

        compare_keys = [
            "fight_id",
            "side",
            "fighter_name",
        ]

        keep = (
            compare_keys
            + [
                "pred_standing_attempted",
                "pred_td_attempted",
                "pred_td_landed",
                "pred_ground_attempted",
                "pred_qualified_control_inflicted_seconds",
            ]
        )

        old = old[
            keep
        ].rename(
            columns={
                col: f"v0_{col}"
                for col in keep
                if col
                not in compare_keys
            }
        )

        joined = test.merge(
            old,
            on=compare_keys,
            how="left",
            validate="one_to_one",
        )

        print()
        print("=" * 145)
        print(
            "DIRECT MODEL — "
            "V0 VS V1"
        )
        print("=" * 145)

        for target, label in (
            (
                "standing_attempted",
                "standing attempts",
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
                "qualified_control_inflicted_seconds",
                "control seconds",
            ),
        ):

            old_col = (
                f"v0_pred_{target}"
            )

            new_col = (
                f"pred_{target}"
            )

            old_r, old_rho, old_mae = (
                metrics(
                    joined[target],
                    joined[old_col],
                )
            )

            new_r, new_rho, new_mae = (
                metrics(
                    joined[target],
                    joined[new_col],
                )
            )

            print()
            print(label.upper())
            print(
                f"V0 | mean="
                f"{joined[old_col].mean():.3f} | "
                f"rho={old_rho:+.4f} | "
                f"MAE={old_mae:.3f}"
            )

            print(
                f"V1 | mean="
                f"{joined[new_col].mean():.3f} | "
                f"rho={new_rho:+.4f} | "
                f"MAE={new_mae:.3f}"
            )

    # ---------------------------------------------------------------------
    # Control sanity
    # ---------------------------------------------------------------------

    fight_control = (
        test.groupby(
            "fight_id",
            as_index=False,
        )
        .agg(
            duration=(
                "duration",
                "first",
            ),
            historical_control=(
                "qualified_control_inflicted_seconds",
                "sum",
            ),
            predicted_control=(
                "pred_qualified_control_inflicted_seconds",
                "sum",
            ),
            simulated_control=(
                "sim_qualified_control_inflicted_seconds",
                "sum",
            ),
        )
    )

    print()
    print("=" * 145)
    print("CONTROL SANITY")
    print("=" * 145)

    print(
        "Historical total qualified "
        f"control/fight: "
        f"{fight_control['historical_control'].mean():.2f}s"
    )

    print(
        "Predicted total control/fight: "
        f"{fight_control['predicted_control'].mean():.2f}s"
    )

    print(
        "Clock-MC total control/fight:  "
        f"{fight_control['simulated_control'].mean():.2f}s"
    )

    print(
        "Clock paths violating total "
        "fight-time invariant: "
        f"{int((fight_control['simulated_control'] > fight_control['duration'] + 1e-9).sum())}"
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test.to_csv(
        OUT,
        index=False,
    )

    print()
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
