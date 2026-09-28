from __future__ import annotations

from math import exp
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.fsr_v2.sources.round_stats import build_paired_rounds

from pipeline.simulation.event_mc_v1.components.fsr_v2_mechanics import (
    TAKEDOWN_ATTACKER_AGE_CENTER_YEARS,
    TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR,
    effective_rate,
    matchup_probability,
)

from pipeline.simulation.event_mc_v1.diagnostics.population_validation import (
    _fight,
    build_cohort,
)

from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
)

from pipeline.simulation.event_mc_v1.diagnostics.stage1_flow_replay import (
    stage1_observed_duration_seconds,
)


# =============================================================================
# CONFIG
# =============================================================================

CUTOFF = pd.Timestamp("2025-03-22")

TRAIN_MAX_FIGHTS = 2500
TEST_FIGHTS = 500
PATHS = 20

SEED = 20260817

RIDGE_ALPHA = 20.0

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage1_500x20.csv"
)

MODEL_TARGETS = (
    "distance_attempted",
    "distance_landed",
    "clinch_attempted",
    "clinch_landed",
    "ground_attempted",
    "ground_landed",
    "td_attempted",
    "td_landed",
    "qualified_control_inflicted_seconds",
)

ATTEMPT_FAMILIES = (
    "distance",
    "clinch",
    "ground",
    "td",
)

FSR_ATTRS = (
    "standing_striking_tendency",
    "standing_striking_suppression",
    "standing_striking_offense",
    "standing_striking_defense",
    "standing_accuracy_baseline",

    "takedown_tendency",
    "takedown_suppression",
    "takedown_offense",
    "takedown_defense",
    "takedown_completion_baseline",

    "escape_offense",
    "escape_defense",
    "escape_population_mean_seconds",

    "ground_striking_tendency",
    "ground_striking_suppression",
    "ground_striking_offense",
    "ground_striking_defense",
    "ground_accuracy_baseline",

    "submission_tendency",
    "submission_suppression",
    "submission_offense",
    "submission_defense",
)


# =============================================================================
# HELPERS
# =============================================================================

def numeric_attr(obj, name):
    value = getattr(obj, name, np.nan)

    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_feature_rows(master, fsr):
    rows = []
    skipped = 0

    for _, master_row in master.iterrows():

        try:
            fight = _fight(master_row, fsr)
        except Exception:
            skipped += 1
            continue

        matchup = fight.fsr_v2_matchup

        if matchup is None:
            skipped += 1
            continue

        duration = float(
            stage1_observed_duration_seconds(
                master_row
            )
        )

        corners = {
            "red": matchup.red,
            "blue": matchup.blue,
        }

        for side, opponent_side in (
            ("red", "blue"),
            ("blue", "red"),
        ):

            fighter = corners[side]
            opponent = corners[opponent_side]

            row = {
                "fight_id": str(master_row["fight_id"]),
                "event_date": pd.Timestamp(
                    master_row["event_date"]
                ),
                "side": side,
                "fighter_name": fighter.fighter_name,
                "opponent_name": opponent.fighter_name,
                "duration": duration,
                "scheduled_rounds": float(
                    master_row["total_rounds"]
                ),
                "fighter_age": numeric_attr(
                    fighter,
                    "age_years",
                ),
                "opponent_age": numeric_attr(
                    opponent,
                    "age_years",
                ),
            }

            for attr in FSR_ATTRS:
                row[f"self_{attr}"] = (
                    numeric_attr(fighter, attr)
                )

                row[f"opp_{attr}"] = (
                    numeric_attr(opponent, attr)
                )

            # -------------------------------------------------------------
            # Useful literal matchup transforms.
            # No fit or calibration here.
            # -------------------------------------------------------------

            row["effective_standing_rate"] = (
                effective_rate(
                    numeric_attr(
                        fighter,
                        "standing_striking_tendency",
                    ),
                    numeric_attr(
                        opponent,
                        "standing_striking_suppression",
                    ),
                )
            )

            row["effective_td_rate"] = (
                effective_rate(
                    numeric_attr(
                        fighter,
                        "takedown_tendency",
                    ),
                    numeric_attr(
                        opponent,
                        "takedown_suppression",
                    ),
                )
            )

            row["effective_ground_rate"] = (
                effective_rate(
                    numeric_attr(
                        fighter,
                        "ground_striking_tendency",
                    ),
                    numeric_attr(
                        opponent,
                        "ground_striking_suppression",
                    ),
                )
            )

            fighter_age = row["fighter_age"]

            if np.isfinite(fighter_age):
                td_age_offset = (
                    TAKEDOWN_ATTACKER_AGE_LOGIT_PER_YEAR
                    * (
                        fighter_age
                        - TAKEDOWN_ATTACKER_AGE_CENTER_YEARS
                    )
                )
            else:
                td_age_offset = 0.0

            row["td_completion_matchup"] = (
                matchup_probability(
                    numeric_attr(
                        fighter,
                        "takedown_completion_baseline",
                    ),
                    numeric_attr(
                        fighter,
                        "takedown_offense",
                    ),
                    numeric_attr(
                        opponent,
                        "takedown_defense",
                    ),
                    td_age_offset,
                )
            )

            row["standing_accuracy_matchup"] = (
                matchup_probability(
                    numeric_attr(
                        fighter,
                        "standing_accuracy_baseline",
                    ),
                    numeric_attr(
                        fighter,
                        "standing_striking_offense",
                    ),
                    numeric_attr(
                        opponent,
                        "standing_striking_defense",
                    ),
                )
            )

            row["ground_accuracy_matchup"] = (
                matchup_probability(
                    numeric_attr(
                        fighter,
                        "ground_accuracy_baseline",
                    ),
                    numeric_attr(
                        fighter,
                        "ground_striking_offense",
                    ),
                    numeric_attr(
                        opponent,
                        "ground_striking_defense",
                    ),
                )
            )

            bottom_population_mean = numeric_attr(
                opponent,
                "escape_population_mean_seconds",
            )

            bottom_escape_offense = numeric_attr(
                opponent,
                "escape_offense",
            )

            top_escape_defense = numeric_attr(
                fighter,
                "escape_defense",
            )

            retention_mean = (
                bottom_population_mean
                * exp(
                    -bottom_escape_offense
                    + top_escape_defense
                )
            )

            row["retention_mean_base"] = (
                retention_mean
            )

            row["successful_td_pressure"] = (
                row["effective_td_rate"]
                * row["td_completion_matchup"]
            )

            row["control_pressure"] = (
                row["successful_td_pressure"]
                * retention_mean
            )

            if (
                np.isfinite(row["fighter_age"])
                and np.isfinite(row["opponent_age"])
            ):
                row["age_edge"] = (
                    row["fighter_age"]
                    - row["opponent_age"]
                )
            else:
                row["age_edge"] = np.nan

            rows.append(row)

    result = pd.DataFrame(rows)

    print(
        f"feature fights built: "
        f"{result['fight_id'].nunique()} | "
        f"fighter-fight rows: {len(result)} | "
        f"skipped fights: {skipped}"
    )

    return result


def build_historical_targets():
    rounds = build_paired_rounds().copy()

    rounds["fight_id"] = (
        rounds["fight_id"].astype(str)
    )

    hist = (
        rounds.groupby(
            ["fight_id", "fighter_name"],
            as_index=False,
        )
        .agg(
            distance_attempted=(
                "distance_attempted",
                "sum",
            ),
            distance_landed=(
                "distance_landed",
                "sum",
            ),
            clinch_attempted=(
                "clinch_attempted",
                "sum",
            ),
            clinch_landed=(
                "clinch_landed",
                "sum",
            ),
            ground_attempted=(
                "ground_attempted",
                "sum",
            ),
            ground_landed=(
                "ground_landed",
                "sum",
            ),
            td_attempted=(
                "td_attempted",
                "sum",
            ),
            td_landed=(
                "td_landed",
                "sum",
            ),
            qualified_control_inflicted_seconds=(
                "qualified_control_inflicted_seconds",
                "sum",
            ),
            ground_entries=(
                "ground_entries",
                "sum",
            ),
        )
    )

    return hist


# =============================================================================
# SIMPLE LEAKAGE-SAFE DIRECT OUTPUT MODEL
#
# Weighted ridge on log(1 + output per 15 min).
#
# This is deliberately simple.
# We are testing architecture, not searching for the best ML algorithm.
# =============================================================================

class LogRateRidge:
    def __init__(self, alpha=20.0):
        self.alpha = float(alpha)

    def fit(
        self,
        x,
        y_rate,
        sample_weight,
    ):
        x = np.asarray(x, dtype=float)
        y_rate = np.asarray(
            y_rate,
            dtype=float,
        )
        w = np.asarray(
            sample_weight,
            dtype=float,
        )

        self.mean_ = np.nanmean(
            x,
            axis=0,
        )

        filled = np.where(
            np.isfinite(x),
            x,
            self.mean_,
        )

        self.std_ = np.nanstd(
            filled,
            axis=0,
        )

        self.std_ = np.where(
            self.std_ > 1e-9,
            self.std_,
            1.0,
        )

        z = (
            filled
            - self.mean_
        ) / self.std_

        design = np.column_stack(
            [
                np.ones(len(z)),
                z,
            ]
        )

        target = np.log1p(
            np.maximum(y_rate, 0.0)
        )

        sqrt_w = np.sqrt(
            np.maximum(w, 1e-6)
        )

        weighted_design = (
            design
            * sqrt_w[:, None]
        )

        weighted_target = (
            target
            * sqrt_w
        )

        penalty = np.eye(
            design.shape[1]
        ) * self.alpha

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

        self.beta_ = np.linalg.solve(
            lhs,
            rhs,
        )

        positive = y_rate[
            np.isfinite(y_rate)
            & (y_rate >= 0)
        ]

        if len(positive):
            self.max_rate_ = max(
                1.0,
                float(
                    np.quantile(
                        positive,
                        0.995,
                    )
                )
                * 1.50,
            )
        else:
            self.max_rate_ = 1.0

        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)

        filled = np.where(
            np.isfinite(x),
            x,
            self.mean_,
        )

        z = (
            filled
            - self.mean_
        ) / self.std_

        design = np.column_stack(
            [
                np.ones(len(z)),
                z,
            ]
        )

        log_pred = (
            design
            @ self.beta_
        )

        pred = np.expm1(log_pred)

        return np.clip(
            pred,
            0.0,
            self.max_rate_,
        )


# =============================================================================
# COMPETING CLOCK ENGINE
#
# NO PHASE.
#
# Active clocks:
# - red/blue distance strike attempt
# - red/blue clinch strike attempt
# - red/blue ground strike attempt
# - red/blue TD attempt
# - red/blue control episode
#
# Control episode is an instantaneous event with a duration mark.
# It does NOT consume simulator time.
#
# Static V0 rates intentionally isolate the architecture.
# =============================================================================

def simulate_path(
    pair,
    rng,
    control_mark_mean,
):
    duration = float(
        pair["duration"].iloc[0]
    )

    fighters = {
        row["side"]: row
        for _, row in pair.iterrows()
    }

    output = {}

    for side in ("red", "blue"):
        for family in ATTEMPT_FAMILIES:
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

    clocks = []

    for side in ("red", "blue"):
        row = fighters[side]

        for family in (
            "distance",
            "clinch",
            "ground",
            "td",
        ):
            expected_attempts = max(
                0.0,
                float(
                    row[
                        f"pred_{family}_attempted"
                    ]
                ),
            )

            rate = (
                expected_attempts
                / duration
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
        clock[2]
        for clock in clocks
    )

    if total_rate <= 0:
        return output

    time = 0.0
    total_control = 0.0

    while time < duration:

        dt = rng.exponential(
            1.0 / total_rate
        )

        time += dt

        if time >= duration:
            break

        draw = rng.random() * total_rate

        running = 0.0
        selected = clocks[-1]

        for clock in clocks:
            running += clock[2]

            if draw <= running:
                selected = clock
                break

        side, family, _ = selected
        row = fighters[side]

        if family == "control":

            remaining_control = max(
                0.0,
                duration
                - total_control,
            )

            if remaining_control <= 0:
                continue

            # Gamma mark:
            # shape=2, same expected duration as control_mark_mean,
            # less extreme than exponential marks.
            mark = rng.gamma(
                shape=2.0,
                scale=control_mark_mean / 2.0,
            )

            mark = min(
                mark,
                remaining_control,
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

        if expected_att > 1e-9:
            p_land = np.clip(
                expected_lnd
                / expected_att,
                0.0,
                0.98,
            )
        else:
            p_land = 0.0

        if rng.random() < p_land:
            output[
                f"{side}_{family}_landed"
            ] += 1.0

    return output


def simulate_fight(
    pair,
    paths,
    seed,
    control_mark_mean,
):
    samples = []

    for path in range(paths):
        rng = np.random.default_rng(
            seed
            + path
        )

        samples.append(
            simulate_path(
                pair,
                rng,
                control_mark_mean,
            )
        )

    keys = samples[0].keys()

    result = {
        key: float(
            np.mean(
                [
                    sample[key]
                    for sample in samples
                ]
            )
        )
        for key in keys
    }

    return result


# =============================================================================
# METRICS
# =============================================================================

def metrics(actual, predicted):
    frame = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
        }
    ).dropna()

    if len(frame) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    return (
        frame["actual"].corr(
            frame["predicted"],
            method="pearson",
        ),
        frame["actual"].corr(
            frame["predicted"],
            method="spearman",
        ),
        (
            frame["actual"]
            - frame["predicted"]
        ).abs().mean(),
    )


def within_bout_direction(
    df,
    actual_col,
    pred_col,
):
    correct = []

    for _, group in df.groupby(
        "fight_id"
    ):

        if len(group) != 2:
            continue

        a, b = group.iloc[0], group.iloc[1]

        actual_diff = (
            float(a[actual_col])
            - float(b[actual_col])
        )

        pred_diff = (
            float(a[pred_col])
            - float(b[pred_col])
        )

        if (
            not np.isfinite(actual_diff)
            or not np.isfinite(pred_diff)
            or actual_diff == 0
            or pred_diff == 0
        ):
            continue

        correct.append(
            np.sign(actual_diff)
            == np.sign(pred_diff)
        )

    if not correct:
        return np.nan, 0

    return (
        float(np.mean(correct)),
        len(correct),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print("EVENT CLOCK MC V0 — DIRECT OUTPUT / NO-PHASE PROTOTYPE")
    print("=" * 120)

    # ---------------------------------------------------------------------
    # Historical targets
    # ---------------------------------------------------------------------

    hist = build_historical_targets()

    # ---------------------------------------------------------------------
    # Leakage-safe earlier training universe
    # ---------------------------------------------------------------------

    train_master, train_fsr = build_cohort(
        start_year=2018,
        limit=None,
    )

    train_master = train_master[
        train_master["event_date"]
        < CUTOFF
    ].copy()

    train_master = (
        train_master
        .sort_values(
            ["event_date", "fight_id"]
        )
        .tail(TRAIN_MAX_FIGHTS)
        .reset_index(drop=True)
    )

    print()
    print(
        f"training fights before cutoff: "
        f"{len(train_master)}"
    )

    train_features = build_feature_rows(
        train_master,
        train_fsr,
    )

    # ---------------------------------------------------------------------
    # Exact fresh 500 evaluation cohort
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
    # Join observed outputs
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
        f"training fighter-fight rows with targets: "
        f"{len(train)}"
    )

    print(
        f"evaluation fighter-fight rows with targets: "
        f"{len(test)}"
    )

    if (
        test["fight_id"].nunique()
        != TEST_FIGHTS
    ):
        raise RuntimeError(
            "Fresh evaluation target join did not "
            "retain exactly 500 fights"
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

    x_train = train[
        feature_cols
    ].to_numpy(float)

    x_test = test[
        feature_cols
    ].to_numpy(float)

    # Longer fights provide more stable observed rates.
    weights = np.clip(
        train["duration"].to_numpy(float)
        / 900.0,
        0.10,
        5.0 / 3.0,
    )

    models = {}

    print()
    print("=" * 120)
    print("DIRECT PREFIGHT OUTPUT MODELS")
    print("=" * 120)

    for target in MODEL_TARGETS:

        y_train_rate = (
            train[target].to_numpy(float)
            / train["duration"].to_numpy(float)
            * 900.0
        )

        model = LogRateRidge(
            alpha=RIDGE_ALPHA
        ).fit(
            x_train,
            y_train_rate,
            weights,
        )

        models[target] = model

        pred_rate = model.predict(
            x_test
        )

        test[
            f"raw_pred_{target}"
        ] = (
            pred_rate
            * test["duration"].to_numpy(float)
            / 900.0
        )

    # ---------------------------------------------------------------------
    # Enforce only unavoidable arithmetic invariants.
    #
    # landed <= attempts
    # total control <= elapsed fight time
    #
    # No phase allocation is introduced.
    # ---------------------------------------------------------------------

    for family in (
        "distance",
        "clinch",
        "ground",
        "td",
    ):

        test[
            f"pred_{family}_attempted"
        ] = np.maximum(
            0.0,
            test[
                f"raw_pred_{family}_attempted"
            ],
        )

        test[
            f"pred_{family}_landed"
        ] = np.minimum(
            test[
                f"pred_{family}_attempted"
            ]
            * 0.98,
            np.maximum(
                0.0,
                test[
                    f"raw_pred_{family}_landed"
                ],
            ),
        )

    test[
        "pred_qualified_control_inflicted_seconds"
    ] = np.maximum(
        0.0,
        test[
            "raw_pred_qualified_control_inflicted_seconds"
        ],
    )

    rescaled_control_fights = 0

    for fight_id, group in test.groupby(
        "fight_id"
    ):

        index = group.index

        duration = float(
            group["duration"].iloc[0]
        )

        total_pred = float(
            group[
                "pred_qualified_control_inflicted_seconds"
            ].sum()
        )

        if total_pred > duration:
            scale = (
                duration
                / total_pred
            )

            test.loc[
                index,
                "pred_qualified_control_inflicted_seconds",
            ] *= scale

            rescaled_control_fights += 1

    print(
        f"control predictions requiring "
        f"physical rescale: "
        f"{rescaled_control_fights}/"
        f"{TEST_FIGHTS}"
    )

    # ---------------------------------------------------------------------
    # Learn ONE global control-duration mark mean from EARLIER history.
    #
    # This changes path variance/frequency, not expected total control.
    # ---------------------------------------------------------------------

    total_train_entries = float(
        train["ground_entries"].sum()
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
        f"control episode duration mark mean "
        f"from training data: "
        f"{control_mark_mean:.2f}s"
    )

    # ---------------------------------------------------------------------
    # Run actual competing-clock MC
    # ---------------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        f"RUNNING COMPETING CLOCKS — "
        f"{TEST_FIGHTS} fights x {PATHS} paths"
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
    ) in enumerate(fight_groups):

        if len(pair) != 2:
            raise RuntimeError(
                f"{fight_id}: expected two corners"
            )

        result = simulate_fight(
            pair,
            PATHS,
            SEED
            + fight_index * 100000,
            control_mark_mean,
        )

        for _, fighter_row in pair.iterrows():

            side = fighter_row["side"]

            out = {
                "fight_id": fight_id,
                "side": side,
                "fighter_name":
                    fighter_row["fighter_name"],
            }

            for family in (
                "distance",
                "clinch",
                "ground",
                "td",
            ):
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
            (fight_index + 1) % 50 == 0
            or fight_index + 1
            == TEST_FIGHTS
        ):
            print(
                f"completed "
                f"{fight_index + 1}/"
                f"{TEST_FIGHTS}"
            )

    sim = pd.DataFrame(sim_rows)

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

    # Useful sum:
    # distance + clinch = all-standing significant attempts.
    test["standing_attempted"] = (
        test["distance_attempted"]
        + test["clinch_attempted"]
    )

    test["pred_standing_attempted"] = (
        test["pred_distance_attempted"]
        + test["pred_clinch_attempted"]
    )

    test["sim_standing_attempted"] = (
        test["sim_distance_attempted"]
        + test["sim_clinch_attempted"]
    )

    test["standing_landed"] = (
        test["distance_landed"]
        + test["clinch_landed"]
    )

    test["pred_standing_landed"] = (
        test["pred_distance_landed"]
        + test["pred_clinch_landed"]
    )

    test["sim_standing_landed"] = (
        test["sim_distance_landed"]
        + test["sim_clinch_landed"]
    )

    # ---------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------

    report_targets = (
        ("standing_attempted", "standing attempts"),
        ("standing_landed", "standing landed"),
        ("distance_attempted", "distance attempts"),
        ("clinch_attempted", "clinch attempts"),
        ("ground_attempted", "ground attempts"),
        ("ground_landed", "ground landed"),
        ("td_attempted", "TD attempts"),
        ("td_landed", "TD landed"),
        (
            "qualified_control_inflicted_seconds",
            "qualified control sec",
        ),
    )

    print()
    print("=" * 140)
    print("EVENT CLOCK V0 — FRESH 500 RESULTS")
    print("=" * 140)

    for target, label in report_targets:

        pred_col = f"pred_{target}"
        sim_col = f"sim_{target}"

        actual_mean = float(
            test[target].mean()
        )

        pred_mean = float(
            test[pred_col].mean()
        )

        sim_mean = float(
            test[sim_col].mean()
        )

        pred_r, pred_rho, pred_mae = (
            metrics(
                test[target],
                test[pred_col],
            )
        )

        sim_r, sim_rho, sim_mae = (
            metrics(
                test[target],
                test[sim_col],
            )
        )

        pred_direction, pred_n = (
            within_bout_direction(
                test,
                target,
                pred_col,
            )
        )

        sim_direction, sim_n = (
            within_bout_direction(
                test,
                target,
                sim_col,
            )
        )

        print()
        print(label.upper())
        print("-" * 140)

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
            f"within-bout={pred_direction:.2%} "
            f"(N={pred_n})"
        )

        print(
            f"CLOCK MC       | "
            f"r={sim_r:+.4f} | "
            f"rho={sim_rho:+.4f} | "
            f"MAE={sim_mae:.3f} | "
            f"within-bout={sim_direction:.2%} "
            f"(N={sim_n})"
        )

    # ---------------------------------------------------------------------
    # Fight-level physical sanity
    # ---------------------------------------------------------------------

    fight_control = (
        test.groupby(
            "fight_id",
            as_index=False,
        )
        .agg(
            duration=("duration", "first"),
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
    print("=" * 140)
    print("CONTROL SANITY")
    print("=" * 140)

    print(
        f"Historical total qualified control/fight: "
        f"{fight_control['historical_control'].mean():.2f}s"
    )

    print(
        f"Predicted total control/fight:            "
        f"{fight_control['predicted_control'].mean():.2f}s"
    )

    print(
        f"Clock-MC total control/fight:             "
        f"{fight_control['simulated_control'].mean():.2f}s"
    )

    print(
        f"Clock paths violating total fight-time "
        f"control invariant: "
        f"{int((fight_control['simulated_control'] > fight_control['duration'] + 1e-9).sum())}"
    )

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------

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
