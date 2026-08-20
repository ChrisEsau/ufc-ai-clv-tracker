from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pipeline.simulation.event_clock_mc_v1.prototype_stage1 import (
    metrics,
    within_bout_direction,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage2 import (
    BINOMIAL_ALPHA,
    BinomialRidge,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import (
    prepare_direct_predictions,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import (
    direct_feature_columns,
    draw_frailty,
    estimate_nb_alpha,
)


PATHS = 20
SEED = 20260817

CONTROL_CHUNK_SECONDS = 10.0

OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage5_competitive_500x20.csv"
)

PATH_OUT = Path(
    "data/diagnostics/event_clock_mc_v1/"
    "prototype_stage5_competitive_paths.csv"
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def logit(p):
    p = np.clip(
        np.asarray(p, dtype=float),
        1e-8,
        1.0 - 1e-8,
    )
    return np.log(
        p / (1.0 - p)
    )


def sigmoid(x):
    x = np.clip(
        np.asarray(x, dtype=float),
        -30.0,
        30.0,
    )
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def physical_hurdle_probability(
    mean_count,
    probability,
):
    """
    Positive integer count constraint:

        E[N] >= P(N > 0)

    because conditional positive mean must be >= 1.
    """
    mu = np.maximum(
        np.asarray(
            mean_count,
            dtype=float,
        ),
        0.0,
    )

    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        0.0,
        0.999999,
    )

    upper = np.minimum(
        mu,
        0.999999,
    )

    p = np.minimum(
        p,
        upper,
    )

    p = np.where(
        mu > 0,
        np.maximum(
            p,
            1e-8,
        ),
        0.0,
    )

    return p


# =============================================================================
# TD / GROUND HURDLES
# =============================================================================

def fit_count_hurdle(
    train,
    test,
    family,
    feature_cols,
):
    target = (
        f"{family}_attempted"
    )

    pred = (
        f"pred_{target}"
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

    occurrence_model = (
        BinomialRidge(
            alpha=BINOMIAL_ALPHA
        )
        .fit(
            x_train,
            successes=(
                train[target]
                .to_numpy(float)
                > 0
            ).astype(float),
            trials=np.ones(
                len(train),
                dtype=float,
            ),
        )
    )

    train_p = (
        occurrence_model
        .predict_probability(
            x_train
        )
    )

    test_p = (
        occurrence_model
        .predict_probability(
            x_test
        )
    )

    train_mu = (
        train[pred]
        .to_numpy(float)
    )

    test_mu = (
        test[pred]
        .to_numpy(float)
    )

    train_p = (
        physical_hurdle_probability(
            train_mu,
            train_p,
        )
    )

    test_p = (
        physical_hurdle_probability(
            test_mu,
            test_p,
        )
    )

    train_cond = np.divide(
        train_mu,
        train_p,
        out=np.zeros_like(
            train_mu
        ),
        where=train_p > 0,
    )

    test_cond = np.divide(
        test_mu,
        test_p,
        out=np.zeros_like(
            test_mu
        ),
        where=test_p > 0,
    )

    train[
        f"pred_{family}_positive_probability"
    ] = train_p

    test[
        f"pred_{family}_positive_probability"
    ] = test_p

    train[
        f"pred_{family}_conditional_attempts"
    ] = train_cond

    test[
        f"pred_{family}_conditional_attempts"
    ] = test_cond

    # ---------------------------------------------------------
    # Once activated, first attempt is guaranteed.
    #
    # Fit overdispersion only to attempts AFTER the first one.
    # ---------------------------------------------------------

    positive = (
        train[target]
        .to_numpy(float)
        > 0
    )

    actual_extra = np.maximum(
        train[target]
        .to_numpy(float)[positive]
        - 1.0,
        0.0,
    )

    predicted_extra = np.maximum(
        train_cond[positive]
        - 1.0,
        1e-6,
    )

    extra_alpha = (
        estimate_nb_alpha(
            actual_extra,
            predicted_extra,
        )
    )

    print()
    print(
        f"{family.upper()} HURDLE"
    )
    print("-" * 100)

    print(
        f"TRAIN historical positive: "
        f"{(train[target] > 0).mean():.2%}"
    )

    print(
        f"TRAIN predicted positive:  "
        f"{train_p.mean():.2%}"
    )

    print(
        f"TEST historical positive:  "
        f"{(test[target] > 0).mean():.2%}"
    )

    print(
        f"TEST predicted positive:   "
        f"{test_p.mean():.2%}"
    )

    print(
        f"positive-path extra-count "
        f"alpha: {extra_alpha:.4f}"
    )

    return extra_alpha


# =============================================================================
# PAIR-LEVEL CONTROL DATA
# =============================================================================

def build_pair_frame(
    fighter_frame,
):
    rows = []

    for fight_id, group in (
        fighter_frame.groupby(
            "fight_id",
            sort=False,
        )
    ):

        if len(group) != 2:
            continue

        red = group[
            group["side"] == "red"
        ]

        blue = group[
            group["side"] == "blue"
        ]

        if (
            len(red) != 1
            or len(blue) != 1
        ):
            continue

        red = red.iloc[0]
        blue = blue.iloc[0]

        duration = float(
            red["duration"]
        )

        actual_red_control = float(
            red[
                "qualified_control_inflicted_seconds"
            ]
        )

        actual_blue_control = float(
            blue[
                "qualified_control_inflicted_seconds"
            ]
        )

        actual_total_control = (
            actual_red_control
            + actual_blue_control
        )

        pred_red_control = float(
            red[
                "pred_qualified_control_inflicted_seconds"
            ]
        )

        pred_blue_control = float(
            blue[
                "pred_qualified_control_inflicted_seconds"
            ]
        )

        pred_total_control = min(
            duration,
            pred_red_control
            + pred_blue_control,
        )

        row = {
            "fight_id":
                str(fight_id),

            "duration":
                duration,

            "scheduled_rounds":
                float(
                    red[
                        "scheduled_rounds"
                    ]
                ),

            "actual_red_control":
                actual_red_control,

            "actual_blue_control":
                actual_blue_control,

            "actual_total_control":
                actual_total_control,

            "actual_red_td_attempted":
                float(
                    red[
                        "td_attempted"
                    ]
                ),

            "actual_blue_td_attempted":
                float(
                    blue[
                        "td_attempted"
                    ]
                ),

            "actual_red_td_landed":
                float(
                    red[
                        "td_landed"
                    ]
                ),

            "actual_blue_td_landed":
                float(
                    blue[
                        "td_landed"
                    ]
                ),

            "pred_red_control":
                pred_red_control,

            "pred_blue_control":
                pred_blue_control,

            "pred_total_control":
                pred_total_control,

            "pred_total_td":
                float(
                    red[
                        "pred_td_attempted"
                    ]
                    + blue[
                        "pred_td_attempted"
                    ]
                ),

            "pred_total_td_landed":
                float(
                    red[
                        "pred_td_landed"
                    ]
                    + blue[
                        "pred_td_landed"
                    ]
                ),

            "pred_total_ground":
                float(
                    red[
                        "pred_ground_attempted"
                    ]
                    + blue[
                        "pred_ground_attempted"
                    ]
                ),

            "pred_control_abs_diff":
                abs(
                    pred_red_control
                    - pred_blue_control
                ),

            "pred_td_abs_diff":
                abs(
                    float(
                        red[
                            "pred_td_attempted"
                        ]
                    )
                    - float(
                        blue[
                            "pred_td_attempted"
                        ]
                    )
                ),

            # -------------------------------------------------
            # Signed Red-vs-Blue ownership features.
            # -------------------------------------------------

            "ownership_control_log_ratio":
                (
                    np.log1p(
                        pred_red_control
                    )
                    - np.log1p(
                        pred_blue_control
                    )
                ),

            "ownership_td_attempt_diff":
                float(
                    red[
                        "pred_td_attempted"
                    ]
                    - blue[
                        "pred_td_attempted"
                    ]
                ),

            "ownership_td_landed_diff":
                float(
                    red[
                        "pred_td_landed"
                    ]
                    - blue[
                        "pred_td_landed"
                    ]
                ),

            "ownership_ground_diff":
                float(
                    red[
                        "pred_ground_attempted"
                    ]
                    - blue[
                        "pred_ground_attempted"
                    ]
                ),

            "ownership_control_pressure_diff":
                float(
                    red[
                        "control_pressure"
                    ]
                    - blue[
                        "control_pressure"
                    ]
                ),

            "ownership_successful_td_pressure_diff":
                float(
                    red[
                        "successful_td_pressure"
                    ]
                    - blue[
                        "successful_td_pressure"
                    ]
                ),

            "ownership_effective_td_rate_diff":
                float(
                    red[
                        "effective_td_rate"
                    ]
                    - blue[
                        "effective_td_rate"
                    ]
                ),

            "ownership_retention_diff":
                float(
                    red[
                        "retention_mean_base"
                    ]
                    - blue[
                        "retention_mean_base"
                    ]
                ),
        }

        rows.append(row)

    return pd.DataFrame(
        rows
    )


CONTROL_OCCURRENCE_FEATURES = [
    "pred_total_control",
    "pred_total_td",
    "pred_total_td_landed",
    "pred_total_ground",
    "pred_control_abs_diff",
    "pred_td_abs_diff",
    "scheduled_rounds",
]


CONTROL_OWNERSHIP_FEATURES = [
    "ownership_control_log_ratio",
    "ownership_td_attempt_diff",
    "ownership_td_landed_diff",
    "ownership_ground_diff",
    "ownership_control_pressure_diff",
    "ownership_successful_td_pressure_diff",
    "ownership_effective_td_rate_diff",
    "ownership_retention_diff",
]


# =============================================================================
# CONTROL OCCURRENCE + OWNERSHIP
# =============================================================================

def fit_control_models(
    train_pair,
    test_pair,
):
    # ---------------------------------------------------------
    # Does meaningful control occur at all?
    # ---------------------------------------------------------

    occurrence = (
        BinomialRidge(
            alpha=20.0
        )
        .fit(
            train_pair[
                CONTROL_OCCURRENCE_FEATURES
            ].to_numpy(float),
            successes=(
                train_pair[
                    "actual_total_control"
                ].to_numpy(float)
                > 0
            ).astype(float),
            trials=np.ones(
                len(train_pair)
            ),
        )
    )

    train_p = (
        occurrence
        .predict_probability(
            train_pair[
                CONTROL_OCCURRENCE_FEATURES
            ].to_numpy(float)
        )
    )

    test_p = (
        occurrence
        .predict_probability(
            test_pair[
                CONTROL_OCCURRENCE_FEATURES
            ].to_numpy(float)
        )
    )

    # ---------------------------------------------------------
    # Physical constraint:
    #
    # E[CTRL] =
    #   P(CTRL>0) *
    #   E[CTRL | positive]
    #
    # and conditional control cannot exceed fight duration.
    #
    # Therefore:
    #
    #   P(CTRL>0) >= E[CTRL] / T
    # ---------------------------------------------------------

    for frame, p in (
        (
            train_pair,
            train_p,
        ),
        (
            test_pair,
            test_p,
        ),
    ):

        floor = np.clip(
            frame[
                "pred_total_control"
            ].to_numpy(float)
            / frame[
                "duration"
            ].to_numpy(float),
            0.0,
            0.999999,
        )

        p[:] = np.maximum(
            p,
            floor,
        )

        p[:] = np.clip(
            p,
            1e-8,
            0.999999,
        )

    train_pair[
        "pred_control_any_probability"
    ] = train_p

    test_pair[
        "pred_control_any_probability"
    ] = test_p

    train_pair[
        "pred_control_conditional_total"
    ] = (
        train_pair[
            "pred_total_control"
        ]
        / train_p
    )

    test_pair[
        "pred_control_conditional_total"
    ] = (
        test_pair[
            "pred_total_control"
        ]
        / test_p
    )

    # ---------------------------------------------------------
    # Who owns the control that occurs?
    #
    # Quasi-binomial:
    #
    # successes = Red control seconds
    # trials    = total control seconds
    #
    # Divide by 10 simply to avoid giving hundreds of seconds
    # enormous likelihood weight relative to the ridge penalty.
    # ---------------------------------------------------------

    positive = (
        train_pair[
            "actual_total_control"
        ] > 0
    )

    ownership = (
        BinomialRidge(
            alpha=20.0
        )
        .fit(
            train_pair.loc[
                positive,
                CONTROL_OWNERSHIP_FEATURES,
            ].to_numpy(float),

            successes=(
                train_pair.loc[
                    positive,
                    "actual_red_control",
                ].to_numpy(float)
                / 10.0
            ),

            trials=(
                train_pair.loc[
                    positive,
                    "actual_total_control",
                ].to_numpy(float)
                / 10.0
            ),
        )
    )

    train_owner = (
        ownership
        .predict_probability(
            train_pair[
                CONTROL_OWNERSHIP_FEATURES
            ].to_numpy(float)
        )
    )

    test_owner = (
        ownership
        .predict_probability(
            test_pair[
                CONTROL_OWNERSHIP_FEATURES
            ].to_numpy(float)
        )
    )

    train_pair[
        "pred_red_control_share"
    ] = train_owner

    test_pair[
        "pred_red_control_share"
    ] = test_owner

    # ---------------------------------------------------------
    # Fit how much each actually landed TD shifts control
    # ownership beyond the prefight ownership prediction.
    #
    # eta =
    #   logit(prefight ownership)
    #   + beta * (Red TD landed - Blue TD landed)
    #
    # In simulation:
    # every Red landed TD adds beta;
    # every Blue landed TD subtracts beta.
    # ---------------------------------------------------------

    pos = train_pair[
        "actual_total_control"
    ] > 0

    base_eta = logit(
        train_pair.loc[
            pos,
            "pred_red_control_share",
        ].to_numpy(float)
    )

    td_diff = (
        train_pair.loc[
            pos,
            "actual_red_td_landed",
        ].to_numpy(float)
        - train_pair.loc[
            pos,
            "actual_blue_td_landed",
        ].to_numpy(float)
    )

    successes = (
        train_pair.loc[
            pos,
            "actual_red_control",
        ].to_numpy(float)
        / 10.0
    )

    trials = (
        train_pair.loc[
            pos,
            "actual_total_control",
        ].to_numpy(float)
        / 10.0
    )

    def objective(beta_array):
        beta = float(
            beta_array[0]
        )

        eta = (
            base_eta
            + beta * td_diff
        )

        loss = np.sum(
            trials
            * np.logaddexp(
                0.0,
                eta,
            )
            - successes * eta
        )

        loss += (
            0.5
            * beta ** 2
        )

        return float(
            loss
        )

    fit = minimize(
        objective,
        np.array(
            [0.0]
        ),
        method="L-BFGS-B",
        bounds=[
            (
                0.0,
                3.0,
            )
        ],
    )

    if not fit.success:
        raise RuntimeError(
            "TD-control interaction fit failed: "
            f"{fit.message}"
        )

    td_control_beta = float(
        fit.x[0]
    )

    # ---------------------------------------------------------
    # Positive total-control overdispersion.
    #
    # Shared control is generated as 10-second chunks.
    # First chunk is guaranteed after activation.
    # Fit variation in the remaining chunks.
    # ---------------------------------------------------------

    positive = (
        train_pair[
            "actual_total_control"
        ] > 0
    )

    actual_chunks = (
        train_pair.loc[
            positive,
            "actual_total_control",
        ].to_numpy(float)
        / CONTROL_CHUNK_SECONDS
    )

    predicted_chunks = (
        train_pair.loc[
            positive,
            "pred_control_conditional_total",
        ].to_numpy(float)
        / CONTROL_CHUNK_SECONDS
    )

    actual_extra = np.maximum(
        actual_chunks - 1.0,
        0.0,
    )

    predicted_extra = np.maximum(
        predicted_chunks - 1.0,
        1e-6,
    )

    control_alpha = (
        estimate_nb_alpha(
            actual_extra,
            predicted_extra,
        )
    )

    print()
    print("=" * 110)
    print(
        "SHARED CONTROL MODEL"
    )
    print("=" * 110)

    print(
        f"TRAIN any-control HIST: "
        f"{(train_pair['actual_total_control'] > 0).mean():.2%}"
    )

    print(
        f"TRAIN any-control PRED: "
        f"{train_p.mean():.2%}"
    )

    print(
        f"TEST any-control HIST:  "
        f"{(test_pair['actual_total_control'] > 0).mean():.2%}"
    )

    print(
        f"TEST any-control PRED:  "
        f"{test_p.mean():.2%}"
    )

    actual_direction = (
        test_pair[
            "actual_red_control"
        ]
        != test_pair[
            "actual_blue_control"
        ]
    ) & (
        test_pair[
            "actual_total_control"
        ] > 0
    )

    owner_correct = np.mean(
        (
            test_pair.loc[
                actual_direction,
                "pred_red_control_share",
            ]
            >= 0.5
        )
        ==
        (
            test_pair.loc[
                actual_direction,
                "actual_red_control",
            ]
            >
            test_pair.loc[
                actual_direction,
                "actual_blue_control",
            ]
        )
    )

    print(
        f"prefight ownership direction: "
        f"{owner_correct:.2%} "
        f"(N={actual_direction.sum()})"
    )

    print(
        f"fitted TD -> control ownership "
        f"logit shift per landed TD: "
        f"{td_control_beta:+.4f}"
    )

    print(
        f"positive-control extra-chunk "
        f"alpha: {control_alpha:.4f}"
    )

    return (
        td_control_beta,
        control_alpha,
    )


# =============================================================================
# COUNT ATTEMPT RESOLUTION
# =============================================================================

def resolve_attempt(
    output,
    row,
    side,
    family,
    rng,
):
    output[
        f"{side}_{family}_attempted"
    ] += 1.0

    pred_attempted = float(
        row[
            f"pred_{family}_attempted"
        ]
    )

    pred_landed = float(
        row[
            f"pred_{family}_landed"
        ]
    )

    if pred_attempted <= 0:
        return False

    p_land = float(
        np.clip(
            pred_landed
            / pred_attempted,
            0.0,
            0.98,
        )
    )

    landed = (
        rng.random()
        < p_land
    )

    if landed:
        output[
            f"{side}_{family}_landed"
        ] += 1.0

    return landed


# =============================================================================
# COMPETITIVE PATH
# =============================================================================

def simulate_path(
    pair,
    pair_info,
    rng,
    always_alpha,
    hurdle_extra_alpha,
    control_alpha,
    td_control_beta,
):
    duration = float(
        pair[
            "duration"
        ].iloc[0]
    )

    fighters = {
        row["side"]: row
        for _, row
        in pair.iterrows()
    }

    output = {}

    # ---------------------------------------------------------
    # Static high-frequency clocks:
    # distance + clinch.
    # ---------------------------------------------------------

    always_rates = {}

    for side in (
        "red",
        "blue",
    ):

        row = fighters[
            side
        ]

        for family in (
            "distance",
            "clinch",
            "ground",
            "td",
        ):
            output[
                f"{side}_{family}_attempted"
            ] = 0.0

            output[
                f"{side}_{family}_landed"
            ] = 0.0

        output[
            f"{side}_qualified_control_inflicted_seconds"
        ] = 0.0

        for family in (
            "distance",
            "clinch",
        ):

            frailty = (
                draw_frailty(
                    rng,
                    always_alpha[
                        family
                    ],
                )
            )

            expected = max(
                0.0,
                float(
                    row[
                        f"pred_{family}_attempted"
                    ]
                )
                * frailty,
            )

            always_rates[
                (
                    side,
                    family,
                )
            ] = (
                expected
                / duration
                if duration > 0
                else 0.0
            )

    # ---------------------------------------------------------
    # TD / ground hurdle states.
    #
    # Until activation:
    #     one activation clock.
    #
    # Activation event IS first attempt.
    #
    # After activation:
    #     extra-attempt clock generates remaining positive count.
    # ---------------------------------------------------------

    hurdle_state = {}

    for side in (
        "red",
        "blue",
    ):

        row = fighters[
            side
        ]

        for family in (
            "td",
            "ground",
        ):

            p = float(
                np.clip(
                    row[
                        f"pred_{family}_positive_probability"
                    ],
                    0.0,
                    0.999999,
                )
            )

            conditional_mean = max(
                1.0,
                float(
                    row[
                        f"pred_{family}_conditional_attempts"
                    ]
                ),
            )

            activation_rate = (
                -np.log1p(-p)
                / duration
                if (
                    p > 0
                    and duration > 0
                )
                else 0.0
            )

            hurdle_state[
                (
                    side,
                    family,
                )
            ] = {
                "active":
                    False,

                "activation_rate":
                    activation_rate,

                "conditional_mean":
                    conditional_mean,

                "extra_rate":
                    0.0,

                "extra_frailty":
                    draw_frailty(
                        rng,
                        hurdle_extra_alpha[
                            family
                        ],
                    ),
            }

    # ---------------------------------------------------------
    # Shared control.
    # ---------------------------------------------------------

    control_p = float(
        np.clip(
            pair_info[
                "pred_control_any_probability"
            ],
            0.0,
            0.999999,
        )
    )

    conditional_control = float(
        pair_info[
            "pred_control_conditional_total"
        ]
    )

    control_activation_rate = (
        -np.log1p(
            -control_p
        )
        / duration
        if (
            control_p > 0
            and duration > 0
        )
        else 0.0
    )

    control_active = False
    control_extra_rate = 0.0

    control_frailty = (
        draw_frailty(
            rng,
            control_alpha,
        )
    )

    base_red_ownership = float(
        np.clip(
            pair_info[
                "pred_red_control_share"
            ],
            1e-6,
            1.0 - 1e-6,
        )
    )

    base_owner_logit = float(
        logit(
            base_red_ownership
        )
    )

    # Actual path interaction state.
    td_control_shift = 0.0

    total_control = 0.0
    control_hit_cap = False

    time = 0.0

    # ---------------------------------------------------------
    # Shared control allocator.
    # ---------------------------------------------------------

    def allocate_control(
        amount,
    ):
        nonlocal total_control
        nonlocal control_hit_cap

        remaining = max(
            0.0,
            duration
            - total_control,
        )

        if remaining <= 0:
            control_hit_cap = True
            return

        actual_amount = min(
            amount,
            remaining,
        )

        if actual_amount < amount:
            control_hit_cap = True

        red_probability = float(
            sigmoid(
                base_owner_logit
                + td_control_shift
            )
        )

        owner = (
            "red"
            if rng.random()
            < red_probability
            else "blue"
        )

        output[
            f"{owner}_qualified_control_inflicted_seconds"
        ] += actual_amount

        total_control += (
            actual_amount
        )

    # ---------------------------------------------------------
    # Event loop.
    # ---------------------------------------------------------

    while time < duration:

        clocks = []

        # Always-on strike clocks.
        for (
            side,
            family,
        ), rate in (
            always_rates.items()
        ):

            if rate > 0:
                clocks.append(
                    (
                        side,
                        family,
                        "normal",
                        rate,
                    )
                )

        # Hurdle clocks.
        for (
            side,
            family,
        ), state in (
            hurdle_state.items()
        ):

            if not state[
                "active"
            ]:

                if (
                    state[
                        "activation_rate"
                    ] > 0
                ):
                    clocks.append(
                        (
                            side,
                            family,
                            "activate",
                            state[
                                "activation_rate"
                            ],
                        )
                    )

            elif (
                state[
                    "extra_rate"
                ] > 0
            ):

                clocks.append(
                    (
                        side,
                        family,
                        "extra",
                        state[
                            "extra_rate"
                        ],
                    )
                )

        # Shared control clock.
        if not control_active:

            if (
                control_activation_rate
                > 0
            ):
                clocks.append(
                    (
                        "shared",
                        "control",
                        "activate",
                        control_activation_rate,
                    )
                )

        elif (
            control_extra_rate
            > 0
            and total_control
            < duration
        ):

            clocks.append(
                (
                    "shared",
                    "control",
                    "extra",
                    control_extra_rate,
                )
            )

        total_rate = sum(
            event[3]
            for event in clocks
        )

        if total_rate <= 0:
            break

        dt = float(
            rng.exponential(
                1.0
                / total_rate
            )
        )

        time += dt

        if time >= duration:
            break

        draw = (
            rng.random()
            * total_rate
        )

        running = 0.0
        selected = clocks[-1]

        for event in clocks:

            running += (
                event[3]
            )

            if draw <= running:
                selected = event
                break

        side, family, kind, _ = (
            selected
        )

        # -----------------------------------------------------
        # CONTROL EVENT
        # -----------------------------------------------------

        if family == "control":

            if kind == "activate":

                control_active = True

                first_amount = min(
                    CONTROL_CHUNK_SECONDS,
                    conditional_control,
                )

                allocate_control(
                    first_amount
                )

                expected_remaining = max(
                    0.0,
                    conditional_control
                    - first_amount,
                )

                remaining_time = max(
                    duration - time,
                    1e-6,
                )

                control_extra_rate = (
                    expected_remaining
                    * control_frailty
                    / CONTROL_CHUNK_SECONDS
                    / remaining_time
                )

            else:

                allocate_control(
                    CONTROL_CHUNK_SECONDS
                )

            continue

        # -----------------------------------------------------
        # TD / GROUND ACTIVATION
        # -----------------------------------------------------

        if kind == "activate":

            state = hurdle_state[
                (
                    side,
                    family,
                )
            ]

            state[
                "active"
            ] = True

            landed = (
                resolve_attempt(
                    output,
                    fighters[
                        side
                    ],
                    side,
                    family,
                    rng,
                )
            )

            # First real fighter-vs-fighter interaction:
            #
            # A landed TD shifts ownership of all FUTURE
            # control chunks toward that fighter.
            if (
                family == "td"
                and landed
            ):

                if side == "red":
                    td_control_shift += (
                        td_control_beta
                    )
                else:
                    td_control_shift -= (
                        td_control_beta
                    )

                td_control_shift = float(
                    np.clip(
                        td_control_shift,
                        -5.0,
                        5.0,
                    )
                )

            expected_extra = max(
                0.0,
                state[
                    "conditional_mean"
                ]
                - 1.0,
            )

            remaining_time = max(
                duration - time,
                1e-6,
            )

            state[
                "extra_rate"
            ] = (
                expected_extra
                * state[
                    "extra_frailty"
                ]
                / remaining_time
            )

            continue

        # -----------------------------------------------------
        # NORMAL OR EXTRA COUNT EVENT
        # -----------------------------------------------------

        landed = (
            resolve_attempt(
                output,
                fighters[
                    side
                ],
                side,
                family,
                rng,
            )
        )

        if (
            family == "td"
            and landed
        ):

            if side == "red":
                td_control_shift += (
                    td_control_beta
                )
            else:
                td_control_shift -= (
                    td_control_beta
                )

            td_control_shift = float(
                np.clip(
                    td_control_shift,
                    -5.0,
                    5.0,
                )
            )

    output[
        "total_control_seconds"
    ] = total_control

    output[
        "control_hit_cap"
    ] = float(
        control_hit_cap
    )

    output[
        "final_td_control_shift"
    ] = (
        td_control_shift
    )

    return output


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "EVENT CLOCK MC V4 — "
        "HURDLE GRAPPLING + "
        "SHARED COMPETITIVE CONTROL"
    )
    print("=" * 120)

    train, test = (
        prepare_direct_predictions()
    )

    # Standing totals for convenience.
    for frame in (
        train,
        test,
    ):

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

    feature_cols = (
        direct_feature_columns()
    )

    # =========================================================
    # TD / GROUND HURDLES
    # =========================================================

    print()
    print("=" * 110)
    print(
        "TD / GROUND HURDLE MODELS"
    )
    print("=" * 110)

    hurdle_extra_alpha = {}

    for family in (
        "td",
        "ground",
    ):

        hurdle_extra_alpha[
            family
        ] = fit_count_hurdle(
            train,
            test,
            family,
            feature_cols,
        )

    # =========================================================
    # DISTANCE / CLINCH OVERDISPERSION
    # =========================================================

    always_alpha = {}

    for family in (
        "distance",
        "clinch",
    ):

        always_alpha[
            family
        ] = estimate_nb_alpha(
            train[
                f"{family}_attempted"
            ],
            train[
                f"pred_{family}_attempted"
            ],
        )

    # =========================================================
    # SHARED CONTROL
    # =========================================================

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

    pair_lookup = {
        str(row["fight_id"]):
            row
        for _, row
        in test_pair.iterrows()
    }

    # =========================================================
    # RUN
    # =========================================================

    print()
    print("=" * 120)
    print(
        f"RUNNING COMPETITIVE CLOCKS — "
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

        pair_info = (
            pair_lookup[
                str(fight_id)
            ]
        )

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

            output = simulate_path(
                pair,
                pair_info,
                rng,
                always_alpha,
                hurdle_extra_alpha,
                control_alpha,
                td_control_beta,
            )

            for _, fighter_row in (
                pair.iterrows()
            ):

                side = (
                    fighter_row[
                        "side"
                    ]
                )

                row = {
                    "fight_id":
                        str(fight_id),

                    "path":
                        path_index,

                    "side":
                        side,

                    "fighter_name":
                        fighter_row[
                            "fighter_name"
                        ],

                    "duration":
                        float(
                            fighter_row[
                                "duration"
                            ]
                        ),

                    "control_hit_cap":
                        output[
                            "control_hit_cap"
                        ],

                    "final_td_control_shift":
                        output[
                            "final_td_control_shift"
                        ],
                }

                for family in (
                    "distance",
                    "clinch",
                    "ground",
                    "td",
                ):

                    row[
                        f"sim_{family}_attempted"
                    ] = output[
                        f"{side}_{family}_attempted"
                    ]

                    row[
                        f"sim_{family}_landed"
                    ] = output[
                        f"{side}_{family}_landed"
                    ]

                row[
                    "sim_qualified_control_inflicted_seconds"
                ] = output[
                    f"{side}_qualified_control_inflicted_seconds"
                ]

                path_rows.append(
                    row
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
    # FIGHTER PATH MEANS
    # =========================================================

    agg = {}

    for family in (
        "distance",
        "clinch",
        "ground",
        "td",
    ):

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
    # MARGINALS
    # =========================================================

    print()
    print("=" * 135)
    print(
        "MARGINAL MEANS / "
        "FIGHTER DISCRIMINATION"
    )
    print("=" * 135)

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
                result[target],
                result[direct],
            )
        )

        _, sim_rho, sim_mae = (
            metrics(
                result[target],
                result[sim],
            )
        )

        print()
        print(label)
        print("-" * 135)

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
    # PATH DISTRIBUTIONS
    # =========================================================

    print()
    print("=" * 135)
    print(
        "PATH DISTRIBUTIONS — "
        "HISTORICAL VS COMPETITIVE CLOCK"
    )
    print("=" * 135)

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
        print("-" * 135)

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
    # COMPETITIVE INTERACTION TEST
    #
    # Does a TD advantage correspond to control ownership?
    # =========================================================

    path_pair_rows = []

    for (
        fight_id,
        path_index,
    ), group in (
        paths.groupby(
            [
                "fight_id",
                "path",
            ]
        )
    ):

        red = group[
            group[
                "side"
            ] == "red"
        ].iloc[0]

        blue = group[
            group[
                "side"
            ] == "blue"
        ].iloc[0]

        red_control = float(
            red[
                "sim_qualified_control_inflicted_seconds"
            ]
        )

        blue_control = float(
            blue[
                "sim_qualified_control_inflicted_seconds"
            ]
        )

        total_control = (
            red_control
            + blue_control
        )

        path_pair_rows.append(
            {
                "fight_id":
                    fight_id,

                "path":
                    path_index,

                "td_diff":
                    float(
                        red[
                            "sim_td_landed"
                        ]
                        - blue[
                            "sim_td_landed"
                        ]
                    ),

                "red_control":
                    red_control,

                "blue_control":
                    blue_control,

                "total_control":
                    total_control,

                "red_control_share":
                    (
                        red_control
                        / total_control
                        if total_control > 0
                        else np.nan
                    ),

                "control_hit_cap":
                    float(
                        red[
                            "control_hit_cap"
                        ]
                    ),
            }
        )

    path_pair = pd.DataFrame(
        path_pair_rows
    )

    hist_interaction = (
        test_pair[
            test_pair[
                "actual_total_control"
            ] > 0
        ]
        .copy()
    )

    hist_interaction[
        "td_diff"
    ] = (
        hist_interaction[
            "actual_red_td_landed"
        ]
        - hist_interaction[
            "actual_blue_td_landed"
        ]
    )

    hist_interaction[
        "red_control_share"
    ] = (
        hist_interaction[
            "actual_red_control"
        ]
        / hist_interaction[
            "actual_total_control"
        ]
    )

    # majority_match() expects common red_control / blue_control names.
    # Historical pair data stores them with actual_ prefixes.
    hist_interaction["red_control"] = (
        hist_interaction["actual_red_control"]
    )
    hist_interaction["blue_control"] = (
        hist_interaction["actual_blue_control"]
    )

    sim_interaction = (
        path_pair[
            path_pair[
                "total_control"
            ] > 0
        ]
        .copy()
    )

    hist_rho = (
        hist_interaction[
            [
                "td_diff",
                "red_control_share",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )

    sim_rho = (
        sim_interaction[
            [
                "td_diff",
                "red_control_share",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[
            0,
            1,
        ]
    )

    def majority_match(
        frame,
    ):
        unequal_td = (
            frame[
                "td_diff"
            ] != 0
        )

        unequal_control = (
            frame[
                "red_control"
            ]
            != frame[
                "blue_control"
            ]
        )

        subset = frame[
            unequal_td
            & unequal_control
        ]

        if len(
            subset
        ) == 0:
            return (
                np.nan,
                0,
            )

        correct = (
            np.sign(
                subset[
                    "td_diff"
                ]
            )
            ==
            np.sign(
                subset[
                    "red_control"
                ]
                - subset[
                    "blue_control"
                ]
            )
        )

        return (
            float(
                correct.mean()
            ),
            len(
                subset
            ),
        )

    hist_match, hist_n = (
        majority_match(
            hist_interaction
        )
    )

    sim_match, sim_n = (
        majority_match(
            sim_interaction
        )
    )

    print()
    print("=" * 135)
    print(
        "FIRST FIGHTER-vs-FIGHTER "
        "INTERACTION TEST"
    )
    print("=" * 135)

    print(
        f"TD landed differential vs "
        f"Red control share Spearman:"
    )

    print(
        f"  HIST = {hist_rho:+.4f}"
    )

    print(
        f"  SIM  = {sim_rho:+.4f}"
    )

    print()

    print(
        f"When TD landed totals differ, "
        f"same fighter owns more control:"
    )

    print(
        f"  HIST = "
        f"{hist_match:.2%} "
        f"(N={hist_n})"
    )

    print(
        f"  SIM  = "
        f"{sim_match:.2%} "
        f"(N={sim_n})"
    )

    print()

    print(
        f"Shared-control paths hitting "
        f"the fight-time cap: "
        f"{path_pair['control_hit_cap'].mean():.2%}"
    )

    # =========================================================
    # CONTROL DIRECTION AT MATCHUP LEVEL
    # =========================================================

    direction, n_direction = (
        within_bout_direction(
            result,
            "qualified_control_inflicted_seconds",
            "sim_qualified_control_inflicted_seconds",
        )
    )

    print(
        f"Clock-MC control within-bout "
        f"direction: "
        f"{direction:.2%} "
        f"(N={n_direction})"
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
