from __future__ import annotations

import math

import numpy as np

from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import (
    draw_frailty,
)

from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    CONTROL_CHUNK_SECONDS,
    logit,
    resolve_attempt,
    sigmoid,
)


def fit_ownership_concentration(train_pair):
    """
    Fit persistent path-to-path variation in control ownership.

    Each path draws one Red control propensity around the prefight mean:

        p_path ~ Beta(p*kappa, (1-p)*kappa)

    Low kappa => persistent/extreme ownership.
    High kappa => ownership stays close to prefight mean.
    """

    positive = (
        train_pair["actual_total_control"]
        > 0
    )

    y = (
        train_pair.loc[
            positive,
            "actual_red_control",
        ].to_numpy(float)
        /
        train_pair.loc[
            positive,
            "actual_total_control",
        ].to_numpy(float)
    )

    p = np.clip(
        train_pair.loc[
            positive,
            "pred_red_control_share",
        ].to_numpy(float),
        1e-4,
        1.0 - 1e-4,
    )

    scaled_error = (
        (y - p) ** 2
        /
        (
            p
            * (1.0 - p)
        )
    )

    ratio = float(
        np.mean(
            scaled_error
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
    print("=" * 105)
    print("PERSISTENT CONTROL OWNERSHIP")
    print("=" * 105)

    print(
        f"ownership beta concentration "
        f"kappa: {kappa:.4f}"
    )

    return kappa


def _draw_control_target(
    pair_info,
    rng,
    control_alpha,
    duration,
):
    """
    Draw total control for this entire path.

    Control remains a direct predicted output.
    The clock determines WHEN those seconds occur.

    Mean:
        P(any control)
        *
        E(control | positive)

    remains anchored to the direct pair prediction.
    """

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

    expected_extra_chunks = max(
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
            expected_extra_chunks
            * frailty
        )
    )

    target = (
        first
        + extra_chunks
        * CONTROL_CHUNK_SECONDS
    )

    return float(
        np.clip(
            target,
            0.0,
            duration,
        )
    )


def _draw_path_ownership(
    base_red_share,
    rng,
    kappa,
):
    p = float(
        np.clip(
            base_red_share,
            1e-5,
            1.0 - 1e-5,
        )
    )

    a = max(
        p * kappa,
        1e-5,
    )

    b = max(
        (1.0 - p) * kappa,
        1e-5,
    )

    return float(
        rng.beta(
            a,
            b,
        )
    )


def simulate_path(
    pair,
    pair_info,
    rng,
    always_alpha,
    hurdle_extra_alpha,
    control_alpha,
    td_control_beta,
    ownership_kappa,
):
    """
    Stage-6 event-clock path.

    Key contract
    ------------
    There is still NO standing/ground phase machine.

    Instead:

      * distance/clinch/TD clocks run during free fight time;
      * control has its own competing clock;
      * when control fires it consumes real fight seconds;
      * distance/clinch/TD clocks are unavailable in those seconds;
      * ground-strike clocks continue to exist during control windows;
      * control ownership is persistent within a path;
      * landed TDs shift future control ownership.

    The direct predictions remain the baseline expectations.
    """

    duration = float(
        pair["duration"].iloc[0]
    )

    fighters = {
        row["side"]: row
        for _, row
        in pair.iterrows()
    }

    output = {}

    for side in (
        "red",
        "blue",
    ):
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

    # ------------------------------------------------------------------
    # Draw the path's total shared control first.
    #
    # The control clock below decides WHEN it occupies the timeline.
    # ------------------------------------------------------------------

    target_control = _draw_control_target(
        pair_info,
        rng,
        control_alpha,
        duration,
    )

    control_remaining = (
        target_control
    )

    predicted_total_control = float(
        np.clip(
            pair_info[
                "pred_total_control"
            ],
            0.0,
            duration,
        )
    )

    # Expected amount of timeline available for free-time events.
    expected_free_time = max(
        duration
        - predicted_total_control,
        30.0,
    )

    # ------------------------------------------------------------------
    # Persistent ownership realization.
    # ------------------------------------------------------------------

    base_red_share = float(
        np.clip(
            pair_info[
                "pred_red_control_share"
            ],
            1e-5,
            1.0 - 1e-5,
        )
    )

    path_red_share = (
        _draw_path_ownership(
            base_red_share,
            rng,
            ownership_kappa,
        )
    )

    owner_logit = float(
        logit(
            path_red_share
        )
    )

    td_control_shift = 0.0

    # ------------------------------------------------------------------
    # Distance + clinch:
    #
    # Direct counts are full-fight predictions.
    # Divide by EXPECTED free time so the expected matchup retains its
    # direct mean, while paths with unusually high control lose volume.
    # ------------------------------------------------------------------

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
        ):
            expected = max(
                0.0,
                float(
                    row[
                        f"pred_{family}_attempted"
                    ]
                ),
            )

            frailty = draw_frailty(
                rng,
                always_alpha[
                    family
                ],
            )

            always_rates[
                (
                    side,
                    family,
                )
            ] = (
                expected
                * frailty
                / expected_free_time
            )

    # ------------------------------------------------------------------
    # Hurdle clocks.
    #
    # TD activation is a FREE-TIME process.
    # Ground activation is an ALL-TIME process.
    # ------------------------------------------------------------------

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

            exposure = (
                expected_free_time
                if family == "td"
                else duration
            )

            activation_rate = (
                -np.log1p(
                    -p
                )
                / exposure
                if (
                    p > 0
                    and exposure > 0
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

    def apply_td_ownership_shift(
        side,
        landed,
    ):
        nonlocal td_control_shift

        if not landed:
            return

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

    def activate_hurdle(
        side,
        family,
        current_time,
    ):
        state = hurdle_state[
            (
                side,
                family,
            )
        ]

        state[
            "active"
        ] = True

        landed = resolve_attempt(
            output,
            fighters[
                side
            ],
            side,
            family,
            rng,
        )

        if family == "td":
            apply_td_ownership_shift(
                side,
                landed,
            )

        expected_extra = max(
            0.0,
            state[
                "conditional_mean"
            ]
            - 1.0,
        )

        if family == "td":
            remaining_exposure = max(
                expected_free_time
                * (
                    1.0
                    - current_time
                    / max(
                        duration,
                        1e-6,
                    )
                ),
                1.0,
            )
        else:
            remaining_exposure = max(
                duration
                - current_time,
                1.0,
            )

        state[
            "extra_rate"
        ] = (
            expected_extra
            * state[
                "extra_frailty"
            ]
            / remaining_exposure
        )

    # ------------------------------------------------------------------
    # Ground clocks run inside control windows as well.
    # ------------------------------------------------------------------

    def simulate_ground_interval(
        start_time,
        interval_seconds,
    ):
        local = 0.0

        while (
            local
            < interval_seconds
        ):
            clocks = []

            for side in (
                "red",
                "blue",
            ):
                state = hurdle_state[
                    (
                        side,
                        "ground",
                    )
                ]

                rate = (
                    state[
                        "extra_rate"
                    ]
                    if state[
                        "active"
                    ]
                    else state[
                        "activation_rate"
                    ]
                )

                if rate > 0:
                    clocks.append(
                        (
                            side,
                            rate,
                        )
                    )

            total_rate = sum(
                rate
                for _, rate
                in clocks
            )

            if total_rate <= 0:
                break

            dt = float(
                rng.exponential(
                    1.0
                    / total_rate
                )
            )

            if (
                local + dt
                >= interval_seconds
            ):
                break

            local += dt

            draw = (
                rng.random()
                * total_rate
            )

            running = 0.0
            selected_side = (
                clocks[-1][0]
            )

            for side, rate in clocks:
                running += rate

                if draw <= running:
                    selected_side = (
                        side
                    )
                    break

            state = hurdle_state[
                (
                    selected_side,
                    "ground",
                )
            ]

            global_time = (
                start_time
                + local
            )

            if not state[
                "active"
            ]:
                activate_hurdle(
                    selected_side,
                    "ground",
                    global_time,
                )
            else:
                resolve_attempt(
                    output,
                    fighters[
                        selected_side
                    ],
                    selected_side,
                    "ground",
                    rng,
                )

    def consume_control_window(
        current_time,
    ):
        nonlocal control_remaining

        amount = min(
            CONTROL_CHUNK_SECONDS,
            control_remaining,
            duration
            - current_time,
        )

        if amount <= 0:
            return 0.0

        red_probability = float(
            sigmoid(
                owner_logit
                + td_control_shift
            )
        )

        owner = (
            "red"
            if (
                rng.random()
                < red_probability
            )
            else "blue"
        )

        output[
            f"{owner}_qualified_control_inflicted_seconds"
        ] += amount

        # Ground clocks remain alive while control consumes the timeline.
        simulate_ground_interval(
            current_time,
            amount,
        )

        control_remaining -= (
            amount
        )

        return amount

    # ------------------------------------------------------------------
    # MAIN EVENT LOOP
    # ------------------------------------------------------------------

    time = 0.0
    free_seconds = 0.0
    control_seconds = 0.0

    while time < duration:

        schedule_remaining = (
            duration
            - time
        )

        if schedule_remaining <= 0:
            break

        # If the remaining control target now fills the entire remaining
        # schedule, consume it directly. No free-time opportunity remains.
        if (
            control_remaining > 0
            and schedule_remaining
            <= control_remaining
            + 1e-9
        ):
            amount = (
                consume_control_window(
                    time
                )
            )

            time += amount
            control_seconds += (
                amount
            )

            continue

        clocks = []

        # Distance / clinch only in free time.
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

        # TD + ground hurdle clocks.
        for (
            side,
            family,
        ), state in (
            hurdle_state.items()
        ):
            rate = (
                state[
                    "extra_rate"
                ]
                if state[
                    "active"
                ]
                else state[
                    "activation_rate"
                ]
            )

            if rate <= 0:
                continue

            clocks.append(
                (
                    side,
                    family,
                    (
                        "extra"
                        if state[
                            "active"
                        ]
                        else "activate"
                    ),
                    rate,
                )
            )

        # Shared control clock.
        #
        # remaining windows / remaining free-time budget
        # gives a self-correcting competing hazard without an escape model.
        if control_remaining > 0:

            windows_remaining = max(
                1,
                int(
                    math.ceil(
                        control_remaining
                        / CONTROL_CHUNK_SECONDS
                    )
                ),
            )

            free_budget = max(
                schedule_remaining
                - control_remaining,
                1e-6,
            )

            control_rate = (
                windows_remaining
                / free_budget
            )

            clocks.append(
                (
                    "shared",
                    "control",
                    "window",
                    control_rate,
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

        if (
            time + dt
            >= duration
        ):
            free_seconds += (
                duration - time
            )

            time = duration
            break

        time += dt
        free_seconds += dt

        draw = (
            rng.random()
            * total_rate
        )

        running = 0.0
        selected = (
            clocks[-1]
        )

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

        # --------------------------------------------------------------
        # CONTROL WINDOW
        # --------------------------------------------------------------

        if family == "control":

            amount = (
                consume_control_window(
                    time
                )
            )

            time += amount

            control_seconds += (
                amount
            )

            continue

        # --------------------------------------------------------------
        # HURDLE ACTIVATION
        # --------------------------------------------------------------

        if kind == "activate":

            activate_hurdle(
                side,
                family,
                time,
            )

            continue

        # --------------------------------------------------------------
        # ORDINARY ATTEMPT
        # --------------------------------------------------------------

        landed = resolve_attempt(
            output,
            fighters[
                side
            ],
            side,
            family,
            rng,
        )

        if family == "td":
            apply_td_ownership_shift(
                side,
                landed,
            )

    output[
        "target_control_seconds"
    ] = target_control

    output[
        "realized_control_seconds"
    ] = control_seconds

    output[
        "free_seconds"
    ] = free_seconds

    output[
        "path_red_control_share_prior"
    ] = path_red_share

    output[
        "final_td_control_shift"
    ] = td_control_shift

    return output
