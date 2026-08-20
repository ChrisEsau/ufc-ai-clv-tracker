from pathlib import Path
from dataclasses import replace
import sys
from collections import Counter

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from scipy.optimize import minimize_scalar

from pipeline.common.paths import (
    MASTER_PATH,
    FSR_V2_PREFIGHT_SNAPSHOTS_PATH,
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
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage7_budget_timeline import (
    add_historical_free_time,
    fit_standing_free_time_model,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage8_grappling_calibration import (
    fit_directional_ownership_kappa,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import (
    fit_ground_alpha_by_shape,
    fit_control_minority_models,
    simulate_stage9_path,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage10d_total_fight_judge import (
    VARIANTS,
    decision_mask,
    fit_model,
    prepare_master,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11_submission_attempts import (
    build_submission_targets,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11c_submission_conversion import (
    clip_probability,
    load_submission_baseline,
    logistic,
    logit,
)

from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import (
    EventClockShadowKOKDModel,
)

from pipeline.simulation.event_mc_v1.diagnostics.population_validation import (
    _fight,
)
from pipeline.simulation.event_mc_v1.calibration import DEFAULT_RESOLVER
from pipeline.simulation.event_mc_v1.components.actions import ActionAttempt
from pipeline.simulation.event_mc_v1.components.profiles import (
    MatchupProfiles,
    Side,
)
from pipeline.simulation.event_mc_v1.finishes import KOTKOFinishModel
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.physiology import (
    ImpactTraumaKnockdownModel,
    PhysiologyTimeAdvanceModel,
)
from pipeline.simulation.event_mc_v1.stamina import StaminaModel
from pipeline.simulation.event_mc_v1.state import (
    FightState,
    StateDelta,
)


FIGHTS = 500
PATHS = 20
SEED = 20260818

# Fight-context translation for the validated paired FSR V2
# striking-power trait.
#
# persisted FSR power is unchanged.
#
# effective_power = persisted_power - 1.15 * (age - 30)
POWER_AGE_CENTER_YEARS = 30.0
POWER_AGE_RATING_POINTS_PER_YEAR = -1.15


# =============================================================================
# EVENT CLOCK PHYSICAL TRANSLATION
# =============================================================================

def event_clock_effective_power_rating(
    persisted_power,
    age_years,
):
    power = float(persisted_power)
    age = float(age_years)

    if not np.isfinite(power):
        raise RuntimeError(
            "Non-finite persisted FSR striking_power."
        )

    if not np.isfinite(age) or age <= 0.0:
        raise RuntimeError(
            "Invalid fight-date fighter age."
        )

    # Deliberately no clipping here.
    # The 35-90 bounds belong to the persisted FSR rating,
    # not to the matchup-context effective rating.
    return (
        power
        + POWER_AGE_RATING_POINTS_PER_YEAR
        * (
            age
            - POWER_AGE_CENTER_YEARS
        )
    )


def event_clock_physics_profiles(fight):
    """
    Build consequence-only Event Clock profiles.

    Stage-9 flow and stamina continue to use fight.profiles.

    Only striking power is translated here for the
    impact / trauma / KD / KO chain.
    """

    matchup = fight.fsr_v2_matchup

    if matchup is None:
        raise RuntimeError(
            "Event Clock full-fight replay requires "
            "FSR V2 matchup context."
        )

    def translated(side):
        profile = fight.profiles.fighter(
            side
        )

        fsr_input = (
            matchup.red
            if side is Side.RED
            else matchup.blue
        )

        effective_power = (
            event_clock_effective_power_rating(
                profile.striking_power,
                fsr_input.age_years,
            )
        )

        changes = {
            "striking_power":
                effective_power,
        }

        # The shared old consequence helper may locally
        # contain an age-aware power adjustment from other
        # work. Event Clock owns the translation above, so
        # downstream age must be neutral to prevent double
        # counting.
        if hasattr(
            profile,
            "age_years",
        ):
            changes["age_years"] = (
                POWER_AGE_CENTER_YEARS
            )

        return replace(
            profile,
            **changes,
        )

    return MatchupProfiles(
        red=translated(Side.RED),
        blue=translated(Side.BLUE),
    )


def event_clock_shadow_ko_kd_profiles(fight):
    """
    Profiles for the empirical shadow KO/KD model.

    IMPORTANT:
    - striking_power stays at the persisted FSR value
    - KD resistance stays at the persisted FSR value
    - canonical fight-date age is supplied explicitly
    - no power-age translation is applied here

    The shadow hazard coefficients were estimated using
    persisted ratings and age as separate predictors.
    """
    matchup = fight.fsr_v2_matchup

    if matchup is None:
        raise RuntimeError(
            "Shadow KO/KD replay requires FSR V2 matchup context."
        )

    def build(side):
        profile = fight.profiles.fighter(side)

        fsr_input = (
            matchup.red
            if side is Side.RED
            else matchup.blue
        )

        return replace(
            profile,
            age_years=float(fsr_input.age_years),
        )

    return MatchupProfiles(
        red=build(Side.RED),
        blue=build(Side.BLUE),
    )



# =============================================================================
# BASIC HELPERS
# =============================================================================

def apply_delta(state, delta):
    for name in (
        "finished",
        "finish_reason",
        "winner",
        "finish_method",
        "red_stamina",
        "blue_stamina",
        "red_cumulative_trauma",
        "blue_cumulative_trauma",
        "red_acute_vulnerability",
        "blue_acute_vulnerability",
    ):
        value = getattr(delta, name, None)
        if value is not None:
            setattr(state, name, value)


def normalize_method(value):
    s = str(value).lower()

    if "decision" in s:
        return "DEC"

    if "submission" in s:
        return "SUB"

    if "ko" in s or "tko" in s:
        return "KO_TKO"

    return "OTHER"


def base_submission_rate(frame):
    tendency = pd.to_numeric(
        frame["self_submission_tendency"],
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    suppression = pd.to_numeric(
        frame["opp_submission_suppression"],
        errors="coerce",
    ).fillna(1.0).clip(lower=0.0)

    return tendency * suppression


def fit_submission_attempt_scale(train):
    raw = (
        train["submission_base_rate"].to_numpy(float)
        * train["historical_duration"].to_numpy(float)
    )

    actual = train["submission_attempted"].to_numpy(float)

    return float(
        actual.sum()
        / max(raw.sum(), 1e-12)
    )


def fit_conversion_offset(train):
    active = train["submission_attempted"].to_numpy(float) > 0

    y = train.loc[
        active,
        "submission_win",
    ].to_numpy(int)

    attempts = train.loc[
        active,
        "submission_attempted",
    ].to_numpy(float)

    baseline = clip_probability(
        train.loc[
            active,
            "submission_conversion_baseline",
        ]
    )

    def objective(offset):
        p = logistic(
            logit(baseline)
            + float(offset)
        )

        q = (
            1.0
            - np.power(
                1.0 - p,
                attempts,
            )
        )

        q = clip_probability(q)

        return float(
            -np.sum(
                y * np.log(q)
                + (1 - y) * np.log(1 - q)
            )
        )

    fit = minimize_scalar(
        objective,
        bounds=(-3.0, 3.0),
        method="bounded",
    )

    if not fit.success:
        raise RuntimeError(
            "submission conversion fit failed"
        )

    return float(fit.x)


# =============================================================================
# SCHEDULE EVENTS
# =============================================================================

def add_budget_events(
    events,
    side,
    family,
    attempted,
    landed,
    horizon,
    rng,
):
    attempted = int(max(attempted, 0))
    landed = int(
        np.clip(
            landed,
            0,
            attempted,
        )
    )

    if attempted <= 0:
        return

    times = rng.uniform(
        0.0,
        horizon,
        attempted,
    )

    flags = np.zeros(
        attempted,
        dtype=bool,
    )

    if landed > 0:
        idx = rng.choice(
            attempted,
            size=landed,
            replace=False,
        )
        flags[idx] = True

    for t, did_land in zip(
        times,
        flags,
    ):
        events.append(
            (
                float(t),
                side,
                family,
                bool(did_land),
            )
        )


def add_submission_events(
    events,
    side,
    rate,
    horizon,
    rng,
):
    count = int(
        rng.poisson(
            max(rate, 0.0)
            * horizon
        )
    )

    if count <= 0:
        return

    times = rng.uniform(
        0.0,
        horizon,
        count,
    )

    for t in times:
        events.append(
            (
                float(t),
                side,
                "submission_attempt",
                None,
            )
        )


# =============================================================================
# PATH SIMULATOR
# =============================================================================

def simulate_integrated_path(
    fight,
    budgets,
    submission_rates,
    conversion_probability,
    judge_model,
    judge_features,
    seed,
):
    rng = np.random.default_rng(seed)

    horizon = float(
        fight.rounds * 300
    )

    key = (
        fight.division
        if fight.division
        in DEFAULT_RESOLVER.weight_classes
        else None
    )

    calibration = (
        DEFAULT_RESOLVER
        .for_weight_class(key)
    )

    shadow_profiles = (
        event_clock_shadow_ko_kd_profiles(
            fight
        )
    )

    stamina = StaminaModel(
        fight.profiles,
        calibration=calibration,
    )

    modifiers = DynamicModifierProvider(
        calibration
    )

    shadow_ko_kd = EventClockShadowKOKDModel(
        shadow_profiles
    )

    # Retained only for stamina/time advancement.
    # Shadow KO/KD does not consume cumulative trauma
    # or acute vulnerability.
    time_advance = PhysiologyTimeAdvanceModel(
        stamina,
        calibration,
    )

    state = FightState()

    # -------------------------------------------------------------
    # Full scheduled path budgets
    # -------------------------------------------------------------

    events = []

    for side_name in (
        "red",
        "blue",
    ):
        side = Side(side_name)

        add_budget_events(
            events,
            side,
            "standing_strike",
            budgets[
                f"{side_name}_standing_attempted"
            ],
            budgets[
                f"{side_name}_standing_landed"
            ],
            horizon,
            rng,
        )

        add_budget_events(
            events,
            side,
            "ground_strike",
            budgets[
                f"{side_name}_ground_attempted"
            ],
            budgets[
                f"{side_name}_ground_landed"
            ],
            horizon,
            rng,
        )

        add_budget_events(
            events,
            side,
            "takedown",
            budgets[
                f"{side_name}_td_attempted"
            ],
            budgets[
                f"{side_name}_td_landed"
            ],
            horizon,
            rng,
        )

        add_submission_events(
            events,
            side,
            submission_rates[
                side_name
            ],
            horizon,
            rng,
        )

    events.sort(
        key=lambda x: x[0]
    )

    # -------------------------------------------------------------
    # Executed path statistics
    # -------------------------------------------------------------

    stats = {
        "red": Counter(),
        "blue": Counter(),
    }

    kds = {
        "red": 0,
        "blue": 0,
    }

    next_boundary = 300.0

    def advance_to(target):
        nonlocal next_boundary

        while (
            next_boundary < target
            and next_boundary < horizon
        ):
            dt = (
                next_boundary
                - state.fight_time_seconds
            )

            delta = time_advance.advance(
                state,
                None,
                dt,
            )

            apply_delta(
                state,
                delta,
            )

            state.fight_time_seconds = (
                next_boundary
            )

            recovery = (
                stamina.recovery_delta(
                    state
                )
            )

            apply_delta(
                state,
                recovery,
            )

            next_boundary += 300.0

        dt = (
            target
            - state.fight_time_seconds
        )

        if dt > 0:
            delta = time_advance.advance(
                state,
                None,
                dt,
            )

            apply_delta(
                state,
                delta,
            )

            state.fight_time_seconds = (
                target
            )

    # -------------------------------------------------------------
    # Chronological event competition
    # -------------------------------------------------------------

    for (
        event_time,
        side,
        family,
        landed,
    ) in events:

        if state.finished:
            break

        advance_to(
            event_time
        )

        side_name = side.value

        stats[
            side_name
        ][f"{family}_attempted"] += 1

        profile = (
            fight.profiles.fighter(
                side
            )
        )

        dynamic = modifiers.modifiers(
            profile,
            state,
            side,
        )

        cost = stamina.action_delta(
            state,
            side,
            family,
        )

        apply_delta(
            state,
            cost,
        )

        # ---------------------------------------------------------
        # Submission
        # ---------------------------------------------------------

        if family == "submission_attempt":

            if rng.random() < conversion_probability:

                state.finished = True
                state.finish_reason = "SUB"
                state.finish_method = "SUB"
                state.winner = side_name

            continue

        # ---------------------------------------------------------
        # TD
        # ---------------------------------------------------------

        if family == "takedown":

            if landed:
                stats[
                    side_name
                ]["td_landed"] += 1

            continue

        # ---------------------------------------------------------
        # Strike
        # ---------------------------------------------------------

        if landed:
            stats[
                side_name
            ][f"{family}_landed"] += 1

        # ---------------------------------------------------------
        # SHADOW KO / KD CONSEQUENCE
        #
        # Only landed significant strikes enter the consequence model.
        #
        # Ordering:
        #   1. KO/TKO?
        #   2. only if no KO/TKO, KD?
        #
        # A survived KD changes the state seen by later strikes.
        # The same strike can never be both KD and KO/TKO.
        # ---------------------------------------------------------

        if landed and family in (
            "standing_strike",
            "ground_strike",
        ):
            consequence = (
                shadow_ko_kd.resolve_landed_strike(
                    state=state,
                    attacker=side,
                    prior_defender_kds=kds[side_name],
                    rng=rng,
                )
            )

            stats[
                side_name
            ]["ko_kd_strike_opportunities"] += 1

            stats[
                side_name
            ]["ko_probability_sum"] += (
                consequence.ko_probability
            )

            if consequence.ko_tko:
                stats[
                    side_name
                ]["ko_events"] += 1

                state.finished = True
                state.finish_reason = "KO_TKO"
                state.finish_method = "KO_TKO"
                state.winner = side_name

                break

            stats[
                side_name
            ]["kd_probability_sum"] += (
                consequence.kd_probability
            )

            if consequence.knockdown:
                kds[
                    side_name
                ] += 1

                stats[
                    side_name
                ]["kd_events"] += 1

    # -------------------------------------------------------------
    # Decision
    # -------------------------------------------------------------

    if not state.finished:

        advance_to(
            horizon
        )

        red_sig = (
            stats["red"][
                "standing_strike_landed"
            ]
            + stats["red"][
                "ground_strike_landed"
            ]
        )

        blue_sig = (
            stats["blue"][
                "standing_strike_landed"
            ]
            + stats["blue"][
                "ground_strike_landed"
            ]
        )

        decision_row = {
            "sig_diff":
                red_sig
                - blue_sig,

            "kd_diff":
                kds["red"]
                - kds["blue"],

            "td_diff":
                stats["red"]["td_landed"]
                - stats["blue"]["td_landed"],

            "sub_diff":
                stats["red"][
                    "submission_attempt_attempted"
                ]
                - stats["blue"][
                    "submission_attempt_attempted"
                ],

            "ctrl_diff":
                budgets["red_control"]
                - budgets["blue_control"],
        }

        p_red = float(
            judge_model.predict_proba(
                pd.DataFrame(
                    [decision_row]
                )[judge_features]
            )[0, 1]
        )

        state.finished = True
        state.finish_reason = "DEC"
        state.finish_method = "DEC"

        state.winner = (
            "red"
            if rng.random() < p_red
            else "blue"
        )

    return {
        "winner":
            state.winner,

        "method":
            state.finish_method,

        "elapsed":
            state.fight_time_seconds,

        "red_kd":
            kds["red"],

        "blue_kd":
            kds["blue"],

        "red_ko_kd_strike_opportunities":
            stats["red"]["ko_kd_strike_opportunities"],

        "blue_ko_kd_strike_opportunities":
            stats["blue"]["ko_kd_strike_opportunities"],

        "red_ko_probability_sum":
            stats["red"]["ko_probability_sum"],

        "blue_ko_probability_sum":
            stats["blue"]["ko_probability_sum"],

        "red_kd_probability_sum":
            stats["red"]["kd_probability_sum"],

        "blue_kd_probability_sum":
            stats["blue"]["kd_probability_sum"],

        "red_shadow_ko_events":
            stats["red"]["ko_events"],

        "blue_shadow_ko_events":
            stats["blue"]["ko_events"],

        "red_sub_attempts":
            stats["red"][
                "submission_attempt_attempted"
            ],

        "blue_sub_attempts":
            stats["blue"][
                "submission_attempt_attempted"
            ],

        "red_standing_attempts":
            stats["red"][
                "standing_strike_attempted"
            ],

        "blue_standing_attempts":
            stats["blue"][
                "standing_strike_attempted"
            ],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 150)
    print(
        "EVENT CLOCK MC — FIRST COMPLETE "
        "END-TO-END PREDICTIVE REPLAY"
    )
    print("=" * 150)

    print(
        "Stage-9 scheduled budgets + existing KO mechanics "
        "+ independent SUB clock + Stage-10D judge"
    )

    print(
        "KO/KD calibration: validated empirical default"
    )

    # -------------------------------------------------------------
    # Direct predictions
    # -------------------------------------------------------------

    train, test = (
        prepare_direct_predictions()
    )

    for frame in (
        train,
        test,
    ):
        frame["fight_id"] = (
            frame["fight_id"]
            .astype(str)
        )

    if (
        test["fight_id"].nunique()
        != FIGHTS
    ):
        raise RuntimeError(
            "fresh cohort != 500 fights"
        )

    # -------------------------------------------------------------
    # Submission targets
    # -------------------------------------------------------------

    targets = (
        build_submission_targets()[
            [
                "fight_id",
                "side",
                "submission_attempted",
                "submission_win",
            ]
        ]
        .copy()
    )

    targets["fight_id"] = (
        targets["fight_id"]
        .astype(str)
    )

    train = train.merge(
        targets,
        on=["fight_id", "side"],
        how="inner",
        validate="one_to_one",
    )

    test = test.merge(
        targets,
        on=["fight_id", "side"],
        how="inner",
        validate="one_to_one",
    )

    # -------------------------------------------------------------
    # Preserve historical duration, then project TEST to
    # scheduled horizon.
    # -------------------------------------------------------------

    for frame in (
        train,
        test,
    ):
        frame[
            "historical_duration"
        ] = frame[
            "duration"
        ].astype(float)

    train = add_historical_free_time(
        train
    )

    test = add_historical_free_time(
        test
    )

    scheduled = (
        test["scheduled_rounds"]
        .astype(float)
        * 300.0
    )

    exposure_ratio = (
        scheduled
        / test["historical_duration"]
        .clip(lower=1.0)
    )

    # Direct Stage-2 models are exposure-linear,
    # so this is the exact scheduled-horizon projection
    # for their expected count outputs.
    for family in (
        "distance",
        "clinch",
        "ground",
        "td",
    ):
        for suffix in (
            "attempted",
            "landed",
        ):
            col = (
                f"pred_{family}_{suffix}"
            )

            test[col] = (
                test[col]
                * exposure_ratio
            )

    test[
        "pred_qualified_control_inflicted_seconds"
    ] = (
        test[
            "pred_qualified_control_inflicted_seconds"
        ]
        * exposure_ratio
    )

    test["duration"] = scheduled

    test[
        "pred_qualified_control_inflicted_seconds"
    ] = np.minimum(
        test[
            "pred_qualified_control_inflicted_seconds"
        ],
        test["duration"],
    )

    # Fight-level control cap.
    for fight_id, group in (
        test.groupby("fight_id")
    ):
        idx = group.index

        total = float(
            test.loc[
                idx,
                "pred_qualified_control_inflicted_seconds",
            ].sum()
        )

        duration = float(
            group["duration"].iloc[0]
        )

        if total > duration:
            test.loc[
                idx,
                "pred_qualified_control_inflicted_seconds",
            ] *= (
                duration
                / total
            )

    # -------------------------------------------------------------
    # Standing aggregates
    # -------------------------------------------------------------

    for frame in (
        train,
        test,
    ):
        frame[
            "standing_attempted"
        ] = (
            frame["distance_attempted"]
            + frame["clinch_attempted"]
        )

        frame[
            "standing_landed"
        ] = (
            frame["distance_landed"]
            + frame["clinch_landed"]
        )

        frame[
            "pred_standing_attempted"
        ] = (
            frame["pred_distance_attempted"]
            + frame["pred_clinch_attempted"]
        )

        frame[
            "pred_standing_landed"
        ] = (
            frame["pred_distance_landed"]
            + frame["pred_clinch_landed"]
        )

    feature_cols = (
        direct_feature_columns()
    )

    # -------------------------------------------------------------
    # Stage-9 models
    # -------------------------------------------------------------

    hurdle_alpha = {}

    hurdle_alpha["td"] = (
        fit_count_hurdle(
            train,
            test,
            "td",
            feature_cols,
        )
    )

    fit_count_hurdle(
        train,
        test,
        "ground",
        feature_cols,
    )

    hurdle_alpha["ground"] = (
        fit_ground_alpha_by_shape(
            train
        )
    )

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

    train_pair = build_pair_frame(
        train
    )

    test_pair = build_pair_frame(
        test
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
        str(row["fight_id"]): row
        for _, row
        in test_pair.iterrows()
    }

    # -------------------------------------------------------------
    # Independent submission clock
    # -------------------------------------------------------------

    train[
        "submission_base_rate"
    ] = base_submission_rate(
        train
    )

    test[
        "submission_base_rate"
    ] = base_submission_rate(
        test
    )

    submission_scale = (
        fit_submission_attempt_scale(
            train
        )
    )

    baseline = (
        load_submission_baseline()
    )

    train = train.merge(
        baseline,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )

    test = test.merge(
        baseline,
        on="fight_id",
        how="left",
        validate="many_to_one",
    )

    conversion_offset = (
        fit_conversion_offset(
            train
        )
    )

    test[
        "submission_clock_rate"
    ] = (
        submission_scale
        * test[
            "submission_base_rate"
        ]
    )

    test[
        "submission_conversion_probability"
    ] = logistic(
        logit(
            clip_probability(
                test[
                    "submission_conversion_baseline"
                ]
            )
        )
        + conversion_offset
    )

    print()
    print("=" * 150)
    print("SUBMISSION SETTINGS")
    print("=" * 150)

    print(
        f"attempt-rate scale: "
        f"{submission_scale:.6f}"
    )

    print(
        f"conversion offset: "
        f"{conversion_offset:+.6f}"
    )

    # -------------------------------------------------------------
    # Frozen Stage-10D judge
    # -------------------------------------------------------------

    master_raw = (
        pd.read_parquet(
            MASTER_PATH
        )
        .drop_duplicates(
            "fight_id"
        )
        .copy()
    )

    master_raw["fight_id"] = (
        master_raw["fight_id"]
        .astype(str)
    )

    # _fight() requires normalized event_date.
    master_raw["event_date"] = (
        pd.to_datetime(
            master_raw["date"],
            errors="raise",
        )
        .dt.normalize()
    )

    master_judge = (
        prepare_master(
            master_raw
        )
    )

    train_ids = set(
        train["fight_id"]
    )

    judge_train = master_judge[
        master_judge["fight_id"]
        .isin(train_ids)
        & decision_mask(
            master_judge
        )
        & master_judge[
            "red_win"
        ].notna()
    ].copy()

    judge_features = (
        VARIANTS["FULL_TOTAL"]
    )

    judge_model = fit_model(
        judge_train,
        judge_features,
    )

    print()
    print(
        f"decision judge training fights: "
        f"{len(judge_train)}"
    )

    # -------------------------------------------------------------
    # Physical FSR fight objects
    # -------------------------------------------------------------

    fsr_all = pd.read_parquet(
        FSR_V2_PREFIGHT_SNAPSHOTS_PATH
    )

    fsr_all["fight_id"] = (
        fsr_all["fight_id"]
        .astype(str)
    )

    fsr_all["event_date"] = (
        pd.to_datetime(
            fsr_all["event_date"],
            errors="raise",
        )
        .dt.normalize()
    )

    master_lookup = {
        str(row["fight_id"]): row
        for _, row
        in master_raw.iterrows()
    }

    # -------------------------------------------------------------
    # RUN 500 x 20
    # -------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        f"RUNNING FULL REPLAY — "
        f"{FIGHTS} fights x {PATHS} paths"
    )
    print("=" * 150)

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

        if fight_index % 50 == 0:
            print(
                f"fight {fight_index}/"
                f"{len(groups)}"
            )

        fight_id = str(
            fight_id
        )

        pair_info = (
            pair_lookup[
                fight_id
            ]
        )

        fight = _fight(
            master_lookup[
                fight_id
            ],
            fsr_all,
        )

        sub_rate = {}

        convert = None

        for side in (
            "red",
            "blue",
        ):
            row = pair[
                pair["side"] == side
            ].iloc[0]

            sub_rate[side] = float(
                row[
                    "submission_clock_rate"
                ]
            )

            if convert is None:
                convert = float(
                    row[
                        "submission_conversion_probability"
                    ]
                )

        for path in range(
            PATHS
        ):
            seed = (
                SEED
                + fight_index * 100000
                + path
            )

            rng = np.random.default_rng(
                seed
            )

            budgets = (
                simulate_stage9_path(
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
            )

            result = (
                simulate_integrated_path(
                    fight,
                    budgets,
                    sub_rate,
                    convert,
                    judge_model,
                    judge_features,
                    seed + 50000000,
                )
            )

            result.update(
                {
                    "fight_id":
                        fight_id,

                    "path":
                        path,
                }
            )

            path_rows.append(
                result
            )

    paths = pd.DataFrame(
        path_rows
    )

    # -------------------------------------------------------------
    # Shadow KO/KD transition diagnostics
    # -------------------------------------------------------------

    total_opportunities = float(
        paths[
            "red_ko_kd_strike_opportunities"
        ].sum()
        + paths[
            "blue_ko_kd_strike_opportunities"
        ].sum()
    )

    total_kds = float(
        paths["red_kd"].sum()
        + paths["blue_kd"].sum()
    )

    total_kos = float(
        paths[
            "red_shadow_ko_events"
        ].sum()
        + paths[
            "blue_shadow_ko_events"
        ].sum()
    )

    expected_kds = float(
        paths[
            "red_kd_probability_sum"
        ].sum()
        + paths[
            "blue_kd_probability_sum"
        ].sum()
    )

    expected_kos = float(
        paths[
            "red_ko_probability_sum"
        ].sum()
        + paths[
            "blue_ko_probability_sum"
        ].sum()
    )

    print()
    print("=" * 150)
    print("SHADOW KO/KD TRANSITION RATES")
    print("=" * 150)

    print(
        f"landed strike opportunities: "
        f"{total_opportunities:,.0f}"
    )

    print(
        f"realized KDs:                "
        f"{total_kds:,.0f}"
    )

    print(
        f"realized KO/TKOs:            "
        f"{total_kos:,.0f}"
    )

    print(
        f"realized KD / landed strike: "
        f"{total_kds / total_opportunities:.6%}"
    )

    print(
        f"expected KD / landed strike: "
        f"{expected_kds / total_opportunities:.6%}"
    )

    print(
        f"historical target KD/strike: "
        f"{0.00559446:.6%}"
    )

    print()

    print(
        f"realized KO / landed strike: "
        f"{total_kos / total_opportunities:.6%}"
    )

    print(
        f"expected KO / landed strike: "
        f"{expected_kos / total_opportunities:.6%}"
    )

    print(
        f"recent historical KO/strike: "
        f"{0.00373261:.6%}"
    )

    # -------------------------------------------------------------
    # Aggregate outcome probabilities
    # -------------------------------------------------------------

    rows = []

    for fight_id, group in (
        paths.groupby(
            "fight_id",
            sort=False,
        )
    ):
        row = {
            "fight_id":
                fight_id
        }

        for side in (
            "red",
            "blue",
        ):
            for method in (
                "DEC",
                "KO_TKO",
                "SUB",
            ):
                row[
                    f"p_{side}_{method.lower()}"
                ] = float(
                    (
                        (
                            group["winner"]
                            == side
                        )
                        & (
                            group["method"]
                            == method
                        )
                    ).mean()
                )

        row["p_red_win"] = (
            row["p_red_dec"]
            + row["p_red_ko_tko"]
            + row["p_red_sub"]
        )

        row["p_blue_win"] = (
            1.0
            - row["p_red_win"]
        )

        row["p_dec"] = (
            row["p_red_dec"]
            + row["p_blue_dec"]
        )

        row["p_ko"] = (
            row["p_red_ko_tko"]
            + row["p_blue_ko_tko"]
        )

        row["p_sub"] = (
            row["p_red_sub"]
            + row["p_blue_sub"]
        )

        row["sim_elapsed"] = float(
            group["elapsed"].mean()
        )

        rows.append(
            row
        )

    result = pd.DataFrame(
        rows
    )

    # -------------------------------------------------------------
    # Historical labels
    # -------------------------------------------------------------

    eval_master = master_judge[
        master_judge["fight_id"]
        .isin(
            result["fight_id"]
        )
    ].copy()

    eval_master[
        "actual_method"
    ] = (
        eval_master["method"]
        .map(
            normalize_method
        )
    )

    result = result.merge(
        eval_master[
            [
                "fight_id",
                "red_win",
                "actual_method",
            ]
        ],
        on="fight_id",
        how="left",
        validate="one_to_one",
    )

    # -------------------------------------------------------------
    # Overall moneyline
    # -------------------------------------------------------------

    y = result[
        "red_win"
    ].astype(int)

    p = np.clip(
        result[
            "p_red_win"
        ].to_numpy(float),
        1e-6,
        1.0 - 1e-6,
    )

    pred = (
        p >= 0.5
    ).astype(int)

    print()
    print("=" * 150)
    print("FULL-FIGHT MONEYLINE")
    print("=" * 150)

    print(
        f"winner accuracy: "
        f"{accuracy_score(y, pred):.2%}"
    )

    print(
        f"AUC:             "
        f"{roc_auc_score(y, p):.4f}"
    )

    print(
        f"Brier:           "
        f"{brier_score_loss(y, p):.4f}"
    )

    print(
        f"log loss:        "
        f"{log_loss(y, p):.4f}"
    )

    # -------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------

    hist_dec = (
        result[
            "actual_method"
        ].eq("DEC").mean()
    )

    hist_ko = (
        result[
            "actual_method"
        ].eq("KO_TKO").mean()
    )

    hist_sub = (
        result[
            "actual_method"
        ].eq("SUB").mean()
    )

    print()
    print("=" * 150)
    print("METHOD SHARES")
    print("=" * 150)

    print(
        f"DEC HIST={hist_dec:.2%} | "
        f"SIM={result['p_dec'].mean():.2%}"
    )

    print(
        f"KO  HIST={hist_ko:.2%} | "
        f"SIM={result['p_ko'].mean():.2%}"
    )

    print(
        f"SUB HIST={hist_sub:.2%} | "
        f"SIM={result['p_sub'].mean():.2%}"
    )

    # -------------------------------------------------------------
    # Fighter-level KO and SUB discrimination
    # -------------------------------------------------------------

    fighter_rows = []

    for _, row in result.iterrows():

        for side in (
            "red",
            "blue",
        ):

            actual_side_win = (
                (
                    int(
                        row["red_win"]
                    )
                    == 1
                    and side == "red"
                )
                or
                (
                    int(
                        row["red_win"]
                    )
                    == 0
                    and side == "blue"
                )
            )

            fighter_rows.append(
                {
                    "fight_id":
                        row["fight_id"],

                    "side":
                        side,

                    "actual_ko_win":
                        int(
                            actual_side_win
                            and row[
                                "actual_method"
                            ]
                            == "KO_TKO"
                        ),

                    "actual_sub_win":
                        int(
                            actual_side_win
                            and row[
                                "actual_method"
                            ]
                            == "SUB"
                        ),

                    "p_ko":
                        row[
                            f"p_{side}_ko_tko"
                        ],

                    "p_sub":
                        row[
                            f"p_{side}_sub"
                        ],
                }
            )

    fighters = pd.DataFrame(
        fighter_rows
    )

    print()
    print("=" * 150)
    print("FINISHER DISCRIMINATION")
    print("=" * 150)

    print(
        f"fighter KO-winner AUC:  "
        f"{roc_auc_score(fighters['actual_ko_win'], fighters['p_ko']):.4f}"
    )

    print(
        f"fighter SUB-winner AUC: "
        f"{roc_auc_score(fighters['actual_sub_win'], fighters['p_sub']):.4f}"
    )

    # -------------------------------------------------------------
    # Historical decision subset
    # -------------------------------------------------------------

    decisions = result[
        result[
            "actual_method"
        ].eq("DEC")
    ].copy()

    if len(decisions):
        dec_p = (
            decisions[
                "p_red_dec"
            ]
            / np.maximum(
                decisions[
                    "p_dec"
                ],
                1e-9,
            )
        )

        print()
        print("=" * 150)
        print(
            "HISTORICAL DECISION FIGHTS — "
            "CONDITIONAL DECISION SIDE"
        )
        print("=" * 150)

        print(
            f"N={len(decisions)} | "
            f"accuracy="
            f"{accuracy_score(decisions['red_win'].astype(int), dec_p >= .5):.2%} | "
            f"AUC="
            f"{roc_auc_score(decisions['red_win'].astype(int), dec_p):.4f}"
        )

    # -------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------

    historical_elapsed = (
        test.groupby("fight_id")[
            "historical_duration"
        ]
        .first()
        .mean()
    )

    print()
    print("=" * 150)
    print("FIGHT LENGTH")
    print("=" * 150)

    print(
        f"historical mean elapsed: "
        f"{historical_elapsed:.1f}s"
    )

    print(
        f"simulated mean elapsed:  "
        f"{result['sim_elapsed'].mean():.1f}s"
    )

    # -------------------------------------------------------------
    # Biggest moneyline misses
    # -------------------------------------------------------------

    result[
        "actual_red_probability_target"
    ] = result[
        "red_win"
    ].astype(float)

    result[
        "absolute_probability_error"
    ] = (
        result[
            "p_red_win"
        ]
        - result[
            "actual_red_probability_target"
        ]
    ).abs()

    names = test[
        [
            "fight_id",
            "side",
            "fighter_name",
        ]
    ].pivot(
        index="fight_id",
        columns="side",
        values="fighter_name",
    ).reset_index()

    result = result.merge(
        names,
        on="fight_id",
        how="left",
    )

    print()
    print("=" * 150)
    print("LARGEST MONEYLINE MISSES")
    print("=" * 150)

    print(
        result[
            [
                "fight_id",
                "red",
                "blue",
                "red_win",
                "actual_method",
                "p_red_win",
                "p_red_dec",
                "p_red_ko_tko",
                "p_red_sub",
                "p_blue_dec",
                "p_blue_ko_tko",
                "p_blue_sub",
            ]
        ]
        .assign(
            error=result[
                "absolute_probability_error"
            ]
        )
        .sort_values(
            "error",
            ascending=False,
        )
        .head(25)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 150)
    print("INTERPRETATION RULE")
    print("=" * 150)

    print(
        "Do not tune anything from this run yet. "
        "First identify whether the dominant error is "
        "KO allocation, submission allocation, decision allocation, "
        "or overall method calibration."
    )


if __name__ == "__main__":
    main()
