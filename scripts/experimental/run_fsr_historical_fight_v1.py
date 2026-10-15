"""Generic historical FSR V1.1 -> MC V2 fight runner.

Usage
-----
PYTHONPATH=. python \
    scripts/experimental/run_fsr_historical_fight_v1.py \
    <fight_id>

Example
-------
PYTHONPATH=. python \
    scripts/experimental/run_fsr_historical_fight_v1.py \
    9e05fe95a4fc63bc

Workflow
--------
1. Resolve target fight metadata from UFCStats round data.
2. Run the frozen 9-skill FSR V0 historical replay.
3. Run the frozen FSR V1.1 power/durability replay.
4. Read each fighter's final PRE-target state from the generated histories.
5. Compute leakage-safe population baselines using only earlier fights.
6. Build matchup-specific FSR -> MC V2 parameters.
7. Run MC V2 using the target fight's actual scheduled-round contract.

Shadow/research only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from math import exp, log
from pathlib import Path

import pandas as pd

from collections import Counter, defaultdict
import inspect

from pipeline.simulation.rfs_mc_v2_shared_state.contracts import (
    FighterSide,
)
from pipeline.simulation.rfs_mc_v2_shared_state.final_fight_result import (
    resolve_final_fight_result,
)
from pipeline.simulation.rfs_mc_v2_shared_state.finish_path_runner import (
    run_finish_enabled_dynamic_path,
)

from pipeline.simulation.rfs_mc_v2_shared_state.dynamic_parameters import (
    FighterDynamicParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.monte_carlo_runner import (
    run_matchup_monte_carlo,
)
from pipeline.simulation.rfs_mc_v2_shared_state.phase_parameters import (
    ClinchRateParameters,
    DistanceRateParameters,
    FighterPhaseParameters,
    GroundDefenderRateParameters,
    GroundOwnerRateParameters,
)
from pipeline.simulation.rfs_mc_v2_shared_state.transition_parameters import (
    FighterTransitionParameters,
)

from scripts.calibrate_rfs_mc_v2_dynamic_workload_v3 import (
    Candidate,
    V1_KNOCKDOWN_BONUS_HAZARD,
    V1_LANDED_KO_HAZARD,
    finish_calibration,
    phase_effect_calibration,
    state_calibration,
    zero_transition_effect_calibration,
)


# =============================================================================
# Paths
# =============================================================================

ROUND_PATH = Path(
    "data/fight_details/ufc_round_stats.parquet"
)

OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state"
)

FSR_V0_SCRIPT = Path(
    "scripts/experimental/fsr_historical_fight_v0.py"
)

POWER_SCRIPT = Path(
    "scripts/experimental/fsr_power_durability_v1_2.py"
)

CARDIO_SCRIPT = Path(
    "scripts/experimental/fsr_cardio_v1.py"
)


# =============================================================================
# Adapter controls -- frozen from current FSR experiment
# =============================================================================

RATING_SCALE = 12.0

RATE_EFFECT_STRENGTH = 0.35
PROBABILITY_EFFECT_STRENGTH = 0.35

SIMULATIONS_DEFAULT = 1000
SEED_DEFAULT = 2026080700


# =============================================================================
# Helpers
# =============================================================================

def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(
        low,
        min(high, value),
    )


def sigmoid(value: float) -> float:
    return 1.0 / (
        1.0 + exp(-value)
    )


def logit(probability: float) -> float:
    probability = clamp(
        probability,
        1e-6,
        1.0 - 1e-6,
    )

    return log(
        probability
        / (1.0 - probability)
    )


def normalized_skill(
    rating: float,
) -> float:
    """Convert 50-centered FSR rating to [0,1]."""

    return sigmoid(
        (rating - 50.0)
        / RATING_SCALE
    )


def intrinsic_power_multiplier(
    rating: float,
) -> float:
    """Persistent power multiplier used by the KO conversion pathway.

    50 -> 1.00x
    55 -> ~1.15x
    60 -> ~1.30x

    Current experimental bounds remain deliberately conservative.
    """

    return clamp(
        1.0
        + 0.03
        * (rating - 50.0),
        0.50,
        1.50,
    )


def intrinsic_ko_vulnerability_multiplier(
    rating: float,
) -> float:
    """Convert FSR chin resistance into acute KO vulnerability.

    Higher chin resistance means lower vulnerability.

    50 -> 1.00x
    55 -> 0.85x
    60 -> 0.70x
    45 -> 1.15x
    40 -> 1.30x
    """

    return clamp(
        1.0
        - 0.03
        * (rating - 50.0),
        0.50,
        1.75,
    )


def matchup_probability(
    *,
    baseline: float,
    offense_rating: float,
    defense_rating: float,
) -> float:
    """Opponent-adjust a population probability."""

    rating_difference = (
        offense_rating
        - defense_rating
    )

    adjusted_logit = (
        logit(baseline)
        + PROBABILITY_EFFECT_STRENGTH
        * rating_difference
        / RATING_SCALE
    )

    return sigmoid(
        adjusted_logit
    )


def matchup_rate(
    *,
    baseline: float,
    offense_rating: float,
    defense_rating: float,
) -> float:
    """Opponent-adjust a nonnegative activity rate."""

    rating_difference = (
        offense_rating
        - defense_rating
    )

    multiplier = exp(
        RATE_EFFECT_STRENGTH
        * rating_difference
        / RATING_SCALE
    )

    return max(
        0.0,
        baseline * multiplier,
    )


# =============================================================================
# Target-fight resolution
# =============================================================================

def load_target_fight(
    fight_id: str,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp,
    dict,
    dict,
    int,
]:
    """Resolve target fight and authoritative RED/BLUE identities."""

    rounds = pd.read_parquet(
        ROUND_PATH
    )

    rounds["event_date"] = pd.to_datetime(
        rounds["event_date"]
    )

    rounds["fight_id"] = (
        rounds["fight_id"]
        .astype(str)
    )

    target = rounds.loc[
        rounds["fight_id"] == fight_id
    ].copy()

    if target.empty:
        raise RuntimeError(
            f"Fight not found: {fight_id}"
        )

    target_date = pd.Timestamp(
        target["event_date"].iloc[0]
    )

    scheduled_rounds = int(
        float(
            target["total_rounds"].iloc[0]
        )
    )

    if scheduled_rounds not in {
        3,
        5,
    }:
        raise RuntimeError(
            "Target fight scheduled rounds "
            f"are invalid: {scheduled_rounds}"
        )

    # Prefer UFCStats corner labels.
    if "corner" not in target.columns:
        raise RuntimeError(
            "round stats are missing the corner column"
        )

    fighters = (
        target[
            [
                "corner",
                "fighter_id",
                "fighter_name",
            ]
        ]
        .drop_duplicates()
    )

    red_rows = fighters.loc[
        fighters["corner"]
        .astype(str)
        .str.upper()
        == "RED"
    ]

    blue_rows = fighters.loc[
        fighters["corner"]
        .astype(str)
        .str.upper()
        == "BLUE"
    ]

    if (
        len(red_rows) != 1
        or len(blue_rows) != 1
    ):
        raise RuntimeError(
            "Could not resolve exactly one RED "
            "and one BLUE fighter."
        )

    red_row = red_rows.iloc[0]
    blue_row = blue_rows.iloc[0]

    red = {
        "fighter_id": str(
            red_row["fighter_id"]
        ),
        "fighter_name": str(
            red_row["fighter_name"]
        ),
    }

    blue = {
        "fighter_id": str(
            blue_row["fighter_id"]
        ),
        "fighter_name": str(
            blue_row["fighter_name"]
        ),
    }

    return (
        rounds,
        target_date,
        red,
        blue,
        scheduled_rounds,
    )


# =============================================================================
# Historical replay
# =============================================================================

def run_rating_builders(
    fight_id: str,
) -> None:
    """Generate both frozen rating-history artifacts."""

    commands = [
        [
            sys.executable,
            str(FSR_V0_SCRIPT),
            fight_id,
        ],
        [
            sys.executable,
            str(POWER_SCRIPT),
            fight_id,
        ],
        [
            sys.executable,
            str(CARDIO_SCRIPT),
            fight_id,
        ],
    ]

    for command in commands:
        subprocess.run(
            command,
            check=True,
        )


# =============================================================================
# Card extraction
# =============================================================================

NINE_SKILLS = [
    "distance_volume",
    "distance_accuracy",
    "distance_defense",
    "td_initiative",
    "td_completion",
    "td_defense",
    "control_imposition",
    "control_resistance",
    "submission_pressure",
]


def load_v0_card(
    fight_id: str,
    fighter_id: str,
) -> tuple[
    dict[str, float],
    int,
]:
    """Read the fighter's latest POST rating before target fight."""

    path = (
        OUTPUT_DIR
        / f"fsr_{fight_id}_v0_rating_history.csv"
    )

    df = pd.read_csv(path)

    df["fighter_id"] = (
        df["fighter_id"]
        .astype(str)
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    rows = df.loc[
        (
            df["fighter_id"]
            == fighter_id
        )
        & (
            df["stage"]
            .astype(str)
            .str.upper()
            == "POST"
        )
    ].copy()

    if rows.empty:
        # Experimental fallback for fighter with no prior UFC history.
        return (
            {
                skill: 50.0
                for skill in NINE_SKILLS
            },
            0,
        )

    row = (
        rows
        .sort_values(
            [
                "date",
                "fight_count",
            ]
        )
        .iloc[-1]
    )

    card = {
        skill: float(
            row[skill]
        )
        for skill in NINE_SKILLS
    }

    return (
        card,
        int(row["fight_count"]),
    )


def load_power_card(
    fight_id: str,
    fighter_id: str,
) -> dict[str, float]:
    """Read authoritative V1.2 PRE-target power/chin/absorption card."""

    path = (
        OUTPUT_DIR
        / (
            f"fsr_{fight_id}"
            "_power_chin_absorption_v1_2_target_card.csv"
        )
    )

    df = pd.read_csv(path)

    df["fighter_id"] = (
        df["fighter_id"]
        .astype(str)
    )

    rows = df.loc[
        df["fighter_id"] == fighter_id
    ].copy()

    if rows.empty:
        return {
            "finishing_power": 50.0,
            "chin_resistance": 50.0,
            "damage_absorption": 50.0,
        }

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one V1.2 target-card row for "
            f"fighter {fighter_id}; found {len(rows)}"
        )

    row = rows.iloc[0]

    return {
        "finishing_power": float(
            row["finishing_power"]
        ),
        "chin_resistance": float(
            row["chin_resistance"]
        ),
        "damage_absorption": float(
            row["damage_absorption"]
        ),
    }


def load_cardio_card(
    fight_id: str,
    fighter_id: str,
) -> dict[str, float]:
    """Read authoritative leakage-safe FSR Cardio V1 target card."""

    path = (
        OUTPUT_DIR
        / f"fsr_{fight_id}_cardio_v1_target_card.csv"
    )

    df = pd.read_csv(path)

    df["fighter_id"] = (
        df["fighter_id"]
        .astype(str)
    )

    rows = df.loc[
        df["fighter_id"] == fighter_id
    ].copy()

    if rows.empty:
        # Neutral fallback preserves the existing benchmark behavior.
        return {
            "fatigue_accumulation_resistance_engine": 0.50,
            "fatigue_performance_resilience_engine": 0.50,
            "recovery_ability_engine": 0.50,
            "fatigue_accumulation_resistance_rating": 50.0,
            "fatigue_performance_resilience_rating": 50.0,
            "recovery_ability_rating": 50.0,
        }

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one FSR Cardio V1 target-card row for "
            f"fighter {fighter_id}; found {len(rows)}"
        )

    row = rows.iloc[0]

    return {
        "fatigue_accumulation_resistance_engine": float(
            row["fatigue_accumulation_resistance_engine"]
        ),
        "fatigue_performance_resilience_engine": float(
            row["fatigue_performance_resilience_engine"]
        ),
        "recovery_ability_engine": float(
            row["recovery_ability_engine"]
        ),
        "fatigue_accumulation_resistance_rating": float(
            row["fatigue_accumulation_resistance_rating"]
        ),
        "fatigue_performance_resilience_rating": float(
            row["fatigue_performance_resilience_rating"]
        ),
        "recovery_ability_rating": float(
            row["recovery_ability_rating"]
        ),
    }


def build_full_card(
    fight_id: str,
    fighter_id: str,
) -> tuple[
    dict[str, float],
    int,
]:
    """Combine 9-skill FSR V0 with V1.2 power/chin/absorption."""

    card, fight_count = (
        load_v0_card(
            fight_id,
            fighter_id,
        )
    )

    card.update(
        load_power_card(
            fight_id,
            fighter_id,
        )
    )

    card.update(
        load_cardio_card(
            fight_id,
            fighter_id,
        )
    )

    return (
        card,
        fight_count,
    )


# =============================================================================
# Leakage-safe population baselines
# =============================================================================

def population_baselines(
    rounds: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict[str, float]:
    """Compute population evidence using only fights before target date."""

    history = rounds.loc[
        rounds["event_date"]
        < target_date
    ].copy()

    if history.empty:
        raise RuntimeError(
            "No historical rows before target fight."
        )

    fighter_rounds = float(
        len(history)
    )

    distance_attempts = float(
        history[
            "distance_str_attempted"
            if "distance_str_attempted" in history.columns
            else "distance_attempted"
        ].sum()
    ) if (
        "distance_str_attempted" in history.columns
        or "distance_attempted" in history.columns
    ) else float(
        history[
            "distance_attempted"
        ].sum()
    )

    # Repository schema currently uses distance_attempted /
    # distance_landed in the round-level table.
    if "distance_attempted" in history.columns:
        distance_attempt_col = (
            "distance_attempted"
        )
        distance_landed_col = (
            "distance_landed"
        )
    else:
        # Support alternate abbreviated schema if encountered.
        distance_attempt_col = (
            "distance_str_attempted"
        )
        distance_landed_col = (
            "distance_str_landed"
        )

    total_distance_attempts = float(
        history[
            distance_attempt_col
        ].sum()
    )

    total_distance_landed = float(
        history[
            distance_landed_col
        ].sum()
    )

    total_td_attempted = float(
        history["td_attempted"].sum()
    )

    total_td_landed = float(
        history["td_landed"].sum()
    )

    total_sub_attempts = float(
        history["sub_att"].sum()
    )

    return {
        "distance_attempts_per_round": (
            total_distance_attempts
            / fighter_rounds
        ),
        "distance_accuracy": (
            total_distance_landed
            / max(
                total_distance_attempts,
                1.0,
            )
        ),
        "td_attempts_per_round": (
            total_td_attempted
            / fighter_rounds
        ),
        "td_completion": (
            total_td_landed
            / max(
                total_td_attempted,
                1.0,
            )
        ),
        "sub_attempts_per_round": (
            total_sub_attempts
            / fighter_rounds
        ),
    }


# =============================================================================
# MC V2 adapter
# =============================================================================

def build_transition(
    fighter: dict[str, float],
) -> FighterTransitionParameters:
    """Translate persistent FSR skills into intrinsic transition strengths."""

    td_entry = normalized_skill(
        fighter["td_initiative"]
    )

    td_completion = normalized_skill(
        fighter["td_completion"]
    )

    td_defense = normalized_skill(
        fighter["td_defense"]
    )

    control = normalized_skill(
        fighter["control_imposition"]
    )

    control_resistance = normalized_skill(
        fighter["control_resistance"]
    )

    distance = normalized_skill(
        fighter["distance_volume"]
    )

    phase_imposition = (
        distance
        + td_entry
        + control
    ) / 3.0

    phase_resistance = (
        td_defense
        + control_resistance
    ) / 2.0

    return FighterTransitionParameters(
        distance_retention=distance,

        clinch_entry_tendency=(
            0.50 * td_entry
            + 0.50 * control
        ),

        clinch_entry_resistance=(
            control_resistance
        ),

        takedown_entry_tendency=td_entry,
        takedown_completion_ability=(
            td_completion
        ),
        takedown_resistance=td_defense,

        takedown_persistence=(
            0.50 * td_entry
            + 0.50 * control
        ),

        failed_takedown_persistence=(
            td_entry
        ),

        clinch_retention=control,
        clinch_escape_ability=(
            control_resistance
        ),

        ground_retention=control,
        ground_escape_ability=(
            control_resistance
        ),

        reversal_ability=0.50,

        phase_imposition=phase_imposition,
        phase_resistance=phase_resistance,
    )


def build_phase(
    fighter: dict[str, float],
    opponent: dict[str, float],
    baselines: dict[str, float],
) -> FighterPhaseParameters:
    """Build matchup-specific physical activity parameters."""

    distance_attempt_rate = (
        matchup_rate(
            baseline=(
                baselines[
                    "distance_attempts_per_round"
                ]
                / 10.0
            ),
            offense_rating=(
                fighter[
                    "distance_volume"
                ]
            ),
            defense_rating=(
                opponent[
                    "distance_defense"
                ]
            ),
        )
    )

    distance_accuracy = (
        matchup_probability(
            baseline=(
                baselines[
                    "distance_accuracy"
                ]
            ),
            offense_rating=(
                fighter[
                    "distance_accuracy"
                ]
            ),
            defense_rating=(
                opponent[
                    "distance_defense"
                ]
            ),
        )
    )

    knockdown_probability = (
        matchup_probability(
            baseline=0.012,
            offense_rating=(
                fighter[
                    "finishing_power"
                ]
            ),
            # V1.2: knockdown generation is driven by offensive
            # finishing power. Defender chin is reserved for subsequent
            # KO/TKO conversion, while damage_absorption is cumulative-
            # damage tolerance and should not affect acute knockdowns.
            defense_rating=50.0,
        )
    )

    control_strength = (
        matchup_probability(
            baseline=0.50,
            offense_rating=(
                fighter[
                    "control_imposition"
                ]
            ),
            defense_rating=(
                opponent[
                    "control_resistance"
                ]
            ),
        )
    )

    clinch_control_seconds = (
        3.0
        + 6.0
        * control_strength
    )

    ground_control_seconds = (
        4.0
        + 8.0
        * control_strength
    )

    submission_attempt_rate = (
        matchup_rate(
            baseline=(
                baselines[
                    "sub_attempts_per_round"
                ]
                / 10.0
            ),
            offense_rating=(
                fighter[
                    "submission_pressure"
                ]
            ),
            defense_rating=(
                opponent[
                    "control_resistance"
                ]
            ),
        )
    )

    return FighterPhaseParameters(
        distance=DistanceRateParameters(
            sig_strike_attempt_rate=(
                distance_attempt_rate
            ),
            sig_strike_accuracy=(
                distance_accuracy
            ),
            knockdown_probability_per_landed=(
                knockdown_probability
            ),
        ),

        clinch=ClinchRateParameters(
            clinch_strike_attempt_rate=0.30,
            clinch_strike_accuracy=0.45,
            control_seconds_mean=(
                clinch_control_seconds
            ),
            damaging_clinch_probability=0.10,
        ),

        ground_owner=GroundOwnerRateParameters(
            ground_strike_attempt_rate=0.35,
            ground_strike_accuracy=0.50,
            control_seconds_mean=(
                ground_control_seconds
            ),
            submission_attempt_rate=(
                submission_attempt_rate
            ),
            position_advancement_probability=0.20,
        ),

        ground_defender=GroundDefenderRateParameters(
            escape_attempt_rate=(
                normalized_skill(
                    fighter[
                        "control_resistance"
                    ]
                )
            ),
            reversal_attempt_rate=0.10,
            scramble_attempt_rate=0.25,
            submission_defense=(
                normalized_skill(
                    fighter[
                        "control_resistance"
                    ]
                )
            ),
        ),
    )


def build_dynamic(
    fighter: dict[str, float],
) -> FighterDynamicParameters:
    """Use FSR cardio plus V1.2 damage absorption dynamic traits."""

    return FighterDynamicParameters(
        fatigue_accumulation_resistance=(
            fighter[
                "fatigue_accumulation_resistance_engine"
            ]
        ),
        fatigue_performance_resilience=(
            fighter[
                "fatigue_performance_resilience_engine"
            ]
        ),
        recovery_ability=(
            fighter[
                "recovery_ability_engine"
            ]
        ),

        damage_resistance=(
            normalized_skill(
                fighter[
                    "damage_absorption"
                ]
            )
        ),

        acute_stress_resistance=0.50,
        acute_stress_recovery=0.50,
    )


# =============================================================================
# Display
# =============================================================================

def print_card(
    name: str,
    fight_count: int,
    card: dict[str, float],
) -> None:
    print()
    print(
        f"{name} "
        f"({fight_count} prior UFC fights)"
    )
    print("-" * 70)

    for key in (
        NINE_SKILLS
        + [
            "finishing_power",
            "chin_resistance",
            "damage_absorption",
            "fatigue_accumulation_resistance_rating",
            "fatigue_performance_resilience_rating",
            "recovery_ability_rating",
        ]
    ):
        print(
            f"{key:<28} "
            f"{card[key]:8.2f}"
        )


def print_matchup_parameters(
    name: str,
    card: dict[str, float],
    transition: FighterTransitionParameters,
    phase: FighterPhaseParameters,
    dynamic: FighterDynamicParameters,
) -> None:
    print()
    print(name)
    print("-" * 70)

    print(
        "distance attempt rate :",
        round(
            phase.distance.sig_strike_attempt_rate,
            4,
        ),
    )

    print(
        "distance accuracy     :",
        round(
            phase.distance.sig_strike_accuracy,
            4,
        ),
    )

    print(
        "TD entry tendency     :",
        round(
            transition.takedown_entry_tendency,
            4,
        ),
    )

    print(
        "TD completion ability :",
        round(
            transition.takedown_completion_ability,
            4,
        ),
    )

    print(
        "TD resistance         :",
        round(
            transition.takedown_resistance,
            4,
        ),
    )

    print(
        "control imposition    :",
        round(
            transition.phase_imposition,
            4,
        ),
    )

    print(
        "finishing power       :",
        round(
            normalized_skill(
                card[
                    "finishing_power"
                ]
            ),
            4,
        ),
    )

    print(
        "damage resistance     :",
        round(
            dynamic.damage_resistance,
            4,
        ),
    )

    print(
        "KD probability/landed :",
        round(
            phase.distance
            .knockdown_probability_per_landed,
            5,
        ),
    )

    print(
        "intrinsic KO power    :",
        round(
            intrinsic_power_multiplier(
                card[
                    "finishing_power"
                ]
            ),
            4,
        ),
    )



# =============================================================================
# Population diagnostics
# =============================================================================

def run_population_diagnostics(
    *,
    fight_id: str,
    red_name: str,
    blue_name: str,
    red_transition,
    blue_transition,
    red_phase,
    blue_phase,
    red_dynamic,
    blue_dynamic,
    red_card: dict[str, float],
    blue_card: dict[str, float],
    scheduled_rounds: int,
    simulation_count: int,
    seed_start: int,
) -> None:
    """Replay identical seeds and retain detailed MC diagnostics.

    This is a diagnostic-only second pass.

    The normal run_matchup_monte_carlo() result remains the authoritative
    population result. This replay uses the same parameters, calibrations,
    simulation count, and seeds so that detailed path information can be
    inspected without modifying the MC engine.
    """

    candidate = Candidate(
        landed_ko_hazard=V1_LANDED_KO_HAZARD,
        knockdown_bonus_hazard=V1_KNOCKDOWN_BONUS_HAZARD,
    )

    dynamic_cal = state_calibration(candidate)
    phase_cal = phase_effect_calibration(candidate)
    transition_cal = zero_transition_effect_calibration()
    finish_cal = finish_calibration(candidate)

    # -------------------------------------------------------------------------
    # Population containers
    # -------------------------------------------------------------------------

    round_totals = {
        r: defaultdict(float)
        for r in range(1, scheduled_rounds + 1)
    }

    round_reached = Counter()
    round_completed = Counter()

    finish_rounds = Counter()
    decision_types = Counter()

    judge_round_votes = {
        r: Counter()
        for r in range(1, scheduled_rounds + 1)
    }

    judge_score_pairs = {
        r: Counter()
        for r in range(1, scheduled_rounds + 1)
    }

    judge_fight_votes = Counter()
    judge_card_totals = Counter()

    scorecard_rows = []
    path_rows = []

    example_scorecards = None
    example_seed = None
    example_decision = None

    # -------------------------------------------------------------------------
    # Support our locally-added intrinsic power pathway while remaining
    # compatible with an older engine checkout.
    # -------------------------------------------------------------------------

    path_signature = inspect.signature(
        run_finish_enabled_dynamic_path
    )

    supports_intrinsic_power = (
        "red_intrinsic_power_multiplier" in path_signature.parameters
        and
        "blue_intrinsic_power_multiplier" in path_signature.parameters
    )

    supports_intrinsic_chin = (
        "red_intrinsic_ko_vulnerability_multiplier"
        in path_signature.parameters
        and
        "blue_intrinsic_ko_vulnerability_multiplier"
        in path_signature.parameters
    )

    red_intrinsic = intrinsic_power_multiplier(
        red_card["finishing_power"]
    )

    blue_intrinsic = intrinsic_power_multiplier(
        blue_card["finishing_power"]
    )

    red_ko_vulnerability = (
        intrinsic_ko_vulnerability_multiplier(
            red_card["chin_resistance"]
        )
    )

    blue_ko_vulnerability = (
        intrinsic_ko_vulnerability_multiplier(
            blue_card["chin_resistance"]
        )
    )

    # -------------------------------------------------------------------------
    # Replay every simulation
    # -------------------------------------------------------------------------

    for simulation_index in range(simulation_count):

        seed = seed_start + simulation_index

        kwargs = dict(
            dynamic_state_calibration=dynamic_cal,
            phase_effect_calibration=phase_cal,
            transition_effect_calibration=transition_cal,
            finish_probability_calibration=finish_cal,
            scheduled_rounds=scheduled_rounds,
            seed=seed,
        )

        if supports_intrinsic_power:
            kwargs.update(
                red_intrinsic_power_multiplier=red_intrinsic,
                blue_intrinsic_power_multiplier=blue_intrinsic,
            )

        if supports_intrinsic_chin:
            kwargs.update(
                red_intrinsic_ko_vulnerability_multiplier=(
                    red_ko_vulnerability
                ),
                blue_intrinsic_ko_vulnerability_multiplier=(
                    blue_ko_vulnerability
                ),
            )

        path_result = run_finish_enabled_dynamic_path(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            red_dynamic,
            blue_dynamic,
            **kwargs,
        )

        segments_by_round = defaultdict(list)

        for segment in path_result.segments:
            segments_by_round[
                segment.state.round_number
            ].append(segment)

        # ---------------------------------------------------------------------
        # Round mechanics
        # ---------------------------------------------------------------------

        for round_number in range(
            1,
            scheduled_rounds + 1,
        ):
            segments = segments_by_round.get(
                round_number,
                [],
            )

            if not segments:
                continue

            round_reached[round_number] += 1
            totals = round_totals[round_number]

            for segment in segments:

                state = segment.state
                activity = segment.activity

                red_activity = activity.red
                blue_activity = activity.blue

                # -------------------------------------------------------------
                # Phase occupancy
                # -------------------------------------------------------------

                phase = state.phase.value

                totals[
                    f"{phase}_seconds"
                ] += 30.0

                # -------------------------------------------------------------
                # Distance striking
                # -------------------------------------------------------------

                totals["red_distance_attempted"] += getattr(
                    red_activity,
                    "sig_str_attempted",
                    0,
                )

                totals["blue_distance_attempted"] += getattr(
                    blue_activity,
                    "sig_str_attempted",
                    0,
                )

                totals["red_distance_landed"] += getattr(
                    red_activity,
                    "sig_str_landed",
                    0,
                )

                totals["blue_distance_landed"] += getattr(
                    blue_activity,
                    "sig_str_landed",
                    0,
                )

                totals["red_knockdowns"] += getattr(
                    red_activity,
                    "knockdowns",
                    0,
                )

                totals["blue_knockdowns"] += getattr(
                    blue_activity,
                    "knockdowns",
                    0,
                )

                # -------------------------------------------------------------
                # Clinch striking
                # -------------------------------------------------------------

                totals["red_clinch_attempted"] += getattr(
                    red_activity,
                    "clinch_str_attempted",
                    0,
                )

                totals["blue_clinch_attempted"] += getattr(
                    blue_activity,
                    "clinch_str_attempted",
                    0,
                )

                totals["red_clinch_landed"] += getattr(
                    red_activity,
                    "clinch_str_landed",
                    0,
                )

                totals["blue_clinch_landed"] += getattr(
                    blue_activity,
                    "clinch_str_landed",
                    0,
                )

                # -------------------------------------------------------------
                # Ground striking
                # -------------------------------------------------------------

                totals["red_ground_attempted"] += getattr(
                    red_activity,
                    "ground_str_attempted",
                    0,
                )

                totals["blue_ground_attempted"] += getattr(
                    blue_activity,
                    "ground_str_attempted",
                    0,
                )

                totals["red_ground_landed"] += getattr(
                    red_activity,
                    "ground_str_landed",
                    0,
                )

                totals["blue_ground_landed"] += getattr(
                    blue_activity,
                    "ground_str_landed",
                    0,
                )

                # -------------------------------------------------------------
                # Control
                # -------------------------------------------------------------

                totals["red_control_seconds"] += getattr(
                    red_activity,
                    "control_seconds",
                    0,
                )

                totals["blue_control_seconds"] += getattr(
                    blue_activity,
                    "control_seconds",
                    0,
                )

                # -------------------------------------------------------------
                # Submission / position activity
                # -------------------------------------------------------------

                totals["red_submission_attempts"] += getattr(
                    red_activity,
                    "submission_attempts",
                    0,
                )

                totals["blue_submission_attempts"] += getattr(
                    blue_activity,
                    "submission_attempts",
                    0,
                )

                totals["red_position_advancements"] += getattr(
                    red_activity,
                    "position_advancements",
                    0,
                )

                totals["blue_position_advancements"] += getattr(
                    blue_activity,
                    "position_advancements",
                    0,
                )

                # -------------------------------------------------------------
                # Successful takedown transitions
                #
                # NOTE:
                # Current V2 path contracts expose successful TAKEDOWN
                # transitions, not a clean failed-TD-attempt count.
                # -------------------------------------------------------------

                transition = segment.transition

                if transition is not None:

                    event = transition.event.value

                    totals[
                        f"transition_{event}"
                    ] += 1

                    if event == "takedown":

                        if transition.actor is FighterSide.RED:
                            totals[
                                "red_takedown_success"
                            ] += 1

                        elif transition.actor is FighterSide.BLUE:
                            totals[
                                "blue_takedown_success"
                            ] += 1

            # -----------------------------------------------------------------
            # End-of-round state.
            #
            # Use dynamic_state_after_activity because this is BEFORE
            # round-break recovery.
            # -----------------------------------------------------------------

            last_segment = segments[-1]

            if last_segment.state.segment_number == 10:

                round_completed[
                    round_number
                ] += 1

                state = (
                    last_segment
                    .dynamic_state_after_activity
                )

                totals["red_fatigue"] += state.red.fatigue
                totals["blue_fatigue"] += state.blue.fatigue

                totals["red_damage"] += state.red.damage
                totals["blue_damage"] += state.blue.damage

                totals["red_stress"] += state.red.acute_stress
                totals["blue_stress"] += state.blue.acute_stress

        # ---------------------------------------------------------------------
        # Resolve final result / judges
        # ---------------------------------------------------------------------

        result = resolve_final_fight_result(
            path_result
        )

        if result.winner is FighterSide.RED:
            winner = red_name
        elif result.winner is FighterSide.BLUE:
            winner = blue_name
        else:
            winner = "DRAW"

        if result.finish is not None:

            method = result.finish.method.value

            finish_rounds[
                result.finish.round_number
            ] += 1

            decision_type = None

        else:

            method = result.decision_type.value
            decision_type = result.decision_type.value

            decision_types[
                decision_type
            ] += 1

        path_rows.append(
            {
                "fight_id": fight_id,
                "seed": seed,
                "winner": winner,
                "branch": result.branch.value,
                "method": method,
                "finish_round": result.finish_round,
                "finish_segment": result.finish_segment,
                "finish_seconds_in_round": (
                    result.elapsed_seconds_in_round
                ),
                "decision_type": decision_type,
            }
        )

        # ---------------------------------------------------------------------
        # Judge scorecards exist only for scheduled-distance simulations.
        # ---------------------------------------------------------------------

        scorecards = result.scorecards

        if scorecards is None:
            continue

        if example_scorecards is None:
            example_scorecards = scorecards
            example_seed = seed
            example_decision = (
                result.decision_type.value
            )

        for card in scorecards:

            if card.winner is FighterSide.RED:
                card_winner = red_name
            elif card.winner is FighterSide.BLUE:
                card_winner = blue_name
            else:
                card_winner = "DRAW"

            judge_fight_votes[
                card_winner
            ] += 1

            judge_card_totals[
                (
                    card.red_total,
                    card.blue_total,
                )
            ] += 1

            round_text = []

            for score in card.rounds:

                pair = (
                    score.red_points,
                    score.blue_points,
                )

                judge_score_pairs[
                    score.round_number
                ][pair] += 1

                if score.winner is FighterSide.RED:
                    vote = "RED"
                elif score.winner is FighterSide.BLUE:
                    vote = "BLUE"
                else:
                    vote = "EVEN"

                judge_round_votes[
                    score.round_number
                ][vote] += 1

                round_text.append(
                    (
                        f"R{score.round_number}:"
                        f"{score.red_points}-"
                        f"{score.blue_points}"
                    )
                )

            scorecard_rows.append(
                {
                    "fight_id": fight_id,
                    "seed": seed,
                    "judge_number": card.judge_number,
                    "decision_type": result.decision_type.value,
                    "red_total": card.red_total,
                    "blue_total": card.blue_total,
                    "winner": card_winner,
                    "round_scores": " | ".join(
                        round_text
                    ),
                }
            )

    # =========================================================================
    # Build round dataframe
    # =========================================================================

    round_rows = []

    for round_number in range(
        1,
        scheduled_rounds + 1,
    ):

        reached = round_reached[
            round_number
        ]

        if reached == 0:
            continue

        completed = round_completed[
            round_number
        ]

        totals = round_totals[
            round_number
        ]

        def per_reached(key):
            return totals[key] / reached

        def per_completed(key):
            if completed == 0:
                return float("nan")
            return totals[key] / completed

        votes = judge_round_votes[
            round_number
        ]

        total_judge_votes = sum(
            votes.values()
        )

        def vote_pct(key):
            if total_judge_votes == 0:
                return float("nan")

            return (
                100.0
                * votes[key]
                / total_judge_votes
            )

        round_rows.append(
            {
                "fight_id": fight_id,
                "round": round_number,

                "paths_reaching_round": reached,

                "population_reaching_pct": (
                    100.0
                    * reached
                    / simulation_count
                ),

                "red_distance_attempted": per_reached(
                    "red_distance_attempted"
                ),

                "blue_distance_attempted": per_reached(
                    "blue_distance_attempted"
                ),

                "red_distance_landed": per_reached(
                    "red_distance_landed"
                ),

                "blue_distance_landed": per_reached(
                    "blue_distance_landed"
                ),

                "red_clinch_landed": per_reached(
                    "red_clinch_landed"
                ),

                "blue_clinch_landed": per_reached(
                    "blue_clinch_landed"
                ),

                "red_ground_landed": per_reached(
                    "red_ground_landed"
                ),

                "blue_ground_landed": per_reached(
                    "blue_ground_landed"
                ),

                "red_takedown_success": per_reached(
                    "red_takedown_success"
                ),

                "blue_takedown_success": per_reached(
                    "blue_takedown_success"
                ),

                "red_control_seconds": per_reached(
                    "red_control_seconds"
                ),

                "blue_control_seconds": per_reached(
                    "blue_control_seconds"
                ),

                "red_knockdowns": per_reached(
                    "red_knockdowns"
                ),

                "blue_knockdowns": per_reached(
                    "blue_knockdowns"
                ),

                "red_submission_attempts": per_reached(
                    "red_submission_attempts"
                ),

                "blue_submission_attempts": per_reached(
                    "blue_submission_attempts"
                ),

                "distance_seconds": per_reached(
                    "distance_seconds"
                ),

                "clinch_seconds": per_reached(
                    "clinch_seconds"
                ),

                "ground_seconds": per_reached(
                    "ground_seconds"
                ),

                "completed_round_paths": completed,

                "red_fatigue_end": per_completed(
                    "red_fatigue"
                ),

                "blue_fatigue_end": per_completed(
                    "blue_fatigue"
                ),

                "red_damage_end": per_completed(
                    "red_damage"
                ),

                "blue_damage_end": per_completed(
                    "blue_damage"
                ),

                "red_stress_end": per_completed(
                    "red_stress"
                ),

                "blue_stress_end": per_completed(
                    "blue_stress"
                ),

                "judge_red_round_pct": vote_pct(
                    "RED"
                ),

                "judge_blue_round_pct": vote_pct(
                    "BLUE"
                ),

                "judge_even_round_pct": vote_pct(
                    "EVEN"
                ),
            }
        )

    round_df = pd.DataFrame(
        round_rows
    )

    scorecard_df = pd.DataFrame(
        scorecard_rows
    )

    path_df = pd.DataFrame(
        path_rows
    )

    # =========================================================================
    # Save artifacts
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    round_path = (
        OUTPUT_DIR
        / f"fsr_{fight_id}_v1_round_diagnostics.csv"
    )

    scorecard_path = (
        OUTPUT_DIR
        / f"fsr_{fight_id}_v1_scorecards.csv"
    )

    path_result_path = (
        OUTPUT_DIR
        / f"fsr_{fight_id}_v1_path_results.csv"
    )

    round_df.to_csv(
        round_path,
        index=False,
    )

    scorecard_df.to_csv(
        scorecard_path,
        index=False,
    )

    path_df.to_csv(
        path_result_path,
        index=False,
    )

    # =========================================================================
    # Terminal output
    # =========================================================================

    print()
    print("=" * 100)
    print("ROUND-BY-ROUND POPULATION DIAGNOSTICS")
    print("=" * 100)

    display = round_df[
        [
            "round",
            "paths_reaching_round",
            "population_reaching_pct",

            "red_distance_attempted",
            "blue_distance_attempted",

            "red_distance_landed",
            "blue_distance_landed",

            "red_takedown_success",
            "blue_takedown_success",

            "red_control_seconds",
            "blue_control_seconds",

            "red_knockdowns",
            "blue_knockdowns",

            "red_submission_attempts",
            "blue_submission_attempts",
        ]
    ].rename(
        columns={
            "round": "R",
            "paths_reaching_round": "Paths",
            "population_reaching_pct": "Reach%",

            "red_distance_attempted": "R Dist Att",
            "blue_distance_attempted": "B Dist Att",

            "red_distance_landed": "R Dist Lnd",
            "blue_distance_landed": "B Dist Lnd",

            "red_takedown_success": "R TD",
            "blue_takedown_success": "B TD",

            "red_control_seconds": "R Ctrl",
            "blue_control_seconds": "B Ctrl",

            "red_knockdowns": "R KD",
            "blue_knockdowns": "B KD",

            "red_submission_attempts": "R SUB",
            "blue_submission_attempts": "B SUB",
        }
    )

    print()
    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print(f"RED  = {red_name}")
    print(f"BLUE = {blue_name}")

    # =========================================================================
    # Phase occupancy
    # =========================================================================

    print()
    print("=" * 100)
    print("PHASE OCCUPANCY BY ROUND")
    print("=" * 100)

    phase_display = round_df[
        [
            "round",
            "distance_seconds",
            "clinch_seconds",
            "ground_seconds",
        ]
    ].rename(
        columns={
            "round": "R",
            "distance_seconds": "Distance",
            "clinch_seconds": "Clinch",
            "ground_seconds": "Ground",
        }
    )

    print()
    print(
        phase_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    print()
    print(
        "Seconds are population means conditional "
        "on reaching the round."
    )

    # =========================================================================
    # Dynamic state
    # =========================================================================

    print()
    print("=" * 100)
    print("END-OF-ROUND DYNAMIC STATE — BEFORE ROUND-BREAK RECOVERY")
    print("=" * 100)

    dynamic_display = round_df[
        [
            "round",
            "completed_round_paths",

            "red_fatigue_end",
            "blue_fatigue_end",

            "red_damage_end",
            "blue_damage_end",

            "red_stress_end",
            "blue_stress_end",
        ]
    ].rename(
        columns={
            "round": "R",
            "completed_round_paths": "Paths",

            "red_fatigue_end": "R Fatigue",
            "blue_fatigue_end": "B Fatigue",

            "red_damage_end": "R Damage",
            "blue_damage_end": "B Damage",

            "red_stress_end": "R Stress",
            "blue_stress_end": "B Stress",
        }
    )

    print()
    print(
        dynamic_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # =========================================================================
    # Finish rounds
    # =========================================================================

    print()
    print("=" * 100)
    print("FINISH ROUND DISTRIBUTION")
    print("=" * 100)

    print()

    for round_number in range(
        1,
        scheduled_rounds + 1,
    ):

        count = finish_rounds[
            round_number
        ]

        pct = (
            100.0
            * count
            / simulation_count
        )

        print(
            f"Round {round_number}: "
            f"{count:4d} ({pct:6.2f}%)"
        )

    # =========================================================================
    # Decision types
    # =========================================================================

    print()
    print("=" * 100)
    print("DECISION TYPE DISTRIBUTION")
    print("=" * 100)

    print()

    if not decision_types:

        print(
            "No simulations reached scheduled distance."
        )

    else:

        distance_count = sum(
            decision_types.values()
        )

        for decision_type, count in (
            decision_types.most_common()
        ):

            pct_population = (
                100.0
                * count
                / simulation_count
            )

            pct_decisions = (
                100.0
                * count
                / distance_count
            )

            print(
                f"{decision_type:<30} "
                f"{count:4d}  "
                f"{pct_population:6.2f}% population  "
                f"{pct_decisions:6.2f}% decisions"
            )

    # =========================================================================
    # Judge round scoring
    # =========================================================================

    print()
    print("=" * 100)
    print("SIMULATED JUDGE ROUND WIN PROBABILITIES")
    print("=" * 100)

    judge_display = round_df[
        [
            "round",
            "judge_red_round_pct",
            "judge_blue_round_pct",
            "judge_even_round_pct",
        ]
    ].rename(
        columns={
            "round": "R",
            "judge_red_round_pct": "RED %",
            "judge_blue_round_pct": "BLUE %",
            "judge_even_round_pct": "EVEN %",
        }
    )

    print()
    print(
        judge_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # =========================================================================
    # Most common judge score totals
    # =========================================================================

    print()
    print("=" * 100)
    print("MOST COMMON JUDGE SCORECARD TOTALS")
    print("=" * 100)

    print()

    total_cards = sum(
        judge_card_totals.values()
    )

    if total_cards == 0:

        print(
            "No simulated judge scorecards."
        )

    else:

        for (
            red_total,
            blue_total,
        ), count in judge_card_totals.most_common(10):

            pct = (
                100.0
                * count
                / total_cards
            )

            print(
                f"{red_total}-{blue_total}: "
                f"{count:5d} ({pct:6.2f}%)"
            )

    # =========================================================================
    # Judge fight votes
    # =========================================================================

    print()
    print("=" * 100)
    print("INDIVIDUAL JUDGE FIGHT VOTES")
    print("=" * 100)

    print()

    total_votes = sum(
        judge_fight_votes.values()
    )

    for name, count in (
        judge_fight_votes.most_common()
    ):

        pct = (
            100.0
            * count
            / total_votes
            if total_votes
            else 0.0
        )

        print(
            f"{name:<28}: "
            f"{count:5d} ({pct:6.2f}%)"
        )

    # =========================================================================
    # Example full simulated scorecard
    # =========================================================================

    print()
    print("=" * 100)
    print("EXAMPLE SIMULATED JUDGES SCORECARDS")
    print("=" * 100)

    if example_scorecards is None:

        print()
        print(
            "No scheduled-distance simulation available."
        )

    else:

        print()
        print(
            f"Seed          : {example_seed}"
        )

        print(
            f"Decision type : {example_decision}"
        )

        print()

        for card in example_scorecards:

            if card.winner is FighterSide.RED:
                winner = red_name

            elif card.winner is FighterSide.BLUE:
                winner = blue_name

            else:
                winner = "DRAW"

            print(
                f"Judge {card.judge_number}: "
                f"{card.red_total}-"
                f"{card.blue_total} "
                f"({winner})"
            )

            for score in card.rounds:

                if score.winner is FighterSide.RED:
                    round_winner = red_name

                elif score.winner is FighterSide.BLUE:
                    round_winner = blue_name

                else:
                    round_winner = "EVEN"

                print(
                    f"    Round {score.round_number}: "
                    f"{score.red_points}-"
                    f"{score.blue_points} "
                    f"({round_winner})"
                )

            print()

    # =========================================================================
    # Saved files
    # =========================================================================

    print("=" * 100)
    print("DIAGNOSTIC ARTIFACTS")
    print("=" * 100)

    print()
    print(
        "Round diagnostics :",
        round_path,
    )

    print(
        "Judge scorecards  :",
        scorecard_path,
    )

    print(
        "Path results      :",
        path_result_path,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "fight_id",
        help="Historical UFCStats fight ID.",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=SIMULATIONS_DEFAULT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )

    args = parser.parse_args()

    fight_id = str(
        args.fight_id
    )

    (
        rounds,
        target_date,
        red_info,
        blue_info,
        scheduled_rounds,
    ) = load_target_fight(
        fight_id
    )

    print()
    print("=" * 100)
    print("GENERIC FSR V1.1 HISTORICAL BENCHMARK")
    print("=" * 100)

    print(
        f"Fight ID          : {fight_id}"
    )

    print(
        f"Fight date        : "
        f"{target_date.date()}"
    )

    print(
        f"RED               : "
        f"{red_info['fighter_name']}"
    )

    print(
        f"BLUE              : "
        f"{blue_info['fighter_name']}"
    )

    print(
        f"Scheduled rounds  : "
        f"{scheduled_rounds}"
    )

    print(
        f"Simulations       : "
        f"{args.simulations}"
    )

    print(
        f"Seed start        : "
        f"{args.seed}"
    )

    # -------------------------------------------------------------------------
    # Build leakage-safe rating histories.
    # -------------------------------------------------------------------------

    run_rating_builders(
        fight_id
    )

    red_card, red_count = (
        build_full_card(
            fight_id,
            red_info[
                "fighter_id"
            ],
        )
    )

    blue_card, blue_count = (
        build_full_card(
            fight_id,
            blue_info[
                "fighter_id"
            ],
        )
    )

    baselines = (
        population_baselines(
            rounds,
            target_date,
        )
    )

    print()
    print("=" * 100)
    print("POPULATION BASELINES")
    print("=" * 100)

    for key, value in (
        baselines.items()
    ):
        print(
            f"{key:<32} "
            f"{value:.6f}"
        )

    print()
    print("=" * 100)
    print("FSR V1.1 FIGHTER CARDS")
    print("=" * 100)

    print_card(
        red_info[
            "fighter_name"
        ],
        red_count,
        red_card,
    )

    print_card(
        blue_info[
            "fighter_name"
        ],
        blue_count,
        blue_card,
    )

    # -------------------------------------------------------------------------
    # Translate FSR -> MC V2.
    # -------------------------------------------------------------------------

    red_transition = (
        build_transition(
            red_card
        )
    )

    blue_transition = (
        build_transition(
            blue_card
        )
    )

    red_phase = (
        build_phase(
            red_card,
            blue_card,
            baselines,
        )
    )

    blue_phase = (
        build_phase(
            blue_card,
            red_card,
            baselines,
        )
    )

    red_dynamic = (
        build_dynamic(
            red_card
        )
    )

    blue_dynamic = (
        build_dynamic(
            blue_card
        )
    )

    print()
    print("=" * 100)
    print("FSR V1.2 -> MC V2 MATCHUP PARAMETERS")
    print("=" * 100)

    print_matchup_parameters(
        red_info[
            "fighter_name"
        ],
        red_card,
        red_transition,
        red_phase,
        red_dynamic,
    )

    print_matchup_parameters(
        blue_info[
            "fighter_name"
        ],
        blue_card,
        blue_transition,
        blue_phase,
        blue_dynamic,
    )

    # -------------------------------------------------------------------------
    # Same MC V2 calibration bundle used throughout benchmarks.
    # -------------------------------------------------------------------------

    candidate = Candidate(
        landed_ko_hazard=(
            V1_LANDED_KO_HAZARD
        ),
        knockdown_bonus_hazard=(
            V1_KNOCKDOWN_BONUS_HAZARD
        ),
    )

    summary = (
        run_matchup_monte_carlo(
            red_transition,
            blue_transition,
            red_phase,
            blue_phase,
            red_dynamic,
            blue_dynamic,

            dynamic_state_calibration=(
                state_calibration(
                    candidate
                )
            ),

            phase_effect_calibration=(
                phase_effect_calibration(
                    candidate
                )
            ),

            transition_effect_calibration=(
                zero_transition_effect_calibration()
            ),

            finish_probability_calibration=(
                finish_calibration(
                    candidate
                )
            ),

            simulation_count=(
                args.simulations
            ),

            seed_start=(
                args.seed
            ),

            scheduled_rounds=(
                scheduled_rounds
            ),

            red_intrinsic_power_multiplier=(
                intrinsic_power_multiplier(
                    red_card[
                        "finishing_power"
                    ]
                )
            ),

            blue_intrinsic_power_multiplier=(
                intrinsic_power_multiplier(
                    blue_card[
                        "finishing_power"
                    ]
                )
            ),

            red_intrinsic_ko_vulnerability_multiplier=(
                intrinsic_ko_vulnerability_multiplier(
                    red_card[
                        "chin_resistance"
                    ]
                )
            ),

            blue_intrinsic_ko_vulnerability_multiplier=(
                intrinsic_ko_vulnerability_multiplier(
                    blue_card[
                        "chin_resistance"
                    ]
                )
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    n = (
        summary.simulation_count
    )

    def pct(count: int) -> float:
        return (
            100.0
            * count
            / n
        )

    red_name = (
        red_info[
            "fighter_name"
        ]
    )

    blue_name = (
        blue_info[
            "fighter_name"
        ]
    )

    print()
    print("=" * 100)
    print("WINNER PROBABILITY")
    print("=" * 100)

    print(
        f"{red_name:<28}: "
        f"{pct(summary.red_win_count):6.2f}%"
    )

    print(
        f"{blue_name:<28}: "
        f"{pct(summary.blue_win_count):6.2f}%"
    )

    print(
        f"{'Draw':<28}: "
        f"{pct(summary.draw_count):6.2f}%"
    )

    print()
    print("=" * 100)
    print("METHOD PROBABILITY")
    print("=" * 100)

    print(
        f"{red_name + ' KO/TKO':<28}: "
        f"{pct(summary.red_ko_tko_count):6.2f}%"
    )

    print(
        f"{blue_name + ' KO/TKO':<28}: "
        f"{pct(summary.blue_ko_tko_count):6.2f}%"
    )

    print(
        f"{red_name + ' SUB':<28}: "
        f"{pct(summary.red_submission_count):6.2f}%"
    )

    print(
        f"{blue_name + ' SUB':<28}: "
        f"{pct(summary.blue_submission_count):6.2f}%"
    )

    print(
        f"{red_name + ' DEC':<28}: "
        f"{pct(summary.red_decision_count):6.2f}%"
    )

    print(
        f"{blue_name + ' DEC':<28}: "
        f"{pct(summary.blue_decision_count):6.2f}%"
    )

    print(
        f"{'Scheduled distance':<28}: "
        f"{pct(summary.scheduled_distance_count):6.2f}%"
    )

    # -------------------------------------------------------------------------
    # Detailed deterministic diagnostic replay.
    #
    # This does not replace or alter the authoritative MC summary above.
    # -------------------------------------------------------------------------

    run_population_diagnostics(
        fight_id=fight_id,

        red_name=red_name,
        blue_name=blue_name,

        red_transition=red_transition,
        blue_transition=blue_transition,

        red_phase=red_phase,
        blue_phase=blue_phase,

        red_dynamic=red_dynamic,
        blue_dynamic=blue_dynamic,

        red_card=red_card,
        blue_card=blue_card,

        scheduled_rounds=scheduled_rounds,

        simulation_count=args.simulations,
        seed_start=args.seed,
    )


if __name__ == "__main__":
    main()
