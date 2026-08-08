"""Experimental FSR V1.2: finishing power, chin resistance, and damage absorption.

Shadow-only research script.

Persistent fighter-state dimensions:

    finishing_power
        Acute offensive ability to create knockdowns / KO-TKO outcomes.

    chin_resistance
        Resistance to an acute knockdown or single-event KO/TKO outcome.

    damage_absorption
        Ability to continue functioning after accumulating repeated incoming
        significant-strike damage.

The concepts are intentionally separated:

    attacker finishing_power
            vs
    defender chin_resistance
            -> acute KD / KO evidence

while:

    accumulated incoming significant strikes
            + opponent finishing power
            + subsequent performance retention
            -> damage_absorption evidence

Important:
- only fights BEFORE the target fight are replayed
- both fighters use PRE-fight ratings when expectations are calculated
- updates are applied simultaneously after each fight
- pre-2018 history may build ratings
- the target/evaluation fight itself is never used
- V1.1 is preserved unchanged for A/B comparison
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from math import exp, log, sqrt, tanh
from pathlib import Path

import pandas as pd


ROUND_PATH = Path(
    "data/fight_details/ufc_round_stats.parquet"
)

MASTER_PATH = Path(
    "data/master/ufc_master.parquet"
)

OUTPUT_DIR = Path(
    "data/simulation/rfs_mc_v2_shared_state"
)


# =============================================================================
# Rating controls
# =============================================================================

BASE_RATING = 50.0
MIN_RATING = 10.0
MAX_RATING = 90.0

BASE_K = 3.5

# Existing V1.1 scales are intentionally preserved.
PROBABILITY_RATING_SCALE = 12.0
RATE_RATING_SCALE = 24.0

# Damage absorption is a noisier signal than a KD event.
# Keep its update somewhat conservative.
ABSORPTION_UPDATE_WEIGHT = 0.75

# Chin resistance is learned from actual KO/TKO loss history rather than
# ordinary knockdowns.
#
# A KO/TKO loss is strong negative evidence.
# Surviving additional UFC fights without another KO/TKO gradually restores
# confidence, but recovery is intentionally slower than the loss penalty.
# Chin-resistance event model.
#
# Each historical KO/TKO loss contributes a vulnerability weight.
# That weight:
#   - is smaller for young fighters
#   - is larger for older fighters
#   - decays faster when the KO occurred at a young age
#   - decays slower when the KO occurred at an older age
#
# Surviving many UFC fights without KO/TKO loss provides modest positive
# evidence, but does not overpower recent or clustered KO losses.

CHIN_EVENT_BASE_PENALTY = 7.0

# Age where KO vulnerability begins to become materially more persistent.
CHIN_AGE_CENTER = 32.0
CHIN_AGE_SCALE = 3.5

# Initial KO-event severity multiplier.
CHIN_MIN_EVENT_MULTIPLIER = 0.55
CHIN_MAX_EVENT_MULTIPLIER = 1.60

# Event decay measured in subsequent UFC fights.
#
# Young KO events fade relatively quickly.
# Older KO events remain relevant for much longer.
CHIN_MIN_DECAY_FIGHTS = 2.5
CHIN_MAX_DECAY_FIGHTS = 8.0

# Modest positive evidence from fights survived without a KO/TKO loss.
CHIN_SURVIVAL_MAX_BONUS = 8.0
CHIN_SURVIVAL_SCALE_FIGHTS = 12.0

# Keep chin ratings within a sensible experimental band for now.
CHIN_MIN_RATING = 20.0
CHIN_MAX_RATING = 70.0

# Effective accumulated-damage scale used by damage_absorption.
DAMAGE_EXPOSURE_SCALE = 30.0

EPS = 1e-6


# =============================================================================
# Math helpers
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


def sigmoid(
    value: float,
) -> float:
    return 1.0 / (
        1.0 + exp(-value)
    )


def logit(
    probability: float,
) -> float:
    probability = clamp(
        probability,
        EPS,
        1.0 - EPS,
    )

    return log(
        probability
        / (1.0 - probability)
    )


def k_factor(
    fight_count: int,
) -> float:
    """Experience-decayed FSR update size."""

    return BASE_K / sqrt(
        1.0
        + fight_count / 6.0
    )


def count_surprise(
    observed_count: float,
    expected_count: float,
) -> float:
    """Bounded residual for sparse count events."""

    residual = (
        observed_count
        - expected_count
    ) / sqrt(
        expected_count + 1.0
    )

    return tanh(residual)


def probability_from_ratings(
    *,
    baseline: float,
    attacker_rating: float,
    defender_rating: float,
) -> float:
    """Opponent-adjusted expected binary-event probability."""

    rating_delta = (
        attacker_rating
        - defender_rating
    )

    return sigmoid(
        logit(baseline)
        + rating_delta
        / PROBABILITY_RATING_SCALE
    )


def rate_from_ratings(
    *,
    baseline: float,
    attacker_rating: float,
    defender_rating: float,
) -> float:
    """Opponent-adjusted expected event rate."""

    rating_delta = (
        attacker_rating
        - defender_rating
    )

    return baseline * exp(
        rating_delta
        / RATE_RATING_SCALE
    )


# =============================================================================
# Target
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "fight_id",
    help=(
        "Historical fight_id whose PRE-fight "
        "ratings should be produced."
    ),
)

args = parser.parse_args()

TARGET_FIGHT_ID = str(
    args.fight_id
)


# =============================================================================
# Load round data
# =============================================================================

rounds = pd.read_parquet(
    ROUND_PATH
)

rounds["event_date"] = pd.to_datetime(
    rounds["event_date"]
)

rounds["fighter_id"] = (
    rounds["fighter_id"]
    .astype(str)
)

rounds["opponent_id"] = (
    rounds["opponent_id"]
    .astype(str)
)

rounds["fight_id"] = (
    rounds["fight_id"]
    .astype(str)
)


target_rows = rounds.loc[
    rounds["fight_id"]
    == TARGET_FIGHT_ID
].copy()

if target_rows.empty:
    raise RuntimeError(
        "Fight not found in round stats: "
        f"{TARGET_FIGHT_ID}"
    )


target_date = pd.Timestamp(
    target_rows[
        "event_date"
    ].iloc[0]
)


target_fighters = (
    target_rows[
        [
            "fighter_id",
            "fighter_name",
        ]
    ]
    .drop_duplicates()
)


if len(target_fighters) != 2:
    raise RuntimeError(
        "Target fight does not contain "
        "exactly two fighters."
    )


target_ids = set(
    target_fighters[
        "fighter_id"
    ].astype(str)
)


target_names = {
    str(row.fighter_id):
    str(row.fighter_name)
    for row
    in target_fighters.itertuples(
        index=False
    )
}


# =============================================================================
# Leakage-safe historical population
# =============================================================================

history = rounds.loc[
    rounds["event_date"]
    < target_date
].copy()


# =============================================================================
# Fight-level acute finishing evidence
# =============================================================================

fighter_fights = (
    history
    .groupby(
        [
            "fight_id",
            "event_date",
            "fighter_id",
            "fighter_name",
            "opponent_id",
            "opponent_name",
        ],
        as_index=False,
    )
    .agg(
        rounds=("round", "nunique"),
        knockdowns=("kd", "sum"),
        sig_landed=(
            "sig_str_landed",
            "sum",
        ),
    )
)


# =============================================================================
# Fight outcomes
# =============================================================================

master = pd.read_parquet(
    MASTER_PATH
)

master["fight_id"] = (
    master["fight_id"]
    .astype(str)
)

# =============================================================================
# Canonical fighter DOB map
# =============================================================================
#
# UFC master stores DOB separately for RED and BLUE corners.
# Normalize both sides into:
#
#     fighter_id -> date_of_birth
#
# Age is then calculated at each historical fight date.
# =============================================================================

red_dob = master[
    [
        "r_id",
        "r_dob",
    ]
].rename(
    columns={
        "r_id": "fighter_id",
        "r_dob": "dob",
    }
)

blue_dob = master[
    [
        "b_id",
        "b_dob",
    ]
].rename(
    columns={
        "b_id": "fighter_id",
        "b_dob": "dob",
    }
)

fighter_dob = pd.concat(
    [
        red_dob,
        blue_dob,
    ],
    ignore_index=True,
)

fighter_dob["fighter_id"] = (
    fighter_dob["fighter_id"]
    .astype(str)
)

fighter_dob["dob"] = pd.to_datetime(
    fighter_dob["dob"],
    errors="coerce",
)

fighter_dob = (
    fighter_dob
    .dropna(
        subset=[
            "fighter_id",
            "dob",
        ]
    )
    .drop_duplicates(
        "fighter_id"
    )
)

DOB_BY_FIGHTER_ID = {
    str(row.fighter_id):
        pd.Timestamp(row.dob)
    for row
    in fighter_dob.itertuples(
        index=False
    )
}


def age_at_fight(
    fighter_id: str,
    fight_date,
) -> float | None:
    """Return fighter age in years on the historical fight date."""

    dob = DOB_BY_FIGHTER_ID.get(
        str(fighter_id)
    )

    if dob is None or pd.isna(dob):
        return None

    fight_date = pd.Timestamp(
        fight_date
    )

    return (
        fight_date - dob
    ).days / 365.2425


master_result = master[
    [
        "fight_id",
        "method",
        "winner_id",
    ]
].copy()

master_result["winner_id"] = (
    master_result["winner_id"]
    .astype(str)
)

master_result = (
    master_result
    .drop_duplicates(
        "fight_id"
    )
)


fighter_fights = (
    fighter_fights
    .merge(
        master_result,
        on="fight_id",
        how="left",
    )
)


fighter_fights["is_ko_tko"] = (
    fighter_fights["method"]
    .fillna("")
    .astype(str)
    .str.contains(
        "KO/TKO",
        regex=False,
    )
)


fighter_fights["ko_win"] = (
    fighter_fights["is_ko_tko"]
    & (
        fighter_fights["winner_id"]
        == fighter_fights["fighter_id"]
    )
).astype(float)


fighter_fights["ko_loss"] = (
    fighter_fights["is_ko_tko"]
    & (
        fighter_fights["winner_id"]
        != fighter_fights["fighter_id"]
    )
).astype(float)


# =============================================================================
# Population baselines for acute power / chin
# =============================================================================

total_kd = float(
    fighter_fights[
        "knockdowns"
    ].sum()
)

total_sig_landed = float(
    fighter_fights[
        "sig_landed"
    ].sum()
)

total_rounds = float(
    fighter_fights[
        "rounds"
    ].sum()
)


POP_KD_PER_SIG_LANDED = (
    total_kd
    / max(
        total_sig_landed,
        1.0,
    )
)


POP_KD_PER_ROUND = (
    total_kd
    / max(
        total_rounds,
        1.0,
    )
)


POP_KO_WIN_PROBABILITY = float(
    fighter_fights[
        "ko_win"
    ].mean()
)


print()
print("=" * 100)
print(
    "FSR V1.2 POWER / CHIN / "
    "DAMAGE-ABSORPTION BASELINES"
)
print("=" * 100)

print(
    "KD / sig landed :",
    round(
        POP_KD_PER_SIG_LANDED,
        6,
    ),
)

print(
    "KD / round      :",
    round(
        POP_KD_PER_ROUND,
        6,
    ),
)

print(
    "KO/TKO win prob :",
    round(
        POP_KO_WIN_PROBABILITY,
        6,
    ),
)


# =============================================================================
# Round-level data for damage absorption
# =============================================================================
#
# damage_absorption is NOT "how many strikes can this fighter get hit by."
#
# We instead ask:
#
#     after accumulated incoming significant strikes,
#     how well does the fighter retain subsequent performance?
#
# The incoming damage load is also adjusted by the opponent's PRE-fight
# finishing-power rating during historical replay.
# =============================================================================

needed_columns = [
    "fight_id",
    "event_date",
    "round",
    "fighter_id",
    "fighter_name",
    "opponent_id",
    "opponent_name",
    "sig_str_attempted",
    "sig_str_landed",
    "td_attempted",
    "ctrl_sec",
]


missing = [
    column
    for column in needed_columns
    if column not in history.columns
]

if missing:
    raise RuntimeError(
        "Missing required round-stat columns "
        "for damage_absorption: "
        + ", ".join(missing)
    )


round_perf = history[
    needed_columns
].copy()


# Add opponent significant strikes landed in the same round.
opponent_damage = round_perf[
    [
        "fight_id",
        "round",
        "fighter_id",
        "sig_str_landed",
    ]
].rename(
    columns={
        "fighter_id":
            "opponent_id",
        "sig_str_landed":
            "incoming_sig_landed",
    }
)


round_perf = round_perf.merge(
    opponent_damage,
    on=[
        "fight_id",
        "round",
        "opponent_id",
    ],
    how="left",
)


round_perf[
    "incoming_sig_landed"
] = (
    round_perf[
        "incoming_sig_landed"
    ]
    .fillna(0.0)
    .astype(float)
)


# =============================================================================
# Population activity scales
# =============================================================================
#
# These normalize different performance components before combining them.
# They are descriptive population scales, not fighter ratings.
# =============================================================================

POP_SIG_ATTEMPTED_PER_ROUND = max(
    float(
        round_perf[
            "sig_str_attempted"
        ].mean()
    ),
    1.0,
)

POP_SIG_LANDED_PER_ROUND = max(
    float(
        round_perf[
            "sig_str_landed"
        ].mean()
    ),
    1.0,
)

POP_TD_ATTEMPTED_PER_ROUND = max(
    float(
        round_perf[
            "td_attempted"
        ].mean()
    ),
    0.25,
)

POP_CONTROL_SECONDS_PER_ROUND = max(
    float(
        round_perf[
            "ctrl_sec"
        ].mean()
    ),
    5.0,
)


def round_performance_index(
    row: pd.Series,
) -> float:
    """Broad measure of how much useful activity a fighter maintains.

    This is intentionally not purely striking based.

    We include:
    - significant-strike attempts
    - significant strikes landed
    - takedown attempts
    - control

    This makes damage_absorption closer to continued functional performance
    than simply measuring whether a fighter keeps throwing punches.
    """

    sig_attempt_component = (
        float(
            row["sig_str_attempted"]
        )
        / POP_SIG_ATTEMPTED_PER_ROUND
    )

    sig_landed_component = (
        float(
            row["sig_str_landed"]
        )
        / POP_SIG_LANDED_PER_ROUND
    )

    td_component = (
        float(
            row["td_attempted"]
        )
        / POP_TD_ATTEMPTED_PER_ROUND
    )

    control_component = (
        float(
            row["ctrl_sec"]
        )
        / POP_CONTROL_SECONDS_PER_ROUND
    )

    # Striking carries most of the weight.
    return (
        0.40 * sig_attempt_component
        + 0.30 * sig_landed_component
        + 0.15 * td_component
        + 0.15 * control_component
    )


round_perf[
    "performance_index"
] = round_perf.apply(
    round_performance_index,
    axis=1,
)


# =============================================================================
# Rating state
# =============================================================================

power = defaultdict(
    lambda: BASE_RATING
)

# Chin is derived from historical KO/TKO events rather than incrementally
# updated as an Elo-like state.
#
# Each fighter stores their prior KO events as:
#
#     {
#         "age_at_ko": ...,
#         "fight_index": ...,
#     }
#
chin_events = defaultdict(list)

damage_absorption = defaultdict(
    lambda: BASE_RATING
)

fight_counts = defaultdict(
    int
)

rating_history: list[dict] = []


# =============================================================================
# Acute finishing / chin evidence
# =============================================================================


def offensive_surprise(
    row: pd.Series,
    *,
    attacker_power: float,
    defender_chin: float,
) -> tuple[
    float,
    dict[str, float],
]:
    """Estimate sudden finishing-power evidence.

    V1.2 definition:

        finishing_power =
            ability to create a sudden strike-based fight-ending event

    Primary evidence:
    - KO/TKO wins
    - knockdowns produced
    - knockdown efficiency per significant strike landed

    Important:
    - KO/TKO wins are the strongest positive signal.
    - A non-KO fight is only weak negative evidence.
    - General striking volume is NOT finishing power.
    """

    sig_landed = max(
        float(row["sig_landed"]),
        0.0,
    )

    knockdowns = max(
        float(row["knockdowns"]),
        0.0,
    )

    ko_win = float(
        row["ko_win"]
    )

    rounds_fought = max(
        float(row["rounds"]),
        1.0,
    )

    # -------------------------------------------------------------------------
    # 1. Expected knockdowns from strike opportunities.
    #
    # Attacker power increases expected KD production.
    # Defender chin resistance suppresses expected KD production.
    # -------------------------------------------------------------------------

    expected_kd_per_landed = (
        rate_from_ratings(
            baseline=(
                POP_KD_PER_SIG_LANDED
            ),
            attacker_rating=(
                attacker_power
            ),
            defender_rating=(
                defender_chin
            ),
        )
    )

    expected_kd_count = (
        expected_kd_per_landed
        * sig_landed
    )

    kd_count_surprise = (
        count_surprise(
            observed_count=(
                knockdowns
            ),
            expected_count=(
                expected_kd_count
            ),
        )
    )

    # -------------------------------------------------------------------------
    # 2. KD efficiency.
    #
    # This asks whether the fighter produces knockdowns efficiently relative
    # to how many significant strikes they actually landed.
    #
    # It is deliberately a smaller signal than actual KO/TKO outcomes.
    # -------------------------------------------------------------------------

    observed_kd_efficiency = (
        knockdowns
        / max(
            sig_landed,
            1.0,
        )
    )

    expected_kd_efficiency = (
        expected_kd_per_landed
    )

    kd_efficiency_residual = (
        observed_kd_efficiency
        - expected_kd_efficiency
    ) / sqrt(
        expected_kd_efficiency
        + 0.01
    )

    kd_efficiency_surprise = tanh(
        kd_efficiency_residual
    )

    # -------------------------------------------------------------------------
    # 3. KO/TKO outcome evidence.
    #
    # This is intentionally the dominant signal.
    #
    # KO win:
    #     strong positive evidence.
    #
    # No KO:
    #     only weak negative evidence.
    #
    # This prevents a dangerous puncher from losing substantial power rating
    # simply because a particular fight reached a decision.
    # -------------------------------------------------------------------------

    expected_ko_probability = (
        probability_from_ratings(
            baseline=(
                POP_KO_WIN_PROBABILITY
            ),
            attacker_rating=(
                attacker_power
            ),
            defender_rating=(
                defender_chin
            ),
        )
    )

    if ko_win > 0.0:

        ko_surprise = (
            1.0
            - expected_ko_probability
        )

    else:

        # Weak negative evidence only.
        #
        # Longer fights without a finish provide somewhat more evidence than
        # a very short fight, but the effect remains intentionally small.
        exposure_factor = min(
            rounds_fought / 3.0,
            1.0,
        )

        ko_surprise = (
            -0.15
            * expected_ko_probability
            * exposure_factor
        )

    # -------------------------------------------------------------------------
    # Final finishing-power surprise.
    #
    # KO/TKO history dominates.
    # Knockdowns remain meaningful.
    # KD efficiency adds a smaller "one-shot danger" component.
    # -------------------------------------------------------------------------

    combined = (
        0.60 * ko_surprise
        + 0.30 * kd_count_surprise
        + 0.10 * kd_efficiency_surprise
    )

    combined = clamp(
        combined,
        -1.0,
        1.0,
    )

    return (
        combined,
        {
            "expected_kd_count":
                expected_kd_count,

            "kd_count_surprise":
                kd_count_surprise,

            "observed_kd_efficiency":
                observed_kd_efficiency,

            "expected_kd_efficiency":
                expected_kd_efficiency,

            "kd_efficiency_surprise":
                kd_efficiency_surprise,

            "expected_ko_probability":
                expected_ko_probability,

            "ko_surprise":
                ko_surprise,

            "finishing_power_surprise":
                combined,
        },
    )


# =============================================================================
# Damage-absorption evidence
# =============================================================================


def absorption_surprise(
    *,
    fight_id: str,
    fighter_id: str,
    opponent_power: float,
    absorption_rating: float,
) -> tuple[
    float,
    dict[str, float],
]:
    """Estimate repeated-damage performance-retention surprise.

    We compare subsequent-round performance against the fighter's opening
    round performance while weighting later observations according to the
    accumulated incoming significant-strike load.

    Important:
    getting hit is not itself rewarded.

    A fighter receives positive evidence only if:
    1. meaningful accumulated damage exposure exists, AND
    2. subsequent functional performance is retained better than expected.

    Opponent finishing power changes the effective damage load.
    """

    fighter_rounds = (
        round_perf.loc[
            (
                round_perf["fight_id"]
                == fight_id
            )
            & (
                round_perf["fighter_id"]
                == fighter_id
            )
        ]
        .sort_values(
            "round"
        )
        .copy()
    )


    # We cannot measure within-fight retention from a one-round fight.
    if len(
        fighter_rounds
    ) < 2:
        return (
            0.0,
            {
                "damage_exposure":
                    0.0,
                "observed_retention_log":
                    0.0,
                "expected_retention_log":
                    0.0,
                "absorption_surprise":
                    0.0,
                "absorption_rounds_used":
                    0,
            },
        )


    opening_performance = max(
        float(
            fighter_rounds.iloc[0][
                "performance_index"
            ]
        ),
        0.10,
    )


    # Stronger hitters make the same number of incoming significant strikes
    # a somewhat larger effective damage exposure.
    opponent_power_multiplier = exp(
        (
            opponent_power
            - BASE_RATING
        )
        / RATE_RATING_SCALE
    )


    cumulative_damage = 0.0

    weighted_surprise_sum = 0.0
    total_weight = 0.0

    weighted_observed_sum = 0.0
    weighted_expected_sum = 0.0

    rounds_used = 0


    for index, row in enumerate(
        fighter_rounds.itertuples(
            index=False
        )
    ):
        incoming = max(
            float(
                row.incoming_sig_landed
            ),
            0.0,
        )


        # Round 1 establishes the initial performance reference.
        if index == 0:
            cumulative_damage += (
                incoming
                * opponent_power_multiplier
            )
            continue


        exposure_strength = tanh(
            cumulative_damage
            / DAMAGE_EXPOSURE_SCALE
        )


        # Tiny exposure contributes almost no rating information.
        if exposure_strength > 0.05:

            current_performance = max(
                float(
                    row.performance_index
                ),
                0.05,
            )


            observed_retention_log = log(
                current_performance
                / opening_performance
            )


            # A higher existing damage_absorption rating means that stronger
            # retention is already expected.
            #
            # This prevents ratings from climbing indefinitely simply because
            # a fighter historically performs well late.
            expected_retention_log = (
                (
                    absorption_rating
                    - BASE_RATING
                )
                / RATE_RATING_SCALE
                * exposure_strength
            )


            residual = (
                observed_retention_log
                - expected_retention_log
            )


            round_surprise = tanh(
                residual
            )


            weighted_surprise_sum += (
                exposure_strength
                * round_surprise
            )

            weighted_observed_sum += (
                exposure_strength
                * observed_retention_log
            )

            weighted_expected_sum += (
                exposure_strength
                * expected_retention_log
            )

            total_weight += (
                exposure_strength
            )

            rounds_used += 1


        cumulative_damage += (
            incoming
            * opponent_power_multiplier
        )


    if total_weight <= EPS:
        surprise = 0.0
        observed_mean = 0.0
        expected_mean = 0.0
    else:
        surprise = (
            weighted_surprise_sum
            / total_weight
        )

        observed_mean = (
            weighted_observed_sum
            / total_weight
        )

        expected_mean = (
            weighted_expected_sum
            / total_weight
        )


    return (
        surprise,
        {
            "damage_exposure":
                cumulative_damage,
            "observed_retention_log":
                observed_mean,
            "expected_retention_log":
                expected_mean,
            "absorption_surprise":
                surprise,
            "absorption_rounds_used":
                rounds_used,
        },
    )



# =============================================================================
# Chin resistance
# =============================================================================


def chin_age_vulnerability(
    age: float | None,
) -> float:
    """Smooth age vulnerability from 0 to 1."""

    if age is None:
        return 0.5

    return sigmoid(
        (
            age
            - CHIN_AGE_CENTER
        )
        / CHIN_AGE_SCALE
    )


def chin_event_multiplier(
    age_at_ko: float | None,
) -> float:
    """Initial severity of a KO/TKO event."""

    vulnerability = (
        chin_age_vulnerability(
            age_at_ko
        )
    )

    return (
        CHIN_MIN_EVENT_MULTIPLIER
        + vulnerability
        * (
            CHIN_MAX_EVENT_MULTIPLIER
            - CHIN_MIN_EVENT_MULTIPLIER
        )
    )


def chin_event_decay_scale(
    age_at_ko: float | None,
) -> float:
    """How many subsequent UFC fights a KO event remains influential.

    Young fighters:
        short decay horizon

    Older fighters:
        long decay horizon
    """

    vulnerability = (
        chin_age_vulnerability(
            age_at_ko
        )
    )

    return (
        CHIN_MIN_DECAY_FIGHTS
        + vulnerability
        * (
            CHIN_MAX_DECAY_FIGHTS
            - CHIN_MIN_DECAY_FIGHTS
        )
    )


def calculate_chin_resistance(
    *,
    fight_count: int,
    ko_events: list[dict],
) -> tuple[float, dict[str, float]]:
    """Calculate current chin resistance from historical KO/TKO events.

    This is not a running Elo update.

    Every prior KO/TKO loss contributes a vulnerability penalty.

    Penalty depends on:
        - age when the KO happened
        - how many UFC fights have occurred since that KO

    Young KO events:
        smaller initial penalty
        faster decay

    Older KO events:
        larger initial penalty
        slower decay

    Survival without KO/TKO provides modest positive evidence.
    """

    total_vulnerability = 0.0

    active_event_count = 0

    most_recent_event_weight = 0.0


    for event in ko_events:

        age_at_ko = event[
            "age_at_ko"
        ]

        event_fight_index = int(
            event[
                "fight_index"
            ]
        )

        fights_since_event = max(
            fight_count
            - event_fight_index
            - 1,
            0,
        )


        severity = (
            CHIN_EVENT_BASE_PENALTY
            * chin_event_multiplier(
                age_at_ko
            )
        )


        decay_scale = (
            chin_event_decay_scale(
                age_at_ko
            )
        )


        decay = exp(
            -float(
                fights_since_event
            )
            / decay_scale
        )


        event_weight = (
            severity
            * decay
        )


        total_vulnerability += (
            event_weight
        )

        active_event_count += 1

        most_recent_event_weight = max(
            most_recent_event_weight,
            event_weight,
        )


    # -------------------------------------------------------------------------
    # Positive survival evidence.
    #
    # Many UFC fights without being stopped should count for something,
    # especially for fighters with no active/recent KO vulnerability.
    #
    # This is intentionally capped and modest.
    # -------------------------------------------------------------------------

    survival_bonus = (
        CHIN_SURVIVAL_MAX_BONUS
        * (
            1.0
            - exp(
                -float(
                    fight_count
                )
                / CHIN_SURVIVAL_SCALE_FIGHTS
            )
        )
    )


    raw_rating = (
        BASE_RATING
        + survival_bonus
        - total_vulnerability
    )


    rating = clamp(
        raw_rating,
        CHIN_MIN_RATING,
        CHIN_MAX_RATING,
    )


    return (
        rating,
        {
            "chin_prior_fights":
                fight_count,

            "chin_ko_event_count":
                active_event_count,

            "chin_total_vulnerability":
                total_vulnerability,

            "chin_survival_bonus":
                survival_bonus,

            "chin_recent_event_weight":
                most_recent_event_weight,

            "chin_raw_rating":
                raw_rating,
        },
    )




# =============================================================================
# Chronological replay
# =============================================================================

ordered_fights = (
    fighter_fights[
        [
            "fight_id",
            "event_date",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "event_date",
            "fight_id",
        ]
    )
)


for fight in ordered_fights.itertuples(
    index=False
):

    fight_id = str(
        fight.fight_id
    )


    rows = fighter_fights.loc[
        fighter_fights[
            "fight_id"
        ]
        == fight_id
    ].copy()


    if len(rows) != 2:
        continue


    a = rows.iloc[0]
    b = rows.iloc[1]

    a_id = str(
        a["fighter_id"]
    )

    b_id = str(
        b["fighter_id"]
    )


    # -------------------------------------------------------------------------
    # Freeze every state before calculating either fighter's evidence.
    # -------------------------------------------------------------------------

    a_power_pre = float(
        power[a_id]
    )

    b_power_pre = float(
        power[b_id]
    )

    # Chin is recalculated from all KO/TKO events known BEFORE this fight.
    a_chin_pre, a_chin_pre_diag = (
        calculate_chin_resistance(
            fight_count=(
                fight_counts[a_id]
            ),
            ko_events=(
                chin_events[a_id]
            ),
        )
    )

    b_chin_pre, b_chin_pre_diag = (
        calculate_chin_resistance(
            fight_count=(
                fight_counts[b_id]
            ),
            ko_events=(
                chin_events[b_id]
            ),
        )
    )

    a_absorption_pre = float(
        damage_absorption[a_id]
    )

    b_absorption_pre = float(
        damage_absorption[b_id]
    )


    # Fighter age is calculated at THIS historical fight date.
    # This keeps age effects temporally correct and leakage-safe.
    a_age = age_at_fight(
        a_id,
        fight.event_date,
    )

    b_age = age_at_fight(
        b_id,
        fight.event_date,
    )


    # -------------------------------------------------------------------------
    # Acute power / chin surprise
    # -------------------------------------------------------------------------

    a_acute_surprise, a_acute_diag = (
        offensive_surprise(
            a,
            attacker_power=(
                a_power_pre
            ),
            defender_chin=(
                b_chin_pre
            ),
        )
    )


    b_acute_surprise, b_acute_diag = (
        offensive_surprise(
            b,
            attacker_power=(
                b_power_pre
            ),
            defender_chin=(
                a_chin_pre
            ),
        )
    )


    # -------------------------------------------------------------------------
    # Accumulated-damage performance-retention surprise
    # -------------------------------------------------------------------------

    a_absorption_surprise, a_abs_diag = (
        absorption_surprise(
            fight_id=fight_id,
            fighter_id=a_id,
            opponent_power=(
                b_power_pre
            ),
            absorption_rating=(
                a_absorption_pre
            ),
        )
    )


    b_absorption_surprise, b_abs_diag = (
        absorption_surprise(
            fight_id=fight_id,
            fighter_id=b_id,
            opponent_power=(
                a_power_pre
            ),
            absorption_rating=(
                b_absorption_pre
            ),
        )
    )


    # Fighter-specific experience-adjusted K.
    a_k = k_factor(
        fight_counts[a_id]
    )

    b_k = k_factor(
        fight_counts[b_id]
    )


    # -------------------------------------------------------------------------
    # SIMULTANEOUS UPDATES
    #
    # Acute pathway:
    #
    # A produces more acute finishing damage than expected:
    #   A finishing_power ↑
    #   B chin_resistance ↓
    #
    # B produces more acute finishing damage than expected:
    #   B finishing_power ↑
    #   A chin_resistance ↓
    #
    # Accumulated pathway:
    #
    # Fighter maintains function after accumulated incoming damage:
    #   own damage_absorption ↑
    #
    # Fighter degrades worse than expected:
    #   own damage_absorption ↓
    # -------------------------------------------------------------------------

    a_power_post = clamp(
        a_power_pre
        + a_k
        * a_acute_surprise,
        MIN_RATING,
        MAX_RATING,
    )


    # Chin does not receive an incremental post-fight rating update.
    # Historical KO events are appended below, then the rating will be
    # recalculated from the full event history before the fighter's next fight.

    a_chin_post = a_chin_pre
    a_chin_diag = dict(
        a_chin_pre_diag
    )


    b_power_post = clamp(
        b_power_pre
        + b_k
        * b_acute_surprise,
        MIN_RATING,
        MAX_RATING,
    )


    b_chin_post = b_chin_pre
    b_chin_diag = dict(
        b_chin_pre_diag
    )


    a_absorption_post = clamp(
        a_absorption_pre
        + (
            ABSORPTION_UPDATE_WEIGHT
            * a_k
            * a_absorption_surprise
        ),
        MIN_RATING,
        MAX_RATING,
    )


    b_absorption_post = clamp(
        b_absorption_pre
        + (
            ABSORPTION_UPDATE_WEIGHT
            * b_k
            * b_absorption_surprise
        ),
        MIN_RATING,
        MAX_RATING,
    )


    # Apply only after all calculations used frozen PRE-fight values.
    power[a_id] = (
        a_power_post
    )

    power[b_id] = (
        b_power_post
    )

    damage_absorption[a_id] = (
        a_absorption_post
    )

    damage_absorption[b_id] = (
        b_absorption_post
    )


    # -----------------------------------------------------------------
    # Record NEW KO/TKO events only after this fight has been evaluated.
    #
    # This preserves leakage safety:
    # the current fight never influences its own pre-fight chin rating.
    # -----------------------------------------------------------------

    if float(a["ko_loss"]) > 0.0:
        chin_events[a_id].append(
            {
                "age_at_ko":
                    a_age,

                "fight_index":
                    fight_counts[a_id],
            }
        )

    if float(b["ko_loss"]) > 0.0:
        chin_events[b_id].append(
            {
                "age_at_ko":
                    b_age,

                "fight_index":
                    fight_counts[b_id],
            }
        )


    fight_counts[a_id] += 1
    fight_counts[b_id] += 1


    # -------------------------------------------------------------------------
    # Audit rows
    # -------------------------------------------------------------------------

    for (
        row,
        fighter_id,
        power_pre,
        power_post,
        chin_pre,
        chin_post,
        absorption_pre,
        absorption_post,
        acute_diag,
        chin_diag,
        absorption_diag,
    ) in (
        (
            a,
            a_id,
            a_power_pre,
            a_power_post,
            a_chin_pre,
            a_chin_post,
            a_absorption_pre,
            a_absorption_post,
            a_acute_diag,
            a_chin_diag,
            a_abs_diag,
        ),
        (
            b,
            b_id,
            b_power_pre,
            b_power_post,
            b_chin_pre,
            b_chin_post,
            b_absorption_pre,
            b_absorption_post,
            b_acute_diag,
            b_chin_diag,
            b_abs_diag,
        ),
    ):

        rating_history.append(
            {
                "fight_id":
                    fight_id,

                "event_date":
                    fight.event_date,

                "fighter_id":
                    fighter_id,

                "fighter_name":
                    row[
                        "fighter_name"
                    ],

                "opponent_id":
                    row[
                        "opponent_id"
                    ],

                "opponent_name":
                    row[
                        "opponent_name"
                    ],

                "finishing_power_pre":
                    power_pre,

                "finishing_power_post":
                    power_post,

                "chin_resistance_pre":
                    chin_pre,

                "chin_resistance_post":
                    chin_post,

                "damage_absorption_pre":
                    absorption_pre,

                "damage_absorption_post":
                    absorption_post,

                **acute_diag,
                **chin_diag,
                **absorption_diag,
            }
        )


# =============================================================================
# Target cards
# =============================================================================

print()
print("=" * 100)
print(
    "FSR V1.2 PRE-FIGHT "
    "POWER / CHIN / DAMAGE ABSORPTION"
)
print("=" * 100)


target_card_rows = []


for fighter_id in target_ids:

    name = target_names[
        fighter_id
    ]

    print()

    print(
        f"{name} "
        f"({fight_counts[fighter_id]} "
        "prior UFC fights)"
    )


    print(
        "finishing_power      ",
        f"{power[fighter_id]:.2f}",
    )

    target_chin, _ = (
        calculate_chin_resistance(
            fight_count=(
                fight_counts[
                    fighter_id
                ]
            ),
            ko_events=(
                chin_events[
                    fighter_id
                ]
            ),
        )
    )

    print(
        "chin_resistance      ",
        f"{target_chin:.2f}",
    )

    print(
        "damage_absorption    ",
        (
            f"{damage_absorption[fighter_id]:.2f}"
        ),
    )

    target_card_rows.append(
        {
            "fighter_id": fighter_id,
            "fighter_name": name,
            "prior_ufc_fights": fight_counts[fighter_id],
            "finishing_power": power[fighter_id],
            "chin_resistance": target_chin,
            "damage_absorption": damage_absorption[fighter_id],
        }
    )


# =============================================================================
# Audit artifact
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


target_card_path = (
    OUTPUT_DIR
    / (
        f"fsr_{TARGET_FIGHT_ID}"
        "_power_chin_absorption_v1_2_target_card.csv"
    )
)

pd.DataFrame(
    target_card_rows
).to_csv(
    target_card_path,
    index=False,
)

print()
print("Saved V1.2 target pre-fight card:")
print(target_card_path)


output_path = (
    OUTPUT_DIR
    / (
        f"fsr_{TARGET_FIGHT_ID}"
        "_power_chin_absorption_v1_2_history.csv"
    )
)


pd.DataFrame(
    rating_history
).to_csv(
    output_path,
    index=False,
)


print()
print(
    "Saved V1.2 power/chin/"
    "damage-absorption history:"
)

print(
    output_path
)
