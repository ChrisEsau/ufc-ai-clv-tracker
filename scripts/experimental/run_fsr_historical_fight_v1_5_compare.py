"""Historical fight comparison for the provisional FSR/MC V1.5 activity stack.

Shadow/research only.

V1.5 intentionally leaves the V1.4 transition/wrestling/submission checkpoint
untouched and changes only the activity adapter:

- RFS Phase Baseline EWM distance attempts/round controls distance strike pace;
- RFS Phase Baseline EWM clinch attempts/round controls clinch strike pace;
- RFS Phase Baseline EWM ground attempts/round controls ground-owner strike pace;
- RFS Phase Baseline EWM control seconds/round controls expected control time
  once the fighter owns a clinch/ground segment;
- FSR Control Imposition vs Control Resistance opponent-adjusts control time;
- FSR Distance Precision vs Distance Defense still controls distance accuracy;
- V1.4 style transitions, x1.75 two-stage chain wrestling, V1.3 KO hazards,
  submission attempt generation 1.00x, submission hazard 0.12, cardio,
  dynamics, scoring, and judging remain unchanged.

The historical whole-round RFS activity observations are converted to the
per-active-30-second units consumed by MC V2 using the same neutral V1.4 phase
exposure reference introduced in V1.2/V1.4. This is an adapter experiment, not
an assertion that UFCStats exposes exact phase seconds.

Usage
-----
PYTHONPATH=. python \
    scripts/experimental/run_fsr_historical_fight_v1_5_compare.py \
    <fight_id> --simulations 500
"""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from pathlib import Path

import pandas as pd

from pipeline.common.fight_time import repair_elapsed_match_time

from scripts.experimental import run_fsr_historical_fight_v1 as base
from scripts.experimental import run_fsr_historical_fight_v1_4_compare as v1_4_compare
from scripts.experimental import run_fsr_historical_fight_locked_v1_2 as v1_2
from scripts.experimental import run_fsr_v1_4_two_fight_full_validation as v1_4


RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")

V1_5_STYLE_COLUMNS = {
    "distance_attempts_per_round": (
        "rfs_phase_base_ewm_distance_attempts_per_round"
    ),
    "clinch_attempts_per_round": (
        "rfs_phase_base_ewm_clinch_attempts_per_round"
    ),
    "ground_attempts_per_round": (
        "rfs_phase_base_ewm_ground_attempts_per_round"
    ),
}

_V1_4_BUILD_FULL_CARD = None
_V1_4_BUILD_PHASE = None
_BASE_PRINT_MATCHUP_PARAMETERS = base.print_matchup_parameters
_V1_5_STYLE_HISTORY: pd.DataFrame | None = None


def _finite_nonnegative(value: object) -> float | None:
    """Return one finite nonnegative style value, else None."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(parsed) or parsed < 0.0:
        return None

    return parsed


def _load_v1_5_style_history() -> pd.DataFrame:
    """Load only the extra Phase Baseline pace fields needed by V1.5."""

    global _V1_5_STYLE_HISTORY

    if _V1_5_STYLE_HISTORY is None:
        history = pd.read_parquet(
            RFS_HISTORY_PATH,
            columns=[
                "fight_id",
                "fighter_id",
                *V1_5_STYLE_COLUMNS.values(),
            ],
        )
        history["fight_id"] = history["fight_id"].astype(str)
        history["fighter_id"] = history["fighter_id"].astype(str)
        _V1_5_STYLE_HISTORY = history

    return _V1_5_STYLE_HISTORY


def build_full_card_v1_5(
    fight_id: str,
    fighter_id: str,
) -> tuple[dict[str, float], int]:
    """Extend the V1.4 card with leakage-safe Phase Baseline pace state."""

    if _V1_4_BUILD_FULL_CARD is None:
        raise RuntimeError("V1.5 adapter was not installed before card build")

    card, fight_count = _V1_4_BUILD_FULL_CARD(
        fight_id,
        fighter_id,
    )

    history = _load_v1_5_style_history()
    rows = history.loc[
        (history["fight_id"] == str(fight_id))
        & (history["fighter_id"] == str(fighter_id))
    ]

    if len(rows) != 1:
        raise RuntimeError(
            "Expected exactly one V1.5 RFS style row for "
            f"fight={fight_id}, fighter={fighter_id}; found {len(rows)}"
        )

    row = rows.iloc[0]
    out = dict(card)

    for short_name, column in V1_5_STYLE_COLUMNS.items():
        value = _finite_nonnegative(row[column])
        out[f"style_{short_name}"] = (
            float("nan") if value is None else value
        )

    return out, fight_count


def _phase_conditioned_rate(
    *,
    whole_round_rate: object,
    reference_segments_per_round: float,
    fallback: float,
) -> float:
    """Convert one whole-round tendency to a per-active-segment MC rate."""

    selected = _finite_nonnegative(whole_round_rate)

    if selected is None:
        return max(0.0, float(fallback))

    denominator = float(reference_segments_per_round)
    if not isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(
            "V1.5 phase-conditioning denominator must be positive and finite"
        )

    return selected / denominator


def _control_means_v1_5(
    *,
    fighter: dict[str, float],
    opponent: dict[str, float],
    baselines: dict[str, float],
    fallback_clinch: float,
    fallback_ground: float,
) -> tuple[float, float]:
    """Convert RFS control tendency into matchup-adjusted owner-segment means.

    UFCStats records combined control time without exact clinch/ground
    allocation. Preserve the existing adapter's neutral 6:8 clinch-to-ground
    control-duration shape, then choose the common scale so the fighter's RFS
    control-seconds-per-round tendency is reproduced under neutral V1.4 owner
    exposure before opponent adjustment.

    The final means remain bounded by the physical 30-second segment contract.
    """

    control_per_round = _finite_nonnegative(
        fighter.get("style_control_seconds_per_round")
    )

    if control_per_round is None:
        return float(fallback_clinch), float(fallback_ground)

    clinch_owner_segments = (
        float(baselines["reference_clinch_segments_per_round"])
        / 2.0
    )
    ground_owner_segments = float(
        baselines["reference_ground_owner_segments_per_fighter_round"]
    )

    weighted_reference = (
        6.0 * clinch_owner_segments
        + 8.0 * ground_owner_segments
    )

    if not isfinite(weighted_reference) or weighted_reference <= 0.0:
        raise RuntimeError(
            "V1.5 control reference exposure must be positive and finite"
        )

    control_scale = control_per_round / weighted_reference
    clinch_style_mean = 6.0 * control_scale
    ground_style_mean = 8.0 * control_scale

    # Skill remains an opponent-adjusted execution layer. At equal 50-centered
    # ratings matchup_rate() preserves the RFS behavioral baseline exactly.
    clinch_matchup_mean = base.matchup_rate(
        baseline=clinch_style_mean,
        offense_rating=fighter["control_imposition"],
        defense_rating=opponent["control_resistance"],
    )
    ground_matchup_mean = base.matchup_rate(
        baseline=ground_style_mean,
        offense_rating=fighter["control_imposition"],
        defense_rating=opponent["control_resistance"],
    )

    return (
        max(0.0, min(30.0, clinch_matchup_mean)),
        max(0.0, min(30.0, ground_matchup_mean)),
    )


def build_phase_v1_5(
    fighter: dict[str, float],
    opponent: dict[str, float],
    baselines: dict[str, float],
):
    """Apply fighter-specific RFS activity tendency to the V1.4 phase adapter."""

    if _V1_4_BUILD_PHASE is None:
        raise RuntimeError("V1.5 adapter was not installed before phase build")

    phase = _V1_4_BUILD_PHASE(
        fighter,
        opponent,
        baselines,
    )

    distance_rate = _phase_conditioned_rate(
        whole_round_rate=fighter.get("style_distance_attempts_per_round"),
        reference_segments_per_round=baselines[
            "reference_distance_segments_per_round"
        ],
        fallback=phase.distance.sig_strike_attempt_rate,
    )

    # Both fighters may strike during every clinch segment, so use total
    # neutral clinch segments as the per-fighter denominator.
    clinch_rate = _phase_conditioned_rate(
        whole_round_rate=fighter.get("style_clinch_attempts_per_round"),
        reference_segments_per_round=baselines[
            "reference_clinch_segments_per_round"
        ],
        fallback=phase.clinch.clinch_strike_attempt_rate,
    )

    # The current MC contract permits ground strikes only for the authoritative
    # owner, so condition historical ground pace on neutral owner exposure.
    ground_rate = _phase_conditioned_rate(
        whole_round_rate=fighter.get("style_ground_attempts_per_round"),
        reference_segments_per_round=baselines[
            "reference_ground_owner_segments_per_fighter_round"
        ],
        fallback=phase.ground_owner.ground_strike_attempt_rate,
    )

    clinch_control, ground_control = _control_means_v1_5(
        fighter=fighter,
        opponent=opponent,
        baselines=baselines,
        fallback_clinch=phase.clinch.control_seconds_mean,
        fallback_ground=phase.ground_owner.control_seconds_mean,
    )

    return replace(
        phase,
        distance=replace(
            phase.distance,
            sig_strike_attempt_rate=distance_rate,
        ),
        clinch=replace(
            phase.clinch,
            clinch_strike_attempt_rate=clinch_rate,
            control_seconds_mean=clinch_control,
        ),
        ground_owner=replace(
            phase.ground_owner,
            ground_strike_attempt_rate=ground_rate,
            control_seconds_mean=ground_control,
        ),
    )


def print_matchup_parameters_v1_5(
    name: str,
    card: dict[str, float],
    transition,
    phase,
    dynamic,
) -> None:
    """Print existing matchup diagnostics plus V1.5 activity-style inputs."""

    _BASE_PRINT_MATCHUP_PARAMETERS(
        name,
        card,
        transition,
        phase,
        dynamic,
    )

    print(
        "RFS distance att/rnd   :",
        round(
            float(card.get("style_distance_attempts_per_round", float("nan"))),
            4,
        ),
    )
    print(
        "RFS clinch att/rnd     :",
        round(
            float(card.get("style_clinch_attempts_per_round", float("nan"))),
            4,
        ),
    )
    print(
        "RFS ground att/rnd     :",
        round(
            float(card.get("style_ground_attempts_per_round", float("nan"))),
            4,
        ),
    )
    print(
        "RFS control sec/rnd    :",
        round(
            float(card.get("style_control_seconds_per_round", float("nan"))),
            4,
        ),
    )
    print(
        "V1.5 clinch att/seg    :",
        round(phase.clinch.clinch_strike_attempt_rate, 4),
    )
    print(
        "V1.5 ground att/owner  :",
        round(phase.ground_owner.ground_strike_attempt_rate, 4),
    )
    print(
        "V1.5 clinch ctrl mean  :",
        round(phase.clinch.control_seconds_mean, 3),
    )
    print(
        "V1.5 ground ctrl mean  :",
        round(phase.ground_owner.control_seconds_mean, 3),
    )


def print_actual_result_v1_5(fight_id: str) -> None:
    """Print actual result using repaired elapsed historical fight time."""

    if not v1_4_compare.MASTER_PATH.exists():
        return

    master = pd.read_parquet(v1_4_compare.MASTER_PATH)
    if "fight_id" not in master.columns:
        return

    master["fight_id"] = master["fight_id"].astype(str)
    rows = master.loc[master["fight_id"] == str(fight_id)].copy()
    if rows.empty:
        return

    rows = repair_elapsed_match_time(rows)
    row = rows.iloc[0]

    winner = row.get("winner", row.get("winner_name", None))
    method = row.get("method", None)
    finish_round = pd.to_numeric(
        pd.Series([row.get("finish_round", None)]),
        errors="coerce",
    ).iloc[0]
    elapsed = pd.to_numeric(
        pd.Series([row.get("match_time_sec", None)]),
        errors="coerce",
    ).iloc[0]

    time_text = ""
    if pd.notna(finish_round) and pd.notna(elapsed):
        round_number = int(finish_round)
        seconds_in_round = max(
            0.0,
            float(elapsed) - 300.0 * (round_number - 1),
        )
        minutes = int(seconds_in_round // 60)
        seconds = int(round(seconds_in_round - 60 * minutes))
        time_text = f" | R{round_number} {minutes}:{seconds:02d}"

    print()
    print("=" * 120)
    print("ACTUAL FIGHT RESULT")
    print("=" * 120)
    print(f"Winner: {winner} | Method: {method}{time_text}")


def run_round_comparison_diagnostics_v1_5(**kwargs) -> None:
    """Reuse V1.4 diagnostics without overwriting the preserved V1.4 CSV."""

    original_output_dir = base.OUTPUT_DIR
    v1_5_output_dir = original_output_dir / "v1_5"
    base.OUTPUT_DIR = v1_5_output_dir

    try:
        v1_4_compare.run_round_comparison_diagnostics(**kwargs)
    finally:
        base.OUTPUT_DIR = original_output_dir

    fight_id = str(kwargs["fight_id"])
    temporary_path = (
        v1_5_output_dir
        / f"fsr_{fight_id}_v1_4_round_actual_comparison.csv"
    )
    final_path = (
        v1_5_output_dir
        / f"fsr_{fight_id}_v1_5_round_actual_comparison.csv"
    )

    if not temporary_path.exists():
        raise RuntimeError(
            "V1.5 diagnostic replay did not create its expected comparison CSV: "
            f"{temporary_path}"
        )

    temporary_path.replace(final_path)

    print()
    print(f"V1.5 round comparison artifact: {final_path}")


def install_v1_5() -> None:
    """Install V1.5 only on top of the preserved V1.4 shadow checkpoint."""

    global _V1_4_BUILD_FULL_CARD, _V1_4_BUILD_PHASE

    v1_4_compare.install_v1_4()

    # Capture the complete V1.4 adapter after all of its overrides are active.
    _V1_4_BUILD_FULL_CARD = base.build_full_card
    _V1_4_BUILD_PHASE = base.build_phase

    # Retain the V1.4 neutral phase exposure denominator explicitly.
    v1_2.neutral_phase_exposure = v1_4.neutral_phase_exposure_v1_4

    base.build_full_card = build_full_card_v1_5
    base.build_phase = build_phase_v1_5
    base.print_matchup_parameters = print_matchup_parameters_v1_5
    base.run_population_diagnostics = run_round_comparison_diagnostics_v1_5

    # The reused comparison diagnostic calls this module-level display helper.
    v1_4_compare._print_actual_result = print_actual_result_v1_5


def main() -> None:
    install_v1_5()

    print()
    print("=" * 120)
    print("CURRENT SHADOW CHECKPOINT: FSR / MC V1.5 ACTIVITY-STYLE COMPARISON")
    print("=" * 120)
    print(
        f"TD scale={v1_4.ATTEMPT_SCALE:.2f} | "
        f"submission hazard={v1_4_compare.SUBMISSION_HAZARD:.2f} | "
        "RFS distance/clinch/ground pace + control tendency enabled"
    )

    # Keep the same fight_id / --simulations / --seed interface.
    base.main()


if __name__ == "__main__":
    main()
