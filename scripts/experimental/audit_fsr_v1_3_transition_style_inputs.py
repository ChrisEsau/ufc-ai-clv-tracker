"""Audit leakage-safe style/tendency inputs for FSR/MC V1.3 transitions.

Shadow/research only.

This script changes no simulator, FSR, adapter, calibration, or data artifacts.
It inspects the point-in-time Phase Baseline state that already exists on each
historical Round Fighter State row entering a target fight.

Purpose
-------
The V1.3 archetype batch showed weak phase differentiation.  Before changing
transition calibration, verify whether the existing leakage-safe RFS state
contains the missing *propensity/style* information that should remain distinct
from persistent FSR skill.

The audited fields are evidence about phase tendency, not exact phase seconds:
- distance/clinch/ground significant-strike attempt shares;
- takedown attempts per observed round;
- failed takedown attempts per observed round;
- takedown persistence ratio;
- control seconds per observed round;
- non-distance clinch/ground shares.

History rows are already point-in-time PRE-fight state, so no current-fight
observation is used.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROUND_STATS_PATH = Path("data/fight_details/ufc_round_stats.parquet")
RFS_HISTORY_PATH = Path("data/features/round_fighter_state_history.parquet")

TARGET_FIGHTS = (
    ("power strikers", "7208e40818401e88"),
    ("wrestler vs striker", "3146e5a47a922976"),
    ("submission / grappling", "40e8bf8ce508c436"),
    ("high power / chin", "bca5d01f8775f852"),
    ("high-volume striker vs striker", "a4817b7e46028b4a"),
    ("wrestler vs wrestler", "31b3ae9352d9389b"),
)

STYLE_COLUMNS = (
    "rfs_phase_base_ewm_distance_attempt_share",
    "rfs_phase_base_ewm_clinch_attempt_share",
    "rfs_phase_base_ewm_ground_attempt_share",
    "rfs_phase_base_ewm_td_attempts_per_round",
    "rfs_phase_base_ewm_failed_td_attempts_per_round",
    "rfs_phase_base_ewm_td_persistence_ratio",
    "rfs_phase_base_ewm_control_seconds_per_round",
    "rfs_phase_base_ewm_non_distance_clinch_share",
    "rfs_phase_base_ewm_non_distance_ground_share",
)

DISPLAY_NAMES = {
    "rfs_phase_base_ewm_distance_attempt_share": "dist_share",
    "rfs_phase_base_ewm_clinch_attempt_share": "clinch_share",
    "rfs_phase_base_ewm_ground_attempt_share": "ground_share",
    "rfs_phase_base_ewm_td_attempts_per_round": "td_att/rnd",
    "rfs_phase_base_ewm_failed_td_attempts_per_round": "failed_td/rnd",
    "rfs_phase_base_ewm_td_persistence_ratio": "td_persist",
    "rfs_phase_base_ewm_control_seconds_per_round": "ctrl_sec/rnd",
    "rfs_phase_base_ewm_non_distance_clinch_share": "non_dist_clinch",
    "rfs_phase_base_ewm_non_distance_ground_share": "non_dist_ground",
}


def main() -> None:
    if not ROUND_STATS_PATH.exists():
        raise FileNotFoundError(ROUND_STATS_PATH)
    if not RFS_HISTORY_PATH.exists():
        raise FileNotFoundError(RFS_HISTORY_PATH)

    rounds = pd.read_parquet(
        ROUND_STATS_PATH,
        columns=[
            "fight_id",
            "fighter_id",
            "fighter_name",
            "event_date",
            "corner",
        ],
    )
    rounds["fight_id"] = rounds["fight_id"].astype(str)
    rounds["fighter_id"] = rounds["fighter_id"].astype(str)

    history = pd.read_parquet(RFS_HISTORY_PATH)
    history["fight_id"] = history["fight_id"].astype(str)
    history["fighter_id"] = history["fighter_id"].astype(str)

    missing = [column for column in STYLE_COLUMNS if column not in history.columns]
    if missing:
        raise RuntimeError(
            "RFS history is missing expected Phase Baseline prior-state columns: "
            f"{missing}"
        )

    rows: list[dict[str, object]] = []

    for archetype, fight_id in TARGET_FIGHTS:
        target = rounds.loc[rounds["fight_id"] == fight_id].copy()
        if target.empty:
            print(f"SKIP {fight_id}: target fight not found in round stats")
            continue

        fighters = (
            target[["fighter_id", "fighter_name", "corner", "event_date"]]
            .drop_duplicates(subset=["fighter_id"])
            .copy()
        )

        for _, fighter in fighters.iterrows():
            fighter_id = str(fighter["fighter_id"])
            state_rows = history.loc[
                (history["fight_id"] == fight_id)
                & (history["fighter_id"] == fighter_id)
            ].copy()

            if len(state_rows) != 1:
                raise RuntimeError(
                    "Expected exactly one point-in-time RFS history row for "
                    f"fight={fight_id}, fighter={fighter_id}; found {len(state_rows)}"
                )

            state = state_rows.iloc[0]
            row: dict[str, object] = {
                "archetype": archetype,
                "fight_id": fight_id,
                "event_date": str(pd.Timestamp(fighter["event_date"]).date()),
                "corner": str(fighter["corner"]),
                "fighter_name": str(fighter["fighter_name"]),
                "fighter_id": fighter_id,
            }

            for column in STYLE_COLUMNS:
                row[DISPLAY_NAMES[column]] = pd.to_numeric(
                    pd.Series([state[column]]), errors="coerce"
                ).iloc[0]

            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No target fighter style rows were resolved.")

    print()
    print("=" * 154)
    print("FSR / MC V1.3 — LEAKAGE-SAFE TRANSITION STYLE INPUT AUDIT")
    print("=" * 154)
    print(
        "These are PRE-fight RFS EWM tendency states. They are not FSR skills "
        "and are not exact phase-time estimates."
    )
    print()

    display_columns = [
        "archetype",
        "fighter_name",
        "dist_share",
        "clinch_share",
        "ground_share",
        "td_att/rnd",
        "failed_td/rnd",
        "td_persist",
        "ctrl_sec/rnd",
        "non_dist_clinch",
        "non_dist_ground",
    ]

    print(
        result[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print()
    print("=" * 154)
    print("PAIRWISE STYLE CONTRASTS")
    print("=" * 154)

    for archetype, group in result.groupby("archetype", sort=False):
        print()
        print(archetype)
        for _, row in group.iterrows():
            print(
                f"  {row['fighter_name']:<22} "
                f"TD/rnd={row['td_att/rnd']:.3f}  "
                f"DistShare={row['dist_share']:.3f}  "
                f"Ctrl/rnd={row['ctrl_sec/rnd']:.1f}  "
                f"GroundShare={row['ground_share']:.3f}"
            )

    output = Path(
        "data/simulation/rfs_mc_v2_shared_state/"
        "fsr_v1_3_transition_style_input_audit.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
