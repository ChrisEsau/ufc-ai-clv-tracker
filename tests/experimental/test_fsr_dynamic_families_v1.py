from __future__ import annotations

import pandas as pd

from scripts.experimental import fsr_dynamic_families_v1 as fsr


def _round_rows(*, adversity: bool = False) -> pd.DataFrame:
    rows = []
    fighter_values = [
        (20, 10, 30, 15, 2, 1, 60, 0, 2, 0),
        (10, 5, 15, 8, 1, 0, 30, 0, 1, 0),
        (18, 9, 26, 13, 2, 1, 50, 0, 2, 0),
    ]
    opponent_values = [
        (15, 7, 20, 10, 1, 0, 20, 0, 1, 0),
        (15, 7, 20, 10, 1, 0, 20, 1 if adversity else 0, 1, 0),
        (15, 7, 20, 10, 1, 0, 20, 0, 1, 0),
    ]

    for fighter_id, values in (("A", fighter_values), ("B", opponent_values)):
        for round_number, values_for_round in enumerate(values, start=1):
            (
                sig_attempted,
                sig_landed,
                total_attempted,
                total_landed,
                td_attempted,
                td_landed,
                ctrl_sec,
                kd,
                head_landed,
                ground_landed,
            ) = values_for_round
            rows.append(
                {
                    "fight_id": "F1",
                    "fighter_id": fighter_id,
                    "round": round_number,
                    "sig_str_landed": sig_landed,
                    "sig_str_attempted": sig_attempted,
                    "total_str_landed": total_landed,
                    "total_str_attempted": total_attempted,
                    "td_landed": td_landed,
                    "td_attempted": td_attempted,
                    "ctrl_sec": ctrl_sec,
                    "kd": kd,
                    "head_landed": head_landed,
                    "ground_landed": ground_landed,
                }
            )
    return pd.DataFrame(rows)


def test_non_adversity_recovery_detects_dip_and_rebound() -> None:
    observations = fsr.build_non_adversity_recovery_observations(
        _round_rows(adversity=False)
    )
    fighter = observations.loc[observations["fighter_id"] == "A"].iloc[0]

    assert fighter["recovery_opportunities"] == 1
    assert 0.0 <= fighter["recovery_workload"] <= 1.0
    assert 0.0 <= fighter["recovery_output"] <= 1.0
    assert fighter["recovery_quality"] > 0.0


def test_adversity_contaminated_triplet_is_not_general_recovery() -> None:
    observations = fsr.build_non_adversity_recovery_observations(
        _round_rows(adversity=True)
    )
    fighter = observations.loc[observations["fighter_id"] == "A"].iloc[0]

    assert fighter["recovery_opportunities"] == 0
    assert fighter["recovery_quality"] == 0.0


def test_one_round_fight_has_no_fatigue_evidence() -> None:
    row = pd.Series({column: 0.0 for column in fsr.C.values()})
    row[fsr.C["rounds"]] = 1.0
    pools = {key: [] for key in fsr.POOL_KEYS}

    bundle = fsr.observation_bundle(row, pools)

    assert bundle["fatigue_accumulation_resistance"][1] == 0.0
    assert bundle["fatigue_performance_resilience"][1] == 0.0


def test_dynamic_expectation_is_population_centered() -> None:
    assert fsr.expected_probability(50.0, 0.35) == 0.35
