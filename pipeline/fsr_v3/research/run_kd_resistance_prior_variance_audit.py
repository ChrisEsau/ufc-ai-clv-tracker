"""Run KD-resistance audit from raw data in the validated V3 POWER regime."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.fsr_v3.research import active_trait_audit_sources as sources
from pipeline.fsr_v3.research import kd_resistance_prior_variance_audit as audit

POWER_VALIDATED_START = pd.Timestamp("2020-01-01")


def _validated_power_observations():
    obs = sources.physical_observations().copy()
    obs["event_date"] = obs["date"]

    legacy = sources.legacy_physical_prefight()[
        ["date", "fight_id", "fighter_id", "knockdown_resistance"]
    ].rename(columns={"date": "event_date", "knockdown_resistance": "legacy_kdres"})
    power = sources.v3_power_prefight()[
        ["event_date", "fight_id", "fighter_id", "striking_power_v3", "validated_regime"]
    ]
    own_power = power.rename(columns={
        "fighter_id": "opponent_id",
        "striking_power_v3": "attacker_power_v3",
        "validated_regime": "attacker_power_validated",
    })

    obs = obs.merge(
        legacy,
        on=["event_date", "fight_id", "fighter_id"],
        how="inner",
        validate="one_to_one",
    )
    obs = obs.merge(
        own_power,
        on=["event_date", "fight_id", "opponent_id"],
        how="inner",
        validate="one_to_one",
    )

    master = sources.raw_master()[["fight_id", "date", "r_id", "b_id", "r_dob", "b_dob"]].copy()
    age_rows = []
    for row in master.itertuples(index=False):
        date = pd.Timestamp(row.date).normalize()
        age_rows.extend([
            {
                "fight_id": str(row.fight_id),
                "fighter_id": str(row.r_id),
                "age": audit.fighter_age_years(row.r_dob, date),
            },
            {
                "fight_id": str(row.fight_id),
                "fighter_id": str(row.b_id),
                "age": audit.fighter_age_years(row.b_dob, date),
            },
        ])
    ages = pd.DataFrame(age_rows)
    obs = obs.merge(
        ages.rename(columns={"age": "defender_age"}),
        on=["fight_id", "fighter_id"],
        how="left",
        validate="many_to_one",
    )
    obs = obs.merge(
        ages.rename(columns={"fighter_id": "opponent_id", "age": "attacker_age"}),
        on=["fight_id", "opponent_id"],
        how="left",
        validate="many_to_one",
    )

    obs["k"] = pd.to_numeric(obs["kd_absorbed"], errors="coerce").fillna(0.0)
    obs["n"] = pd.to_numeric(obs["sig_absorbed"], errors="coerce").fillna(0.0)
    obs["k"] = np.minimum(obs["k"].clip(lower=0.0), obs["n"].clip(lower=0.0))
    obs = obs[
        (obs["n"] > 0.0)
        & obs["attacker_power_validated"].astype(bool)
        & (obs["event_date"] >= POWER_VALIDATED_START)
    ].copy()

    appearances = obs[["event_date", "fighter_id"]].drop_duplicates()
    counts = (
        appearances.groupby(["fighter_id", "event_date"], as_index=False)
        .size()
        .sort_values(["fighter_id", "event_date"])
    )
    counts["prior_ufc_fights"] = (
        counts.groupby("fighter_id")["size"].cumsum() - counts["size"]
    )
    obs = obs.merge(
        counts[["fighter_id", "event_date", "prior_ufc_fights"]],
        on=["fighter_id", "event_date"],
        how="left",
        validate="many_to_one",
    )
    obs["prior_bucket"] = obs["prior_ufc_fights"].map(audit._bucket)

    c = audit.ShadowKOKDCalibration()
    obs["context_offset"] = (
        obs["attacker_power_v3"].astype(float)
        + c.kd_attacker_age_beta * (obs["attacker_age"].astype(float) - 30.0)
        + c.kd_defender_age_beta * (obs["defender_age"].astype(float) - 30.0)
    )
    return obs.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def main():
    audit.build_observations = _validated_power_observations
    audit.main()


if __name__ == "__main__":
    main()
