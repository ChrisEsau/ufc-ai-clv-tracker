"""Run nine-fight KO V3 cohort with O50 + positive-only D50 + age.

Research-only wrapper. Production remains untouched.

KO architecture:
    logit(h) = logit(p_att_O50)
             + max(0, logit(p_def_D50) - logit(population))
             + chronological age delta

Defender susceptibility may increase KO hazard but can never suppress attacker
KO offense below its own O50 base (before age adjustment).
"""
from __future__ import annotations

import numpy as np

from pipeline.simulation.event_clock_mc_v2.diagnostics import ko_v3_age_total_ko_nine_fight_cohort as cohort

O_STRENGTH = 50.0
D_STRENGTH = 50.0
cohort.KO_PRIOR_STRENGTH = O_STRENGTH


def positive_only_total_hazards_for_fight(frame, fight_id, cutoff, beta_att, beta_def):
    target = frame[frame.fight_id.astype(str).eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"Expected 2 rows for {fight_id}, got {len(target)}")
    p0, p0_audit = cohort.population_prior(frame, cutoff)
    out = {}
    for row in target.itertuples(index=False):
        att_n = float(row.prior_sig_landed)
        def_n = float(row.opp_prior_sig_absorbed)
        att_k = float(row.prior_ko_wins)
        def_k = float(row.opp_prior_ko_losses)
        raw_att = att_k / att_n if att_n > 0 else 0.0
        raw_def = def_k / def_n if def_n > 0 else 0.0
        literal_union = 1.0 - (1.0 - raw_att) * (1.0 - raw_def)

        p_att = (att_k + O_STRENGTH * p0) / (att_n + O_STRENGTH)
        p_def = (def_k + D_STRENGTH * p0) / (def_n + D_STRENGTH)
        defender_delta = cohort.logit(p_def) - cohort.logit(p0)
        positive_delta = max(0.0, defender_delta)
        pre_age = float(cohort.sigmoid(cohort.logit(p_att) + positive_delta))
        age_delta = beta_att * (float(row.attacker_age) - 30.0) + beta_def * (float(row.defender_age) - 30.0)
        p_age = float(cohort.sigmoid(cohort.logit(pre_age) + age_delta))

        out[str(row.fighter_id)] = {
            "fighter_name": str(row.fighter_name),
            "attacker_age": float(row.attacker_age),
            "defender_age": float(row.defender_age),
            "population_ko_per_sig": p0,
            "prior_strength_sig_strikes": O_STRENGTH,
            "defender_prior_strength_sig_strikes": D_STRENGTH,
            "raw_attacker_ko_per_sig": raw_att,
            "raw_defender_ko_loss_per_sig": raw_def,
            "literal_union_raw_total_ko_per_landed": literal_union,
            "shrunk_attacker_ko_per_sig": p_att,
            "shrunk_defender_ko_loss_per_sig": p_def,
            "defender_logit_delta": defender_delta,
            "defender_positive_only_logit_delta": positive_delta,
            "pre_age_total_ko_per_landed": pre_age,
            "age_logodds_delta": age_delta,
            "total_ko_per_landed": p_age,
            **p0_audit,
        }
    return out


cohort.total_hazards_for_fight = positive_only_total_hazards_for_fight

if __name__ == "__main__":
    cohort.main()
