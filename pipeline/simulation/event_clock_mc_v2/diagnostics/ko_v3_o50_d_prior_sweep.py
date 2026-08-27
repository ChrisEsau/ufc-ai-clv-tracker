"""Nine-fight KO V3 shadow with offense prior fixed S50 and defender prior swept.

Research only. Set D_PRIOR_STRENGTH in the environment. All other cohort
mechanics remain frozen; production is untouched.
"""
from __future__ import annotations
import os

from pipeline.simulation.event_clock_mc_v2.diagnostics import ko_v3_age_total_ko_nine_fight_cohort as cohort

O_PRIOR_STRENGTH = 50.0
D_PRIOR_STRENGTH = float(os.environ.get("D_PRIOR_STRENGTH", "50"))


def total_hazards_for_fight(frame, fight_id, cutoff, beta_att, beta_def):
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

        p_att = (att_k + O_PRIOR_STRENGTH * p0) / (att_n + O_PRIOR_STRENGTH)
        p_def = (def_k + D_PRIOR_STRENGTH * p0) / (def_n + D_PRIOR_STRENGTH)
        pre_age = float(
            cohort.sigmoid(
                cohort.logit(p0)
                + (cohort.logit(p_att) - cohort.logit(p0))
                + (cohort.logit(p_def) - cohort.logit(p0))
            )
        )
        delta = beta_att * (float(row.attacker_age) - 30.0) + beta_def * (float(row.defender_age) - 30.0)
        p_age = float(cohort.sigmoid(cohort.logit(pre_age) + delta))
        out[str(row.fighter_id)] = {
            "fighter_name": str(row.fighter_name),
            "attacker_age": float(row.attacker_age),
            "defender_age": float(row.defender_age),
            "population_ko_per_sig": p0,
            "prior_strength_sig_strikes": O_PRIOR_STRENGTH,
            "offense_prior_strength_sig_strikes": O_PRIOR_STRENGTH,
            "defense_prior_strength_sig_strikes": D_PRIOR_STRENGTH,
            "raw_attacker_ko_per_sig": raw_att,
            "raw_defender_ko_loss_per_sig": raw_def,
            "literal_union_raw_total_ko_per_landed": literal_union,
            "shrunk_attacker_ko_per_sig": p_att,
            "shrunk_defender_ko_loss_per_sig": p_def,
            "pre_age_total_ko_per_landed": pre_age,
            "age_logodds_delta": delta,
            "total_ko_per_landed": p_age,
            **p0_audit,
        }
    return out


cohort.KO_PRIOR_STRENGTH = O_PRIOR_STRENGTH
cohort.total_hazards_for_fight = total_hazards_for_fight

if __name__ == "__main__":
    cohort.main()
