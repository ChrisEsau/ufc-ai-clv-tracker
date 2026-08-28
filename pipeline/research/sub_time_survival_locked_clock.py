"""Fight-agnostic locked submission finish hazard inputs.

Uses the OOS-selected submission survival architecture from
pipeline.research.sub_time_survival_oos: fighter offense x opponent submission
vulnerability with a piecewise 5-minute population baseline and 1.0 equivalent
prior event of EB shrinkage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.research import sub_time_survival_oos as surv

PRIOR_EVENTS = 1.0


def time_clock_inputs(fight_id: str):
    ff = surv.add_prefight(surv.load_fighter_fights())
    target = ff[ff.fight_id.astype(str).eq(str(fight_id))].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected two target fighter rows, got {len(target)}")
    cutoff = pd.Timestamp(target.event_date.iloc[0]).normalize()
    train = ff[ff.event_date < cutoff].copy()
    p0, piece = surv.train_baselines(train)
    prior_sec = PRIOR_EVENTS / p0
    by_name = {}
    for r in target.itertuples(index=False):
        att_rate = (float(r.prior_sub_win) + PRIOR_EVENTS) / (float(r.prior_seconds) + prior_sec)
        def_rate = (float(r.opp_prior_sub_loss) + PRIOR_EVENTS) / (float(r.opp_prior_seconds) + prior_sec)
        rr = float(np.clip(att_rate * def_rate / (p0 * p0), 0.05, 20.0))
        hazards = np.asarray(piece, float) * rr
        by_name[str(r.fighter_name)] = {
            "prior_sub_wins": float(r.prior_sub_win),
            "prior_seconds": float(r.prior_seconds),
            "opponent_prior_sub_losses": float(r.opp_prior_sub_loss),
            "opponent_prior_seconds": float(r.opp_prior_seconds),
            "attacker_rate_per_minute": float(att_rate * 60.0),
            "defender_vulnerability_per_minute": float(def_rate * 60.0),
            "rate_ratio": rr,
            "hazards_per_second": hazards,
        }
    return cutoff, float(p0), np.asarray(piece, float), by_name
