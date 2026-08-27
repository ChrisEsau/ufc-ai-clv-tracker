"""Research-only exact prefight KO time-clock values for Allen vs Shahbazyan."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pipeline.research import ko_time_survival_oos as s

FIGHT_ID = "419fff06f338f5c6"
PRIOR_EVENTS = 2.0


def main():
    ff = s.add_prefight(s.load_fighter_fights())
    target = ff[ff.fight_id.astype(str).eq(FIGHT_ID)].copy()
    if len(target) != 2:
        raise RuntimeError(f"expected 2 target rows, got {len(target)}")
    cutoff = pd.Timestamp(target.event_date.iloc[0]).normalize()
    train = ff[ff.event_date < cutoff].copy()
    p0, piece = s.train_baselines(train)
    prior_sec = PRIOR_EVENTS / p0
    rows = []
    for r in target.itertuples(index=False):
        att_rate = (float(r.prior_ko_win) + PRIOR_EVENTS) / (float(r.prior_seconds) + prior_sec)
        def_rate = (float(r.opp_prior_ko_loss) + PRIOR_EVENTS) / (float(r.opp_prior_seconds) + prior_sec)
        rr = float(np.clip(att_rate * def_rate / (p0 * p0), 0.05, 20.0))
        fighter_piece = piece * rr
        # conditional probability of a KO during each full 5-min interval if still alive at interval start
        interval_p = 1.0 - np.exp(-fighter_piece * 300.0)
        cumhaz = np.cumsum(fighter_piece * 300.0)
        cumulative_p = 1.0 - np.exp(-cumhaz)
        rows.append({
            "fighter": str(r.fighter_name),
            "opponent_id": str(r.opponent_id),
            "prior_ko_wins": float(r.prior_ko_win),
            "prior_fight_seconds": float(r.prior_seconds),
            "opponent_prior_ko_losses": float(r.opp_prior_ko_loss),
            "opponent_prior_fight_seconds": float(r.opp_prior_seconds),
            "population_hazard_per_second": float(p0),
            "population_hazard_per_minute": float(p0*60.0),
            "attacker_ko_rate_per_second": float(att_rate),
            "attacker_ko_rate_per_minute": float(att_rate*60.0),
            "defender_vulnerability_rate_per_second": float(def_rate),
            "defender_vulnerability_rate_per_minute": float(def_rate*60.0),
            "matchup_rate_ratio": rr,
            **{f"baseline_r{i+1}_per_second": float(piece[i]) for i in range(5)},
            **{f"matchup_r{i+1}_hazard_per_second": float(fighter_piece[i]) for i in range(5)},
            **{f"matchup_r{i+1}_conditional_ko_prob": float(interval_p[i]) for i in range(5)},
            **{f"cumulative_ko_prob_through_r{i+1}": float(cumulative_p[i]) for i in range(5)},
        })
    out = pd.DataFrame(rows)
    outdir = s.OUT / "allen_shahbazyan_values"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "allen_shahbazyan_ko_time_values.csv", index=False)
    payload = {"fight_id":FIGHT_ID,"cutoff":str(cutoff.date()),"prior_events":PRIOR_EVENTS,"rows":rows}
    (outdir / "allen_shahbazyan_ko_time_values.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
