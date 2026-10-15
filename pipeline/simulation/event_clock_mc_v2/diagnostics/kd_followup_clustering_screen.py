"""Research-only men's post-KD follow-up clustering screen.

Keeps the validated consequence-side power curve fixed:
    power_offset(t) = clip(35 - t/12, -40, 35)

No extra hurt/KO bonus is applied. Instead, after a survived KD this diagnostic
pulls the attacker's next existing strike event(s) forward into a short burst.
This preserves each path's pre-drawn strike attempt and landing budgets exactly;
it changes only temporal placement of already-existing events.

Arms:
  control  : no clustering
  cluster1 : next attacker strike moved to KD+3s
  cluster2 : next two attacker strikes moved to KD+2s and KD+5s

Frozen V1 source is not modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1 import run_event_or_fight as runner
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel as FrozenShadow
from pipeline.simulation.event_clock_mc_v2.diagnostics import canonical_c_validation as canonical
from pipeline.simulation.event_clock_mc_v2.diagnostics import shared_power_decay_grid as shared
from pipeline.simulation.event_clock_mc_v2.diagnostics import weight_class_audit as wc_audit
from pipeline.simulation.event_mc_v1.calibration import DEFAULT_RESOLVER
from pipeline.simulation.event_mc_v1.components.profiles import MatchupProfiles, Side
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.physiology import PhysiologyTimeAdvanceModel
from pipeline.simulation.event_mc_v1.stamina import StaminaModel
from pipeline.simulation.event_mc_v1.state import FightState

INTERCEPT = 35.0
DENOMINATOR = 12.0
LOWER_CAP = -40.0
UPPER_CAP = 35.0

_MODE = "control"
ARMS = {
    "control": (),
    "cluster1": (3.0,),
    "cluster2": (2.0, 5.0),
}


class ContinuousPowerShadow(FrozenShadow):
    def resolve_landed_strike(self, *, state, attacker, prior_defender_kds, rng):
        offset = float(np.clip(INTERCEPT - float(state.fight_time_seconds) / DENOMINATOR, LOWER_CAP, UPPER_CAP))

        def shifted(side: Side):
            p = self.profiles.fighter(side)
            return replace(p, striking_power=float(p.striking_power) + offset)

        model = FrozenShadow(
            profiles=MatchupProfiles(red=shifted(Side.RED), blue=shifted(Side.BLUE)),
            calibration=self.calibration,
        )
        return model.resolve_landed_strike(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
            rng=rng,
        )


def _cluster_future_strikes(events, start_idx: int, attacker: Side, now: float, horizon: float, offsets: tuple[float, ...]) -> None:
    if not offsets:
        return
    chosen = []
    for j in range(start_idx, len(events)):
        t, side, family, landed = events[j]
        if side is attacker and family in ("standing_strike", "ground_strike") and t > now:
            chosen.append(j)
            if len(chosen) == len(offsets):
                break
    if not chosen:
        return
    for j, dt in zip(chosen, offsets):
        t, side, family, landed = events[j]
        events[j] = (min(now + dt, horizon - 1e-9), side, family, landed)
    events[start_idx:] = sorted(events[start_idx:], key=lambda x: x[0])


def clustered_simulate_detailed_path(fight, budgets, submission_rates, conversion_probability, judge_model, judge_features, seed):
    global _MODE
    rng = np.random.default_rng(seed)
    horizon = float(fight.rounds * 300)
    key = fight.division if fight.division in DEFAULT_RESOLVER.weight_classes else None
    calibration = DEFAULT_RESOLVER.for_weight_class(key)

    shadow_profiles = runner.event_clock_shadow_ko_kd_profiles(fight)
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    modifiers = DynamicModifierProvider(calibration)
    shadow_ko_kd = ContinuousPowerShadow(shadow_profiles)
    time_advance = PhysiologyTimeAdvanceModel(stamina, calibration)

    state = FightState()
    events = []
    for side_name in ("red", "blue"):
        side = Side(side_name)
        runner.add_budget_events(events, side, "standing_strike", budgets[f"{side_name}_standing_attempted"], budgets[f"{side_name}_standing_landed"], horizon, rng)
        runner.add_budget_events(events, side, "ground_strike", budgets[f"{side_name}_ground_attempted"], budgets[f"{side_name}_ground_landed"], horizon, rng)
        runner.add_budget_events(events, side, "takedown", budgets[f"{side_name}_td_attempted"], budgets[f"{side_name}_td_landed"], horizon, rng)
        runner.add_submission_events(events, side, submission_rates[side_name], horizon, rng)
    events.sort(key=lambda x: x[0])

    stats = {"red": Counter(), "blue": Counter()}
    kds = {"red": 0, "blue": 0}
    next_boundary = 300.0

    def advance_to(target):
        nonlocal next_boundary
        while next_boundary < target and next_boundary < horizon:
            dt = next_boundary - state.fight_time_seconds
            runner.apply_delta(state, time_advance.advance(state, None, dt))
            state.fight_time_seconds = next_boundary
            runner.apply_delta(state, stamina.recovery_delta(state))
            next_boundary += 300.0
        dt = target - state.fight_time_seconds
        if dt > 0:
            runner.apply_delta(state, time_advance.advance(state, None, dt))
            state.fight_time_seconds = target

    i = 0
    while i < len(events):
        event_time, side, family, landed = events[i]
        if state.finished:
            break
        advance_to(event_time)
        side_name = side.value
        stats[side_name][f"{family}_attempted"] += 1
        profile = fight.profiles.fighter(side)
        modifiers.modifiers(profile, state, side)
        runner.apply_delta(state, stamina.action_delta(state, side, family))

        if family == "submission_attempt":
            if rng.random() < conversion_probability:
                state.finished = True; state.finish_reason = "SUB"; state.finish_method = "SUB"; state.winner = side_name
            i += 1; continue
        if family == "takedown":
            if landed: stats[side_name]["td_landed"] += 1
            i += 1; continue
        if landed:
            stats[side_name][f"{family}_landed"] += 1
        if landed and family in ("standing_strike", "ground_strike"):
            consequence = shadow_ko_kd.resolve_landed_strike(state=state, attacker=side, prior_defender_kds=kds[side_name], rng=rng)
            if consequence.ko_tko:
                state.finished = True; state.finish_reason = "KO_TKO"; state.finish_method = "KO_TKO"; state.winner = side_name
                break
            if consequence.knockdown:
                kds[side_name] += 1
                _cluster_future_strikes(events, i + 1, side, float(state.fight_time_seconds), horizon, ARMS[_MODE])
        i += 1

    if not state.finished:
        advance_to(horizon)
        red_sig = stats["red"]["standing_strike_landed"] + stats["red"]["ground_strike_landed"]
        blue_sig = stats["blue"]["standing_strike_landed"] + stats["blue"]["ground_strike_landed"]
        decision_row = {
            "sig_diff": red_sig - blue_sig,
            "kd_diff": kds["red"] - kds["blue"],
            "td_diff": stats["red"]["td_landed"] - stats["blue"]["td_landed"],
            "sub_diff": stats["red"]["submission_attempt_attempted"] - stats["blue"]["submission_attempt_attempted"],
            "ctrl_diff": budgets["red_control"] - budgets["blue_control"],
        }
        p_red = float(judge_model.predict_proba(pd.DataFrame([decision_row])[judge_features])[0, 1])
        state.finished = True; state.finish_reason = "DEC"; state.finish_method = "DEC"; state.winner = "red" if rng.random() < p_red else "blue"

    elapsed = float(state.fight_time_seconds)
    exposure_fraction = min(max(elapsed / max(horizon, 1.0), 0.0), 1.0)
    out = {"winner": state.winner, "method": state.finish_method, "elapsed": elapsed, "finish_round": int(max(elapsed - 1e-12, 0.0) // 300) + 1}
    for side_name in ("red", "blue"):
        sa = stats[side_name]["standing_strike_attempted"]; sl = stats[side_name]["standing_strike_landed"]
        ga = stats[side_name]["ground_strike_attempted"]; gl = stats[side_name]["ground_strike_landed"]
        out.update({
            f"{side_name}_sig_attempted": sa + ga,
            f"{side_name}_sig_landed": sl + gl,
            f"{side_name}_standing_attempted": sa,
            f"{side_name}_standing_landed": sl,
            f"{side_name}_ground_attempted": ga,
            f"{side_name}_ground_landed": gl,
            f"{side_name}_td_attempted": stats[side_name]["takedown_attempted"],
            f"{side_name}_td_landed": stats[side_name]["td_landed"],
            f"{side_name}_sub_attempts": stats[side_name]["submission_attempt_attempted"],
            f"{side_name}_kd": kds[side_name],
            f"{side_name}_control_seconds": float(budgets[f"{side_name}_control"]) * exposure_fraction,
        })
    return out


def _thin(cohort, n):
    if n <= 0 or len(cohort) <= n: return cohort.reset_index(drop=True)
    idx = np.linspace(0, len(cohort) - 1, num=n, dtype=int)
    return cohort.iloc[np.unique(idx)].reset_index(drop=True)


def _install_summary_wrapper():
    original = canonical.summarize_fight
    def wrapped(fight_id, pair, rows, master_row):
        out = original(fight_id, pair, rows, master_row)
        p = pd.DataFrame(rows); nondec = p["method"].ne("DEC")
        for threshold in (300, 600, 900):
            out[f"p_nondec_by_{threshold}"] = float((nondec & p["elapsed"].le(threshold)).mean())
        return out
    canonical.summarize_fight = wrapped


def _historical_targets(target_n):
    cohorts = [wc_audit.select_cohort(d, target_n)[0] for d in shared.MEN_DIVISIONS]
    c = pd.concat(cohorts, ignore_index=True)
    method = c["method"].map(wc_audit.normalize_method)
    elapsed = pd.to_numeric(c["match_time_sec"], errors="raise")
    return float(method.eq("KO_TKO").mean()), {t: float(((method != "DEC") & elapsed.le(t)).mean()) for t in (300,600,900)}, float(elapsed.mean())


def _run_arm(target_n, sim_n, paths, seed, arm):
    global _MODE
    _MODE = arm
    runner.simulate_detailed_path = clustered_simulate_detailed_path
    canonical.simulate_detailed_path = clustered_simulate_detailed_path
    frames = []
    for i, division in enumerate(shared.MEN_DIVISIONS):
        cohort = _thin(wc_audit.select_cohort(division, target_n)[0], sim_n)
        print(f"ARM {arm} | {division} | fights={len(cohort)} paths={paths}")
        s = canonical._simulate_c(cohort, paths, seed + i * 100_000_000)
        s["division"] = division; frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _summarize(summary, hist_ko, hist_finish, hist_elapsed, paths, arm):
    y = summary["actual_winner"].eq("red").astype(float).to_numpy(); p = summary["p_red_win"].to_numpy(float)
    winner_p = np.where(y > .5, p, 1-p)
    rec = {"arm":arm,"n_fights":len(summary),"paths_per_fight":paths,"ml_accuracy":float(summary["ml_correct"].mean()),"ml_brier":float(np.mean((p-y)**2)),"ml_logloss":float(-np.mean(np.log(np.clip(winner_p,1e-9,1)))),"method_accuracy":float(summary["method_correct"].mean()),"historical_ko_share":hist_ko,"simulated_ko_share":float(summary["p_fight_ko_tko"].mean()),"ko_share_bias":float(summary["p_fight_ko_tko"].mean()-hist_ko),"historical_mean_elapsed":hist_elapsed,"simulated_mean_elapsed":float(summary["sim_mean_elapsed"].mean()),"duration_relative_bias":float(summary["sim_mean_elapsed"].mean()/hist_elapsed-1)}
    for t in (300,600,900):
        sim=float(summary[f"p_nondec_by_{t}"].mean()); rec[f"hist_nondec_by_{t}"]=hist_finish[t]; rec[f"sim_nondec_by_{t}"]=sim; rec[f"bias_nondec_by_{t}"]=sim-hist_finish[t]
    return rec


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--target-n",type=int,default=100); ap.add_argument("--sim-n-per-division",type=int,default=20); ap.add_argument("--paths",type=int,default=10); ap.add_argument("--seed",type=int,default=20260823); ap.add_argument("--out-dir",type=Path,default=Path("data/diagnostics/event_clock_mc_v2/kd_followup_clustering_screen")); args=ap.parse_args()
    _install_summary_wrapper(); hist_ko,hist_finish,hist_elapsed=_historical_targets(args.target_n)
    metrics=[]; summaries=[]
    for arm in ARMS:
        s=_run_arm(args.target_n,args.sim_n_per_division,args.paths,args.seed,arm); s["arm"]=arm; summaries.append(s); metrics.append(_summarize(s,hist_ko,hist_finish,hist_elapsed,args.paths,arm))
    m=pd.DataFrame(metrics); all_s=pd.concat(summaries,ignore_index=True); args.out_dir.mkdir(parents=True,exist_ok=True); m.to_csv(args.out_dir/"arm_metrics.csv",index=False); all_s.to_csv(args.out_dir/"fight_summaries.csv",index=False)
    print("\nHISTORICAL MEN TARGETS"); print(f"KO share={hist_ko:.5f} | mean elapsed={hist_elapsed:.2f}s"); [print(f"nondecision by {t}s={hist_finish[t]:.5f}") for t in (300,600,900)]; print("\nARM METRICS"); print(m.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print(f"\nOUTPUT: {args.out_dir}")

if __name__ == "__main__": main()
