"""Measurement-only R1/R2/R3 lethality bucket diagnostic for canonical Event Clock C.

No tuning, no stamina changes, no FSR changes, and no frozen mechanics changes.
Historical buckets use UFCStats round data. Simulated buckets use an instrumented
copy of the exact frozen detailed path to record round exposure, sig attempts,
sig landings, KDs, and KO/TKO finishes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.round_stats.build_round_fighter_state import (
    read_round_stats,
    standardize_round_stats_input,
    validate_round_stats_input,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_full_fight_predictive_replay_shadow_ko_kd import (
    add_budget_events,
    add_submission_events,
    apply_delta,
    event_clock_shadow_ko_kd_profiles,
)
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    fight_with_kd_resistance,
    historical_kd_resistance_row,
    load_kd_resistance_history,
    sample_kd_resistance_latent,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.canonical_c_validation import _load_core_uncertainty
from pipeline.simulation.event_clock_mc_v2.diagnostics.pace_survival_decomposition import DIVISIONS
from pipeline.simulation.event_clock_mc_v2.diagnostics.weight_class_audit import select_cohort
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    historical_uncertainty_rows,
    initialize_path_matchup,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.inference import load_submission_baseline_v3, predict_feature_frame_v3
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    EPISTEMIC_SEED_OFFSET,
    _draw_budgets,
    _submission_inputs,
    load_frozen_context,
)
from pipeline.simulation.event_mc_v1.calibration import DEFAULT_RESOLVER
from pipeline.simulation.event_mc_v1.components.profiles import Side
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight, normalize_method
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.physiology import PhysiologyTimeAdvanceModel
from pipeline.simulation.event_mc_v1.stamina import StaminaModel
from pipeline.simulation.event_mc_v1.state import FightState

MAX_ROUND = 3


def simulate_bucket_path(fight, budgets, submission_rates, conversion_probability, judge_model, judge_features, seed):
    """Exact frozen detailed path with measurement-only round counters."""
    rng = np.random.default_rng(seed)
    horizon = float(fight.rounds * 300)
    key = fight.division if fight.division in DEFAULT_RESOLVER.weight_classes else None
    calibration = DEFAULT_RESOLVER.for_weight_class(key)
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    modifiers = DynamicModifierProvider(calibration)
    shadow = EventClockShadowKOKDModel(event_clock_shadow_ko_kd_profiles(fight))
    time_advance = PhysiologyTimeAdvanceModel(stamina, calibration)
    state = FightState()

    events = []
    for side_name in ("red", "blue"):
        side = Side(side_name)
        add_budget_events(events, side, "standing_strike", budgets[f"{side_name}_standing_attempted"], budgets[f"{side_name}_standing_landed"], horizon, rng)
        add_budget_events(events, side, "ground_strike", budgets[f"{side_name}_ground_attempted"], budgets[f"{side_name}_ground_landed"], horizon, rng)
        add_budget_events(events, side, "takedown", budgets[f"{side_name}_td_attempted"], budgets[f"{side_name}_td_landed"], horizon, rng)
        add_submission_events(events, side, submission_rates[side_name], horizon, rng)
    events.sort(key=lambda x: x[0])

    stats = {"red": Counter(), "blue": Counter()}
    round_stats = {r: Counter() for r in range(1, MAX_ROUND + 1)}
    kds = {"red": 0, "blue": 0}
    next_boundary = 300.0

    def advance_to(target):
        nonlocal next_boundary
        while next_boundary < target and next_boundary < horizon:
            dt = next_boundary - state.fight_time_seconds
            apply_delta(state, time_advance.advance(state, None, dt))
            state.fight_time_seconds = next_boundary
            apply_delta(state, stamina.recovery_delta(state))
            next_boundary += 300.0
        dt = target - state.fight_time_seconds
        if dt > 0:
            apply_delta(state, time_advance.advance(state, None, dt))
            state.fight_time_seconds = target

    for event_time, side, family, landed in events:
        if state.finished:
            break
        advance_to(event_time)
        side_name = side.value
        round_no = int(event_time // 300.0) + 1
        stats[side_name][f"{family}_attempted"] += 1
        if round_no <= MAX_ROUND and family in ("standing_strike", "ground_strike"):
            round_stats[round_no]["sig_attempted"] += 1

        profile = fight.profiles.fighter(side)
        modifiers.modifiers(profile, state, side)
        apply_delta(state, stamina.action_delta(state, side, family))

        if family == "submission_attempt":
            if rng.random() < conversion_probability:
                state.finished = True
                state.finish_reason = "SUB"
                state.finish_method = "SUB"
                state.winner = side_name
            continue

        if family == "takedown":
            if landed:
                stats[side_name]["td_landed"] += 1
            continue

        if landed:
            stats[side_name][f"{family}_landed"] += 1
            if round_no <= MAX_ROUND and family in ("standing_strike", "ground_strike"):
                round_stats[round_no]["sig_landed"] += 1

        if landed and family in ("standing_strike", "ground_strike"):
            consequence = shadow.resolve_landed_strike(
                state=state,
                attacker=side,
                prior_defender_kds=kds[side_name],
                rng=rng,
            )
            if consequence.ko_tko:
                if round_no <= MAX_ROUND:
                    round_stats[round_no]["ko_finish"] += 1
                state.finished = True
                state.finish_reason = "KO_TKO"
                state.finish_method = "KO_TKO"
                state.winner = side_name
                break
            if consequence.knockdown:
                kds[side_name] += 1
                if round_no <= MAX_ROUND:
                    round_stats[round_no]["kd"] += 1

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
        state.finished = True
        state.finish_reason = "DEC"
        state.finish_method = "DEC"
        state.winner = "red" if rng.random() < p_red else "blue"

    elapsed = float(state.fight_time_seconds)
    rows = []
    prior_kd = 0
    for round_no in range(1, MAX_ROUND + 1):
        start = (round_no - 1) * 300.0
        exposure = min(300.0, max(0.0, elapsed - start))
        if exposure <= 0:
            break
        rs = round_stats[round_no]
        rows.append({
            "round": round_no,
            "round_exposure_seconds": exposure,
            "sig_attempted": int(rs["sig_attempted"]),
            "sig_landed": int(rs["sig_landed"]),
            "kd": int(rs["kd"]),
            "ko_finish": int(rs["ko_finish"]),
            "prior_kd_present": int(prior_kd > 0),
            "round_kd_present": int(rs["kd"] > 0),
        })
        prior_kd += int(rs["kd"])
    return rows


def historical_round_rows(cohort: pd.DataFrame, division: str, round_stats_all: pd.DataFrame) -> pd.DataFrame:
    ids = set(cohort["fight_id"].astype(str))
    raw = round_stats_all[round_stats_all["fight_id"].astype(str).isin(ids)].copy()
    raw = raw[pd.to_numeric(raw["round"], errors="coerce").between(1, MAX_ROUND)]
    agg = (
        raw.groupby(["fight_id", "round"], as_index=False)
        .agg(
            round_exposure_seconds=("round_exposure_seconds", "first"),
            sig_attempted=("sig_str_attempted", "sum"),
            sig_landed=("sig_str_landed", "sum"),
            kd=("kd", "sum"),
        )
        .sort_values(["fight_id", "round"])
    )
    master = cohort.set_index(cohort["fight_id"].astype(str))
    rows = []
    for fight_id, group in agg.groupby("fight_id", sort=False):
        m = master.loc[str(fight_id)]
        method = normalize_method(m["method"])
        finish_round = int(m["finish_round"])
        prior_kd = 0
        for r in group.itertuples(index=False):
            rows.append({
                "division": division,
                "source": "historical",
                "fight_id": str(fight_id),
                "round": int(r.round),
                "round_exposure_seconds": float(r.round_exposure_seconds),
                "sig_attempted": float(r.sig_attempted),
                "sig_landed": float(r.sig_landed),
                "kd": float(r.kd),
                "ko_finish": int(method == "KO_TKO" and finish_round == int(r.round)),
                "prior_kd_present": int(prior_kd > 0),
                "round_kd_present": int(float(r.kd) > 0),
            })
            prior_kd += float(r.kd)
    return pd.DataFrame(rows)


def simulate_round_rows(cohort: pd.DataFrame, division: str, paths: int, seed0: int) -> pd.DataFrame:
    context = load_frozen_context(V2_BUNDLE_PATH)
    fsr_v3 = load_prefight_snapshots()
    uncertainty = _load_core_uncertainty()
    kd_history = load_kd_resistance_history()
    submission_baseline = load_submission_baseline_v3()
    rows = []

    for fight_index, master_row in cohort.reset_index(drop=True).iterrows():
        fight_id = str(master_row["fight_id"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        base_fight = _fight(master_row, context["fsr_all"])
        red_row, blue_row = historical_fighter_rows(
            fsr_v3, event_date=event_date, fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        red_unc = historical_uncertainty_rows(uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"]))
        blue_unc = historical_uncertainty_rows(uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"]))
        red_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"]))
        blue_kd = historical_kd_resistance_row(kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"]))

        if fight_index % 10 == 0:
            print(f"[{fight_index + 1}/{len(cohort)}] {master_row['r_name']} vs {master_row['b_name']}")

        for path in range(paths):
            seed = seed0 + fight_index * 1_000_000 + path
            epistemic_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
            matchup = initialize_path_matchup(red_row, blue_row, red_unc, blue_unc, rng=epistemic_rng, sample_epistemic=True)
            path_fight = fight_with_kd_resistance(
                base_fight,
                red_native_resistance=sample_kd_resistance_latent(red_kd, epistemic_rng),
                blue_native_resistance=sample_kd_resistance_latent(blue_kd, epistemic_rng),
            )
            features = build_sampled_fight_feature_rows_v3(
                master_row,
                red_record=red_row.to_dict(), blue_record=blue_row.to_dict(),
                red_traits=matchup.red, blue_traits=matchup.blue,
            )
            pair_c, control_c = predict_feature_frame_v3(
                features,
                context["inference_models"], context["submission_scale"], context["conversion_offset"],
                submission_baseline=submission_baseline,
            )
            sub_c, conv_c = _submission_inputs(pair_c)
            budgets = _draw_budgets(pair_c, control_c.iloc[0], context, np.random.default_rng(seed))
            bucket_rows = simulate_bucket_path(
                path_fight, budgets, sub_c, conv_c,
                context["judge_model"], context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            for rec in bucket_rows:
                rec.update({
                    "division": division,
                    "source": "simulated",
                    "fight_id": fight_id,
                    "path": path,
                })
                rows.append(rec)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (division, source, round_no), g in rows.groupby(["division", "source", "round"], sort=False):
        exposure_min = float(g["round_exposure_seconds"].sum()) / 60.0
        landed = float(g["sig_landed"].sum())
        kd = float(g["kd"].sum())
        ko = float(g["ko_finish"].sum())
        prior = g[g["prior_kd_present"] > 0]
        no_prior = g[g["prior_kd_present"] == 0]
        round_kd = g[g["round_kd_present"] > 0]
        out.append({
            "division": division,
            "source": source,
            "round": int(round_no),
            "at_risk_rounds": int(len(g)),
            "exposure_minutes": exposure_min,
            "sig_attempts_per_min": float(g["sig_attempted"].sum()) / exposure_min,
            "sig_landed_per_min": landed / exposure_min,
            "kd_per_100_sig_landed": kd / landed * 100.0 if landed > 0 else np.nan,
            "ko_finishes_per_100_sig_landed": ko / landed * 100.0 if landed > 0 else np.nan,
            "ko_finish_hazard": ko / len(g),
            "p_ko_given_prior_kd": float(prior["ko_finish"].mean()) if len(prior) else np.nan,
            "p_ko_given_no_prior_kd": float(no_prior["ko_finish"].mean()) if len(no_prior) else np.nan,
            "p_ko_given_kd_in_round": float(round_kd["ko_finish"].mean()) if len(round_kd) else np.nan,
        })
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--paths", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/lethality_round_bucket"))
    args = parser.parse_args()

    round_stats_all = standardize_round_stats_input(read_round_stats())
    validate_round_stats_input(round_stats_all)
    round_stats_all["fight_id"] = round_stats_all["fight_id"].astype(str)

    all_rows = []
    for i, division in enumerate(DIVISIONS):
        cohort, eligible = select_cohort(division, args.target_n)
        print("=" * 120)
        print(f"LETHALITY ROUND BUCKET — {division.upper()} | eligible={eligible} selected={len(cohort)} paths={args.paths}")
        hist = historical_round_rows(cohort, division, round_stats_all)
        sim = simulate_round_rows(cohort, division, args.paths, args.seed + i * 100_000_000)
        all_rows.extend([hist, sim])

    rows = pd.concat(all_rows, ignore_index=True)
    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_dir / "round_rows.csv", index=False)
    summary.to_csv(args.out_dir / "round_bucket_summary.csv", index=False)

    print("\nROUND BUCKET SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nOUTPUT")
    print(args.out_dir)


if __name__ == "__main__":
    main()
