from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, FSR_V2_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v1.prototype_stage3_correlation import prepare_direct_predictions
from pipeline.simulation.event_clock_mc_v1.prototype_stage4_marginals import direct_feature_columns
from pipeline.simulation.event_clock_mc_v1.prototype_stage5_competitive import (
    build_pair_frame,
    fit_control_models,
    fit_count_hurdle,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage7_budget_timeline import (
    add_historical_free_time,
    fit_standing_free_time_model,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage8_grappling_calibration import fit_directional_ownership_kappa
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import (
    fit_ground_alpha_by_shape,
    fit_control_minority_models,
    simulate_stage9_path,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage10d_total_fight_judge import (
    VARIANTS,
    decision_mask,
    fit_model,
    prepare_master,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11_submission_attempts import build_submission_targets
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage11c_submission_conversion import (
    clip_probability,
    load_submission_baseline,
    logistic,
    logit,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_full_fight_predictive_replay_shadow_ko_kd import (
    add_budget_events,
    add_submission_events,
    apply_delta,
    base_submission_rate,
    event_clock_shadow_ko_kd_profiles,
    fit_conversion_offset,
    fit_submission_attempt_scale,
    normalize_method,
)
from pipeline.simulation.event_clock_mc_v1.ko_kd_shadow import EventClockShadowKOKDModel
from pipeline.simulation.event_mc_v1.calibration import DEFAULT_RESOLVER
from pipeline.simulation.event_mc_v1.components.profiles import Side
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight
from pipeline.simulation.event_mc_v1.modifiers import DynamicModifierProvider
from pipeline.simulation.event_mc_v1.physiology import PhysiologyTimeAdvanceModel
from pipeline.simulation.event_mc_v1.stamina import StaminaModel
from pipeline.simulation.event_mc_v1.state import FightState


SEED = 20260820
DEFAULT_PATHS = 2000
OUT_DIR = Path("data/diagnostics/event_clock_mc_v1/event_predictions")


def _num(row, col):
    return float(pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0])


def _slug(value: str) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "prediction"


def select_target(master: pd.DataFrame, args) -> pd.DataFrame:
    target = master.copy()

    if args.fight_id:
        target = target[target["fight_id"].astype(str) == str(args.fight_id)]
        label = f"fight_{args.fight_id}"
    elif args.event:
        mask = target["event_name"].astype(str).str.contains(args.event, case=False, regex=False, na=False)
        target = target[mask]
        label = args.event
    elif args.event_date:
        date = pd.Timestamp(args.event_date).normalize()
        target = target[target["event_date"] == date]
        label = str(date.date())
    elif args.fighter:
        fighter_mask = (
            target["r_name"].astype(str).str.contains(args.fighter, case=False, regex=False, na=False)
            | target["b_name"].astype(str).str.contains(args.fighter, case=False, regex=False, na=False)
        )
        target = target[fighter_mask]
        if args.opponent:
            opp_mask = (
                target["r_name"].astype(str).str.contains(args.opponent, case=False, regex=False, na=False)
                | target["b_name"].astype(str).str.contains(args.opponent, case=False, regex=False, na=False)
            )
            target = target[opp_mask]
        label = args.fighter if not args.opponent else f"{args.fighter}_vs_{args.opponent}"
    else:
        raise RuntimeError("Specify --event, --event-date, --fight-id, or --fighter.")

    if target.empty:
        raise RuntimeError("No matching fight(s) found in UFC master.")

    target = target.sort_values(["event_date", "fight_id"]).copy()
    target.attrs["label"] = label
    return target


def simulate_detailed_path(
    fight,
    budgets,
    submission_rates,
    conversion_probability,
    judge_model,
    judge_features,
    seed,
):
    """Exact shadow-KO/KD Event Clock path mechanics with expanded reporting."""
    rng = np.random.default_rng(seed)
    horizon = float(fight.rounds * 300)

    key = fight.division if fight.division in DEFAULT_RESOLVER.weight_classes else None
    calibration = DEFAULT_RESOLVER.for_weight_class(key)

    shadow_profiles = event_clock_shadow_ko_kd_profiles(fight)
    stamina = StaminaModel(fight.profiles, calibration=calibration)
    modifiers = DynamicModifierProvider(calibration)
    shadow_ko_kd = EventClockShadowKOKDModel(shadow_profiles)
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
        stats[side_name][f"{family}_attempted"] += 1

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

        if landed and family in ("standing_strike", "ground_strike"):
            consequence = shadow_ko_kd.resolve_landed_strike(
                state=state,
                attacker=side,
                prior_defender_kds=kds[side_name],
                rng=rng,
            )
            stats[side_name]["ko_kd_strike_opportunities"] += 1
            stats[side_name]["ko_probability_sum"] += consequence.ko_probability

            if consequence.ko_tko:
                stats[side_name]["ko_events"] += 1
                state.finished = True
                state.finish_reason = "KO_TKO"
                state.finish_method = "KO_TKO"
                state.winner = side_name
                break

            stats[side_name]["kd_probability_sum"] += consequence.kd_probability
            if consequence.knockdown:
                kds[side_name] += 1
                stats[side_name]["kd_events"] += 1

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
    exposure_fraction = min(max(elapsed / max(horizon, 1.0), 0.0), 1.0)

    out = {
        "winner": state.winner,
        "method": state.finish_method,
        "elapsed": elapsed,
        "finish_round": int(max(elapsed - 1e-12, 0.0) // 300) + 1,
    }

    for side_name in ("red", "blue"):
        standing_a = stats[side_name]["standing_strike_attempted"]
        standing_l = stats[side_name]["standing_strike_landed"]
        ground_a = stats[side_name]["ground_strike_attempted"]
        ground_l = stats[side_name]["ground_strike_landed"]
        out.update({
            f"{side_name}_sig_attempted": standing_a + ground_a,
            f"{side_name}_sig_landed": standing_l + ground_l,
            f"{side_name}_standing_attempted": standing_a,
            f"{side_name}_standing_landed": standing_l,
            f"{side_name}_ground_attempted": ground_a,
            f"{side_name}_ground_landed": ground_l,
            f"{side_name}_td_attempted": stats[side_name]["takedown_attempted"],
            f"{side_name}_td_landed": stats[side_name]["td_landed"],
            f"{side_name}_sub_attempts": stats[side_name]["submission_attempt_attempted"],
            f"{side_name}_kd": kds[side_name],
            f"{side_name}_control_seconds": float(budgets[f"{side_name}_control"]) * exposure_fraction,
        })

    return out


def build_context(target_ids: set[str]):
    train, test = prepare_direct_predictions()

    for frame in (train, test):
        frame["fight_id"] = frame["fight_id"].astype(str)

    missing = sorted(target_ids - set(test["fight_id"]))
    if missing:
        raise RuntimeError(
            "Target fight(s) are outside the current frozen fresh predictive cohort: "
            + ", ".join(missing)
        )

    targets = build_submission_targets()[["fight_id", "side", "submission_attempted", "submission_win"]].copy()
    targets["fight_id"] = targets["fight_id"].astype(str)
    train = train.merge(targets, on=["fight_id", "side"], how="inner", validate="one_to_one")
    test = test.merge(targets, on=["fight_id", "side"], how="inner", validate="one_to_one")

    for frame in (train, test):
        frame["historical_duration"] = frame["duration"].astype(float)

    train = add_historical_free_time(train)
    test = add_historical_free_time(test)

    scheduled = test["scheduled_rounds"].astype(float) * 300.0
    exposure_ratio = scheduled / test["historical_duration"].clip(lower=1.0)

    for family in ("distance", "clinch", "ground", "td"):
        for suffix in ("attempted", "landed"):
            col = f"pred_{family}_{suffix}"
            test[col] = test[col] * exposure_ratio

    test["pred_qualified_control_inflicted_seconds"] *= exposure_ratio
    test["duration"] = scheduled
    test["pred_qualified_control_inflicted_seconds"] = np.minimum(
        test["pred_qualified_control_inflicted_seconds"], test["duration"]
    )

    for fight_id, group in test.groupby("fight_id"):
        idx = group.index
        total = float(test.loc[idx, "pred_qualified_control_inflicted_seconds"].sum())
        duration = float(group["duration"].iloc[0])
        if total > duration:
            test.loc[idx, "pred_qualified_control_inflicted_seconds"] *= duration / total

    for frame in (train, test):
        frame["standing_attempted"] = frame["distance_attempted"] + frame["clinch_attempted"]
        frame["standing_landed"] = frame["distance_landed"] + frame["clinch_landed"]
        frame["pred_standing_attempted"] = frame["pred_distance_attempted"] + frame["pred_clinch_attempted"]
        frame["pred_standing_landed"] = frame["pred_distance_landed"] + frame["pred_clinch_landed"]

    feature_cols = direct_feature_columns()
    hurdle_alpha = {}
    hurdle_alpha["td"] = fit_count_hurdle(train, test, "td", feature_cols)
    fit_count_hurdle(train, test, "ground", feature_cols)
    hurdle_alpha["ground"] = fit_ground_alpha_by_shape(train)
    train, test, _, standing_alpha = fit_standing_free_time_model(train, test, feature_cols)

    train_pair = build_pair_frame(train)
    test_pair = build_pair_frame(test)
    td_control_beta, control_alpha = fit_control_models(train_pair, test_pair)
    dominance_kappa = fit_directional_ownership_kappa(train, train_pair, td_control_beta)
    minority_classifier, minority_share_model, minority_residual_sigma = fit_control_minority_models(
        train, train_pair, td_control_beta
    )
    pair_lookup = {str(row["fight_id"]): row for _, row in test_pair.iterrows()}

    train["submission_base_rate"] = base_submission_rate(train)
    test["submission_base_rate"] = base_submission_rate(test)
    submission_scale = fit_submission_attempt_scale(train)

    baseline = load_submission_baseline()
    train = train.merge(baseline, on="fight_id", how="left", validate="many_to_one")
    test = test.merge(baseline, on="fight_id", how="left", validate="many_to_one")
    conversion_offset = fit_conversion_offset(train)
    test["submission_clock_rate"] = submission_scale * test["submission_base_rate"]
    test["submission_conversion_probability"] = logistic(
        logit(clip_probability(test["submission_conversion_baseline"])) + conversion_offset
    )

    master_raw = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master_raw["fight_id"] = master_raw["fight_id"].astype(str)
    master_raw["event_date"] = pd.to_datetime(master_raw["date"], errors="raise").dt.normalize()
    master_judge = prepare_master(master_raw)
    train_ids = set(train["fight_id"])
    judge_train = master_judge[
        master_judge["fight_id"].isin(train_ids)
        & decision_mask(master_judge)
        & master_judge["red_win"].notna()
    ].copy()
    judge_features = VARIANTS["FULL_TOTAL"]
    judge_model = fit_model(judge_train, judge_features)

    fsr_all = pd.read_parquet(FSR_V2_PREFIGHT_SNAPSHOTS_PATH)
    fsr_all["fight_id"] = fsr_all["fight_id"].astype(str)
    fsr_all["event_date"] = pd.to_datetime(fsr_all["event_date"], errors="raise").dt.normalize()

    return {
        "test": test[test["fight_id"].isin(target_ids)].copy(),
        "pair_lookup": pair_lookup,
        "hurdle_alpha": hurdle_alpha,
        "control_alpha": control_alpha,
        "dominance_kappa": dominance_kappa,
        "td_control_beta": td_control_beta,
        "standing_alpha": standing_alpha,
        "minority_classifier": minority_classifier,
        "minority_share_model": minority_share_model,
        "minority_residual_sigma": minority_residual_sigma,
        "judge_model": judge_model,
        "judge_features": judge_features,
        "master_lookup": {str(row["fight_id"]): row for _, row in master_raw.iterrows()},
        "fsr_all": fsr_all,
    }


def summarize_fight(fight_id, pair, rows, master_row):
    paths = pd.DataFrame(rows)
    red_name = str(master_row["r_name"])
    blue_name = str(master_row["b_name"])

    p = {}
    for side in ("red", "blue"):
        p[f"p_{side}_win"] = float((paths["winner"] == side).mean())
        for method in ("DEC", "KO_TKO", "SUB"):
            p[f"p_{side}_{method.lower()}"] = float(((paths["winner"] == side) & (paths["method"] == method)).mean())

    p["p_fight_dec"] = float((paths["method"] == "DEC").mean())
    p["p_fight_ko_tko"] = float((paths["method"] == "KO_TKO").mean())
    p["p_fight_sub"] = float((paths["method"] == "SUB").mean())

    actual_winner = "red" if master_row["winner"] == red_name else "blue"
    actual_method = normalize_method(master_row["method"])
    actual_elapsed = (int(master_row["finish_round"]) - 1) * 300 + float(master_row["match_time_sec"])

    pred_winner = "red" if p["p_red_win"] >= p["p_blue_win"] else "blue"
    method_probs = {"DEC": p["p_fight_dec"], "KO_TKO": p["p_fight_ko_tko"], "SUB": p["p_fight_sub"]}
    pred_method = max(method_probs, key=method_probs.get)

    joint_probs = {
        (side, method): p[f"p_{side}_{method.lower()}"]
        for side in ("red", "blue")
        for method in ("DEC", "KO_TKO", "SUB")
    }
    pred_joint = max(joint_probs, key=joint_probs.get)

    row = {
        "fight_id": fight_id,
        "event_name": master_row.get("event_name", ""),
        "event_date": str(pd.Timestamp(master_row["event_date"]).date()),
        "red": red_name,
        "blue": blue_name,
        "actual_winner": actual_winner,
        "actual_method": actual_method,
        "actual_finish_round": int(master_row["finish_round"]),
        "actual_elapsed": actual_elapsed,
        "predicted_winner": pred_winner,
        "predicted_method": pred_method,
        "predicted_joint": f"{pred_joint[0]}_{pred_joint[1]}",
        "ml_correct": int(pred_winner == actual_winner),
        "method_correct": int(pred_method == actual_method),
        "winner_method_correct": int(pred_joint == (actual_winner, actual_method)),
        **p,
        "sim_mean_elapsed": float(paths["elapsed"].mean()),
    }

    hist_map = {
        "sig_attempted": "sig_str_atmpted",
        "sig_landed": "sig_str_landed",
        "td_attempted": "td_atmpted",
        "td_landed": "td_landed",
        "kd": "kd",
        "sub_attempts": "sub_att",
        "control_seconds": "ctrl",
    }

    for side, prefix in (("red", "r"), ("blue", "b")):
        for stat, hist_suffix in hist_map.items():
            row[f"sim_{side}_{stat}"] = float(paths[f"{side}_{stat}"].mean())
            row[f"hist_{side}_{stat}"] = _num(master_row, f"{prefix}_{hist_suffix}")
            row[f"error_{side}_{stat}"] = row[f"sim_{side}_{stat}"] - row[f"hist_{side}_{stat}"]

        for stat in ("standing_attempted", "standing_landed", "ground_attempted", "ground_landed"):
            row[f"sim_{side}_{stat}"] = float(paths[f"{side}_{stat}"].mean())

    return row


def main():
    parser = argparse.ArgumentParser(description="Run frozen Event Clock MC for any eligible event or fight in the current predictive cohort.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event", help="Case-insensitive substring of UFC master event_name.")
    group.add_argument("--event-date", help="Event date, e.g. 2026-06-14.")
    group.add_argument("--fight-id")
    group.add_argument("--fighter", help="Case-insensitive fighter-name substring.")
    parser.add_argument("--opponent", help="Optional opponent filter used with --fighter.")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="raise").dt.normalize()

    target = select_target(master, args)
    target_ids = set(target["fight_id"].astype(str))
    context = build_context(target_ids)
    test = context["test"]

    print("=" * 150)
    print("EVENT CLOCK MC — GENERIC EVENT / FIGHT PREDICTION")
    print("=" * 150)
    print(f"target fights: {len(target_ids)} | paths/fight: {args.paths}")
    print("KO/KD calibration: validated empirical default")

    all_path_rows = []
    summary_rows = []

    groups = list(test.groupby("fight_id", sort=False))
    for fight_index, (fight_id, pair) in enumerate(groups):
        fight_id = str(fight_id)
        master_row = context["master_lookup"][fight_id]
        fight = _fight(master_row, context["fsr_all"])
        pair_info = context["pair_lookup"][fight_id]

        sub_rate = {}
        convert = None
        for side in ("red", "blue"):
            row = pair[pair["side"] == side].iloc[0]
            sub_rate[side] = float(row["submission_clock_rate"])
            if convert is None:
                convert = float(row["submission_conversion_probability"])

        fight_rows = []
        print(f"[{fight_index + 1}/{len(groups)}] {master_row['r_name']} vs {master_row['b_name']}")

        for path in range(args.paths):
            seed = args.seed + fight_index * 1000000 + path
            rng = np.random.default_rng(seed)
            budgets = simulate_stage9_path(
                pair,
                pair_info,
                context["hurdle_alpha"],
                context["control_alpha"],
                context["dominance_kappa"],
                context["td_control_beta"],
                context["standing_alpha"],
                context["minority_classifier"],
                context["minority_share_model"],
                context["minority_residual_sigma"],
                rng,
            )
            result = simulate_detailed_path(
                fight,
                budgets,
                sub_rate,
                convert,
                context["judge_model"],
                context["judge_features"],
                seed + 50000000,
            )
            result.update({"fight_id": fight_id, "path": path})
            fight_rows.append(result)
            all_path_rows.append(result)

        summary_rows.append(summarize_fight(fight_id, pair, fight_rows, master_row))

    summary = pd.DataFrame(summary_rows)
    paths = pd.DataFrame(all_path_rows)

    label = target.attrs.get("label", "prediction")
    prefix = args.output_prefix or _slug(label)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"{prefix}_{args.paths}paths_summary.csv"
    paths_path = OUT_DIR / f"{prefix}_{args.paths}paths_paths.csv"
    summary.to_csv(summary_path, index=False)
    paths.to_csv(paths_path, index=False)

    display_cols = [
        "red", "blue", "actual_winner", "actual_method",
        "p_red_win", "p_blue_win", "p_red_dec", "p_red_ko_tko", "p_red_sub",
        "p_blue_dec", "p_blue_ko_tko", "p_blue_sub",
        "ml_correct", "method_correct", "winner_method_correct",
    ]
    print()
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"ML accuracy:              {summary['ml_correct'].mean():.2%}")
    print(f"Method accuracy:          {summary['method_correct'].mean():.2%}")
    print(f"Winner+method accuracy:   {summary['winner_method_correct'].mean():.2%}")
    print(f"summary CSV: {summary_path}")
    print(f"path CSV:    {paths_path}")


if __name__ == "__main__":
    main()
