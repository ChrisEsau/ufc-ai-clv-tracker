"""Measurement-only Event Clock V2 pace/survival decomposition.

Purpose:
- explain universal positive duration bias without tuning mechanics;
- compare historical vs simulated observable event cadence;
- compare historical vs simulated early-finish hazard by elapsed minute;
- measure KD/landing -> KO conversion at fight/path level.

Important limitation: UFCStats does not provide trustworthy phase-time or exact
within-fight event timestamps. This diagnostic therefore does NOT fabricate
historical distance/clinch/ground occupancy or exact inter-event timestamps.
Cadence uses exposure/count implied intervals only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import simulate_detailed_path
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    fight_with_kd_resistance,
    historical_kd_resistance_row,
    load_kd_resistance_history,
    sample_kd_resistance_latent,
)
from pipeline.simulation.event_clock_mc_v2.diagnostics.canonical_c_validation import _load_core_uncertainty
from pipeline.simulation.event_clock_mc_v2.diagnostics.weight_class_audit import select_cohort
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    historical_uncertainty_rows,
    initialize_path_matchup,
    load_prefight_snapshots,
)
from pipeline.simulation.event_clock_mc_v2.inference import (
    load_submission_baseline_v3,
    predict_feature_frame_v3,
)
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    EPISTEMIC_SEED_OFFSET,
    _draw_budgets,
    _submission_inputs,
    load_frozen_context,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight, normalize_method

DIVISIONS = (
    "flyweight",
    "bantamweight",
    "featherweight",
    "lightweight",
    "welterweight",
    "middleweight",
    "light heavyweight",
    "heavyweight",
    "women's strawweight",
    "women's flyweight",
    "women's bantamweight",
)


def _num(row, name: str) -> float:
    return float(pd.to_numeric(pd.Series([row.get(name, np.nan)]), errors="coerce").iloc[0])


def simulate_paths(target: pd.DataFrame, paths: int, seed0: int, division: str) -> pd.DataFrame:
    context = load_frozen_context(V2_BUNDLE_PATH)
    fsr_v3 = load_prefight_snapshots()
    uncertainty = _load_core_uncertainty()
    kd_history = load_kd_resistance_history()
    submission_baseline = load_submission_baseline_v3()

    rows: list[dict] = []
    for fight_index, master_row in target.reset_index(drop=True).iterrows():
        fight_id = str(master_row["fight_id"])
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        base_fight = _fight(master_row, context["fsr_all"])

        red_row, blue_row = historical_fighter_rows(
            fsr_v3,
            event_date=event_date,
            fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        red_unc = historical_uncertainty_rows(
            uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"])
        )
        blue_unc = historical_uncertainty_rows(
            uncertainty, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"])
        )
        red_kd = historical_kd_resistance_row(
            kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["r_id"])
        )
        blue_kd = historical_kd_resistance_row(
            kd_history, event_date=event_date, fight_id=fight_id, fighter_id=str(master_row["b_id"])
        )

        if fight_index % 10 == 0:
            print(f"[{fight_index + 1}/{len(target)}] {master_row['r_name']} vs {master_row['b_name']}")

        for path in range(paths):
            seed = seed0 + fight_index * 1_000_000 + path
            epistemic_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
            matchup = initialize_path_matchup(
                red_row,
                blue_row,
                red_unc,
                blue_unc,
                rng=epistemic_rng,
                sample_epistemic=True,
            )
            path_fight = fight_with_kd_resistance(
                base_fight,
                red_native_resistance=sample_kd_resistance_latent(red_kd, epistemic_rng),
                blue_native_resistance=sample_kd_resistance_latent(blue_kd, epistemic_rng),
            )
            features = build_sampled_fight_feature_rows_v3(
                master_row,
                red_record=red_row.to_dict(),
                blue_record=blue_row.to_dict(),
                red_traits=matchup.red,
                blue_traits=matchup.blue,
            )
            pair_c, control_c = predict_feature_frame_v3(
                features,
                context["inference_models"],
                context["submission_scale"],
                context["conversion_offset"],
                submission_baseline=submission_baseline,
            )
            info_c = control_c.iloc[0]
            sub_c, conv_c = _submission_inputs(pair_c)
            budgets = _draw_budgets(pair_c, info_c, context, np.random.default_rng(seed))
            result = simulate_detailed_path(
                path_fight,
                budgets,
                sub_c,
                conv_c,
                context["judge_model"],
                context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            result.update(
                {
                    "division": division,
                    "fight_id": fight_id,
                    "path": path,
                    "scheduled_seconds": float(master_row["total_rounds"]) * 300.0,
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def historical_rows(cohort: pd.DataFrame, division: str) -> pd.DataFrame:
    rows = []
    for _, r in cohort.iterrows():
        sig_a = _num(r, "r_sig_str_atmpted") + _num(r, "b_sig_str_atmpted")
        sig_l = _num(r, "r_sig_str_landed") + _num(r, "b_sig_str_landed")
        td_a = _num(r, "r_td_atmpted") + _num(r, "b_td_atmpted")
        sub_a = _num(r, "r_sub_att") + _num(r, "b_sub_att")
        kd = _num(r, "r_kd") + _num(r, "b_kd")
        elapsed = float(r["match_time_sec"])
        rows.append(
            {
                "division": division,
                "fight_id": str(r["fight_id"]),
                "elapsed": elapsed,
                "scheduled_seconds": float(r["total_rounds"]) * 300.0,
                "method": normalize_method(r["method"]),
                "sig_attempted": sig_a,
                "sig_landed": sig_l,
                "td_attempted": td_a,
                "sub_attempts": sub_a,
                "kd": kd,
                "meaningful_events": sig_a + td_a + sub_a,
            }
        )
    return pd.DataFrame(rows)


def simulated_derived(paths: pd.DataFrame) -> pd.DataFrame:
    p = paths.copy()
    p["sig_attempted"] = p["red_sig_attempted"] + p["blue_sig_attempted"]
    p["sig_landed"] = p["red_sig_landed"] + p["blue_sig_landed"]
    p["standing_attempted"] = p["red_standing_attempted"] + p["blue_standing_attempted"]
    p["ground_attempted"] = p["red_ground_attempted"] + p["blue_ground_attempted"]
    p["td_attempted"] = p["red_td_attempted"] + p["blue_td_attempted"]
    p["sub_attempts"] = p["red_sub_attempts"] + p["blue_sub_attempts"]
    p["kd"] = p["red_kd"] + p["blue_kd"]
    p["meaningful_events"] = p["sig_attempted"] + p["td_attempted"] + p["sub_attempts"]
    return p


def cadence_table(hist: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for division in DIVISIONS:
        h = hist[hist["division"] == division]
        s = sim[sim["division"] == division]
        for label, frame in (("historical", h), ("simulated", s)):
            exposure = float(frame["elapsed"].sum())
            events = float(frame["meaningful_events"].sum())
            sig_a = float(frame["sig_attempted"].sum())
            sig_l = float(frame["sig_landed"].sum())
            rows.append(
                {
                    "division": division,
                    "source": label,
                    "n": int(len(frame)),
                    "mean_elapsed": float(frame["elapsed"].mean()),
                    "meaningful_events_per_min": events / exposure * 60.0,
                    "implied_seconds_per_meaningful_event": exposure / events if events > 0 else np.nan,
                    "sig_attempts_per_min": sig_a / exposure * 60.0,
                    "sig_landed_per_min": sig_l / exposure * 60.0,
                    "td_attempts_per_min": float(frame["td_attempted"].sum()) / exposure * 60.0,
                    "sub_attempts_per_min": float(frame["sub_attempts"].sum()) / exposure * 60.0,
                    "kd_per_min": float(frame["kd"].sum()) / exposure * 60.0,
                }
            )
    return pd.DataFrame(rows)


def conversion_table(hist: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for division in DIVISIONS:
        for source, frame in (("historical", hist[hist["division"] == division]), ("simulated", sim[sim["division"] == division])):
            kd_mask = frame["kd"] > 0
            ko_mask = frame["method"] == "KO_TKO"
            landed = float(frame["sig_landed"].sum())
            rows.append(
                {
                    "division": division,
                    "source": source,
                    "n": int(len(frame)),
                    "share_with_kd": float(kd_mask.mean()),
                    "ko_share": float(ko_mask.mean()),
                    "p_ko_given_kd_present": float(ko_mask[kd_mask].mean()) if kd_mask.any() else np.nan,
                    "p_ko_given_no_kd": float(ko_mask[~kd_mask].mean()) if (~kd_mask).any() else np.nan,
                    "ko_finishes_per_100_sig_landed": float(ko_mask.sum()) / landed * 100.0 if landed > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def hazard_table(hist: pd.DataFrame, sim: pd.DataFrame, bin_seconds: int = 60) -> pd.DataFrame:
    rows = []
    max_seconds = int(max(hist["scheduled_seconds"].max(), sim["scheduled_seconds"].max()))
    for division in DIVISIONS:
        for source, frame in (("historical", hist[hist["division"] == division]), ("simulated", sim[sim["division"] == division])):
            for start in range(0, max_seconds, bin_seconds):
                end = start + bin_seconds
                at_risk = frame[frame["scheduled_seconds"] > start]
                at_risk = at_risk[at_risk["elapsed"] > start]
                finishes = at_risk[
                    (at_risk["method"] != "DEC")
                    & (at_risk["elapsed"] > start)
                    & (at_risk["elapsed"] <= end)
                ]
                if len(at_risk) == 0:
                    continue
                rows.append(
                    {
                        "division": division,
                        "source": source,
                        "bin_start_sec": start,
                        "bin_end_sec": end,
                        "at_risk": int(len(at_risk)),
                        "nondecision_finishes": int(len(finishes)),
                        "finish_hazard": float(len(finishes) / len(at_risk)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--out-dir", type=Path, default=Path("data/diagnostics/event_clock_mc_v2/pace_survival_decomposition"))
    args = parser.parse_args()

    all_hist, all_sim = [], []
    seed_stride = 100_000_000
    for i, division in enumerate(DIVISIONS):
        cohort, eligible = select_cohort(division, args.target_n)
        print("=" * 120)
        print(f"PACE/SURVIVAL DECOMPOSITION — {division.upper()} | eligible={eligible} selected={len(cohort)} paths={args.paths}")
        hist = historical_rows(cohort, division)
        sim = simulate_paths(cohort, args.paths, args.seed + i * seed_stride, division)
        all_hist.append(hist)
        all_sim.append(sim)

    hist = pd.concat(all_hist, ignore_index=True)
    sim = simulated_derived(pd.concat(all_sim, ignore_index=True))
    cadence = cadence_table(hist, sim)
    conversion = conversion_table(hist, sim)
    hazard = hazard_table(hist, sim)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist.to_csv(args.out_dir / "historical_fights.csv", index=False)
    sim.to_csv(args.out_dir / "simulated_paths.csv", index=False)
    cadence.to_csv(args.out_dir / "cadence.csv", index=False)
    conversion.to_csv(args.out_dir / "finish_conversion.csv", index=False)
    hazard.to_csv(args.out_dir / "finish_hazard_60s.csv", index=False)

    print("\nCADENCE")
    print(cadence.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nKD / KO CONVERSION")
    print(conversion.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nOUTPUT")
    print(args.out_dir)


if __name__ == "__main__":
    main()
