"""Run the upcoming UFC live card with final canonical Event Clock C.

Operational rules:
- use final FSR V3 latest fighter posteriors for fighters with UFC history;
- sample the four validated positive flow traits once per fighter/path;
- sample validated native V3 KD resistance once per fighter/path;
- preserve frozen Event Clock detailed mechanics;
- if a fighter has no UFC FSR history, use an explicit population-prior fallback
  and flag that row.  No scouting, odds, or hand-entered fighter ratings are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, UPCOMING_FIGHTS_PATH
from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.fsr_v3.paths import (
    FSR_V3_LATEST_PATH,
    GROUND_SUPPRESSION_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
    KD_RESISTANCE_HISTORY_PATH,
    STANDING_SUPPRESSION_HISTORY_PATH,
    STANDING_TENDENCY_HISTORY_PATH,
    TAKEDOWN_SUPPRESSION_HISTORY_PATH,
    TAKEDOWN_TENDENCY_HISTORY_PATH,
)
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import (
    SEED,
    simulate_detailed_path,
)
from pipeline.simulation.event_clock_mc_v2.canonical_c import (
    fight_with_kd_resistance,
    sample_kd_resistance_latent,
)
from pipeline.simulation.event_clock_mc_v2.feature_builder import (
    build_sampled_fight_feature_rows_v3,
)
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import (
    DEFAULT_BUNDLE_PATH,
    legacy_power_equivalent,
)
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    initialize_path_matchup,
)
from pipeline.simulation.event_clock_mc_v2.inference import predict_feature_frame_v3
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    EPISTEMIC_SEED_OFFSET,
    _draw_budgets,
    _submission_inputs,
    load_frozen_context,
)
from pipeline.simulation.event_mc_v1.single_fight import (
    fight_from_fsr_v2_rows,
    fighter_age_years,
)

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/live_card_canonical_c")
CORE_HISTORY = {
    "standing_striking_tendency": STANDING_TENDENCY_HISTORY_PATH,
    "standing_striking_suppression": STANDING_SUPPRESSION_HISTORY_PATH,
    "takedown_tendency": TAKEDOWN_TENDENCY_HISTORY_PATH,
    "takedown_suppression": TAKEDOWN_SUPPRESSION_HISTORY_PATH,
}


def _read_history(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
    frame["fight_id"] = frame["fight_id"].astype(str)
    frame["fighter_id"] = frame["fighter_id"].astype(str)
    return frame.sort_values(["event_date", "fight_id", "fighter_id"]).reset_index(drop=True)


def _latest_rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    latest = frame.groupby("fighter_id", as_index=False).tail(1)
    return {str(row["fighter_id"]): row for _, row in latest.iterrows()}


def _last_number(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise RuntimeError(f"no finite {column} in history")
    return float(values.iloc[-1])


def _dob_lookup(master: pd.DataFrame) -> dict[str, object]:
    rows = []
    for prefix in ("r", "b"):
        if f"{prefix}_id" in master.columns and f"{prefix}_dob" in master.columns:
            part = master[[f"{prefix}_id", f"{prefix}_dob"]].rename(
                columns={f"{prefix}_id": "fighter_id", f"{prefix}_dob": "dob"}
            )
            rows.append(part)
    if not rows:
        return {}
    all_rows = pd.concat(rows, ignore_index=True).dropna(subset=["fighter_id"])
    all_rows["fighter_id"] = all_rows["fighter_id"].astype(str)
    all_rows = all_rows.dropna(subset=["dob"]).drop_duplicates("fighter_id", keep="last")
    return dict(zip(all_rows["fighter_id"], all_rows["dob"]))


def _select_card(event_date: str) -> pd.DataFrame:
    upcoming = pd.read_parquet(UPCOMING_FIGHTS_PATH).copy()
    upcoming["event_date"] = pd.to_datetime(upcoming["event_date"], errors="coerce").dt.normalize()
    date = pd.Timestamp(event_date).normalize()
    card = upcoming[upcoming["event_date"].eq(date)].copy()
    if card.empty:
        raise RuntimeError(f"no upcoming fights found for {date.date()} in {UPCOMING_FIGHTS_PATH}")
    required = [
        "fight_id", "red_fighter", "blue_fighter", "red_fighter_id", "blue_fighter_id",
        "weight_class", "total_rounds", "event_name", "event_date",
    ]
    missing = [c for c in required if c not in card.columns]
    if missing:
        raise RuntimeError(f"upcoming card missing columns: {missing}")
    card = card.dropna(subset=["fight_id", "red_fighter_id", "blue_fighter_id"]).copy()
    card["fight_id"] = card["fight_id"].astype(str)
    card["red_fighter_id"] = card["red_fighter_id"].astype(str)
    card["blue_fighter_id"] = card["blue_fighter_id"].astype(str)
    order = "fight_order" if "fight_order" in card.columns else "fight_id"
    return card.sort_values(order).reset_index(drop=True)


def _target_master(card: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    dobs = _dob_lookup(master)
    rows = []
    for row in card.to_dict("records"):
        rows.append({
            "fight_id": str(row["fight_id"]),
            "event_name": str(row["event_name"]),
            "event_date": pd.Timestamp(row["event_date"]).normalize(),
            "r_id": str(row["red_fighter_id"]),
            "b_id": str(row["blue_fighter_id"]),
            "r_name": str(row["red_fighter"]),
            "b_name": str(row["blue_fighter"]),
            "division": str(row["weight_class"]),
            "total_rounds": int(float(row["total_rounds"])),
            "r_dob": dobs.get(str(row["red_fighter_id"])),
            "b_dob": dobs.get(str(row["blue_fighter_id"])),
        })
    return pd.DataFrame(rows)


def _population_profile_template(latest: pd.DataFrame, histories: dict[str, pd.DataFrame]) -> dict:
    numeric = latest.select_dtypes(include=[np.number]).median(numeric_only=True).to_dict()
    record = {k: float(v) for k, v in numeric.items() if np.isfinite(v)}

    st = histories["standing_striking_tendency"]
    ss = histories["standing_striking_suppression"]
    tt = histories["takedown_tendency"]
    ts = histories["takedown_suppression"]
    gt = _read_history(GROUND_TENDENCY_HISTORY_PATH)
    gs = _read_history(GROUND_SUPPRESSION_HISTORY_PATH)

    record.update({
        "standing_striking_tendency": _last_number(st, "population_rate_15m"),
        "standing_striking_suppression": _last_number(ss, "population_multiplier"),
        "standing_striking_offense": 0.0,
        "standing_striking_defense": 0.0,
        "takedown_tendency": _last_number(tt, "population_rate_15m"),
        "takedown_suppression": _last_number(ts, "population_multiplier"),
        "takedown_offense": 0.0,
        "takedown_defense": 0.0,
        "escape_offense": 0.0,
        "escape_defense": 0.0,
        "ground_striking_tendency": _last_number(gt, "population_rate_15m"),
        "ground_striking_suppression": _last_number(gs, "population_multiplier"),
        "ground_striking_offense": 0.0,
        "ground_striking_burst_baseline": _last_number(gt, "population_burst"),
        "ground_striking_population_slope_15m": _last_number(gt, "population_rate_15m"),
        "submission_offense": 0.0,
        "submission_defense": 0.0,
        "head_strike_tendency": 0.80,
        "body_strike_tendency": 0.20,
        "leg_strike_tendency": 0.20,
        "stamina_capacity": 100.0,
        "stamina_depletion_resistance": 50.0,
        "stamina_performance_resilience": 50.0,
        "damage_durability": 50.0,
        "knockdown_resistance": 50.0,
        "striking_power_v3": 0.0,
        "knockdown_resistance_v3": 0.0,
    })
    # Population baselines are global in the latest publication. Median is a
    # deterministic way to read them without tying a debut to any one fighter.
    for col in (
        "standing_accuracy_baseline", "takedown_completion_baseline",
        "ground_accuracy_baseline", "submission_conversion_baseline",
        "escape_population_mean_seconds", "submission_tendency", "submission_suppression",
    ):
        if col not in record or not np.isfinite(record[col]):
            values = pd.to_numeric(latest[col], errors="coerce").dropna()
            if values.empty:
                raise RuntimeError(f"cannot construct population fallback for {col}")
            record[col] = float(values.median())
    return record


def _fighter_record(
    fighter_id: str,
    fighter_name: str,
    latest_lookup: dict[str, pd.Series],
    fallback: dict,
) -> tuple[dict, str]:
    if fighter_id in latest_lookup:
        record = latest_lookup[fighter_id].to_dict()
        source = "latest_v3"
    else:
        record = dict(fallback)
        source = "population_fallback"
    record["fighter_id"] = str(fighter_id)
    record["fighter_name"] = str(fighter_name)
    # V3 deliberately rejects this compatibility-only field.
    record.pop("ground_striking_defense", None)
    record.pop("striking_power", None)
    return record, source


def _core_uncertainty(
    fighter_id: str,
    histories: dict[str, pd.DataFrame],
    latest_history_rows: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    rows = []
    for trait, history in histories.items():
        found = latest_history_rows[trait].get(fighter_id)
        if found is not None:
            mean = float(found["post_rating"])
            sd = float(found["post_posterior_sd"])
            multiplier = float(found["variance_multiplier"])
        elif trait.endswith("tendency"):
            mean = _last_number(history, "population_rate_15m")
            prior_seconds = _last_number(history, "prior_seconds")
            shape = max(mean * prior_seconds / 900.0, 1e-9)
            sd = mean / np.sqrt(shape)
            multiplier = 1.0
        else:
            mean = _last_number(history, "population_multiplier")
            prior_shape = _last_number(history, "prior_shape")
            sd = mean / np.sqrt(max(prior_shape, 1e-9))
            multiplier = 1.0
        rows.append({
            "trait": trait,
            "posterior_mean": mean,
            "posterior_sd": sd,
            "variance_multiplier": multiplier,
            "sampling_enabled": True,
        })
    return pd.DataFrame(rows)


def _kd_state(
    fighter_id: str,
    kd_latest: dict[str, pd.Series],
) -> pd.Series:
    found = kd_latest.get(fighter_id)
    if found is None:
        config = ActiveTraitConfig()
        return pd.Series({
            "pre_rating": 0.0,
            "pre_posterior_sd": config.kd_resistance_sigma,
            "variance_multiplier": config.kd_resistance_variance_multiplier,
            "validated_regime": True,
        })
    return pd.Series({
        "pre_rating": float(found["post_rating"]),
        "pre_posterior_sd": float(found["post_posterior_sd"]),
        "variance_multiplier": 1.0,
        "validated_regime": True,
    })


def _mechanics_record(record: dict, age_years: float) -> dict:
    out = dict(record)
    out["ground_striking_defense"] = 0.0
    out["striking_power"] = float(legacy_power_equivalent(float(record["striking_power_v3"])))
    out["age_years"] = float(age_years)
    return out


def _summarize(master_row: pd.Series, results: list[dict], red_source: str, blue_source: str) -> dict:
    paths = pd.DataFrame(results)
    p = {}
    for side in ("red", "blue"):
        p[f"p_{side}_win"] = float(paths["winner"].eq(side).mean())
        for method in ("DEC", "KO_TKO", "SUB"):
            p[f"p_{side}_{method.lower()}"] = float(
                (paths["winner"].eq(side) & paths["method"].eq(method)).mean()
            )
    p["p_fight_dec"] = float(paths["method"].eq("DEC").mean())
    p["p_fight_ko_tko"] = float(paths["method"].eq("KO_TKO").mean())
    p["p_fight_sub"] = float(paths["method"].eq("SUB").mean())
    pred_side = "red" if p["p_red_win"] >= p["p_blue_win"] else "blue"
    pred_name = str(master_row["r_name"] if pred_side == "red" else master_row["b_name"])
    method = max(
        {"DEC": p["p_fight_dec"], "KO_TKO": p["p_fight_ko_tko"], "SUB": p["p_fight_sub"]},
        key=lambda x: {"DEC": p["p_fight_dec"], "KO_TKO": p["p_fight_ko_tko"], "SUB": p["p_fight_sub"]}[x],
    )
    return {
        "fight_id": str(master_row["fight_id"]),
        "event_name": str(master_row["event_name"]),
        "event_date": str(pd.Timestamp(master_row["event_date"]).date()),
        "red": str(master_row["r_name"]),
        "blue": str(master_row["b_name"]),
        "red_profile_source": red_source,
        "blue_profile_source": blue_source,
        "predicted_winner": pred_name,
        "predicted_method": method,
        **p,
        "sim_mean_elapsed": float(paths["elapsed"].mean()),
        "sim_red_sig_landed": float(paths["red_sig_landed"].mean()),
        "sim_blue_sig_landed": float(paths["blue_sig_landed"].mean()),
        "sim_red_td_landed": float(paths["red_td_landed"].mean()),
        "sim_blue_td_landed": float(paths["blue_td_landed"].mean()),
        "sim_red_control_seconds": float(paths["red_control_seconds"].mean()),
        "sim_blue_control_seconds": float(paths["blue_control_seconds"].mean()),
        "sim_red_kd": float(paths["red_kd"].mean()),
        "sim_blue_kd": float(paths["blue_kd"].mean()),
        "sim_red_sub_attempts": float(paths["red_sub_attempts"].mean()),
        "sim_blue_sub_attempts": float(paths["blue_sub_attempts"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-date", required=True)
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    card = _select_card(args.event_date)
    master = pd.read_parquet(MASTER_PATH).copy()
    target = _target_master(card, master)
    context = load_frozen_context(DEFAULT_BUNDLE_PATH)

    latest = pd.read_parquet(FSR_V3_LATEST_PATH).copy()
    latest["fighter_id"] = latest["fighter_id"].astype(str)
    latest_lookup = {str(row["fighter_id"]): row for _, row in latest.iterrows()}

    histories = {trait: _read_history(path) for trait, path in CORE_HISTORY.items()}
    latest_history_rows = {trait: _latest_rows(frame) for trait, frame in histories.items()}
    kd_history = _read_history(KD_RESISTANCE_HISTORY_PATH)
    kd_latest = _latest_rows(kd_history)
    fallback = _population_profile_template(latest, histories)

    submission_baseline = float(
        pd.to_numeric(latest["submission_conversion_baseline"], errors="coerce").dropna().median()
    )
    baseline_frame = pd.DataFrame({
        "fight_id": target["fight_id"].astype(str),
        "submission_conversion_baseline": submission_baseline,
    })

    print("=" * 150)
    print("EVENT CLOCK MC V2 — LIVE CARD CANONICAL C")
    print("=" * 150)
    print(f"event: {target['event_name'].iloc[0]} | date: {args.event_date}")
    print(f"fights: {len(target)} | paths/fight: {args.paths}")
    print("model: canonical C only; final V3 + validated epistemic draws + frozen mechanics")

    summaries = []
    for fight_index, (_, master_row) in enumerate(target.iterrows()):
        red_id, blue_id = str(master_row["r_id"]), str(master_row["b_id"])
        red_record, red_source = _fighter_record(
            red_id, str(master_row["r_name"]), latest_lookup, fallback
        )
        blue_record, blue_source = _fighter_record(
            blue_id, str(master_row["b_name"]), latest_lookup, fallback
        )
        red_unc = _core_uncertainty(red_id, histories, latest_history_rows)
        blue_unc = _core_uncertainty(blue_id, histories, latest_history_rows)
        red_kd = _kd_state(red_id, kd_latest)
        blue_kd = _kd_state(blue_id, kd_latest)

        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_age = fighter_age_years(master_row.get("r_dob"), event_date)
        blue_age = fighter_age_years(master_row.get("b_dob"), event_date)
        base_fight = fight_from_fsr_v2_rows(
            _mechanics_record(red_record, red_age),
            _mechanics_record(blue_record, blue_age),
            fight_id=str(master_row["fight_id"]),
            date=str(event_date.date()),
            division=str(master_row["division"]),
            rounds=int(master_row["total_rounds"]),
            red_age_years=red_age,
            blue_age_years=blue_age,
        )

        print(
            f"[{fight_index + 1}/{len(target)}] {master_row['r_name']} vs {master_row['b_name']} "
            f"| sources={red_source}/{blue_source}"
        )
        path_results = []
        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            epistemic_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
            matchup = initialize_path_matchup(
                red_record,
                blue_record,
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
                red_record=red_record,
                blue_record=blue_record,
                red_traits=matchup.red,
                blue_traits=matchup.blue,
            )
            pair, control = predict_feature_frame_v3(
                features,
                context["inference_models"],
                context["submission_scale"],
                context["conversion_offset"],
                submission_baseline=baseline_frame,
            )
            sub_rates, conversion = _submission_inputs(pair)
            budgets = _draw_budgets(pair, control.iloc[0], context, np.random.default_rng(seed))
            result = simulate_detailed_path(
                path_fight,
                budgets,
                sub_rates,
                conversion,
                context["judge_model"],
                context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            path_results.append(result)

        summaries.append(_summarize(master_row, path_results, red_source, blue_source))

    summary = pd.DataFrame(summaries)
    output = args.output or (
        OUT_DIR / f"canonical_c_{pd.Timestamp(args.event_date).date()}_{args.paths}paths.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)

    display = summary[[
        "red", "blue", "predicted_winner", "p_red_win", "p_blue_win",
        "p_red_dec", "p_red_ko_tko", "p_red_sub",
        "p_blue_dec", "p_blue_ko_tko", "p_blue_sub",
        "red_profile_source", "blue_profile_source",
    ]]
    print("\nCANONICAL C CARD")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nsummary CSV: {output}")


if __name__ == "__main__":
    main()
