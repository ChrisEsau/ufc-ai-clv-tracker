"""Future-card Canonical C reach A/B validation.

Validation-only runner for upcoming cards.  It preserves the Canonical C path:
latest canonical V3 state, validated standing/TD epistemic sampling, native V3
KD-resistance sampling, V3 power translated into the frozen detailed profile,
and frozen Event Clock mechanics.

The A/B arms use common random numbers.  The only difference is that arm OFF
sets the validated distance-reach multiplier to 1.0 before forward inference.
Arm ON uses the promoted reach translation unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH, UPCOMING_FIGHTS_PATH
from pipeline.fsr_v2.physical import STAMINA_CAPACITY
from pipeline.fsr_v3.active_config import ActiveTraitConfig
from pipeline.fsr_v3.paths import (
    FSR_V3_LATEST_PATH,
    GROUND_EFFECTIVENESS_HISTORY_PATH,
    GROUND_TENDENCY_HISTORY_PATH,
    KD_RESISTANCE_HISTORY_PATH,
    STANDING_EFFECTIVENESS_HISTORY_PATH,
    STANDING_SUPPRESSION_HISTORY_PATH,
    STANDING_TENDENCY_HISTORY_PATH,
    TAKEDOWN_EFFECTIVENESS_HISTORY_PATH,
    TAKEDOWN_SUPPRESSION_HISTORY_PATH,
    TAKEDOWN_TENDENCY_HISTORY_PATH,
)
from pipeline.common.paths import FSR_V2_LATEST_PATH
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
    fighter_age_years,
    fight_from_fsr_v2_rows,
)
from scrapers.ufcstats_fighter_profiles import scrape_fighter_profile

OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/event_predictions")
ARMS = ("reach_off", "reach_on")
SAMPLABLE = (
    "takedown_tendency",
    "takedown_suppression",
    "standing_striking_tendency",
    "standing_striking_suppression",
)


def _latest_by_fighter(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="raise").dt.normalize()
    out["fighter_id"] = out["fighter_id"].astype(str)
    return (
        out.sort_values(["event_date", "fight_id"])
        .groupby("fighter_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _median_numeric_row(frame: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {}
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            row[column] = float(values.median())
    return row


def _population_v2(latest: pd.DataFrame) -> dict[str, object]:
    """Neutral no-UFC-evidence row for the frozen detailed/profile boundary."""
    row = _median_numeric_row(latest)
    row.update(
        {
            "stamina_capacity": STAMINA_CAPACITY,
            "stamina_depletion_resistance": 50.0,
            "stamina_performance_resilience": 50.0,
            "striking_power": 50.0,
            "damage_durability": 50.0,
            "knockdown_resistance": 50.0,
        }
    )
    for name in (
        "standing_striking_offense", "standing_striking_defense",
        "takedown_offense", "takedown_defense",
        "escape_offense", "escape_defense",
        "ground_striking_offense", "ground_striking_defense",
        "submission_offense", "submission_defense",
    ):
        row[name] = 0.0
    for name in (
        "standing_striking_suppression", "takedown_suppression",
        "ground_striking_suppression", "submission_suppression",
    ):
        row[name] = 1.0
    head = float(row.get("head_strike_tendency", 0.65))
    body = float(row.get("body_strike_tendency", 0.35))
    total = max(head + body, 1e-12)
    row["head_strike_tendency"] = head / total
    row["body_strike_tendency"] = body / total
    row["leg_strike_tendency"] = float(np.clip(row.get("leg_strike_tendency", 0.15), 0.0, 1.0))
    return row


def _last_population_value(path: Path, column: str, *, trait: str | None = None) -> float:
    frame = pd.read_parquet(path).copy()
    if trait is not None and "trait" in frame.columns:
        frame = frame[frame["trait"].astype(str).eq(trait)]
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise")
    frame = frame.sort_values(["event_date", "fight_id"])
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        raise RuntimeError(f"No population value {column} in {path}")
    return float(values.iloc[-1])


def _population_v3(v2_pop: dict[str, object], v3_latest: pd.DataFrame) -> dict[str, object]:
    row = dict(v2_pop)
    # Validated V3 families use their current population coordinates.
    row.update(
        {
            "standing_striking_tendency": _last_population_value(
                STANDING_TENDENCY_HISTORY_PATH, "population_rate_15m"
            ),
            "standing_striking_suppression": 1.0,
            "standing_striking_offense": 0.0,
            "standing_striking_defense": 0.0,
            "standing_accuracy_baseline": _last_population_value(
                STANDING_EFFECTIVENESS_HISTORY_PATH, "population_baseline",
                trait="standing_striking_offense",
            ),
            "takedown_tendency": _last_population_value(
                TAKEDOWN_TENDENCY_HISTORY_PATH, "population_rate_15m"
            ),
            "takedown_suppression": 1.0,
            "takedown_offense": 0.0,
            "takedown_defense": 0.0,
            "takedown_completion_baseline": _last_population_value(
                TAKEDOWN_EFFECTIVENESS_HISTORY_PATH, "population_baseline",
                trait="takedown_offense",
            ),
            "ground_striking_tendency": _last_population_value(
                GROUND_TENDENCY_HISTORY_PATH, "population_rate_15m"
            ),
            "ground_striking_suppression": 1.0,
            "ground_striking_offense": 0.0,
            "ground_accuracy_baseline": _last_population_value(
                GROUND_EFFECTIVENESS_HISTORY_PATH, "population_baseline"
            ),
            "ground_striking_burst_baseline": _last_population_value(
                GROUND_TENDENCY_HISTORY_PATH, "population_burst"
            ),
            "ground_striking_population_slope_15m": _last_population_value(
                GROUND_TENDENCY_HISTORY_PATH, "population_rate_15m"
            ),
            "escape_offense": 0.0,
            "escape_defense": 0.0,
            "striking_power_v3": 0.0,
            "knockdown_resistance_v3": 0.0,
        }
    )
    # Active/inherited fight-level population quantities are constant in latest.
    for name in (
        "escape_population_mean_seconds",
        "submission_conversion_baseline",
        "submission_tendency", "submission_suppression",
        "submission_offense", "submission_defense",
    ):
        if name in v3_latest.columns:
            vals = pd.to_numeric(v3_latest[name], errors="coerce").dropna()
            if not vals.empty:
                row[name] = float(vals.median())
    row["submission_offense"] = 0.0
    row["submission_defense"] = 0.0
    row["submission_suppression"] = 1.0
    row.pop("ground_striking_defense", None)
    row.pop("striking_power", None)
    return row


def _uncertainty_table() -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    paths = {
        "takedown_tendency": TAKEDOWN_TENDENCY_HISTORY_PATH,
        "takedown_suppression": TAKEDOWN_SUPPRESSION_HISTORY_PATH,
        "standing_striking_tendency": STANDING_TENDENCY_HISTORY_PATH,
        "standing_striking_suppression": STANDING_SUPPRESSION_HISTORY_PATH,
    }
    by_fighter: dict[str, list[dict[str, object]]] = defaultdict(list)
    pop_sd: dict[str, float] = {}
    for trait, path in paths.items():
        frame = pd.read_parquet(path).copy()
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="raise").dt.normalize()
        frame["fighter_id"] = frame["fighter_id"].astype(str)
        if "trait" in frame.columns:
            frame = frame[frame["trait"].astype(str).eq(trait)].copy()
        latest = _latest_by_fighter(frame)
        for rec in latest.to_dict("records"):
            by_fighter[str(rec["fighter_id"])].append(
                {
                    "trait": trait,
                    "posterior_mean": float(rec["post_rating"]),
                    "posterior_sd": float(rec["post_posterior_sd"]),
                    "variance_multiplier": float(rec.get("variance_multiplier", 1.0)),
                    "sampling_enabled": bool(rec.get("sampling_enabled", True)),
                }
            )
        # First UFC appearance pre-SD is the closest native no-evidence prior SD.
        first = (
            frame.sort_values(["event_date", "fight_id"])
            .groupby("fighter_id", as_index=False)
            .head(1)
        )
        vals = pd.to_numeric(first["pre_posterior_sd"], errors="coerce").dropna()
        pop_sd[trait] = float(vals.median()) if not vals.empty else 0.0
    return {fid: pd.DataFrame(rows) for fid, rows in by_fighter.items()}, pop_sd


def _population_uncertainty(pop_v3: dict[str, object], pop_sd: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trait": trait,
                "posterior_mean": float(pop_v3[trait]),
                "posterior_sd": float(pop_sd.get(trait, 0.0)),
                "variance_multiplier": 1.0,
                "sampling_enabled": True,
            }
            for trait in SAMPLABLE
        ]
    )


def _kd_latest() -> dict[str, pd.Series]:
    frame = pd.read_parquet(KD_RESISTANCE_HISTORY_PATH).copy()
    latest = _latest_by_fighter(frame)
    out = {}
    for rec in latest.to_dict("records"):
        out[str(rec["fighter_id"])] = pd.Series(
            {
                "pre_rating": float(rec["post_rating"]),
                "pre_posterior_sd": float(rec["post_posterior_sd"]),
                "variance_multiplier": float(rec.get("variance_multiplier", 1.0)),
                "validated_regime": bool(rec.get("validated_regime", True)),
            }
        )
    return out


def _population_kd() -> pd.Series:
    cfg = ActiveTraitConfig()
    return pd.Series(
        {
            "pre_rating": 0.0,
            "pre_posterior_sd": float(cfg.kd_resistance_sigma),
            "variance_multiplier": float(cfg.kd_resistance_variance_multiplier),
            "validated_regime": True,
        }
    )


def _historical_physical_lookup() -> dict[str, dict[str, object]]:
    master = pd.read_parquet(MASTER_PATH).copy()
    date_col = "date" if "date" in master.columns else "event_date"
    master["_date"] = pd.to_datetime(master[date_col], errors="coerce")
    rows: dict[str, dict[str, object]] = {}
    for _, rec in master.sort_values("_date").iterrows():
        for prefix in ("r", "b"):
            fid = rec.get(f"{prefix}_id")
            if pd.isna(fid):
                continue
            fid = str(fid)
            item = rows.setdefault(fid, {})
            for src, dst in ((f"{prefix}_dob", "dob"), (f"{prefix}_reach", "reach")):
                if src in master.columns and pd.notna(rec.get(src)):
                    item[dst] = rec.get(src)
    return rows


def _profile_context(card: pd.DataFrame) -> dict[str, dict[str, object]]:
    history = _historical_physical_lookup()
    result: dict[str, dict[str, object]] = {}
    for rec in card.to_dict("records"):
        for prefix in ("red", "blue"):
            fid = str(rec[f"{prefix}_fighter_id"])
            if fid in result:
                continue
            context = dict(history.get(fid, {}))
            url = rec.get(f"{prefix}_fighter_url")
            if pd.notna(url) and str(url).strip():
                try:
                    profile = scrape_fighter_profile(str(url), fighter_id=fid).iloc[0].to_dict()
                    if pd.notna(profile.get("dob")):
                        context["dob"] = profile["dob"]
                    reach_cm = pd.to_numeric(pd.Series([profile.get("reach")]), errors="coerce").iloc[0]
                    if pd.notna(reach_cm):
                        context["reach"] = float(reach_cm) / 2.54
                except Exception as exc:
                    print(f"WARN profile scrape failed for {fid}: {exc}")
            result[fid] = context
    return result


def _select_card(event_date: str, event_contains: str) -> pd.DataFrame:
    frame = pd.read_parquet(UPCOMING_FIGHTS_PATH).copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    mask = frame["event_date"].eq(pd.Timestamp(event_date).normalize())
    if event_contains:
        mask &= frame["event_name"].astype(str).str.contains(
            event_contains, case=False, regex=False, na=False
        )
    card = frame.loc[mask].copy()
    if card.empty:
        raise RuntimeError(
            f"No upcoming card matched date={event_date} event~={event_contains!r}"
        )
    card = card.sort_values("fight_order", na_position="last").reset_index(drop=True)
    print(f"Future card selected: {card['event_name'].iloc[0]} | fights={len(card)}")
    return card


def _build_target(card: pd.DataFrame, physical: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for rec in card.to_dict("records"):
        r_id, b_id = str(rec["red_fighter_id"]), str(rec["blue_fighter_id"])
        rows.append(
            {
                "fight_id": str(rec["fight_id"]),
                "event_name": str(rec["event_name"]),
                "event_date": pd.Timestamp(rec["event_date"]).normalize(),
                "r_id": r_id,
                "b_id": b_id,
                "r_name": str(rec["red_fighter"]),
                "b_name": str(rec["blue_fighter"]),
                "division": str(rec.get("division") or rec.get("weight_class") or "unknown"),
                "total_rounds": int(rec.get("total_rounds") or 3),
                "r_dob": physical.get(r_id, {}).get("dob"),
                "b_dob": physical.get(b_id, {}).get("dob"),
                "r_reach": physical.get(r_id, {}).get("reach", np.nan),
                "b_reach": physical.get(b_id, {}).get("reach", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _future_rows(
    target: pd.DataFrame,
    v2_latest: pd.DataFrame,
    v3_latest: pd.DataFrame,
    v2_pop: dict[str, object],
    v3_pop: dict[str, object],
) -> tuple[dict[str, dict[str, dict]], pd.DataFrame]:
    v2_lookup = {str(r["fighter_id"]): r for r in v2_latest.to_dict("records")}
    v3_lookup = {str(r["fighter_id"]): r for r in v3_latest.to_dict("records")}
    per_fight: dict[str, dict[str, dict]] = {}
    audit = []
    for rec in target.to_dict("records"):
        fight_id = str(rec["fight_id"])
        sides = {}
        for side, prefix in (("red", "r"), ("blue", "b")):
            fid = str(rec[f"{prefix}_id"])
            name = str(rec[f"{prefix}_name"])
            source = "V3" if fid in v3_lookup and fid in v2_lookup else "POP"
            v2 = dict(v2_lookup[fid]) if source == "V3" else dict(v2_pop)
            v3 = dict(v3_lookup[fid]) if source == "V3" else dict(v3_pop)
            for row in (v2, v3):
                row["fighter_id"] = fid
                row["fighter_name"] = name
            # Canonical C: V3 power is the only V3 replacement at the frozen
            # detailed-profile boundary; all other detailed fields stay V2.
            native_power = float(v3.get("striking_power_v3", 0.0))
            v2["striking_power"] = float(legacy_power_equivalent([native_power])[0])
            sides[side] = {"v2": v2, "v3": v3, "source": source}
            audit.append(
                {
                    "fight_id": fight_id,
                    "side": side,
                    "fighter_id": fid,
                    "fighter_name": name,
                    "profile_source": source,
                    "reach_inches": rec.get(f"{prefix}_reach", np.nan),
                }
            )
        per_fight[fight_id] = sides
    return per_fight, pd.DataFrame(audit)


def _summary(rows: list[dict], master_row: pd.Series, arm: str, profile_rows: dict) -> dict:
    paths = pd.DataFrame(rows)
    out = {
        "fight_id": str(master_row["fight_id"]),
        "event_name": str(master_row["event_name"]),
        "event_date": str(pd.Timestamp(master_row["event_date"]).date()),
        "arm": arm,
        "red": str(master_row["r_name"]),
        "blue": str(master_row["b_name"]),
        "red_profile": profile_rows["red"]["source"],
        "blue_profile": profile_rows["blue"]["source"],
        "red_reach_inches": master_row.get("r_reach", np.nan),
        "blue_reach_inches": master_row.get("b_reach", np.nan),
    }
    for side in ("red", "blue"):
        out[f"p_{side}_win"] = float((paths["winner"] == side).mean())
        for method in ("DEC", "KO_TKO", "SUB"):
            out[f"p_{side}_{method.lower()}"] = float(
                ((paths["winner"] == side) & (paths["method"] == method)).mean()
            )
        out[f"mean_{side}_standing_attempted"] = float(paths[f"{side}_standing_attempted"].mean())
        out[f"mean_{side}_standing_landed"] = float(paths[f"{side}_standing_landed"].mean())
    out["p_fight_dec"] = float((paths["method"] == "DEC").mean())
    out["p_fight_ko_tko"] = float((paths["method"] == "KO_TKO").mean())
    out["p_fight_sub"] = float((paths["method"] == "SUB").mean())
    out["mean_elapsed"] = float(paths["elapsed"].mean())
    return out


def run(args) -> pd.DataFrame:
    context = load_frozen_context(args.bundle)
    card = _select_card(args.event_date, args.event_contains)
    physical = _profile_context(card)
    target = _build_target(card, physical)

    v2_latest = pd.read_parquet(FSR_V2_LATEST_PATH).copy()
    v2_latest["fighter_id"] = v2_latest["fighter_id"].astype(str)
    v3_latest = pd.read_parquet(FSR_V3_LATEST_PATH).copy()
    v3_latest["fighter_id"] = v3_latest["fighter_id"].astype(str)
    v2_pop = _population_v2(v2_latest)
    v3_pop = _population_v3(v2_pop, v3_latest)
    future, profile_audit = _future_rows(target, v2_latest, v3_latest, v2_pop, v3_pop)

    uncertainty, pop_sd = _uncertainty_table()
    pop_unc = _population_uncertainty(v3_pop, pop_sd)
    kd_latest = _kd_latest()
    pop_kd = _population_kd()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_audit.to_csv(OUT_DIR / "sacramento_reach_profile_audit.csv", index=False)
    print(profile_audit.to_string(index=False))

    summaries = []
    for fight_index, master_row in target.iterrows():
        fight_id = str(master_row["fight_id"])
        profiles = future[fight_id]
        red_v2, blue_v2 = profiles["red"]["v2"], profiles["blue"]["v2"]
        red_v3, blue_v3 = profiles["red"]["v3"], profiles["blue"]["v3"]
        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        base_fight = fight_from_fsr_v2_rows(
            red_v2,
            blue_v2,
            fight_id=fight_id,
            date=event_date.date().isoformat(),
            division=str(master_row["division"]),
            rounds=int(master_row["total_rounds"]),
            red_age_years=fighter_age_years(master_row.get("r_dob"), event_date),
            blue_age_years=fighter_age_years(master_row.get("b_dob"), event_date),
        )

        red_id, blue_id = str(master_row["r_id"]), str(master_row["b_id"])
        red_unc = uncertainty.get(red_id, pop_unc)
        blue_unc = uncertainty.get(blue_id, pop_unc)
        red_kd = kd_latest.get(red_id, pop_kd)
        blue_kd = kd_latest.get(blue_id, pop_kd)
        submission_baseline = pd.DataFrame(
            [{
                "fight_id": fight_id,
                "submission_conversion_baseline": float(np.median([
                    red_v3["submission_conversion_baseline"],
                    blue_v3["submission_conversion_baseline"],
                ])),
            }]
        )

        print(
            f"[{fight_index + 1}/{len(target)}] {master_row['r_name']} vs {master_row['b_name']} "
            f"profiles={profiles['red']['source']}/{profiles['blue']['source']}"
        )
        arm_rows = {arm: [] for arm in ARMS}
        for path in range(args.paths):
            seed = args.seed + fight_index * 1_000_000 + path
            for arm in ARMS:
                epistemic_rng = np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET)
                matchup = initialize_path_matchup(
                    red_v3,
                    blue_v3,
                    red_unc,
                    blue_unc,
                    rng=epistemic_rng,
                    sample_epistemic=True,
                )
                red_kd_draw = sample_kd_resistance_latent(red_kd, epistemic_rng)
                blue_kd_draw = sample_kd_resistance_latent(blue_kd, epistemic_rng)
                path_fight = fight_with_kd_resistance(
                    base_fight,
                    red_native_resistance=red_kd_draw,
                    blue_native_resistance=blue_kd_draw,
                )
                features = build_sampled_fight_feature_rows_v3(
                    master_row,
                    red_record=red_v3,
                    blue_record=blue_v3,
                    red_traits=matchup.red,
                    blue_traits=matchup.blue,
                )
                if arm == "reach_off":
                    features["distance_reach_multiplier"] = 1.0
                pair, control = predict_feature_frame_v3(
                    features,
                    context["inference_models"],
                    context["submission_scale"],
                    context["conversion_offset"],
                    submission_baseline=submission_baseline,
                )
                sub_rate, convert = _submission_inputs(pair)
                budgets = _draw_budgets(
                    pair,
                    control.iloc[0],
                    context,
                    np.random.default_rng(seed),
                )
                result = simulate_detailed_path(
                    path_fight,
                    budgets,
                    sub_rate,
                    convert,
                    context["judge_model"],
                    context["judge_features"],
                    seed + DETAILED_PATH_SEED_OFFSET,
                )
                arm_rows[arm].append(result)

        for arm in ARMS:
            summaries.append(_summary(arm_rows[arm], master_row, arm, profiles))

    summary = pd.DataFrame(summaries)
    off = summary[summary["arm"].eq("reach_off")].set_index("fight_id")
    on = summary[summary["arm"].eq("reach_on")].set_index("fight_id")
    delta_rows = []
    for fight_id in off.index:
        row = {
            "fight_id": fight_id,
            "red": off.loc[fight_id, "red"],
            "blue": off.loc[fight_id, "blue"],
            "red_profile": off.loc[fight_id, "red_profile"],
            "blue_profile": off.loc[fight_id, "blue_profile"],
            "red_reach_inches": off.loc[fight_id, "red_reach_inches"],
            "blue_reach_inches": off.loc[fight_id, "blue_reach_inches"],
        }
        for col in (
            "p_red_win", "p_blue_win",
            "p_red_dec", "p_red_ko_tko", "p_red_sub",
            "p_blue_dec", "p_blue_ko_tko", "p_blue_sub",
            "p_fight_dec", "p_fight_ko_tko", "p_fight_sub",
            "mean_red_standing_attempted", "mean_blue_standing_attempted",
            "mean_red_standing_landed", "mean_blue_standing_landed",
            "mean_elapsed",
        ):
            row[f"off_{col}"] = float(off.loc[fight_id, col])
            row[f"on_{col}"] = float(on.loc[fight_id, col])
            row[f"delta_{col}"] = float(on.loc[fight_id, col] - off.loc[fight_id, col])
        delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)

    summary_path = OUT_DIR / "sacramento_reach_ab_2000paths_summary.csv"
    delta_path = OUT_DIR / "sacramento_reach_ab_2000paths_delta.csv"
    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)
    print("\nREACH A/B DELTA")
    print(
        delta[[
            "red", "blue", "red_reach_inches", "blue_reach_inches",
            "off_p_red_win", "on_p_red_win", "delta_p_red_win",
            "delta_mean_red_standing_attempted", "delta_mean_blue_standing_attempted",
        ]].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"summary: {summary_path}")
    print(f"delta:   {delta_path}")
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-date", default="2026-08-22")
    parser.add_argument("--event-contains", default="Hernandez vs. Rodrigues")
    parser.add_argument("--paths", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
