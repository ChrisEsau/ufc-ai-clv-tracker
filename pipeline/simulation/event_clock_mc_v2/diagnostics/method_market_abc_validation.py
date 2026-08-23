"""Held-out A/B/C validation of Event Clock fighter-method probabilities vs market.

A = frozen Event Clock V1 + frozen FSR V2
B = Event Clock V2 + FSR V3 posterior means
C = Event Clock V2 + FSR V3 validated path-level epistemic draws

The primary target is the six mutually exclusive fighter-method outcomes:
red DEC / KO_TKO / SUB and blue DEC / KO_TKO / SUB.

The diagnostic also joins historical fighter-method prices, proportionally normalizes
all six implied probabilities within each fight as a no-vig market reference, and
stratifies results by prior UFC fight experience to test the cold-start hypothesis.
No production mechanics are changed.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common.paths import MASTER_PATH
from pipeline.fsr_v3.paths import (
    FSR_V3_PREFIGHT_SNAPSHOTS_PATH,
    FSR_V3_PREFIGHT_UNCERTAINTY_PATH,
)
from pipeline.simulation.event_clock_mc_v1.diagnostics_stage9_final_flow import simulate_stage9_path
from pipeline.simulation.event_clock_mc_v1.frozen_inference import predict_target as predict_target_v1
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight import (
    SEED,
    simulate_detailed_path,
    summarize_fight,
)
from pipeline.simulation.event_clock_mc_v1.run_event_or_fight_frozen import (
    DEFAULT_BUNDLE_PATH as V1_BUNDLE_PATH,
    load_frozen_context as load_v1_context,
)
from pipeline.simulation.event_clock_mc_v2.feature_builder import build_sampled_fight_feature_rows_v3
from pipeline.simulation.event_clock_mc_v2.fit_event_clock_bundle import DEFAULT_BUNDLE_PATH as V2_BUNDLE_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import (
    historical_fighter_rows,
    historical_uncertainty_rows,
    initialize_path_matchup,
    load_prefight_snapshots,
    load_prefight_uncertainty,
)
from pipeline.simulation.event_clock_mc_v2.inference import (
    load_submission_baseline_v3,
    predict_feature_frame_v3,
    predict_target_v3,
)
from pipeline.simulation.event_clock_mc_v2.run_event_or_fight_frozen import (
    DETAILED_PATH_SEED_OFFSET,
    EPISTEMIC_SEED_OFFSET,
    _draw_budgets as draw_v2_budgets,
    _submission_inputs as submission_inputs_v2,
    load_frozen_context as load_v2_context,
)
from pipeline.simulation.event_mc_v1.diagnostics.fresh_100_fight_predictive_replay import (
    select_fresh_cohort,
)
from pipeline.simulation.event_mc_v1.diagnostics.population_validation import _fight

MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")
OUT_DIR = Path("data/diagnostics/event_clock_mc_v2/method_market_abc")

ARMS = ("A_v2_ecv1", "B_v3_means", "C_v3_epistemic")
METHOD_KEYS = {
    "DEC": "win_by_decision",
    "KO_TKO": "win_by_ko_tko_dq",
    "SUB": "win_by_submission",
}
METHOD_SUFFIX = {
    "DEC": "dec",
    "KO_TKO": "ko_tko",
    "SUB": "sub",
}
EDGE_THRESHOLDS = (0.0, 0.025, 0.05, 0.10, 0.15)


def _american_profit_per_1(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return float("nan")


def _prior_ufc_counts(master: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Prior UFC appearances by exact event date; same-event rows do not count."""
    rows = master[["fight_id", "event_date", "r_id", "b_id"]].copy()
    rows = rows.dropna(subset=["event_date", "r_id", "b_id"])
    rows["fight_id"] = rows["fight_id"].astype(str)
    rows["r_id"] = rows["r_id"].astype(str)
    rows["b_id"] = rows["b_id"].astype(str)
    counts: dict[str, int] = {}
    output: dict[tuple[str, str], int] = {}
    for _, day in rows.sort_values(["event_date", "fight_id"]).groupby("event_date", sort=True):
        for row in day.itertuples(index=False):
            output[(str(row.fight_id), str(row.r_id))] = counts.get(str(row.r_id), 0)
            output[(str(row.fight_id), str(row.b_id))] = counts.get(str(row.b_id), 0)
        for row in day.itertuples(index=False):
            counts[str(row.r_id)] = counts.get(str(row.r_id), 0) + 1
            counts[str(row.b_id)] = counts.get(str(row.b_id), 0) + 1
    return output


def _fight_bucket(red_prior: int, blue_prior: int) -> str:
    minimum = min(int(red_prior), int(blue_prior))
    if minimum == 0:
        return "debut_involved"
    if minimum <= 2:
        return "low_1_2"
    return "established_3plus"


def _fighter_bucket(prior: int) -> str:
    prior = int(prior)
    if prior == 0:
        return "debut_0"
    if prior <= 2:
        return "low_1_2"
    return "established_3plus"


def _prepare_market() -> tuple[pd.DataFrame, set[str]]:
    market = pd.read_parquet(MARKET_PATH).copy()
    market["fight_id"] = market["fight_id"].astype(str)
    market["outcome_fighter_id"] = market["outcome_fighter_id"].astype(str)
    market["market_key"] = market["market_key"].astype(str)
    market = market[market["market_key"].isin(METHOD_KEYS.values())].copy()
    market["implied_probability"] = pd.to_numeric(market["implied_probability"], errors="coerce")
    market["american_odds"] = pd.to_numeric(market["american_odds"], errors="coerce")
    market["won"] = market["won"].astype("boolean")
    market = market.dropna(subset=["implied_probability", "american_odds", "won"])

    identity = ["fight_id", "market_key", "outcome_fighter_id"]
    market = market.drop_duplicates(identity, keep="last")

    def valid_group(g: pd.DataFrame) -> bool:
        if len(g) != 6:
            return False
        fighters = list(g["outcome_fighter_id"].unique())
        if len(fighters) != 2:
            return False
        for fighter_id in fighters:
            keys = set(g.loc[g["outcome_fighter_id"] == fighter_id, "market_key"])
            if keys != set(METHOD_KEYS.values()):
                return False
        return True

    valid_ids = {
        str(fight_id)
        for fight_id, group in market.groupby("fight_id", sort=False)
        if valid_group(group)
    }
    market = market[market["fight_id"].isin(valid_ids)].copy()
    totals = market.groupby("fight_id")["implied_probability"].transform("sum")
    market["market_novig_probability"] = market["implied_probability"] / totals
    return market, valid_ids


def _select_cohort(
    master: pd.DataFrame,
    market_ids: set[str],
    quota_per_bucket: int,
) -> pd.DataFrame:
    # Canonical fresh universe: downstream calibration cohorts excluded by helper.
    fresh, _, _ = select_fresh_cohort(500, offset=0)
    fresh = fresh.copy()
    fresh["fight_id"] = fresh["fight_id"].astype(str)
    fresh["event_date"] = pd.to_datetime(fresh["event_date"], errors="raise").dt.normalize()
    fresh = fresh[fresh["fight_id"].isin(market_ids)].copy()

    prior = _prior_ufc_counts(master)
    red_prior, blue_prior, buckets = [], [], []
    for row in fresh.itertuples(index=False):
        rp = prior.get((str(row.fight_id), str(row.r_id)), 0)
        bp = prior.get((str(row.fight_id), str(row.b_id)), 0)
        red_prior.append(rp)
        blue_prior.append(bp)
        buckets.append(_fight_bucket(rp, bp))
    fresh["red_prior_ufc_fights"] = red_prior
    fresh["blue_prior_ufc_fights"] = blue_prior
    fresh["fight_evidence_bucket"] = buckets

    selected = []
    for bucket in ("debut_involved", "low_1_2", "established_3plus"):
        group = fresh[fresh["fight_evidence_bucket"] == bucket].sort_values(["event_date", "fight_id"])
        selected.append(group.head(quota_per_bucket))
    out = pd.concat(selected, ignore_index=True) if selected else fresh.iloc[0:0].copy()
    out = out.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No held-out fights with complete six-way method market coverage")
    return out


def _submission_inputs_v1(pair: pd.DataFrame) -> tuple[dict[str, float], float]:
    rates: dict[str, float] = {}
    conversion = None
    for side in ("red", "blue"):
        row = pair[pair["side"] == side].iloc[0]
        rates[side] = float(row["submission_clock_rate"])
        if conversion is None:
            conversion = float(row["submission_conversion_probability"])
    if conversion is None:
        raise RuntimeError("missing V1 submission conversion")
    return rates, float(conversion)


def _draw_v1_budgets(pair: pd.DataFrame, pair_info: pd.Series, context: dict, rng: np.random.Generator) -> dict:
    return simulate_stage9_path(
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


def _simulate_abc(target: pd.DataFrame, paths: int, seed0: int) -> pd.DataFrame:
    v1_context = load_v1_context(V1_BUNDLE_PATH)
    v2_context = load_v2_context(V2_BUNDLE_PATH)
    fsr_v3 = load_prefight_snapshots(FSR_V3_PREFIGHT_SNAPSHOTS_PATH)
    uncertainty = load_prefight_uncertainty(FSR_V3_PREFIGHT_UNCERTAINTY_PATH)
    submission_baseline = load_submission_baseline_v3()

    v1_test, v1_control = predict_target_v1(
        target,
        v1_context["fsr_all"],
        v1_context["inference_models"],
        v1_context["submission_scale"],
        v1_context["conversion_offset"],
    )
    v2_test, v2_control = predict_target_v3(
        target,
        fsr_v3,
        v2_context["inference_models"],
        v2_context["submission_scale"],
        v2_context["conversion_offset"],
    )

    v1_groups = {str(fid): g.copy() for fid, g in v1_test.groupby("fight_id", sort=False)}
    v2_groups = {str(fid): g.copy() for fid, g in v2_test.groupby("fight_id", sort=False)}
    v1_info = {str(row["fight_id"]): row for _, row in v1_control.iterrows()}
    v2_info = {str(row["fight_id"]): row for _, row in v2_control.iterrows()}
    master_lookup = {str(row["fight_id"]): row for _, row in target.iterrows()}

    summaries: list[dict] = []
    fight_ids = target["fight_id"].astype(str).tolist()
    for fight_index, fight_id in enumerate(fight_ids):
        master_row = master_lookup[fight_id]
        pair_a = v1_groups[fight_id]
        pair_b = v2_groups[fight_id]
        info_a = v1_info[fight_id]
        info_b = v2_info[fight_id]
        fight_a = _fight(master_row, v1_context["fsr_all"])
        fight_b = _fight(master_row, v2_context["fsr_all"])
        sub_a, conv_a = _submission_inputs_v1(pair_a)
        sub_b, conv_b = submission_inputs_v2(pair_b)

        event_date = pd.Timestamp(master_row["event_date"]).normalize()
        red_row, blue_row = historical_fighter_rows(
            fsr_v3,
            event_date=event_date,
            fight_id=fight_id,
            fighter_ids=(str(master_row["r_id"]), str(master_row["b_id"])),
        )
        red_unc = historical_uncertainty_rows(
            uncertainty,
            event_date=event_date,
            fight_id=fight_id,
            fighter_id=str(master_row["r_id"]),
        )
        blue_unc = historical_uncertainty_rows(
            uncertainty,
            event_date=event_date,
            fight_id=fight_id,
            fighter_id=str(master_row["b_id"]),
        )

        if fight_index % 10 == 0:
            print(f"[{fight_index + 1}/{len(fight_ids)}] {master_row['r_name']} vs {master_row['b_name']}")

        arm_paths = {arm: [] for arm in ARMS}
        for path in range(paths):
            seed = seed0 + fight_index * 1_000_000 + path

            budget_a = _draw_v1_budgets(pair_a, info_a, v1_context, np.random.default_rng(seed))
            result_a = simulate_detailed_path(
                fight_a,
                budget_a,
                sub_a,
                conv_a,
                v1_context["judge_model"],
                v1_context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            arm_paths["A_v2_ecv1"].append(result_a)

            budget_b = draw_v2_budgets(pair_b, info_b, v2_context, np.random.default_rng(seed))
            result_b = simulate_detailed_path(
                fight_b,
                budget_b,
                sub_b,
                conv_b,
                v2_context["judge_model"],
                v2_context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            arm_paths["B_v3_means"].append(result_b)

            matchup = initialize_path_matchup(
                red_row,
                blue_row,
                red_unc,
                blue_unc,
                rng=np.random.default_rng(seed + EPISTEMIC_SEED_OFFSET),
                sample_epistemic=True,
            )
            path_features = build_sampled_fight_feature_rows_v3(
                master_row,
                red_record=red_row.to_dict(),
                blue_record=blue_row.to_dict(),
                red_traits=matchup.red,
                blue_traits=matchup.blue,
            )
            pair_c, control_c = predict_feature_frame_v3(
                path_features,
                v2_context["inference_models"],
                v2_context["submission_scale"],
                v2_context["conversion_offset"],
                submission_baseline=submission_baseline,
            )
            info_c = control_c.iloc[0]
            sub_c, conv_c = submission_inputs_v2(pair_c)
            budget_c = draw_v2_budgets(pair_c, info_c, v2_context, np.random.default_rng(seed))
            result_c = simulate_detailed_path(
                fight_b,
                budget_c,
                sub_c,
                conv_c,
                v2_context["judge_model"],
                v2_context["judge_features"],
                seed + DETAILED_PATH_SEED_OFFSET,
            )
            arm_paths["C_v3_epistemic"].append(result_c)

        for arm in ARMS:
            pair = pair_a if arm == "A_v2_ecv1" else pair_b
            summary = summarize_fight(fight_id, pair, arm_paths[arm], master_row)
            summary["arm"] = arm
            summary["event_date"] = master_row["event_date"]
            summary["red_id"] = str(master_row["r_id"])
            summary["blue_id"] = str(master_row["b_id"])
            summary["red_prior_ufc_fights"] = int(master_row["red_prior_ufc_fights"])
            summary["blue_prior_ufc_fights"] = int(master_row["blue_prior_ufc_fights"])
            summary["fight_evidence_bucket"] = master_row["fight_evidence_bucket"]
            summaries.append(summary)
    return pd.DataFrame(summaries)


def _sixway_columns() -> list[tuple[str, str, str]]:
    return [
        ("red", "DEC", "p_red_dec"),
        ("red", "KO_TKO", "p_red_ko_tko"),
        ("red", "SUB", "p_red_sub"),
        ("blue", "DEC", "p_blue_dec"),
        ("blue", "KO_TKO", "p_blue_ko_tko"),
        ("blue", "SUB", "p_blue_sub"),
    ]


def _fight_scores(summary: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market_lookup = {
        (str(row.fight_id), str(row.outcome_fighter_id), str(row.market_key)): float(row.market_novig_probability)
        for row in market.itertuples(index=False)
    }
    rows: list[dict] = []
    for row in summary.itertuples(index=False):
        probs = []
        market_probs = []
        labels = []
        for side, method, col in _sixway_columns():
            fighter_id = str(row.red_id if side == "red" else row.blue_id)
            probs.append(float(getattr(row, col)))
            market_probs.append(market_lookup[(str(row.fight_id), fighter_id, METHOD_KEYS[method])])
            labels.append((side, method))
        p = np.asarray(probs, dtype=float)
        q = np.asarray(market_probs, dtype=float)
        # Minor path rounding guard.
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        q = np.clip(q, 1e-12, 1.0)
        q = q / q.sum()
        actual = (str(row.actual_winner), str(row.actual_method))
        idx = labels.index(actual)
        y = np.zeros(6, dtype=float)
        y[idx] = 1.0
        rows.append({
            "fight_id": str(row.fight_id),
            "arm": str(row.arm),
            "fight_evidence_bucket": str(row.fight_evidence_bucket),
            "actual_class": f"{actual[0]}_{actual[1]}",
            "p_actual": float(p[idx]),
            "market_p_actual": float(q[idx]),
            "sixway_log_loss": float(-math.log(p[idx])),
            "market_sixway_log_loss": float(-math.log(q[idx])),
            "sixway_brier": float(np.sum((p - y) ** 2)),
            "market_sixway_brier": float(np.sum((q - y) ** 2)),
            "top1_correct": int(int(np.argmax(p)) == idx),
            "market_top1_correct": int(int(np.argmax(q)) == idx),
            "entropy": float(-np.sum(p * np.log(p))),
            "market_entropy": float(-np.sum(q * np.log(q))),
            "max_probability": float(p.max()),
            "market_max_probability": float(q.max()),
        })
    return pd.DataFrame(rows)


def _outcome_rows(summary: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market_keyed = market.set_index(["fight_id", "outcome_fighter_id", "market_key"], drop=False)
    records: list[dict] = []
    for row in summary.itertuples(index=False):
        for side, method, col in _sixway_columns():
            fighter_id = str(row.red_id if side == "red" else row.blue_id)
            fighter = str(row.red if side == "red" else row.blue)
            prior = int(row.red_prior_ufc_fights if side == "red" else row.blue_prior_ufc_fights)
            key = (str(row.fight_id), fighter_id, METHOD_KEYS[method])
            m = market_keyed.loc[key]
            model_p = float(getattr(row, col))
            raw_implied = float(m["implied_probability"])
            novig = float(m["market_novig_probability"])
            records.append({
                "fight_id": str(row.fight_id),
                "arm": str(row.arm),
                "fight_evidence_bucket": str(row.fight_evidence_bucket),
                "side": side,
                "fighter_id": fighter_id,
                "fighter": fighter,
                "fighter_prior_ufc_fights": prior,
                "fighter_evidence_bucket": _fighter_bucket(prior),
                "method": method,
                "market_key": METHOD_KEYS[method],
                "model_probability": model_p,
                "market_implied_probability": raw_implied,
                "market_novig_probability": novig,
                "raw_edge": model_p - raw_implied,
                "novig_edge": model_p - novig,
                "american_odds": float(m["american_odds"]),
                "won": int(bool(m["won"])),
            })
    out = pd.DataFrame(records)
    out["model_binary_brier"] = (out["model_probability"] - out["won"]) ** 2
    out["market_binary_brier"] = (out["market_novig_probability"] - out["won"]) ** 2
    eps = 1e-12
    out["model_binary_log_loss"] = -(
        out["won"] * np.log(np.clip(out["model_probability"], eps, 1 - eps))
        + (1 - out["won"]) * np.log(np.clip(1 - out["model_probability"], eps, 1 - eps))
    )
    out["market_binary_log_loss"] = -(
        out["won"] * np.log(np.clip(out["market_novig_probability"], eps, 1 - eps))
        + (1 - out["won"]) * np.log(np.clip(1 - out["market_novig_probability"], eps, 1 - eps))
    )
    return out


def _aggregate_fight_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, bucket), group in scores.groupby(["arm", "fight_evidence_bucket"], sort=False):
        rows.append({
            "arm": arm,
            "fight_evidence_bucket": bucket,
            "fights": len(group),
            "sixway_log_loss": group["sixway_log_loss"].mean(),
            "sixway_brier": group["sixway_brier"].mean(),
            "mean_p_actual": group["p_actual"].mean(),
            "top1_accuracy": group["top1_correct"].mean(),
            "mean_entropy": group["entropy"].mean(),
            "mean_max_probability": group["max_probability"].mean(),
            "market_log_loss": group["market_sixway_log_loss"].mean(),
            "market_brier": group["market_sixway_brier"].mean(),
            "market_top1_accuracy": group["market_top1_correct"].mean(),
        })
    for arm, group in scores.groupby("arm", sort=False):
        rows.append({
            "arm": arm,
            "fight_evidence_bucket": "ALL",
            "fights": len(group),
            "sixway_log_loss": group["sixway_log_loss"].mean(),
            "sixway_brier": group["sixway_brier"].mean(),
            "mean_p_actual": group["p_actual"].mean(),
            "top1_accuracy": group["top1_correct"].mean(),
            "mean_entropy": group["entropy"].mean(),
            "mean_max_probability": group["max_probability"].mean(),
            "market_log_loss": group["market_sixway_log_loss"].mean(),
            "market_brier": group["market_sixway_brier"].mean(),
            "market_top1_accuracy": group["market_top1_correct"].mean(),
        })
    return pd.DataFrame(rows)


def _outcome_metrics(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, bucket), g in outcomes.groupby(["arm", "fighter_evidence_bucket"], sort=False):
        row = {
            "arm": arm,
            "fighter_evidence_bucket": bucket,
            "outcomes": len(g),
            "model_binary_brier": g["model_binary_brier"].mean(),
            "market_binary_brier": g["market_binary_brier"].mean(),
            "brier_gain_vs_market": g["market_binary_brier"].mean() - g["model_binary_brier"].mean(),
            "model_binary_log_loss": g["model_binary_log_loss"].mean(),
            "market_binary_log_loss": g["market_binary_log_loss"].mean(),
            "log_loss_gain_vs_market": g["market_binary_log_loss"].mean() - g["model_binary_log_loss"].mean(),
            "mean_abs_novig_edge": g["novig_edge"].abs().mean(),
        }
        for threshold in (0.30, 0.40, 0.50):
            mask = g["model_probability"] >= threshold
            row[f"count_p_ge_{int(threshold*100)}"] = int(mask.sum())
            row[f"false_rate_p_ge_{int(threshold*100)}"] = float(1 - g.loc[mask, "won"].mean()) if mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _bet_metrics(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, bucket), group in outcomes.groupby(["arm", "fighter_evidence_bucket"], sort=False):
        for threshold in EDGE_THRESHOLDS:
            bets = group[group["raw_edge"] >= threshold].copy()
            if bets.empty:
                rows.append({
                    "arm": arm,
                    "fighter_evidence_bucket": bucket,
                    "raw_edge_threshold": threshold,
                    "bets": 0,
                    "win_rate": np.nan,
                    "mean_raw_edge": np.nan,
                    "mean_novig_edge": np.nan,
                    "flat_roi": np.nan,
                })
                continue
            profit = []
            for r in bets.itertuples(index=False):
                profit.append(_american_profit_per_1(r.american_odds) if int(r.won) else -1.0)
            rows.append({
                "arm": arm,
                "fighter_evidence_bucket": bucket,
                "raw_edge_threshold": threshold,
                "bets": len(bets),
                "win_rate": bets["won"].mean(),
                "mean_raw_edge": bets["raw_edge"].mean(),
                "mean_novig_edge": bets["novig_edge"].mean(),
                "flat_roi": float(np.mean(profit)),
            })
    return pd.DataFrame(rows)


def _edge_bins(outcomes: pd.DataFrame) -> pd.DataFrame:
    bins = [-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 1.0]
    labels = ["<-15", "-15:-10", "-10:-5", "-5:0", "0:5", "5:10", "10:15", ">=15"]
    temp = outcomes.copy()
    temp["edge_bin"] = pd.cut(temp["novig_edge"], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for (arm, bucket, edge_bin), g in temp.groupby(["arm", "fighter_evidence_bucket", "edge_bin"], observed=True):
        rows.append({
            "arm": arm,
            "fighter_evidence_bucket": bucket,
            "edge_bin": str(edge_bin),
            "outcomes": len(g),
            "mean_model_probability": g["model_probability"].mean(),
            "mean_market_novig_probability": g["market_novig_probability"].mean(),
            "mean_edge": g["novig_edge"].mean(),
            "actual_rate": g["won"].mean(),
            "realized_minus_market": g["won"].mean() - g["market_novig_probability"].mean(),
        })
    return pd.DataFrame(rows)


def _fighter_profile_metrics(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, fight_id, fighter_id), g in outcomes.groupby(["arm", "fight_id", "fighter_id"], sort=False):
        total = g["model_probability"].sum()
        conditional = g["model_probability"].to_numpy(float) / max(float(total), 1e-12)
        market_total = g["market_novig_probability"].sum()
        market_cond = g["market_novig_probability"].to_numpy(float) / max(float(market_total), 1e-12)
        rows.append({
            "arm": arm,
            "fight_id": fight_id,
            "fighter_id": fighter_id,
            "fighter": g["fighter"].iloc[0],
            "fighter_prior_ufc_fights": int(g["fighter_prior_ufc_fights"].iloc[0]),
            "fighter_evidence_bucket": g["fighter_evidence_bucket"].iloc[0],
            "fighter_win_probability": float(total),
            "max_method_probability": float(g["model_probability"].max()),
            "conditional_method_max": float(np.max(conditional)),
            "conditional_method_entropy": float(-np.sum(np.clip(conditional, 1e-12, 1.0) * np.log(np.clip(conditional, 1e-12, 1.0)))),
            "market_fighter_win_probability": float(market_total),
            "market_conditional_method_max": float(np.max(market_cond)),
        })
    profiles = pd.DataFrame(rows)
    return profiles.groupby(["arm", "fighter_evidence_bucket"], as_index=False).agg(
        fighters=("fighter_id", "count"),
        mean_fighter_win_probability=("fighter_win_probability", "mean"),
        mean_max_method_probability=("max_method_probability", "mean"),
        mean_conditional_method_max=("conditional_method_max", "mean"),
        mean_conditional_method_entropy=("conditional_method_entropy", "mean"),
        market_mean_conditional_method_max=("market_conditional_method_max", "mean"),
    )


def _bootstrap_deltas(scores: pd.DataFrame, reps: int = 5000, seed: int = 9191) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    comparisons = [
        ("B_v3_means", "A_v2_ecv1"),
        ("C_v3_epistemic", "A_v2_ecv1"),
        ("C_v3_epistemic", "B_v3_means"),
    ]
    for bucket in ["ALL", "debut_involved", "low_1_2", "established_3plus"]:
        frame = scores if bucket == "ALL" else scores[scores["fight_evidence_bucket"] == bucket]
        for left, right in comparisons:
            l = frame[frame["arm"] == left].set_index("fight_id")
            r = frame[frame["arm"] == right].set_index("fight_id")
            common = l.index.intersection(r.index)
            if len(common) == 0:
                continue
            for metric in ("sixway_log_loss", "sixway_brier", "p_actual"):
                delta = (l.loc[common, metric] - r.loc[common, metric]).to_numpy(float)
                samples = np.empty(reps, dtype=float)
                for i in range(reps):
                    idx = rng.integers(0, len(delta), size=len(delta))
                    samples[i] = delta[idx].mean()
                rows.append({
                    "fight_evidence_bucket": bucket,
                    "comparison": f"{left}_minus_{right}",
                    "metric": metric,
                    "fights": len(delta),
                    "mean_delta": float(delta.mean()),
                    "ci_2_5": float(np.quantile(samples, 0.025)),
                    "ci_97_5": float(np.quantile(samples, 0.975)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--quota-per-bucket", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    master = pd.read_parquet(MASTER_PATH).drop_duplicates("fight_id").copy()
    master["fight_id"] = master["fight_id"].astype(str)
    master["event_date"] = pd.to_datetime(master["date"], errors="coerce").dt.normalize()
    market, market_ids = _prepare_market()
    cohort = _select_cohort(master, market_ids, args.quota_per_bucket)

    print("=" * 160)
    print("FSR V2 / V3 METHOD-MARKET A/B/C VALIDATION")
    print("=" * 160)
    print(f"selected fights: {len(cohort)} | paths/fight/arm: {args.paths}")
    print("selection: canonical fresh-500 universe intersect complete six-way method markets, then deterministic evidence quotas")
    print("method market reference: legacy_consensus; proportional six-way no-vig normalization")
    print("fight evidence counts:")
    print(cohort["fight_evidence_bucket"].value_counts().to_string())
    print("date range:", cohort["event_date"].min().date(), "through", cohort["event_date"].max().date())

    summary = _simulate_abc(cohort, args.paths, args.seed)
    market_selected = market[market["fight_id"].isin(set(cohort["fight_id"].astype(str)))].copy()
    scores = _fight_scores(summary, market_selected)
    outcomes = _outcome_rows(summary, market_selected)
    fight_metrics = _aggregate_fight_scores(scores)
    outcome_metrics = _outcome_metrics(outcomes)
    bet_metrics = _bet_metrics(outcomes)
    edge_bins = _edge_bins(outcomes)
    fighter_profiles = _fighter_profile_metrics(outcomes)
    bootstrap = _bootstrap_deltas(scores)

    print("\nSIX-WAY FIGHT METRICS")
    display = fight_metrics.sort_values(["fight_evidence_bucket", "arm"])
    print(display.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nOUTCOME-LEVEL MODEL VS MARKET METRICS")
    print(outcome_metrics.sort_values(["fighter_evidence_bucket", "arm"]).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nFIGHTER METHOD SHARPNESS / COLD-START PROFILE")
    print(fighter_profiles.sort_values(["fighter_evidence_bucket", "arm"]).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nRAW-EDGE BET DIAGNOSTIC (5% and 10% thresholds)")
    focus_bets = bet_metrics[bet_metrics["raw_edge_threshold"].isin([0.05, 0.10])]
    print(focus_bets.sort_values(["fighter_evidence_bucket", "raw_edge_threshold", "arm"]).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nPAIRED BOOTSTRAP DELTAS (negative is better for log loss/Brier; positive is better for p_actual)")
    print(bootstrap.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Headline B vs C cold-start comparison.
    debut_scores = fight_metrics[fight_metrics["fight_evidence_bucket"] == "debut_involved"].set_index("arm")
    debut_outcomes = outcome_metrics[outcome_metrics["fighter_evidence_bucket"] == "debut_0"].set_index("arm")
    debut_profiles = fighter_profiles[fighter_profiles["fighter_evidence_bucket"] == "debut_0"].set_index("arm")
    print("\nCOLD-START HEADLINE")
    for arm in ARMS:
        if arm in debut_scores.index:
            print(
                arm,
                f"debut-fight sixway LL={debut_scores.loc[arm, 'sixway_log_loss']:.5f}",
                f"Brier={debut_scores.loc[arm, 'sixway_brier']:.5f}",
                f"P(actual)={debut_scores.loc[arm, 'mean_p_actual']:.5f}",
            )
        if arm in debut_outcomes.index and arm in debut_profiles.index:
            print(
                "   debut fighter outcomes:",
                f"binary LL={debut_outcomes.loc[arm, 'model_binary_log_loss']:.5f}",
                f"Brier={debut_outcomes.loc[arm, 'model_binary_brier']:.5f}",
                f"mean max method={debut_profiles.loc[arm, 'mean_max_method_probability']:.5f}",
                f"conditional max={debut_profiles.loc[arm, 'mean_conditional_method_max']:.5f}",
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"abc_{len(cohort)}f_{args.paths}paths"
    files = {
        "cohort": cohort,
        "summary": summary,
        "fight_scores": scores,
        "fight_metrics": fight_metrics,
        "outcomes": outcomes,
        "outcome_metrics": outcome_metrics,
        "bet_metrics": bet_metrics,
        "edge_bins": edge_bins,
        "fighter_profiles": fighter_profiles,
        "bootstrap": bootstrap,
    }
    for name, frame in files.items():
        path = args.out_dir / f"{stem}_{name}.csv"
        frame.to_csv(path, index=False)
        print(path)


if __name__ == "__main__":
    main()
