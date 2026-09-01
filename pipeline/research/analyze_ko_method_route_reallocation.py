from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method

OUT = Path("data/research/prop_mispricing")
MASTER = Path("data/master/ufc_master.parquet")

ROWS_OUT = OUT / "ko_method_route_reallocation_side_rows.csv"
WINNERS_OUT = OUT / "ko_method_route_reallocation_winners.csv"
THRESHOLDS_OUT = OUT / "ko_method_route_reallocation_thresholds.csv"
FEATURE_BINS_OUT = OUT / "ko_method_route_reallocation_feature_bins.csv"
GROUP_PERIODS_OUT = OUT / "ko_method_route_reallocation_group_periods.csv"
GROUP_SUMMARY_OUT = OUT / "ko_method_route_reallocation_group_summary.csv"
PAIR_OUT = OUT / "ko_method_route_reallocation_route_pairs.csv"
SUMMARY_OUT = OUT / "ko_method_route_reallocation_summary.json"

DEV_START = pd.Timestamp("2021-01-01")
DEV_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-12-31")
CUTOFF = VAL_END
YEARS = [2021, 2022, 2023, 2024]
EPS = 1e-12
LOW_Q = 0.40
HIGH_Q = 0.60

TARGET_METHOD = {
    0: ("red", "KO"),
    1: ("red", "SUB"),
    2: ("red", "DEC"),
    3: ("blue", "KO"),
    4: ("blue", "SUB"),
    5: ("blue", "DEC"),
}

RAW_FEATURE_SPECS = {
    "ewm_str_acc": ("r_ewm_str_acc", "b_ewm_str_acc"),
    "ewm_splm": ("r_ewm_splm", "b_ewm_splm"),
    "ewm_sapm": ("r_ewm_sapm", "b_ewm_sapm"),
    "ewm_str_def": ("r_ewm_str_def", "b_ewm_str_def"),
    "ewm_td_avg": ("r_ewm_td_avg", "b_ewm_td_avg"),
    "ewm_td_acc": ("r_ewm_td_acc", "b_ewm_td_acc"),
    "ewm_td_def": ("r_ewm_td_def", "b_ewm_td_def"),
    "ewm_sub_avg": ("r_ewm_sub_avg", "b_ewm_sub_avg"),
    "ewm_ctrl_per_min": ("r_ewm_ctrl_per_min", "b_ewm_ctrl_per_min"),
    "ewm_ctrl_against_per_min": ("r_ewm_ctrl_against_per_min", "b_ewm_ctrl_against_per_min"),
    "ewm_kd_avg": ("r_ewm_kd_avg", "b_ewm_kd_avg"),
    "ewm_kd_absorbed_avg": ("r_ewm_kd_absorbed_avg", "b_ewm_kd_absorbed_avg"),
    "ewm_recent_splm": ("r_ewm_recent_splm", "b_ewm_recent_splm"),
    "ewm_recent_sapm": ("r_ewm_recent_sapm", "b_ewm_recent_sapm"),
    "ewm_recent_td_avg": ("r_ewm_recent_td_avg", "b_ewm_recent_td_avg"),
    "ewm_recent_finish_rate": ("r_ewm_recent_finish_rate", "b_ewm_recent_finish_rate"),
    "pre_splm": ("r_pre_splm", "b_pre_splm"),
    "pre_str_acc": ("r_pre_str_acc", "b_pre_str_acc"),
    "pre_str_def": ("r_pre_str_def", "b_pre_str_def"),
    "pre_td_avg": ("r_pre_td_avg", "b_pre_td_avg"),
    "pre_sub_avg": ("r_pre_sub_avg", "b_pre_sub_avg"),
    "pre_ctrl_per_min": ("r_pre_ctrl_per_min", "b_pre_ctrl_per_min"),
    "pre_ko_rate": ("r_pre_ko_rate", "b_pre_ko_rate"),
    "pre_finish_rate": ("r_pre_finish_rate", "b_pre_finish_rate"),
    "pre_decision_dependency": ("r_pre_decision_dependency", "b_pre_decision_dependency"),
    "pre_ko_dependency": ("r_pre_ko_dependency", "b_pre_ko_dependency"),
    "pre_submission_dependency": ("r_pre_submission_dependency", "b_pre_submission_dependency"),
    "pre_avg_fight_time": ("r_pre_avg_fight_time", "b_pre_avg_fight_time"),
    "pre_win_streak": ("r_pre_win_streak", "b_pre_win_streak"),
    "recent_form_win_streak": ("r_recent_form_win_streak", "b_recent_form_win_streak"),
    "style_ko_finisher_score": ("r_pre_style_ko_finisher_score", "b_pre_style_ko_finisher_score"),
    "style_submission_grappler_score": ("r_pre_style_submission_grappler_score", "b_pre_style_submission_grappler_score"),
    "style_control_wrestler_score": ("r_pre_style_control_wrestler_score", "b_pre_style_control_wrestler_score"),
    "style_decision_technician_score": ("r_pre_style_decision_technician_score", "b_pre_style_decision_technician_score"),
}

BIN_FEATURES = [
    "ewm_str_acc_diff",
    "height_diff",
    "reach_diff",
    "ewm_splm_diff",
    "fighter_est_sig_att_per_min",
    "fighter_ewm_td_avg",
    "fighter_ewm_ctrl_per_min",
    "fighter_ewm_sub_avg",
    "fighter_pre_decision_dependency",
    "fighter_pre_ko_rate",
    "opponent_ewm_str_def",
    "opponent_ewm_sub_avg",
    "fighter_ewm_kd_avg",
    "opponent_ewm_kd_absorbed_avg",
    "win_streak_diff",
    "recent_form_win_streak_diff",
    "prior_r1_ko_wins",
    "recent5_r1_ko_wins",
]


def _is_ko(method_name: object) -> bool:
    s = str(method_name).upper()
    return "KO" in s or "TKO" in s


def _build_prefight_r1(cutoff: pd.Timestamp) -> pd.DataFrame:
    m = pd.read_parquet(MASTER, filters=[("date", "<=", cutoff)]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["date", "fight_id", "r_id", "b_id"])
    m = m[m["date"] <= cutoff].sort_values(["date", "fight_id"])
    if (m["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered prefight R1 history builder")

    stats = defaultdict(
        lambda: {
            "fights": 0,
            "r1_ko_wins": 0,
            "r1_ko_losses": 0,
            "ko_wins": 0,
            "recent5": deque(maxlen=5),
        }
    )
    rows = []
    for r in m.itertuples(index=False):
        fid = str(r.fight_id)
        rid = str(r.r_id)
        bid = str(r.b_id)
        for side, fighter, opp in [("red", rid, bid), ("blue", bid, rid)]:
            s = stats[fighter]
            o = stats[opp]
            rows.append(
                {
                    "fight_id": fid,
                    "side": side,
                    "prior_ufc_fights": s["fights"],
                    "prior_r1_ko_wins": s["r1_ko_wins"],
                    "prior_r1_ko_win_rate": s["r1_ko_wins"] / s["fights"] if s["fights"] else 0.0,
                    "prior_ko_wins": s["ko_wins"],
                    "prior_r1_share_of_ko_wins": s["r1_ko_wins"] / s["ko_wins"] if s["ko_wins"] else 0.0,
                    "recent5_r1_ko_wins": int(sum(s["recent5"])),
                    "opp_prior_r1_ko_losses": o["r1_ko_losses"],
                    "opp_prior_r1_ko_loss_rate": o["r1_ko_losses"] / o["fights"] if o["fights"] else 0.0,
                }
            )

        winner = str(getattr(r, "winner_id", None))
        ko = _is_ko(getattr(r, "method", None))
        finish_round = pd.to_numeric(getattr(r, "finish_round", np.nan), errors="coerce")
        r1 = bool(ko and finish_round == 1)
        for fighter in [rid, bid]:
            stats[fighter]["fights"] += 1
            stats[fighter]["recent5"].append(1 if (r1 and fighter == winner) else 0)
        if ko and winner in (rid, bid):
            loser = bid if winner == rid else rid
            stats[winner]["ko_wins"] += 1
            if r1:
                stats[winner]["r1_ko_wins"] += 1
                stats[loser]["r1_ko_losses"] += 1

    return pd.DataFrame(rows)


def _read_ml_market() -> pd.DataFrame:
    m = pd.read_parquet(method.MARKET_PATH, filters=[("date", "<=", CUTOFF)]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[
        (m["date"] <= CUTOFF)
        & (m["bookmaker"] == "legacy_consensus")
        & (m["result_status"] == "graded")
        & (m["market_key"] == "moneyline")
        & m["outcome_side"].astype(str).isin(["red", "blue"])
    ].copy()
    if (m["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered moneyline market reader")
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m = m.dropna(subset=["fight_id", "implied_probability"])
    m["fight_id"] = m["fight_id"].astype(str)
    counts = m.groupby(["fight_id", "outcome_side"]).size()
    good = set((str(fid), str(side)) for fid, side in counts[counts.eq(1)].index)
    m = m[m.apply(lambda r: (str(r["fight_id"]), str(r["outcome_side"])) in good, axis=1)]
    piv = m.pivot(index="fight_id", columns="outcome_side", values="implied_probability")
    piv = piv.dropna(subset=["red", "blue"])
    total = piv["red"] + piv["blue"]
    out = pd.DataFrame(
        {
            "fight_id": piv.index.astype(str),
            "market_ml_probability_red": piv["red"].to_numpy(float) / total.to_numpy(float),
            "market_ml_probability_blue": piv["blue"].to_numpy(float) / total.to_numpy(float),
            "market_ml_overround": total.to_numpy(float),
        }
    )
    return out.reset_index(drop=True)


def _read_raw_side_features() -> tuple[pd.DataFrame, list[str], list[str]]:
    fv = pd.read_parquet(method.FEATURE_PATH, filters=[("date", "<=", CUTOFF)]).copy()
    fv["date"] = pd.to_datetime(fv["date"], errors="coerce")
    fv = fv[fv["date"] <= CUTOFF].copy()
    if (fv["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered raw side-feature reader")
    fv["fight_id"] = fv["fight_id"].astype(str)

    available = []
    missing_concepts = []
    for concept, (r_col, b_col) in RAW_FEATURE_SPECS.items():
        if r_col in fv.columns and b_col in fv.columns:
            available.extend([r_col, b_col])
        else:
            missing_concepts.append(concept)
    available = sorted(set(available))
    raw = fv[["fight_id"] + available].drop_duplicates("fight_id")
    if raw["fight_id"].duplicated().any():
        raise RuntimeError("duplicate raw feature-view fight_id")
    return raw, available, missing_concepts


def _accuracy_decimal(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    x = x.where(x.abs() <= 1.5, x / 100.0)
    return x.where((x > 0.02) & (x <= 1.0))


def _build_side_rows() -> tuple[pd.DataFrame, list[str], list[str]]:
    original = method.DEV_CUTOFF
    method.DEV_CUTOFF = CUTOFF
    try:
        fights, features, _ = method._build_rows(True, True)
    finally:
        method.DEV_CUTOFF = original

    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["date"] = pd.to_datetime(fights["date"], errors="coerce")
    fights = fights[fights["date"].between(DEV_START, VAL_END)].copy()
    if (fights["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered route-reallocation base rows")

    ml = _read_ml_market()
    raw, raw_columns, missing_raw = _read_raw_side_features()
    fights = fights.merge(ml, on="fight_id", how="inner", validate="one_to_one")
    fights = fights.merge(raw, on="fight_id", how="left", validate="one_to_one")

    parts = []
    for side, sign in [("red", 1.0), ("blue", -1.0)]:
        other = "blue" if side == "red" else "red"
        base_cols = [
            "fight_id",
            "date",
            "event_name",
            "red_fighter",
            "blue_fighter",
            "target",
            "betting_eligible",
            "market_overround",
            "market_ml_overround",
            f"market_{side}_ko",
            f"market_{side}_sub",
            f"market_{side}_dec",
            f"market_ml_probability_{side}",
        ]
        x = fights[base_cols + features + raw_columns].copy()
        x["side"] = side
        x["fighter"] = x[f"{side}_fighter"]
        x["opponent"] = x[f"{other}_fighter"]
        x["actual_win"] = np.where(side == "red", x["target"].astype(int) < 3, x["target"].astype(int) >= 3).astype(int)
        x["actual_fight_method"] = x["target"].astype(int).map(lambda t: TARGET_METHOD[int(t)][1])
        x["actual_side_method"] = np.where(x["actual_win"].eq(1), x["actual_fight_method"], "LOSS")
        x["actual_ko"] = ((x["actual_win"] == 1) & x["actual_fight_method"].eq("KO")).astype(int)
        x["actual_sub"] = ((x["actual_win"] == 1) & x["actual_fight_method"].eq("SUB")).astype(int)
        x["actual_dec"] = ((x["actual_win"] == 1) & x["actual_fight_method"].eq("DEC")).astype(int)

        x["market_ml_probability"] = pd.to_numeric(x[f"market_ml_probability_{side}"], errors="coerce")
        x["market_exact_ko_probability"] = pd.to_numeric(x[f"market_{side}_ko"], errors="coerce")
        x["market_exact_sub_probability"] = pd.to_numeric(x[f"market_{side}_sub"], errors="coerce")
        x["market_exact_dec_probability"] = pd.to_numeric(x[f"market_{side}_dec"], errors="coerce")
        x["market_method_side_win_probability"] = x[
            ["market_exact_ko_probability", "market_exact_sub_probability", "market_exact_dec_probability"]
        ].sum(axis=1)
        denom = x["market_method_side_win_probability"].clip(lower=EPS)
        x["market_conditional_ko_given_win"] = x["market_exact_ko_probability"] / denom
        x["market_conditional_sub_given_win"] = x["market_exact_sub_probability"] / denom
        x["market_conditional_dec_given_win"] = x["market_exact_dec_probability"] / denom

        cond_sum = x[
            ["market_conditional_ko_given_win", "market_conditional_sub_given_win", "market_conditional_dec_given_win"]
        ].sum(axis=1)
        if float((cond_sum - 1.0).abs().max()) > 1e-8:
            raise RuntimeError("fighter-side conditional method probabilities do not sum to one")

        for f in features:
            x[f] = sign * pd.to_numeric(x[f], errors="coerce")

        for concept, (r_col, b_col) in RAW_FEATURE_SPECS.items():
            if r_col not in raw_columns or b_col not in raw_columns:
                continue
            fighter_col = r_col if side == "red" else b_col
            opponent_col = b_col if side == "red" else r_col
            x[f"fighter_{concept}"] = pd.to_numeric(x[fighter_col], errors="coerce")
            x[f"opponent_{concept}"] = pd.to_numeric(x[opponent_col], errors="coerce")

        parts.append(x)

    rows = pd.concat(parts, ignore_index=True)
    keep_drop = [c for c in raw_columns if c in rows.columns]
    keep_drop += [
        "market_red_ko",
        "market_red_sub",
        "market_red_dec",
        "market_blue_ko",
        "market_blue_sub",
        "market_blue_dec",
        "market_ml_probability_red",
        "market_ml_probability_blue",
    ]
    rows = rows.drop(columns=keep_drop, errors="ignore")

    if {"fighter_ewm_splm", "fighter_ewm_str_acc"}.issubset(rows.columns):
        acc = _accuracy_decimal(rows["fighter_ewm_str_acc"])
        rows["fighter_est_sig_att_per_min"] = rows["fighter_ewm_splm"] / acc
    if {"opponent_ewm_splm", "opponent_ewm_str_acc"}.issubset(rows.columns):
        acc = _accuracy_decimal(rows["opponent_ewm_str_acc"])
        rows["opponent_est_sig_att_per_min"] = rows["opponent_ewm_splm"] / acc
    if {"fighter_est_sig_att_per_min", "opponent_est_sig_att_per_min"}.issubset(rows.columns):
        rows["est_sig_att_per_min_diff"] = rows["fighter_est_sig_att_per_min"] - rows["opponent_est_sig_att_per_min"]

    r1 = _build_prefight_r1(CUTOFF)
    rows = rows.merge(r1, on=["fight_id", "side"], how="left", validate="one_to_one")
    rows["year"] = rows["date"].dt.year.astype(int)
    for method_name, actual_col in [("ko", "actual_ko"), ("sub", "actual_sub"), ("dec", "actual_dec")]:
        market_col = f"market_conditional_{method_name}_given_win"
        rows[f"winner_conditional_{method_name}_residual"] = np.where(
            rows["actual_win"].eq(1), rows[actual_col].astype(float) - rows[market_col].astype(float), np.nan
        )

    rows = rows.sort_values(["date", "fight_id", "side"]).reset_index(drop=True)
    return rows, features, missing_raw


def _route_stats(winners: pd.DataFrame) -> dict:
    if winners.empty:
        return {
            "n": 0,
            "actual_ko_share": None,
            "market_ko_share": None,
            "ko_residual": None,
            "actual_sub_share": None,
            "market_sub_share": None,
            "sub_residual": None,
            "actual_dec_share": None,
            "market_dec_share": None,
            "dec_residual": None,
            "dominant_ko_donor": None,
            "residual_sum": None,
        }
    a_ko = float(winners["actual_ko"].mean())
    a_sub = float(winners["actual_sub"].mean())
    a_dec = float(winners["actual_dec"].mean())
    m_ko = float(winners["market_conditional_ko_given_win"].mean())
    m_sub = float(winners["market_conditional_sub_given_win"].mean())
    m_dec = float(winners["market_conditional_dec_given_win"].mean())
    r_ko, r_sub, r_dec = a_ko - m_ko, a_sub - m_sub, a_dec - m_dec
    donor = "DEC" if r_dec < r_sub else "SUB"
    if min(r_sub, r_dec) >= 0:
        donor = "NONE"
    return {
        "n": int(len(winners)),
        "actual_ko_share": a_ko,
        "market_ko_share": m_ko,
        "ko_residual": r_ko,
        "actual_sub_share": a_sub,
        "market_sub_share": m_sub,
        "sub_residual": r_sub,
        "actual_dec_share": a_dec,
        "market_dec_share": m_dec,
        "dec_residual": r_dec,
        "dominant_ko_donor": donor,
        "residual_sum": r_ko + r_sub + r_dec,
    }


def _build_thresholds(rows: pd.DataFrame) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    dev = rows[rows["date"].between(DEV_START, DEV_END) & rows["betting_eligible"].astype(bool)].copy()
    thresholds = {}
    records = []
    concepts = [c for c in BIN_FEATURES if c in dev.columns]
    for extra in ["opponent_ewm_str_def", "opponent_ewm_sub_avg", "fighter_ewm_kd_avg", "opponent_ewm_kd_absorbed_avg"]:
        if extra in dev.columns and extra not in concepts:
            concepts.append(extra)

    for c in concepts:
        s = pd.to_numeric(dev[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 100 or s.nunique() < 4:
            continue
        vals = {
            "q20": float(s.quantile(0.20)),
            "q40": float(s.quantile(LOW_Q)),
            "q60": float(s.quantile(HIGH_Q)),
            "q80": float(s.quantile(0.80)),
        }
        thresholds[c] = vals
        records.append(
            {
                "feature": c,
                "source_period": "2021-2024 all betting-eligible side rows",
                "n_non_null": int(len(s)),
                **vals,
            }
        )
    return thresholds, pd.DataFrame(records)


def _feature_bins(rows: pd.DataFrame, thresholds: dict[str, dict[str, float]]) -> pd.DataFrame:
    winners = rows[rows["actual_win"].eq(1) & rows["betting_eligible"].astype(bool)].copy()
    out = []
    for feature in [c for c in BIN_FEATURES if c in thresholds]:
        q = thresholds[feature]
        cuts = [-np.inf, q["q20"], q["q40"], q["q60"], q["q80"], np.inf]
        if len(set(cuts[1:-1])) < 4:
            continue
        for period, mask in [
            ("dev_2021_2024", winners["date"].between(DEV_START, DEV_END)),
            ("validation_2025", winners["date"].between(VAL_START, VAL_END)),
        ]:
            z = winners.loc[
                mask,
                [
                    feature,
                    "actual_ko",
                    "actual_sub",
                    "actual_dec",
                    "market_conditional_ko_given_win",
                    "market_conditional_sub_given_win",
                    "market_conditional_dec_given_win",
                ],
            ].copy()
            z["bucket"] = pd.cut(z[feature], cuts, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], include_lowest=True)
            for bucket, g in z.groupby("bucket", observed=True):
                st = _route_stats(g)
                out.append(
                    {
                        "period": period,
                        "feature": feature,
                        "bucket": str(bucket),
                        "feature_mean": float(pd.to_numeric(g[feature], errors="coerce").mean()),
                        **st,
                    }
                )
    return pd.DataFrame(out)


def _build_group_masks(rows: pd.DataFrame, t: dict[str, dict[str, float]]) -> tuple[dict[str, pd.Series], dict[str, str]]:
    idx = rows.index
    masks = {}
    defs = {}

    def available(feature: str) -> bool:
        return feature in rows.columns and feature in t

    def hi(feature: str) -> pd.Series:
        return pd.to_numeric(rows[feature], errors="coerce").ge(t[feature]["q60"])

    def lo(feature: str) -> pd.Series:
        return pd.to_numeric(rows[feature], errors="coerce").le(t[feature]["q40"])

    def add(name: str, mask: pd.Series, definition: str) -> None:
        masks[name] = mask.fillna(False).reindex(idx, fill_value=False)
        defs[name] = definition

    singles = [
        ("high_striking_accuracy", "ewm_str_acc_diff", "high"),
        ("high_striking_land_rate", "ewm_splm_diff", "high"),
        ("high_estimated_strike_attempt_rate", "fighter_est_sig_att_per_min", "high"),
        ("low_takedown_activity", "fighter_ewm_td_avg", "low"),
        ("high_takedown_activity", "fighter_ewm_td_avg", "high"),
        ("low_control_rate", "fighter_ewm_ctrl_per_min", "low"),
        ("high_control_rate", "fighter_ewm_ctrl_per_min", "high"),
        ("low_submission_activity", "fighter_ewm_sub_avg", "low"),
        ("high_submission_activity", "fighter_ewm_sub_avg", "high"),
        ("low_decision_dependency", "fighter_pre_decision_dependency", "low"),
        ("high_decision_dependency", "fighter_pre_decision_dependency", "high"),
        ("low_historical_ko_rate", "fighter_pre_ko_rate", "low"),
        ("high_historical_ko_rate", "fighter_pre_ko_rate", "high"),
        ("opponent_low_striking_defense", "opponent_ewm_str_def", "low"),
        ("opponent_high_striking_defense", "opponent_ewm_str_def", "high"),
        ("opponent_high_submission_activity", "opponent_ewm_sub_avg", "high"),
        ("height_advantage", "height_diff", "high"),
        ("reach_advantage", "reach_diff", "high"),
    ]
    for name, feature, direction in singles:
        if available(feature):
            cut = t[feature]["q60" if direction == "high" else "q40"]
            op = ">=" if direction == "high" else "<="
            qlab = "q60" if direction == "high" else "q40"
            add(name, hi(feature) if direction == "high" else lo(feature), f"{feature} {op} dev_{qlab} ({cut})")

    if "prior_r1_ko_wins" in rows.columns:
        add("no_prior_r1_ko", rows["prior_r1_ko_wins"].eq(0), "prior_r1_ko_wins == 0")
        add("any_prior_r1_ko", rows["prior_r1_ko_wins"].ge(1), "prior_r1_ko_wins >= 1")
    if "recent5_r1_ko_wins" in rows.columns:
        add("no_recent5_r1_ko", rows["recent5_r1_ko_wins"].eq(0), "recent5_r1_ko_wins == 0")
        add("any_recent5_r1_ko", rows["recent5_r1_ko_wins"].ge(1), "recent5_r1_ko_wins >= 1")
    if "win_streak_diff" in rows.columns:
        add("current_win_streak_advantage", pd.to_numeric(rows["win_streak_diff"], errors="coerce").gt(0), "win_streak_diff > 0")
    if "recent_form_win_streak_diff" in rows.columns:
        add(
            "ewm_minus_current_win_streak_delta_positive",
            pd.to_numeric(rows["recent_form_win_streak_diff"], errors="coerce").gt(0),
            "recent_form_win_streak_diff > 0; provenance = side difference of [EWM(win_streak) - current win_streak]",
        )

    if available("ewm_str_acc_diff"):
        acc = hi("ewm_str_acc_diff")
        if available("ewm_splm_diff"):
            add("high_accuracy__high_land_rate", acc & hi("ewm_splm_diff"), "high striking accuracy AND high striking land rate")
        if available("fighter_est_sig_att_per_min"):
            add("high_accuracy__high_est_attempt_rate", acc & hi("fighter_est_sig_att_per_min"), "high striking accuracy AND high estimated sig-strike attempt rate")
        if available("fighter_ewm_td_avg"):
            add("high_accuracy__low_td", acc & lo("fighter_ewm_td_avg"), "high striking accuracy AND low fighter TD activity")
            add("high_accuracy__high_td", acc & hi("fighter_ewm_td_avg"), "high striking accuracy AND high fighter TD activity")
        if available("fighter_ewm_ctrl_per_min"):
            add("high_accuracy__low_control", acc & lo("fighter_ewm_ctrl_per_min"), "high striking accuracy AND low control rate")
            add("high_accuracy__high_control", acc & hi("fighter_ewm_ctrl_per_min"), "high striking accuracy AND high control rate")
        if available("fighter_ewm_sub_avg"):
            add("high_accuracy__low_sub_activity", acc & lo("fighter_ewm_sub_avg"), "high striking accuracy AND low submission activity")
            add("high_accuracy__high_sub_activity", acc & hi("fighter_ewm_sub_avg"), "high striking accuracy AND high submission activity")
        if available("fighter_pre_decision_dependency"):
            add("high_accuracy__low_decision_dependency", acc & lo("fighter_pre_decision_dependency"), "high striking accuracy AND low decision dependency")
            add("high_accuracy__high_decision_dependency", acc & hi("fighter_pre_decision_dependency"), "high striking accuracy AND high decision dependency")
        if available("opponent_ewm_str_def"):
            add("high_accuracy__opponent_low_str_def", acc & lo("opponent_ewm_str_def"), "high striking accuracy AND opponent low striking defense")
            add("high_accuracy__opponent_high_str_def", acc & hi("opponent_ewm_str_def"), "high striking accuracy AND opponent high striking defense")
        if available("opponent_ewm_sub_avg"):
            add("high_accuracy__opponent_high_sub_activity", acc & hi("opponent_ewm_sub_avg"), "high striking accuracy AND opponent high submission activity")
        if "recent5_r1_ko_wins" in rows.columns:
            add("high_accuracy__no_recent5_r1_ko", acc & rows["recent5_r1_ko_wins"].eq(0), "high striking accuracy AND no R1 KO win in previous 5 UFC fights")
            add("high_accuracy__any_recent5_r1_ko", acc & rows["recent5_r1_ko_wins"].ge(1), "high striking accuracy AND >=1 R1 KO win in previous 5 UFC fights")
        if available("fighter_pre_ko_rate"):
            add("high_accuracy__low_historical_ko_rate", acc & lo("fighter_pre_ko_rate"), "high striking accuracy AND low historical KO rate")
            add("high_accuracy__high_historical_ko_rate", acc & hi("fighter_pre_ko_rate"), "high striking accuracy AND high historical KO rate")

        for geo_name, geo_feature in [("height", "height_diff"), ("reach", "reach_diff")]:
            if available(geo_feature):
                geo = hi(geo_feature)
                add(f"{geo_name}_adv__high_accuracy", geo & acc, f"{geo_name} advantage AND high striking accuracy")
                if available("fighter_ewm_td_avg"):
                    add(f"{geo_name}_adv__high_accuracy__low_td", geo & acc & lo("fighter_ewm_td_avg"), f"{geo_name} advantage AND high accuracy AND low TD activity")
                    add(f"{geo_name}_adv__high_accuracy__high_td", geo & acc & hi("fighter_ewm_td_avg"), f"{geo_name} advantage AND high accuracy AND high TD activity")
                if "recent5_r1_ko_wins" in rows.columns:
                    add(f"{geo_name}_adv__high_accuracy__no_recent5_r1_ko", geo & acc & rows["recent5_r1_ko_wins"].eq(0), f"{geo_name} advantage AND high accuracy AND no recent-5 R1 KO")
                    add(f"{geo_name}_adv__high_accuracy__any_recent5_r1_ko", geo & acc & rows["recent5_r1_ko_wins"].ge(1), f"{geo_name} advantage AND high accuracy AND recent-5 R1 KO")

        if available("fighter_ewm_kd_avg") and available("opponent_ewm_kd_absorbed_avg"):
            add(
                "high_accuracy__high_kd_output__opponent_high_kd_absorption",
                acc & hi("fighter_ewm_kd_avg") & hi("opponent_ewm_kd_absorbed_avg"),
                "high accuracy AND high fighter KD output AND opponent high KD absorption",
            )

    return masks, defs


def _group_periods(rows: pd.DataFrame, masks: dict[str, pd.Series], defs: dict[str, str]) -> pd.DataFrame:
    eligible_winners = rows["betting_eligible"].astype(bool) & rows["actual_win"].eq(1)
    periods = [
        ("dev_2021_2024", rows["date"].between(DEV_START, DEV_END)),
        ("2021", rows["year"].eq(2021)),
        ("2022", rows["year"].eq(2022)),
        ("2023", rows["year"].eq(2023)),
        ("2024", rows["year"].eq(2024)),
        ("validation_2025", rows["date"].between(VAL_START, VAL_END)),
    ]
    out = []
    for name, group_mask in masks.items():
        for period, period_mask in periods:
            g = rows[eligible_winners & group_mask & period_mask].copy()
            out.append({"group": name, "definition": defs[name], "period": period, **_route_stats(g)})
    return pd.DataFrame(out)


def _group_summary(periods: pd.DataFrame) -> pd.DataFrame:
    out = []
    for group, g in periods.groupby("group", sort=False):
        by_period = {r.period: r for r in g.itertuples(index=False)}
        dev = by_period.get("dev_2021_2024")
        val = by_period.get("validation_2025")
        if dev is None:
            continue
        year_resids = []
        positive_years = 0
        same_sign_years = 0
        for year in YEARS:
            r = by_period.get(str(year))
            v = None if r is None else r.ko_residual
            year_resids.append(v)
            if v is not None and not pd.isna(v):
                if v > 0:
                    positive_years += 1
                if dev.ko_residual is not None and not pd.isna(dev.ko_residual) and np.sign(v) == np.sign(dev.ko_residual) and v != 0:
                    same_sign_years += 1
        out.append(
            {
                "group": group,
                "definition": dev.definition,
                "dev_n": dev.n,
                "dev_actual_ko_share": dev.actual_ko_share,
                "dev_market_ko_share": dev.market_ko_share,
                "dev_ko_residual": dev.ko_residual,
                "dev_sub_residual": dev.sub_residual,
                "dev_dec_residual": dev.dec_residual,
                "dev_dominant_ko_donor": dev.dominant_ko_donor,
                "positive_ko_residual_dev_years": positive_years,
                "same_direction_dev_years": same_sign_years,
                "year_2021_ko_residual": year_resids[0],
                "year_2022_ko_residual": year_resids[1],
                "year_2023_ko_residual": year_resids[2],
                "year_2024_ko_residual": year_resids[3],
                "validation_n": None if val is None else val.n,
                "validation_actual_ko_share": None if val is None else val.actual_ko_share,
                "validation_market_ko_share": None if val is None else val.market_ko_share,
                "validation_ko_residual": None if val is None else val.ko_residual,
                "validation_sub_residual": None if val is None else val.sub_residual,
                "validation_dec_residual": None if val is None else val.dec_residual,
                "validation_dominant_ko_donor": None if val is None else val.dominant_ko_donor,
                "validation_same_direction": None
                if val is None or val.ko_residual is None or pd.isna(val.ko_residual) or dev.ko_residual is None or pd.isna(dev.ko_residual)
                else bool(np.sign(val.ko_residual) == np.sign(dev.ko_residual) and val.ko_residual != 0),
            }
        )
    d = pd.DataFrame(out)
    if not d.empty:
        d = d.sort_values(
            ["same_direction_dev_years", "dev_ko_residual", "dev_n"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return d


def _pair_summary(periods: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("accuracy_route_td", "high_accuracy__low_td", "high_accuracy__high_td"),
        ("accuracy_route_control", "high_accuracy__low_control", "high_accuracy__high_control"),
        ("accuracy_route_submission", "high_accuracy__low_sub_activity", "high_accuracy__high_sub_activity"),
        ("height_accuracy_route_td", "height_adv__high_accuracy__low_td", "height_adv__high_accuracy__high_td"),
        ("reach_accuracy_route_td", "reach_adv__high_accuracy__low_td", "reach_adv__high_accuracy__high_td"),
        ("accuracy_decision_dependency", "high_accuracy__low_decision_dependency", "high_accuracy__high_decision_dependency"),
        ("accuracy_opponent_str_def", "high_accuracy__opponent_low_str_def", "high_accuracy__opponent_high_str_def"),
        ("accuracy_recent_r1_reputation", "high_accuracy__no_recent5_r1_ko", "high_accuracy__any_recent5_r1_ko"),
        ("height_accuracy_recent_r1_reputation", "height_adv__high_accuracy__no_recent5_r1_ko", "height_adv__high_accuracy__any_recent5_r1_ko"),
        ("accuracy_historical_ko_reputation", "high_accuracy__low_historical_ko_rate", "high_accuracy__high_historical_ko_rate"),
    ]
    groups = set(periods["group"])
    out = []
    for pair_name, a, b in pairs:
        if a not in groups or b not in groups:
            continue
        for period in ["dev_2021_2024", "validation_2025"]:
            ga = periods[(periods["group"] == a) & (periods["period"] == period)]
            gb = periods[(periods["group"] == b) & (periods["period"] == period)]
            if ga.empty or gb.empty:
                continue
            ra, rb = ga.iloc[0], gb.iloc[0]
            out.append(
                {
                    "pair": pair_name,
                    "period": period,
                    "group_a": a,
                    "group_b": b,
                    "a_n": int(ra["n"]),
                    "b_n": int(rb["n"]),
                    "a_actual_ko_share": ra["actual_ko_share"],
                    "b_actual_ko_share": rb["actual_ko_share"],
                    "actual_ko_share_a_minus_b": ra["actual_ko_share"] - rb["actual_ko_share"] if pd.notna(ra["actual_ko_share"]) and pd.notna(rb["actual_ko_share"]) else None,
                    "a_market_ko_share": ra["market_ko_share"],
                    "b_market_ko_share": rb["market_ko_share"],
                    "market_ko_share_a_minus_b": ra["market_ko_share"] - rb["market_ko_share"] if pd.notna(ra["market_ko_share"]) and pd.notna(rb["market_ko_share"]) else None,
                    "a_ko_residual": ra["ko_residual"],
                    "b_ko_residual": rb["ko_residual"],
                    "ko_residual_a_minus_b": ra["ko_residual"] - rb["ko_residual"] if pd.notna(ra["ko_residual"]) and pd.notna(rb["ko_residual"]) else None,
                    "a_sub_residual": ra["sub_residual"],
                    "b_sub_residual": rb["sub_residual"],
                    "a_dec_residual": ra["dec_residual"],
                    "b_dec_residual": rb["dec_residual"],
                }
            )
    return pd.DataFrame(out)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, all_features, missing_raw = _build_side_rows()
    if (rows["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered KO method route reallocation audit")

    rows.to_csv(ROWS_OUT, index=False)
    winners = rows[rows["actual_win"].eq(1)].copy()
    winners.to_csv(WINNERS_OUT, index=False)

    thresholds, threshold_df = _build_thresholds(rows)
    threshold_df.to_csv(THRESHOLDS_OUT, index=False)
    bins = _feature_bins(rows, thresholds)
    bins.to_csv(FEATURE_BINS_OUT, index=False)

    masks, definitions = _build_group_masks(rows, thresholds)
    periods = _group_periods(rows, masks, definitions)
    periods.to_csv(GROUP_PERIODS_OUT, index=False)
    group_summary = _group_summary(periods)
    group_summary.to_csv(GROUP_SUMMARY_OUT, index=False)
    pairs = _pair_summary(periods)
    pairs.to_csv(PAIR_OUT, index=False)

    dev_winners = winners[winners["date"].between(DEV_START, DEV_END) & winners["betting_eligible"].astype(bool)]
    val_winners = winners[winners["date"].between(VAL_START, VAL_END) & winners["betting_eligible"].astype(bool)]
    baseline_dev = _route_stats(dev_winners)
    baseline_val = _route_stats(val_winners)

    stable = group_summary[
        (group_summary["dev_n"] >= 40)
        & (group_summary["same_direction_dev_years"] >= 3)
        & (group_summary["dev_ko_residual"] > 0)
        & (group_summary["validation_n"].fillna(0) >= 10)
        & group_summary["validation_same_direction"].eq(True)
    ].copy()
    stable = stable.sort_values(["same_direction_dev_years", "dev_ko_residual", "dev_n"], ascending=[False, False, False]).head(20)

    payload = {
        "experiment": "ko_method_route_reallocation_audit_v1",
        "purpose": "Winner-conditioned descriptive audit of how sportsbook side-win probability is allocated across KO/SUB/DEC. No predictive model is trained.",
        "development_window": "2021-2024",
        "validation_window": "2025 fixed after development definitions",
        "reads_2026_plus": False,
        "roi_used": False,
        "new_model_trained": False,
        "winner_knowledge_use": "diagnostic only; actual winner is never proposed as a deployable feature",
        "market_conditional_formula": "side exact method probability / (side KO + side SUB + side DEC)",
        "threshold_policy": "Q40/Q60 descriptive group cuts defined on 2021-2024 all betting-eligible side rows; individual bins use dev quintiles; 2025 never defines a threshold",
        "striking_attempt_rate_provenance": "No native sig-strike attempt/min feature exists in the current moneyline state family. fighter_est_sig_att_per_min is a transparent diagnostic approximation = EWM SPLM / EWM striking accuracy.",
        "recent_form_win_streak_provenance": "recent_form_win_streak_diff is the side difference of form_delta_win_streak, where form_delta_win_streak = EWM(win_streak) - current win_streak. It is not equivalent to current consecutive UFC win streak.",
        "baseline_dev_winners": baseline_dev,
        "baseline_validation_winners": baseline_val,
        "all_side_rows": int(len(rows)),
        "all_winner_rows": int(len(winners)),
        "dev_eligible_winner_rows": int(len(dev_winners)),
        "validation_eligible_winner_rows": int(len(val_winners)),
        "signed_diff_feature_count_available": int(len(all_features)),
        "raw_side_feature_concepts_missing": missing_raw,
        "descriptive_group_count": int(len(definitions)),
        "stable_positive_descriptive_groups": stable[
            [
                "group",
                "definition",
                "dev_n",
                "dev_actual_ko_share",
                "dev_market_ko_share",
                "dev_ko_residual",
                "dev_sub_residual",
                "dev_dec_residual",
                "same_direction_dev_years",
                "validation_n",
                "validation_actual_ko_share",
                "validation_market_ko_share",
                "validation_ko_residual",
                "validation_sub_residual",
                "validation_dec_residual",
            ]
        ].to_dict("records"),
        "artifacts": [
            str(ROWS_OUT),
            str(WINNERS_OUT),
            str(THRESHOLDS_OUT),
            str(FEATURE_BINS_OUT),
            str(GROUP_PERIODS_OUT),
            str(GROUP_SUMMARY_OUT),
            str(PAIR_OUT),
        ],
    }
    SUMMARY_OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
