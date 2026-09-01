from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method

OUT = Path("data/research/prop_mispricing")
FREEZE_PATH = OUT / "ko_market_archetype_2026_freeze.json"
QUALIFIERS_OUT = OUT / "ko_market_archetype_2026_qualifiers.csv"
SUMMARY_OUT = OUT / "ko_market_archetype_2026_confirmation_summary.json"
METRICS_OUT = OUT / "ko_market_archetype_2026_confirmation_metrics.csv"

CUTOFF = pd.Timestamp("2026-12-31")
START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-12-31")
EPS = 1e-12
REQUIRED_FEATURES = [
    "height_diff",
    "ewm_str_acc_diff",
    "aggression_index_diff",
    "recent_form_win_streak_diff",
]


def _load_freeze() -> dict:
    freeze = json.loads(FREEZE_PATH.read_text())
    if not freeze.get("frozen_before_2026_scoring"):
        raise RuntimeError("freeze does not assert pre-2026 status")
    if freeze.get("untouched_confirmation_period") != "calendar 2026":
        raise RuntimeError("unexpected confirmation period")
    rule = freeze["rule"]
    expected = {
        "height_diff_min_cm": 5.0800000000000125,
        "ewm_str_acc_diff_min": 0.08503716030259571,
        "aggression_index_diff_min": 2.563945,
        "recent_form_win_streak_diff_min_exclusive": 0.0,
    }
    for k, v in expected.items():
        if not np.isclose(float(rule[k]), v, rtol=0.0, atol=1e-15):
            raise RuntimeError(f"frozen threshold changed: {k}={rule[k]} expected={v}")
    return freeze


def _raw_ko_prices() -> pd.DataFrame:
    m = pd.read_parquet(method.MARKET_PATH, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[
        (m["bookmaker"] == "legacy_consensus")
        & (m["market_key"] == "win_by_ko_tko_dq")
        & m["outcome_side"].astype(str).isin(["red", "blue"])
    ].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["profit_per_100"] = pd.to_numeric(m["profit_per_100"], errors="coerce")
    counts = m.groupby(["fight_id", "outcome_side"]).size()
    good = set((str(fid), str(side)) for fid, side in counts[counts.eq(1)].index)
    m = m[m.apply(lambda r: (str(r["fight_id"]), str(r["outcome_side"])) in good, axis=1)].copy()
    return m[["fight_id", "outcome_side", "implied_probability", "profit_per_100"]].rename(
        columns={"implied_probability": "raw_ko_implied_probability"}
    )


def _build_2026_side_rows() -> pd.DataFrame:
    original_cutoff = method.DEV_CUTOFF
    method.DEV_CUTOFF = CUTOFF
    try:
        fights, features, _ = method._build_rows(
            development_only=True,
            include_targets=True,
            forced_features=REQUIRED_FEATURES,
        )
    finally:
        method.DEV_CUTOFF = original_cutoff

    if features != REQUIRED_FEATURES:
        raise RuntimeError(f"unexpected feature order: {features}")

    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["date"] = pd.to_datetime(fights["date"], errors="coerce")
    fights = fights[(fights["date"] >= START) & (fights["date"] <= END)].copy()
    if fights.empty:
        raise RuntimeError("no resolved complete six-way 2026 method-market fights found")

    raw_ko = _raw_ko_prices()
    parts = []
    for side, sign, ko_target in [("red", 1.0, 0), ("blue", -1.0, 3)]:
        x = fights[[
            "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
            "betting_eligible", "cold_start", "min_prior_ufc_fights", "market_overround",
        ] + REQUIRED_FEATURES].copy()
        x["side"] = side
        x["fighter"] = np.where(side == "red", x["red_fighter"], x["blue_fighter"])
        x["opponent"] = np.where(side == "red", x["blue_fighter"], x["red_fighter"])
        x["actual_win"] = np.where(side == "red", x["target"].astype(int) < 3, x["target"].astype(int) >= 3).astype(int)
        x["actual_ko_win"] = (x["target"].astype(int) == ko_target).astype(int)
        x["market_exact_ko_p"] = fights[f"market_{side}_ko"].to_numpy(float)
        side_sum = (
            fights[f"market_{side}_ko"].to_numpy(float)
            + fights[f"market_{side}_sub"].to_numpy(float)
            + fights[f"market_{side}_dec"].to_numpy(float)
        )
        x["market_side_win_p_from_method"] = side_sum
        x["market_conditional_ko_given_win"] = x["market_exact_ko_p"].to_numpy(float) / np.clip(side_sum, EPS, None)
        for feature in REQUIRED_FEATURES:
            x[feature] = pd.to_numeric(x[feature], errors="coerce") * sign
        parts.append(x)

    rows = pd.concat(parts, ignore_index=True)
    rows = rows.merge(
        raw_ko,
        left_on=["fight_id", "side"],
        right_on=["fight_id", "outcome_side"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["outcome_side"])
    rows["decimal_ko_odds"] = 1.0 + rows["profit_per_100"] / 100.0
    rows["flat_ko_profit_units"] = np.where(
        rows["actual_ko_win"].eq(1), rows["decimal_ko_odds"] - 1.0, -1.0
    )
    rows["year"] = rows["date"].dt.year.astype(int)
    if not rows["year"].eq(2026).all():
        raise RuntimeError("non-2026 row entered confirmation frame")
    return rows.sort_values(["date", "fight_id", "side"]).reset_index(drop=True)


def _apply_frozen_rule(rows: pd.DataFrame, freeze: dict) -> pd.DataFrame:
    r = freeze["rule"]
    mask = (
        rows["betting_eligible"].astype(bool)
        & rows["height_diff"].ge(float(r["height_diff_min_cm"]))
        & rows["ewm_str_acc_diff"].ge(float(r["ewm_str_acc_diff_min"]))
        & rows["aggression_index_diff"].ge(float(r["aggression_index_diff_min"]))
        & rows["recent_form_win_streak_diff"].gt(float(r["recent_form_win_streak_diff_min_exclusive"]))
    )
    return rows.loc[mask].copy().sort_values(["date", "fight_id", "side"]).reset_index(drop=True)


def _metrics(qualifiers: pd.DataFrame, all_rows: pd.DataFrame) -> dict:
    winners = qualifiers[qualifiers["actual_win"].eq(1)].copy()
    actual_share = float(winners["actual_ko_win"].mean()) if len(winners) else None
    market_share = float(winners["market_conditional_ko_given_win"].mean()) if len(winners) else None
    residual = actual_share - market_share if actual_share is not None and market_share is not None else None
    return {
        "confirmation_period": "2026 year-to-date available resolved data",
        "data_first_resolved_date": all_rows["date"].min().date().isoformat(),
        "data_last_resolved_date": all_rows["date"].max().date().isoformat(),
        "resolved_complete_six_way_fights_2026": int(all_rows["fight_id"].nunique()),
        "eligible_frozen_rule_qualifiers": int(len(qualifiers)),
        "qualifier_wins": int(qualifiers["actual_win"].sum()),
        "qualifier_ko_wins": int(qualifiers["actual_ko_win"].sum()),
        "qualifier_win_rate": float(qualifiers["actual_win"].mean()) if len(qualifiers) else None,
        "actual_ko_share_among_qualifier_wins": actual_share,
        "market_conditional_ko_share_among_qualifier_wins": market_share,
        "conditional_ko_residual": residual,
        "conditional_ko_residual_pp": 100.0 * residual if residual is not None else None,
        "mean_market_exact_ko_p_all_qualifiers": float(qualifiers["market_exact_ko_p"].mean()) if len(qualifiers) else None,
        "actual_exact_ko_rate_all_qualifiers": float(qualifiers["actual_ko_win"].mean()) if len(qualifiers) else None,
        "diagnostic_flat_ko_profit_units": float(qualifiers["flat_ko_profit_units"].sum()) if len(qualifiers) else 0.0,
        "diagnostic_flat_ko_roi": float(qualifiers["flat_ko_profit_units"].mean()) if len(qualifiers) else None,
        "primary_direction_confirmed": bool(residual is not None and residual > 0.0),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freeze = _load_freeze()
    all_rows = _build_2026_side_rows()
    qualifiers = _apply_frozen_rule(all_rows, freeze)
    metrics = _metrics(qualifiers, all_rows)

    qualifier_cols = [
        "date", "event_name", "fight_id", "fighter", "opponent", "side",
        "actual_win", "actual_ko_win", "market_exact_ko_p",
        "market_side_win_p_from_method", "market_conditional_ko_given_win",
        "raw_ko_implied_probability", "profit_per_100", "flat_ko_profit_units",
        "height_diff", "ewm_str_acc_diff", "aggression_index_diff",
        "recent_form_win_streak_diff", "min_prior_ufc_fights", "market_overround",
    ]
    qualifiers[qualifier_cols].to_csv(QUALIFIERS_OUT, index=False)
    pd.DataFrame([metrics]).to_csv(METRICS_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps({
        "freeze": freeze,
        "metrics": metrics,
        "qualifiers": qualifiers[qualifier_cols].assign(date=lambda x: x["date"].astype(str)).to_dict(orient="records"),
    }, indent=2, default=str))

    print(json.dumps(metrics, indent=2))
    print("\n2026 qualifiers:")
    if qualifiers.empty:
        print("NONE")
    else:
        print(qualifiers[qualifier_cols].to_string(index=False))


if __name__ == "__main__":
    main()
