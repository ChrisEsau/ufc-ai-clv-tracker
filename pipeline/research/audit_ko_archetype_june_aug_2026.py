from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_PATH = Path("data/features/moneyline_feature_view.parquet")
MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")
FREEZE_PATH = Path("data/research/prop_mispricing/ko_market_archetype_2026_freeze.json")
OUT_ALL = Path("data/research/prop_mispricing/ko_archetype_june_aug_2026_all_sides.csv")
OUT_NEAR = Path("data/research/prop_mispricing/ko_archetype_june_aug_2026_exact_and_near_misses.csv")
OUT_SUMMARY = Path("data/research/prop_mispricing/ko_archetype_june_aug_2026_summary.json")

START = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-08-22")
FEATURES = ["height_diff", "ewm_str_acc_diff", "aggression_index_diff", "recent_form_win_streak_diff"]
METHOD_MAP = {
    "win_by_ko_tko_dq": "KO",
    "win_by_submission": "SUB",
    "win_by_decision": "DEC",
}


def _market_names_and_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_parquet(MARKET_PATH, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["fight_id"] = m["fight_id"].astype(str)

    # Names are metadata only; do not use any price field for selection/ranking.
    names = (
        m[m["outcome_side"].astype(str).isin(["red", "blue"])]
        .sort_values(["date", "fight_id"])
        .groupby(["fight_id", "outcome_side"], as_index=False)
        .agg(fighter=("outcome_label", "first"), event_name=("event_name", "first"))
    )
    red = names[names["outcome_side"].eq("red")][["fight_id", "fighter", "event_name"]].rename(columns={"fighter": "red_fighter"})
    blue = names[names["outcome_side"].eq("blue")][["fight_id", "fighter"]].rename(columns={"fighter": "blue_fighter"})
    fight_names = red.merge(blue, on="fight_id", how="outer")

    # Outcome is attached only after feature-based selection is defined.
    graded = m[(m["result_status"] == "graded") & m["won"].notna()].copy()
    graded["won_bool"] = graded["won"].astype(bool)
    won = graded[graded["won_bool"] & graded["market_key"].isin(METHOD_MAP)].copy()
    won["win_method"] = won["market_key"].map(METHOD_MAP)
    results = (
        won.sort_values(["date", "fight_id"])
        .groupby("fight_id", as_index=False)
        .agg(winner_side=("outcome_side", "first"), win_method=("win_method", "first"))
    )
    return fight_names, results


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    rule = freeze["rule"]

    df = pd.read_parquet(FEATURE_PATH, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fight_id"] = df["fight_id"].astype(str)
    required = ["fight_id", "date", "r_pre_fights", "b_pre_fights"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature-view columns: {missing}")
    df = df[required].drop_duplicates("fight_id").sort_values(["date", "fight_id"]).reset_index(drop=True)

    names, results = _market_names_and_results()
    df = df.merge(names, on="fight_id", how="left")

    parts = []
    for side, sign in [("red", 1.0), ("blue", -1.0)]:
        x = df.copy()
        x["side"] = side
        x["fighter"] = np.where(side == "red", x.get("red_fighter"), x.get("blue_fighter"))
        x["opponent"] = np.where(side == "red", x.get("blue_fighter"), x.get("red_fighter"))
        for c in FEATURES:
            x[c] = pd.to_numeric(x[c], errors="coerce") * sign
        x["min_prior_ufc_fights"] = pd.concat([
            pd.to_numeric(x["r_pre_fights"], errors="coerce"),
            pd.to_numeric(x["b_pre_fights"], errors="coerce"),
        ], axis=1).min(axis=1)
        x["betting_eligible"] = x["min_prior_ufc_fights"].ge(2)
        parts.append(x)

    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows["betting_eligible"]].copy()

    thresholds = {
        "height": ("height_diff", float(rule["height_diff_min_cm"]), "ge"),
        "accuracy": ("ewm_str_acc_diff", float(rule["ewm_str_acc_diff_min"]), "ge"),
        "aggression": ("aggression_index_diff", float(rule["aggression_index_diff_min"]), "ge"),
        "streak": ("recent_form_win_streak_diff", float(rule["recent_form_win_streak_diff_min_exclusive"]), "gt"),
    }
    for name, (col, threshold, op) in thresholds.items():
        rows[f"pass_{name}"] = rows[col].ge(threshold) if op == "ge" else rows[col].gt(threshold)
    pass_cols = [f"pass_{x}" for x in thresholds]
    rows["conditions_passed"] = rows[pass_cols].sum(axis=1).astype(int)
    rows["exact_qualifier"] = rows["conditions_passed"].eq(4)
    rows["failed_conditions"] = rows.apply(
        lambda r: ",".join(name for name in thresholds if not bool(r[f"pass_{name}"])), axis=1
    )

    rows["height_margin"] = rows["height_diff"] - thresholds["height"][1]
    rows["accuracy_margin"] = rows["ewm_str_acc_diff"] - thresholds["accuracy"][1]
    rows["aggression_margin"] = rows["aggression_index_diff"] - thresholds["aggression"][1]
    rows["streak_margin"] = rows["recent_form_win_streak_diff"] - thresholds["streak"][1]

    # Attach outcomes only after all feature-based qualification columns are complete.
    rows = rows.merge(results, on="fight_id", how="left")
    rows["actual_result"] = np.where(
        rows["winner_side"].isna(),
        np.nan,
        np.where(rows["side"].eq(rows["winner_side"]), "WIN_" + rows["win_method"].astype(str), "LOSS"),
    )

    cols = [
        "date", "event_name", "fight_id", "fighter", "opponent", "side",
        "min_prior_ufc_fights", "conditions_passed", "exact_qualifier", "failed_conditions",
        "actual_result",
        "height_diff", "height_margin", "pass_height",
        "ewm_str_acc_diff", "accuracy_margin", "pass_accuracy",
        "aggression_index_diff", "aggression_margin", "pass_aggression",
        "recent_form_win_streak_diff", "streak_margin", "pass_streak",
    ]
    all_rows = rows[cols].sort_values(["date", "fight_id", "side"]).reset_index(drop=True)
    near = all_rows[all_rows["conditions_passed"].ge(3)].copy()
    near = near.sort_values(["exact_qualifier", "conditions_passed", "date"], ascending=[False, False, True]).reset_index(drop=True)

    all_rows.to_csv(OUT_ALL, index=False)
    near.to_csv(OUT_NEAR, index=False)

    summary = {
        "window_start": str(START.date()),
        "window_end": str(END.date()),
        "feature_fights": int(df["fight_id"].nunique()),
        "eligible_fighter_sides": int(len(all_rows)),
        "exact_qualifiers": int(all_rows["exact_qualifier"].sum()),
        "three_of_four_near_misses": int((all_rows["conditions_passed"] == 3).sum()),
        "near_miss_failed_condition_counts": (
            all_rows.loc[all_rows["conditions_passed"].eq(3), "failed_conditions"].value_counts().to_dict()
        ),
        "exact_rows": all_rows[all_rows["exact_qualifier"]].assign(date=lambda z: z["date"].astype(str)).to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(OUT_SUMMARY.read_text())
    print("\nExact + 3/4 rows:")
    print(near.to_string(index=False))


if __name__ == "__main__":
    main()
