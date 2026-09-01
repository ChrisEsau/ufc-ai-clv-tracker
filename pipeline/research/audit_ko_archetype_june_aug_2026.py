from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed-rule descriptive audit only: no threshold tuning in this script.
FEATURE_PATH = Path("data/features/moneyline_feature_view.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
FREEZE_PATH = Path("data/research/prop_mispricing/ko_market_archetype_2026_freeze.json")
OUT_ALL = Path("data/research/prop_mispricing/ko_archetype_june_aug_2026_all_sides.csv")
OUT_NEAR = Path("data/research/prop_mispricing/ko_archetype_june_aug_2026_exact_and_near_misses.csv")
OUT_SUMMARY = Path("data/research/prop_mispricing/ko_archetype_june_aug_2026_summary.json")

START = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-08-22")
FEATURES = ["height_diff", "ewm_str_acc_diff", "aggression_index_diff", "recent_form_win_streak_diff"]


def _method_bucket(value: object) -> str | None:
    text = str(value).upper()
    if "KO" in text or "TKO" in text:
        return "KO"
    if "SUB" in text:
        return "SUB"
    if "DEC" in text:
        return "DEC"
    return None


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    rule = freeze["rule"]

    df = pd.read_parquet(FEATURE_PATH, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["fight_id"] = df["fight_id"].astype(str)
    if "state_fight_id" not in df.columns:
        df["state_fight_id"] = df["fight_id"]
    df["state_fight_id"] = df["state_fight_id"].astype(str)
    required = ["fight_id", "state_fight_id", "date", "r_id", "b_id", "r_pre_fights", "b_pre_fights"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature-view columns: {missing}")
    df = df[required].drop_duplicates("fight_id").sort_values(["date", "fight_id"]).reset_index(drop=True)

    master = pd.read_parquet(MASTER_PATH, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    master["fight_id"] = master["fight_id"].astype(str)
    keep = [c for c in ["fight_id", "event_name", "r_id", "b_id", "r_name", "b_name", "winner_id", "method", "division"] if c in master.columns]
    master = master[keep].drop_duplicates("fight_id").rename(columns={"fight_id": "master_fight_id"})
    df = df.merge(master, left_on="state_fight_id", right_on="master_fight_id", how="left", suffixes=("", "_master"))

    parts = []
    for side, sign in [("red", 1.0), ("blue", -1.0)]:
        x = df.copy()
        x["side"] = side
        x["fighter"] = np.where(side == "red", x.get("r_name"), x.get("b_name"))
        x["opponent"] = np.where(side == "red", x.get("b_name"), x.get("r_name"))
        x["side_fighter_id"] = np.where(side == "red", x["r_id"].astype(str), x["b_id"].astype(str))
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

    # Outcome is descriptive and attached only after feature qualification is determined.
    rows["winner_id_str"] = rows.get("winner_id").astype(str) if "winner_id" in rows.columns else np.nan
    rows["win_method"] = rows.get("method").map(_method_bucket) if "method" in rows.columns else np.nan
    rows["actual_result"] = np.where(
        rows["winner_id_str"].isin(["nan", "None"]),
        np.nan,
        np.where(rows["side_fighter_id"].eq(rows["winner_id_str"]), "WIN_" + rows["win_method"].astype(str), "LOSS"),
    )

    cols = [
        "date", "event_name", "division", "fight_id", "state_fight_id", "fighter", "opponent", "side",
        "min_prior_ufc_fights", "conditions_passed", "exact_qualifier", "failed_conditions", "actual_result",
        "height_diff", "height_margin", "pass_height",
        "ewm_str_acc_diff", "accuracy_margin", "pass_accuracy",
        "aggression_index_diff", "aggression_margin", "pass_aggression",
        "recent_form_win_streak_diff", "streak_margin", "pass_streak",
    ]
    cols = [c for c in cols if c in rows.columns]
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
        "near_miss_failed_condition_counts": all_rows.loc[all_rows["conditions_passed"].eq(3), "failed_conditions"].value_counts().to_dict(),
        "exact_rows": all_rows[all_rows["exact_qualifier"]].assign(date=lambda z: z["date"].astype(str)).to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(OUT_SUMMARY.read_text())
    print("\nExact + 3/4 rows:")
    print(near.to_string(index=False))


if __name__ == "__main__":
    main()
