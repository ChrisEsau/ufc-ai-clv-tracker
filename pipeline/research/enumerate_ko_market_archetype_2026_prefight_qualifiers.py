from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURE_PATH = Path("data/features/moneyline_feature_view.parquet")
FREEZE_PATH = Path("data/research/prop_mispricing/ko_market_archetype_2026_freeze.json")
OUT = Path("data/research/prop_mispricing/ko_market_archetype_2026_prefight_qualifiers.csv")
SUMMARY = Path("data/research/prop_mispricing/ko_market_archetype_2026_prefight_qualifiers_summary.json")

START = pd.Timestamp("2026-01-01")
END = pd.Timestamp("2026-12-31")
FEATURES = ["height_diff", "ewm_str_acc_diff", "aggression_index_diff", "recent_form_win_streak_diff"]


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    if not freeze.get("frozen_before_2026_scoring"):
        raise RuntimeError("freeze contract missing")
    rule = freeze["rule"]

    df = pd.read_parquet(FEATURE_PATH, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    required = ["fight_id", "date", "r_pre_fights", "b_pre_fights"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature-view columns: {missing}")
    df = df[required].drop_duplicates("fight_id").sort_values(["date", "fight_id"]).reset_index(drop=True)

    parts = []
    for side, sign in [("red", 1.0), ("blue", -1.0)]:
        x = df.copy()
        x["side"] = side
        for c in FEATURES:
            x[c] = pd.to_numeric(x[c], errors="coerce") * sign
        x["min_prior_ufc_fights"] = pd.concat([
            pd.to_numeric(x["r_pre_fights"], errors="coerce"),
            pd.to_numeric(x["b_pre_fights"], errors="coerce"),
        ], axis=1).min(axis=1)
        x["betting_eligible"] = x["min_prior_ufc_fights"].ge(2)
        parts.append(x)

    rows = pd.concat(parts, ignore_index=True)
    mask = (
        rows["betting_eligible"]
        & rows["height_diff"].ge(float(rule["height_diff_min_cm"]))
        & rows["ewm_str_acc_diff"].ge(float(rule["ewm_str_acc_diff_min"]))
        & rows["aggression_index_diff"].ge(float(rule["aggression_index_diff_min"]))
        & rows["recent_form_win_streak_diff"].gt(float(rule["recent_form_win_streak_diff_min_exclusive"]))
    )
    q = rows.loc[mask].copy().sort_values(["date", "fight_id", "side"]).reset_index(drop=True)

    cols = ["date", "fight_id", "side", "min_prior_ufc_fights"] + FEATURES
    q[cols].to_csv(OUT, index=False)
    SUMMARY.write_text(json.dumps({
        "selection_used_outcomes": False,
        "selection_used_market_prices": False,
        "feature_view_first_2026_date": df["date"].min().date().isoformat() if len(df) else None,
        "feature_view_last_2026_date": df["date"].max().date().isoformat() if len(df) else None,
        "feature_view_2026_fights": int(len(df)),
        "frozen_rule_qualifiers": int(len(q)),
        "qualifiers": q[cols].assign(date=lambda z: z["date"].astype(str)).to_dict(orient="records"),
    }, indent=2, default=str))
    print(SUMMARY.read_text())


if __name__ == "__main__":
    main()
