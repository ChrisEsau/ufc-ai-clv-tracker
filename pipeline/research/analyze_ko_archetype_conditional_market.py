from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method

OUT = Path("data/research/prop_mispricing")
SUMMARY = OUT / "ko_archetype_conditional_market_summary.csv"
ROWS = OUT / "ko_archetype_conditional_market_rows.csv"
JSON_OUT = OUT / "ko_archetype_conditional_market_summary.json"

# Fixed archetype thresholds discovered in the prior 2021-2024 residual audit.
SNIPER_SPLM_MAX = -2.218309890006355
EWM_STR_ACC_MIN = 0.08503716030259571
DECISION_DEP_MIN = 0.405714
HEIGHT_MIN_CM = 5.0800000000000125
AGGRESSION_MIN = 2.563945


def orient(df: pd.DataFrame, side: str, features: list[str]) -> pd.DataFrame:
    sign = 1.0 if side == "red" else -1.0
    out = df[[
        "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
        "betting_eligible",
        "market_red_ko", "market_red_sub", "market_red_dec",
        "market_blue_ko", "market_blue_sub", "market_blue_dec",
    ]].copy()
    out["side"] = side
    out["fighter"] = np.where(side == "red", out["red_fighter"], out["blue_fighter"])
    out["actual_win"] = np.where(side == "red", out["target"].to_numpy(int) < 3, out["target"].to_numpy(int) >= 3).astype(int)
    out["actual_ko_win"] = np.where(side == "red", out["target"].to_numpy(int) == 0, out["target"].to_numpy(int) == 3).astype(int)
    if side == "red":
        ko = out["market_red_ko"].to_numpy(float)
        side_total = out[["market_red_ko", "market_red_sub", "market_red_dec"]].sum(axis=1).to_numpy(float)
    else:
        ko = out["market_blue_ko"].to_numpy(float)
        side_total = out[["market_blue_ko", "market_blue_sub", "market_blue_dec"]].sum(axis=1).to_numpy(float)
    out["market_method_side_win_p"] = side_total
    out["market_exact_ko_p"] = ko
    out["market_cond_ko_given_side_win"] = ko / np.clip(side_total, 1e-12, None)
    for f in features:
        out[f] = sign * pd.to_numeric(df[f], errors="coerce")
    return out


def summarize(g: pd.DataFrame, label: str, period: str) -> dict:
    winners = g[g["actual_win"].eq(1)].copy()
    if g.empty:
        return {"archetype": label, "period": period, "n": 0}
    return {
        "archetype": label,
        "period": period,
        "n": int(len(g)),
        "unique_fighters": int(g["fighter"].nunique()),
        "actual_win_rate": float(g["actual_win"].mean()),
        "market_method_side_win_p": float(g["market_method_side_win_p"].mean()),
        "winner_n": int(len(winners)),
        "ko_wins": int(g["actual_ko_win"].sum()),
        "ko_rate_all_rows": float(g["actual_ko_win"].mean()),
        "market_exact_ko_p": float(g["market_exact_ko_p"].mean()),
        "exact_ko_residual": float((g["actual_ko_win"] - g["market_exact_ko_p"]).mean()),
        "ko_share_among_actual_wins": float(winners["actual_ko_win"].mean()) if len(winners) else None,
        "market_cond_ko_given_side_win_all_rows": float(g["market_cond_ko_given_side_win"].mean()),
        "market_cond_ko_given_side_win_actual_winners": float(winners["market_cond_ko_given_side_win"].mean()) if len(winners) else None,
        "conditional_ko_residual_on_actual_winners": float(
            winners["actual_ko_win"].mean() - winners["market_cond_ko_given_side_win"].mean()
        ) if len(winners) else None,
    }


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df, features, _ = method._build_rows(False, True)
    df["date"] = pd.to_datetime(df["date"])
    # Explicit hard stop: this diagnostic may use 2025 validation, never 2026.
    df = df[(df["date"] >= "2021-01-01") & (df["date"] <= "2025-12-31")].copy()
    if (df["date"] >= "2026-01-01").any():
        raise RuntimeError("2026 entered KO archetype diagnostic")

    needed = [
        "recent_splm_diff", "ewm_str_acc_diff", "decision_dependency_diff",
        "height_diff", "aggression_index_diff", "td_avg_diff", "ewm_td_avg_diff",
        "ewm_avg_opponent_elo_diff", "reach_diff", "ewm_finish_loss_rate_diff",
        "ewm_kd_avg_diff", "ewm_kd_absorbed_avg_diff", "str_def_diff",
        "recent_form_win_streak_diff", "style_ko_finisher_score_diff",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing archetype fields: {missing}")

    side_rows = pd.concat([orient(df, "red", needed), orient(df, "blue", needed)], ignore_index=True)
    side_rows = side_rows[side_rows["betting_eligible"].astype(bool)].copy()
    side_rows["year"] = side_rows["date"].dt.year.astype(int)
    side_rows["sniper_decision"] = (
        (side_rows["recent_splm_diff"] <= SNIPER_SPLM_MAX)
        & (side_rows["ewm_str_acc_diff"] >= EWM_STR_ACC_MIN)
        & (side_rows["decision_dependency_diff"] >= DECISION_DEP_MIN)
    )
    side_rows["height_accuracy_aggression"] = (
        (side_rows["height_diff"] >= HEIGHT_MIN_CM)
        & (side_rows["ewm_str_acc_diff"] >= EWM_STR_ACC_MIN)
        & (side_rows["aggression_index_diff"] >= AGGRESSION_MIN)
    )

    rows = []
    for arch in ["sniper_decision", "height_accuracy_aggression"]:
        for period, mask in [
            ("2021-2024", side_rows["year"].between(2021, 2024)),
            ("2025", side_rows["year"].eq(2025)),
        ]:
            g = side_rows[side_rows[arch] & mask].copy()
            rows.append(summarize(g, arch, period))
            if period == "2021-2024":
                first = g.sort_values("date").drop_duplicates("fighter", keep="first")
                last = g.sort_values("date").drop_duplicates("fighter", keep="last")
                rows.append(summarize(first, arch, "2021-2024_unique_fighter_first"))
                rows.append(summarize(last, arch, "2021-2024_unique_fighter_last"))
                for year in [2021, 2022, 2023, 2024]:
                    rows.append(summarize(g[g["year"].ne(year)], arch, f"2021-2024_leave_out_{year}"))

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)
    keep = side_rows[side_rows[["sniper_decision", "height_accuracy_aggression"]].any(axis=1)].copy()
    keep.to_csv(ROWS, index=False)

    payload = {
        "experiment": "KO market archetype conditional-method audit",
        "new_model_trained": False,
        "development_window": "2021-2024 archetype thresholds fixed from prior residual audit",
        "validation_window": "calendar 2025 only",
        "reads_2026": False,
        "purpose": "separate exact-KO market residual from winner-side method-market residual",
        "thresholds": {
            "sniper_recent_splm_max": SNIPER_SPLM_MAX,
            "ewm_str_acc_min": EWM_STR_ACC_MIN,
            "decision_dependency_min": DECISION_DEP_MIN,
            "height_min_cm": HEIGHT_MIN_CM,
            "aggression_min": AGGRESSION_MIN,
        },
        "artifacts": [str(SUMMARY), str(ROWS)],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()
