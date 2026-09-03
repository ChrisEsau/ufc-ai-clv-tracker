"""Compare ALL-BUT quadratic MC winner predictions with prefight market odds.

Cohort
------
- frozen 34-fight validation card
- immediately preceding 8-fight mature event replay (2026-06-06)

Model variant
-------------
ALL-BUT = all prefight FSR observations through the target-fight prefight row,
drop the first initialization point, degree-2 fit, extrapolate N+1.

Market source
-------------
data/market/historical_moneyline_odds.parquet, built from the legacy consensus
prefight moneyline source. Market probabilities are de-vigged within each fight.

Outputs
-------
data/experimental/all_but_vs_prefight_market_42/
  all_but_vs_prefight_market_42.csv
  all_but_vs_prefight_market_42_summary.csv
  all_but_vs_prefight_market_42.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_34 = Path(
    "data/experimental/validation_poly2_all_but_initial_fsr_mc/"
    "fsr_mc_card_validation_all_but_initial_poly2_v1.csv"
)
EVENT_8 = Path(
    "data/experimental/next_event_base_all_allbut_poly2/"
    "event_2026-06-06_comparison.csv"
)
MARKET = Path("data/market/historical_moneyline_odds.parquet")
OUT_DIR = Path("data/experimental/all_but_vs_prefight_market_42")
OUT_CSV = OUT_DIR / "all_but_vs_prefight_market_42.csv"
OUT_SUMMARY = OUT_DIR / "all_but_vs_prefight_market_42_summary.csv"
OUT_MD = OUT_DIR / "all_but_vs_prefight_market_42.md"


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _load_model_rows() -> pd.DataFrame:
    _require(FROZEN_34)
    _require(EVENT_8)

    a = pd.read_csv(FROZEN_34)
    b = pd.read_csv(EVENT_8)

    need_a = {"bout_id", "red", "blue", "actual_winner", "new_p_red_win", "new_p_blue_win"}
    need_b = {"bout_id", "red", "blue", "actual_winner", "all_but_p_red"}
    if missing := sorted(need_a - set(a.columns)):
        raise RuntimeError(f"34-fight ALL-BUT file missing columns: {missing}")
    if missing := sorted(need_b - set(b.columns)):
        raise RuntimeError(f"8-fight ALL-BUT file missing columns: {missing}")

    a = pd.DataFrame({
        "cohort": "frozen_34",
        "bout_id": a["bout_id"].astype(str),
        "event_date": a.get("event_date"),
        "event_name": a.get("event_name"),
        "red": a["red"].astype(str),
        "blue": a["blue"].astype(str),
        "actual_winner": a["actual_winner"].astype(str),
        "model_p_red": pd.to_numeric(a["new_p_red_win"], errors="raise"),
        "model_p_blue": pd.to_numeric(a["new_p_blue_win"], errors="raise"),
    })

    b_red = pd.to_numeric(b["all_but_p_red"], errors="raise")
    b = pd.DataFrame({
        "cohort": "preceding_event_8",
        "bout_id": b["bout_id"].astype(str),
        "event_date": b.get("event_date"),
        "event_name": b.get("event_name"),
        "red": b["red"].astype(str),
        "blue": b["blue"].astype(str),
        "actual_winner": b["actual_winner"].astype(str),
        "model_p_red": b_red,
        "model_p_blue": 1.0 - b_red,
    })

    out = pd.concat([a, b], ignore_index=True)
    if len(out) != 42:
        raise RuntimeError(f"expected 42 model fights, found {len(out)}")
    if out["bout_id"].duplicated().any():
        dupes = out.loc[out["bout_id"].duplicated(keep=False), "bout_id"].tolist()
        raise RuntimeError(f"duplicate bout_ids in 42-fight cohort: {dupes}")

    out["model_favorite"] = np.where(out["model_p_red"] >= out["model_p_blue"], out["red"], out["blue"])
    out["model_correct"] = (out["model_favorite"] == out["actual_winner"]).astype(int)
    return out


def _load_market() -> pd.DataFrame:
    _require(MARKET)
    m = pd.read_parquet(MARKET).copy()
    required = {"fight_id", "market_key", "outcome_side", "american_odds", "implied_probability"}
    if missing := sorted(required - set(m.columns)):
        raise RuntimeError(f"historical moneyline file missing columns: {missing}")

    m = m.loc[m["market_key"].astype(str).eq("moneyline")].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["outcome_side"] = m["outcome_side"].astype(str).str.lower()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["american_odds"] = pd.to_numeric(m["american_odds"], errors="coerce")

    # One historical consensus price per fight/side is expected. If duplicates exist,
    # retain the last deterministic row after sorting by available source metadata.
    sort_cols = [c for c in ["date", "historical_market_timestamp", "legacy_row_number"] if c in m.columns]
    if sort_cols:
        m = m.sort_values(sort_cols)
    m = m.drop_duplicates(["fight_id", "outcome_side"], keep="last")

    keep = ["fight_id", "outcome_side", "american_odds", "implied_probability"]
    p = m[keep].pivot(index="fight_id", columns="outcome_side")
    p.columns = [f"market_{metric}_{side}" for metric, side in p.columns]
    p = p.reset_index().rename(columns={"fight_id": "bout_id"})

    for side in ("red", "blue"):
        col = f"market_implied_probability_{side}"
        if col not in p.columns:
            p[col] = np.nan
        odds_col = f"market_american_odds_{side}"
        if odds_col not in p.columns:
            p[odds_col] = np.nan

    raw_sum = p["market_implied_probability_red"] + p["market_implied_probability_blue"]
    p["market_overround"] = raw_sum - 1.0
    p["market_novig_p_red"] = p["market_implied_probability_red"] / raw_sum
    p["market_novig_p_blue"] = p["market_implied_probability_blue"] / raw_sum
    return p


def _actual_probability(row: pd.Series, red_col: str, blue_col: str) -> float:
    if row["actual_winner"] == row["red"]:
        return float(row[red_col])
    if row["actual_winner"] == row["blue"]:
        return float(row[blue_col])
    return np.nan


def main() -> None:
    model = _load_model_rows()
    market = _load_market()
    out = model.merge(market, on="bout_id", how="left", validate="one_to_one")

    out["market_available"] = (
        out["market_novig_p_red"].notna() & out["market_novig_p_blue"].notna()
    ).astype(int)
    out["market_favorite"] = np.where(
        out["market_available"].eq(1),
        np.where(out["market_novig_p_red"] >= out["market_novig_p_blue"], out["red"], out["blue"]),
        pd.NA,
    )
    out["market_correct"] = np.where(
        out["market_available"].eq(1),
        (out["market_favorite"] == out["actual_winner"]).astype(int),
        np.nan,
    )
    out["model_market_agree"] = np.where(
        out["market_available"].eq(1),
        (out["model_favorite"] == out["market_favorite"]).astype(int),
        np.nan,
    )
    out["model_actual_probability"] = out.apply(
        lambda r: _actual_probability(r, "model_p_red", "model_p_blue"), axis=1
    )
    out["market_actual_probability"] = out.apply(
        lambda r: _actual_probability(r, "market_novig_p_red", "market_novig_p_blue")
        if r["market_available"] == 1 else np.nan,
        axis=1,
    )
    out["model_minus_market_actual_probability"] = (
        out["model_actual_probability"] - out["market_actual_probability"]
    )
    out["model_minus_market_p_red"] = out["model_p_red"] - out["market_novig_p_red"]

    matched = out.loc[out["market_available"].eq(1)].copy()
    disagreements = matched.loc[matched["model_market_agree"].eq(0)].copy()

    n = len(out)
    nm = len(matched)
    model_correct_all = int(out["model_correct"].sum())
    model_correct_matched = int(matched["model_correct"].sum()) if nm else 0
    market_correct = int(matched["market_correct"].sum()) if nm else 0
    agree = int(matched["model_market_agree"].sum()) if nm else 0
    model_wins_disagree = int(((disagreements["model_correct"] == 1) & (disagreements["market_correct"] == 0)).sum())
    market_wins_disagree = int(((disagreements["model_correct"] == 0) & (disagreements["market_correct"] == 1)).sum())

    summary = pd.DataFrame([{
        "cohort_fights": n,
        "market_matched_fights": nm,
        "market_coverage": nm / n if n else np.nan,
        "all_but_correct_all_42": model_correct_all,
        "all_but_accuracy_all_42": model_correct_all / n if n else np.nan,
        "all_but_correct_market_matched": model_correct_matched,
        "all_but_accuracy_market_matched": model_correct_matched / nm if nm else np.nan,
        "market_correct": market_correct,
        "market_accuracy": market_correct / nm if nm else np.nan,
        "model_market_agreements": agree,
        "model_market_agreement_rate": agree / nm if nm else np.nan,
        "model_market_disagreements": len(disagreements),
        "all_but_right_market_wrong_on_disagreements": model_wins_disagree,
        "market_right_all_but_wrong_on_disagreements": market_wins_disagree,
        "both_wrong_on_disagreements": int(((disagreements["model_correct"] == 0) & (disagreements["market_correct"] == 0)).sum()),
        "mean_abs_model_market_probability_gap": float((matched["model_p_red"] - matched["market_novig_p_red"]).abs().mean()) if nm else np.nan,
        "median_abs_model_market_probability_gap": float((matched["model_p_red"] - matched["market_novig_p_red"]).abs().median()) if nm else np.nan,
        "mean_market_overround": float(matched["market_overround"].mean()) if nm else np.nan,
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    s = summary.iloc[0]
    lines = [
        "# ALL-BUT vs Prefight Market — 42-Fight Benchmark",
        "",
        "Model: all prefight FSR points through target, first initialization point removed, degree-2, N+1.",
        "Market: historical legacy-consensus prefight moneyline; probabilities de-vigged per fight.",
        "",
        f"- Cohort fights: {int(s.cohort_fights)}",
        f"- Market matched: {int(s.market_matched_fights)}/{int(s.cohort_fights)} ({s.market_coverage:.1%})",
        f"- ALL-BUT accuracy, all 42: {int(s.all_but_correct_all_42)}/{int(s.cohort_fights)} ({s.all_but_accuracy_all_42:.1%})",
    ]
    if nm:
        lines += [
            f"- ALL-BUT accuracy, market-matched: {int(s.all_but_correct_market_matched)}/{nm} ({s.all_but_accuracy_market_matched:.1%})",
            f"- Market-favorite accuracy: {int(s.market_correct)}/{nm} ({s.market_accuracy:.1%})",
            f"- Model/market favorite agreement: {int(s.model_market_agreements)}/{nm} ({s.model_market_agreement_rate:.1%})",
            f"- Favorite disagreements: {int(s.model_market_disagreements)}",
            f"- On disagreements — ALL-BUT right / market wrong: {int(s.all_but_right_market_wrong_on_disagreements)}",
            f"- On disagreements — market right / ALL-BUT wrong: {int(s.market_right_all_but_wrong_on_disagreements)}",
            f"- Mean absolute model-vs-market red probability gap: {s.mean_abs_model_market_probability_gap:.1%}",
        ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 108)
    print("ALL-BUT vs PREFIGHT MARKET — 34 + 8 = 42 FIGHTS")
    print("=" * 108)
    print(f"market coverage: {nm}/{n} = {nm/n:.1%}")
    print(f"ALL-BUT all-42 accuracy: {model_correct_all}/{n} = {model_correct_all/n:.1%}")
    if nm:
        print(f"ALL-BUT matched accuracy: {model_correct_matched}/{nm} = {model_correct_matched/nm:.1%}")
        print(f"MARKET favorite accuracy: {market_correct}/{nm} = {market_correct/nm:.1%}")
        print(f"model/market agreement: {agree}/{nm} = {agree/nm:.1%}")
        print(f"disagreements: {len(disagreements)} | ALL-BUT right={model_wins_disagree} | market right={market_wins_disagree}")
        print("\nDISAGREEMENTS")
        cols = [
            "cohort", "red", "blue", "actual_winner", "model_favorite", "market_favorite",
            "model_p_red", "market_novig_p_red", "model_correct", "market_correct",
        ]
        if len(disagreements):
            print(disagreements[cols].to_string(index=False))
        else:
            print("none")
    print(f"\nwrote: {OUT_CSV}")
    print(f"wrote: {OUT_SUMMARY}")
    print(f"wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
