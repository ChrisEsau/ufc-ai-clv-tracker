"""Compare ALL-BUT winner predictions against web-sourced prefight moneyline favorites for 42 fights."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from scripts.experimental.compare_all_but_vs_prefight_market_42 import _load_model_rows

ODDS = Path("data/experimental/all_but_vs_prefight_market_42/web_prefight_moneylines_42.csv")
OUT_DIR = Path("data/experimental/all_but_vs_prefight_market_42")
OUT = OUT_DIR / "all_but_vs_web_moneyline_42.csv"
SUMMARY = OUT_DIR / "all_but_vs_web_moneyline_42_summary.csv"
MD = OUT_DIR / "all_but_vs_web_moneyline_42.md"


def implied(odds: float) -> float:
    odds = float(odds)
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def main() -> None:
    model = _load_model_rows().copy()
    odds = pd.read_csv(ODDS).copy()
    odds["event_date"] = pd.to_datetime(odds["event_date"]).dt.strftime("%Y-%m-%d")
    model["event_date"] = pd.to_datetime(model["event_date"]).dt.strftime("%Y-%m-%d")

    if len(odds) != 42:
        raise RuntimeError(f"expected 42 web odds rows, found {len(odds)}")

    out = model.merge(
        odds,
        on=["event_date", "red", "blue"],
        how="left",
        validate="one_to_one",
    )
    missing = out.loc[out["red_odds"].isna(), ["event_date", "red", "blue"]]
    if len(missing):
        print("UNMATCHED MODEL ROWS")
        print(missing.to_string(index=False))
        raise RuntimeError(f"web moneyline coverage {42-len(missing)}/42")

    out["market_raw_p_red"] = out["red_odds"].map(implied)
    out["market_raw_p_blue"] = out["blue_odds"].map(implied)
    total = out["market_raw_p_red"] + out["market_raw_p_blue"]
    out["market_novig_p_red"] = out["market_raw_p_red"] / total
    out["market_novig_p_blue"] = out["market_raw_p_blue"] / total
    out["market_favorite"] = np.where(
        out["market_novig_p_red"] >= out["market_novig_p_blue"], out["red"], out["blue"]
    )
    out["market_correct"] = (out["market_favorite"] == out["actual_winner"]).astype(int)
    out["model_market_agree"] = (out["model_favorite"] == out["market_favorite"]).astype(int)

    disagree = out.loc[out["model_market_agree"].eq(0)].copy()
    model_right = int(((disagree["model_correct"] == 1) & (disagree["market_correct"] == 0)).sum())
    market_right = int(((disagree["model_correct"] == 0) & (disagree["market_correct"] == 1)).sum())

    summary = pd.DataFrame([{
        "fights": len(out),
        "all_but_correct": int(out["model_correct"].sum()),
        "all_but_accuracy": float(out["model_correct"].mean()),
        "market_correct": int(out["market_correct"].sum()),
        "market_accuracy": float(out["market_correct"].mean()),
        "model_market_agreements": int(out["model_market_agree"].sum()),
        "model_market_agreement_rate": float(out["model_market_agree"].mean()),
        "disagreements": len(disagree),
        "all_but_right_market_wrong": model_right,
        "market_right_all_but_wrong": market_right,
    }])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    summary.to_csv(SUMMARY, index=False)

    s = summary.iloc[0]
    lines = [
        "# ALL-BUT vs Web Prefight Moneyline — 42 Fights",
        "",
        f"- ALL-BUT: {int(s.all_but_correct)}/{int(s.fights)} = {s.all_but_accuracy:.1%}",
        f"- Market favorite: {int(s.market_correct)}/{int(s.fights)} = {s.market_accuracy:.1%}",
        f"- Agreement: {int(s.model_market_agreements)}/{int(s.fights)} = {s.model_market_agreement_rate:.1%}",
        f"- Disagreements: {int(s.disagreements)}",
        f"- ALL-BUT right / market wrong: {int(s.all_but_right_market_wrong)}",
        f"- Market right / ALL-BUT wrong: {int(s.market_right_all_but_wrong)}",
        "",
        "Moneylines were manually captured from web event-odds pages; source and source date are stored per fight in web_prefight_moneylines_42.csv.",
    ]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 110)
    print("ALL-BUT vs WEB PREFIGHT MONEYLINE — 42 FIGHTS")
    print("=" * 110)
    print(f"ALL-BUT: {int(s.all_but_correct)}/{int(s.fights)} = {s.all_but_accuracy:.1%}")
    print(f"MARKET:  {int(s.market_correct)}/{int(s.fights)} = {s.market_accuracy:.1%}")
    print(f"agreement: {int(s.model_market_agreements)}/{int(s.fights)} = {s.model_market_agreement_rate:.1%}")
    print(f"disagreements: {int(s.disagreements)} | ALL-BUT right={model_right} | market right={market_right}")
    print("\nDISAGREEMENTS")
    cols = ["event_date", "red", "blue", "actual_winner", "model_favorite", "market_favorite", "red_odds", "blue_odds", "model_correct", "market_correct"]
    print(disagree[cols].to_string(index=False) if len(disagree) else "none")
    print(f"\nwrote: {OUT}")
    print(f"wrote: {SUMMARY}")
    print(f"wrote: {MD}")


if __name__ == "__main__":
    main()
