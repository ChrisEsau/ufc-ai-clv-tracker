from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.research import xgboost_method_market_offset as method

OUT = Path("data/research/prop_mispricing")
SUMMARY = OUT / "ko_market_residual_feature_audit_summary.json"
FEATURES_OUT = OUT / "ko_market_residual_feature_summary.csv"
BINS_OUT = OUT / "ko_market_residual_feature_bins.csv"
INTERACTIONS_OUT = OUT / "ko_market_residual_feature_interactions.csv"
ROWS_OUT = OUT / "ko_market_residual_side_rows.csv"

CUTOFF = pd.Timestamp("2025-12-31")
DEV_START = pd.Timestamp("2021-01-01")
DEV_END = pd.Timestamp("2024-12-31")
VAL_START = pd.Timestamp("2025-01-01")
VAL_END = pd.Timestamp("2025-12-31")
YEARS = [2021, 2022, 2023, 2024]
EPS = 1e-9
MIN_TAIL_N = 80
MIN_INTERACTION_N = 40
TOP_INTERACTION_FEATURES = 15


def _market_aux() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_parquet(method.MARKET_PATH, filters=[("date", "<=", CUTOFF)]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[(m["bookmaker"] == "legacy_consensus") & (m["result_status"] == "graded")].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["profit_per_100"] = pd.to_numeric(m["profit_per_100"], errors="coerce")
    m = m.dropna(subset=["fight_id", "date", "implied_probability"]).copy()
    m["fight_id"] = m["fight_id"].astype(str)

    ml = m[(m["market_key"] == "moneyline") & m["outcome_side"].astype(str).isin(["red", "blue"])].copy()
    counts = ml.groupby(["fight_id", "outcome_side"]).size()
    good = set((str(a), str(b)) for a, b in counts[counts.eq(1)].index)
    ml = ml[ml.apply(lambda r: (str(r["fight_id"]), str(r["outcome_side"])) in good, axis=1)].copy()
    piv = ml.pivot(index="fight_id", columns="outcome_side", values="implied_probability").dropna(subset=["red", "blue"])
    piv["ml_overround"] = piv["red"] + piv["blue"]
    piv["market_ml_p_red"] = piv["red"] / piv["ml_overround"]
    ml_out = piv[["market_ml_p_red", "ml_overround"]].reset_index()

    ko = m[(m["market_key"] == "win_by_ko_tko_dq") & m["outcome_side"].astype(str).isin(["red", "blue"])].copy()
    counts = ko.groupby(["fight_id", "outcome_side"]).size()
    good = set((str(a), str(b)) for a, b in counts[counts.eq(1)].index)
    ko = ko[ko.apply(lambda r: (str(r["fight_id"]), str(r["outcome_side"])) in good, axis=1)].copy()
    ko = ko[["fight_id", "outcome_side", "implied_probability", "profit_per_100"]]
    ko = ko.rename(columns={"implied_probability": "raw_ko_p"})
    ko["decimal_odds"] = 1.0 + ko["profit_per_100"].astype(float) / 100.0
    return ml_out, ko


def _side_rows() -> tuple[pd.DataFrame, list[str]]:
    # Research-only local override so the builder reads through 2025 but never 2026.
    original = method.DEV_CUTOFF
    method.DEV_CUTOFF = CUTOFF
    try:
        fights, features, _ = method._build_rows(True, True)
    finally:
        method.DEV_CUTOFF = original

    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["date"] = pd.to_datetime(fights["date"])
    fights = fights[(fights["date"] >= DEV_START) & (fights["date"] <= VAL_END)].copy()
    ml, raw_ko = _market_aux()
    fights = fights.merge(ml, on="fight_id", how="inner", validate="one_to_one")

    parts = []
    for side, sign, target_class in [("red", 1.0, 0), ("blue", -1.0, 3)]:
        x = fights[[
            "fight_id", "date", "event_name", "red_fighter", "blue_fighter", "target",
            "betting_eligible", "market_overround", "market_ml_p_red", "ml_overround"
        ] + features].copy()
        x["side"] = side
        x["fighter"] = np.where(side == "red", x["red_fighter"], x["blue_fighter"])
        x["actual_ko_win"] = (x["target"].astype(int) == target_class).astype(int)
        x["market_exact_ko_p"] = fights[f"market_{side}_ko"].to_numpy(float)
        x["market_ml_p_side"] = np.where(side == "red", x["market_ml_p_red"], 1.0 - x["market_ml_p_red"])
        x["market_ml_favorite"] = x["market_ml_p_side"] >= 0.5
        for f in features:
            x[f] = pd.to_numeric(x[f], errors="coerce") * sign
        parts.append(x)

    rows = pd.concat(parts, ignore_index=True)
    rows = rows.merge(raw_ko, left_on=["fight_id", "side"], right_on=["fight_id", "outcome_side"], how="inner", validate="one_to_one")
    rows["year"] = rows["date"].dt.year.astype(int)
    rows["market_residual"] = rows["actual_ko_win"].astype(float) - rows["market_exact_ko_p"].astype(float)
    rows["profit_units"] = np.where(rows["actual_ko_win"].eq(1), rows["decimal_odds"] - 1.0, -1.0)
    return rows.sort_values(["date", "fight_id", "side"]).reset_index(drop=True), features


def _tail_stats(x: pd.DataFrame) -> dict:
    if x.empty:
        return {"n": 0, "actual": None, "market": None, "residual": None, "roi": None}
    return {
        "n": int(len(x)),
        "actual": float(x["actual_ko_win"].mean()),
        "market": float(x["market_exact_ko_p"].mean()),
        "residual": float(x["market_residual"].mean()),
        "roi": float(x["profit_units"].mean()),
    }


def _feature_audit(rows: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = rows[(rows["date"] >= DEV_START) & (rows["date"] <= DEV_END) & rows["betting_eligible"]].copy()
    val = rows[(rows["date"] >= VAL_START) & (rows["date"] <= VAL_END) & rows["betting_eligible"]].copy()
    summary_rows = []
    bin_rows = []

    for f in features:
        z = dev[[f, "actual_ko_win", "market_exact_ko_p", "market_residual", "profit_units", "year"]].dropna(subset=[f]).copy()
        if len(z) < 5 * MIN_TAIL_N or z[f].nunique() < 5:
            continue
        q20, q40, q60, q80 = [float(z[f].quantile(q)) for q in [0.2, 0.4, 0.6, 0.8]]
        if len({q20, q40, q60, q80}) < 4:
            continue
        cuts = [-np.inf, q20, q40, q60, q80, np.inf]
        labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        z["bucket"] = pd.cut(z[f], bins=cuts, labels=labels, include_lowest=True)
        for b, g in z.groupby("bucket", observed=True):
            st = _tail_stats(g)
            bin_rows.append({
                "period": "dev_2021_2024", "feature": f, "bucket": str(b),
                "lower": cuts[labels.index(str(b))], "upper": cuts[labels.index(str(b)) + 1],
                "mean_feature": float(g[f].mean()), **st,
            })

        low = z[z[f] <= q20]
        high = z[z[f] >= q80]
        lo = _tail_stats(low); hi = _tail_stats(high)
        if lo["n"] < MIN_TAIL_N or hi["n"] < MIN_TAIL_N:
            continue
        underreaction = float(hi["residual"] - lo["residual"])
        actual_delta = float(hi["actual"] - lo["actual"])
        market_delta = float(hi["market"] - lo["market"])

        year_deltas = {}
        same_sign_years = 0
        for year in YEARS:
            gy = z[z["year"].eq(year)]
            l = gy[gy[f] <= q20]; h = gy[gy[f] >= q80]
            if len(l) >= 15 and len(h) >= 15:
                delta = float(h["market_residual"].mean() - l["market_residual"].mean())
                year_deltas[str(year)] = delta
                if np.sign(delta) == np.sign(underreaction) and delta != 0:
                    same_sign_years += 1
            else:
                year_deltas[str(year)] = None

        vz = val[[f, "actual_ko_win", "market_exact_ko_p", "market_residual", "profit_units"]].dropna(subset=[f]).copy()
        vlo = _tail_stats(vz[vz[f] <= q20]); vhi = _tail_stats(vz[vz[f] >= q80])
        val_delta = None
        val_same = None
        if vlo["n"] >= 10 and vhi["n"] >= 10:
            val_delta = float(vhi["residual"] - vlo["residual"])
            val_same = bool(np.sign(val_delta) == np.sign(underreaction) and val_delta != 0)

        selected_tail = "high" if hi["residual"] >= lo["residual"] else "low"
        selected = hi if selected_tail == "high" else lo
        selected_cut = q80 if selected_tail == "high" else q20
        vselected = vhi if selected_tail == "high" else vlo

        summary_rows.append({
            "feature": f,
            "q20": q20,
            "q80": q80,
            "low_n": lo["n"], "low_actual": lo["actual"], "low_market": lo["market"], "low_residual": lo["residual"], "low_roi_diag": lo["roi"],
            "high_n": hi["n"], "high_actual": hi["actual"], "high_market": hi["market"], "high_residual": hi["residual"], "high_roi_diag": hi["roi"],
            "actual_high_minus_low": actual_delta,
            "market_high_minus_low": market_delta,
            "market_underreaction_high_minus_low": underreaction,
            "same_direction_dev_years": same_sign_years,
            "year_2021_delta": year_deltas["2021"], "year_2022_delta": year_deltas["2022"],
            "year_2023_delta": year_deltas["2023"], "year_2024_delta": year_deltas["2024"],
            "selected_tail": selected_tail,
            "selected_cut": selected_cut,
            "selected_tail_n": selected["n"],
            "selected_tail_residual": selected["residual"],
            "selected_tail_roi_diag": selected["roi"],
            "validation_low_n": vlo["n"], "validation_low_residual": vlo["residual"],
            "validation_high_n": vhi["n"], "validation_high_residual": vhi["residual"],
            "validation_underreaction_high_minus_low": val_delta,
            "validation_same_direction": val_same,
            "validation_selected_tail_n": vselected["n"],
            "validation_selected_tail_residual": vselected["residual"],
            "validation_selected_tail_roi_diag": vselected["roi"],
        })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["abs_underreaction"] = summary["market_underreaction_high_minus_low"].abs()
        summary = summary.sort_values(
            ["same_direction_dev_years", "abs_underreaction", "selected_tail_residual"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary, pd.DataFrame(bin_rows)


def _interaction_audit(rows: pd.DataFrame, feature_summary: pd.DataFrame) -> pd.DataFrame:
    dev = rows[(rows["date"] >= DEV_START) & (rows["date"] <= DEV_END) & rows["betting_eligible"]].copy()
    val = rows[(rows["date"] >= VAL_START) & (rows["date"] <= VAL_END) & rows["betting_eligible"]].copy()
    if feature_summary.empty:
        return pd.DataFrame()

    pool = feature_summary[
        (feature_summary["same_direction_dev_years"] >= 3)
        & (feature_summary["selected_tail_residual"] > 0)
    ].head(TOP_INTERACTION_FEATURES).copy()
    specs = {r.feature: (r.selected_tail, float(r.selected_cut)) for r in pool.itertuples(index=False)}
    out = []

    def mask(frame, feature):
        tail, cut = specs[feature]
        return frame[feature].ge(cut) if tail == "high" else frame[feature].le(cut)

    for a, b in combinations(specs, 2):
        md = mask(dev, a) & mask(dev, b) & dev[a].notna() & dev[b].notna()
        g = dev[md]
        if len(g) < MIN_INTERACTION_N:
            continue
        st = _tail_stats(g)
        year_pos = 0
        year_stats = {}
        for year in YEARS:
            gy = g[g["year"].eq(year)]
            yr = _tail_stats(gy)
            year_stats[str(year)] = yr["residual"]
            if yr["n"] >= 8 and yr["residual"] is not None and yr["residual"] > 0:
                year_pos += 1
        mv = mask(val, a) & mask(val, b) & val[a].notna() & val[b].notna()
        vst = _tail_stats(val[mv])
        out.append({
            "feature_a": a, "feature_b": b,
            "rule_a": f"{a} {'>=' if specs[a][0]=='high' else '<='} {specs[a][1]}",
            "rule_b": f"{b} {'>=' if specs[b][0]=='high' else '<='} {specs[b][1]}",
            "dev_n": st["n"], "dev_actual": st["actual"], "dev_market": st["market"], "dev_residual": st["residual"], "dev_roi_diag": st["roi"],
            "positive_dev_years": year_pos,
            "year_2021_residual": year_stats["2021"], "year_2022_residual": year_stats["2022"],
            "year_2023_residual": year_stats["2023"], "year_2024_residual": year_stats["2024"],
            "validation_n": vst["n"], "validation_actual": vst["actual"], "validation_market": vst["market"],
            "validation_residual": vst["residual"], "validation_roi_diag": vst["roi"],
        })
    d = pd.DataFrame(out)
    if not d.empty:
        d = d.sort_values(["positive_dev_years", "dev_residual", "dev_n"], ascending=[False, False, False]).reset_index(drop=True)
    return d


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, features = _side_rows()
    if (rows["date"] > VAL_END).any():
        raise RuntimeError("2026+ entered KO market residual audit")
    rows.to_csv(ROWS_OUT, index=False)
    feature_summary, bins = _feature_audit(rows, features)
    interactions = _interaction_audit(rows, feature_summary)
    feature_summary.to_csv(FEATURES_OUT, index=False)
    bins.to_csv(BINS_OUT, index=False)
    interactions.to_csv(INTERACTIONS_OUT, index=False)

    dev = rows[(rows["date"] >= DEV_START) & (rows["date"] <= DEV_END) & rows["betting_eligible"]]
    val = rows[(rows["date"] >= VAL_START) & (rows["date"] <= VAL_END) & rows["betting_eligible"]]
    shortlist = feature_summary[
        (feature_summary["same_direction_dev_years"] >= 3)
        & (feature_summary["selected_tail_residual"] > 0.015)
        & feature_summary["validation_same_direction"].eq(True)
        & (feature_summary["validation_selected_tail_residual"] > 0)
    ].head(20)
    summary = {
        "experiment": "ko_market_residual_feature_audit_v1",
        "purpose": "No new predictive model. Identify prefight traits for which actual exact-KO frequency departs from the normalized six-way market KO probability.",
        "development_window": "2021-2024",
        "validation_window": "2025 only",
        "reads_2026_plus": False,
        "model_used_for_selection": False,
        "roi_used_for_feature_ranking": False,
        "orientation": "two rows per fight; all signed-difference features oriented to the priced fighter (positive = more of trait than opponent)",
        "market_baseline": "legacy_consensus six-way normalized exact-KO probability; raw KO odds used only for diagnostic ROI",
        "cold_start_rule": "betting_eligible only for feature ranking and validation",
        "dev_side_rows": int(len(dev)),
        "validation_side_rows": int(len(val)),
        "feature_count": int(len(features)),
        "shortlist_count": int(len(shortlist)),
        "shortlist": shortlist[[
            "feature", "selected_tail", "selected_cut", "selected_tail_n", "selected_tail_residual",
            "market_underreaction_high_minus_low", "same_direction_dev_years",
            "validation_selected_tail_n", "validation_selected_tail_residual", "validation_underreaction_high_minus_low"
        ]].to_dict("records"),
        "artifacts": [str(FEATURES_OUT), str(BINS_OUT), str(INTERACTIONS_OUT), str(ROWS_OUT)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
