from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/research/prop_mispricing"
PRED = OUT / "xgboost_method_hierarchical_v5_oof_predictions.csv"
MARKET = ROOT / "data/market/historical_market_outcomes.parquet"

CANDIDATES_CSV = OUT / "xgboost_method_hierarchical_v5_edge_candidates.csv"
SIGNAL_BINS_CSV = OUT / "xgboost_method_hierarchical_v5_edge_signal_bins.csv"
INTERACTIONS_CSV = OUT / "xgboost_method_hierarchical_v5_edge_interactions.csv"
RULES_CSV = OUT / "xgboost_method_hierarchical_v5_edge_rules.csv"
ROBUSTNESS_CSV = OUT / "xgboost_method_hierarchical_v5_edge_robustness.csv"
SUMMARY_JSON = OUT / "xgboost_method_hierarchical_v5_edge_discovery_summary.json"

EPS = 1e-12
YEARS = [2021, 2022, 2023, 2024]
SLUGS = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]
META = {
    "red_ko": ("red", "win_by_ko_tko_dq", 0, "KO"),
    "red_sub": ("red", "win_by_submission", 1, "SUB"),
    "red_dec": ("red", "win_by_decision", 2, "DEC"),
    "blue_ko": ("blue", "win_by_ko_tko_dq", 3, "KO"),
    "blue_sub": ("blue", "win_by_submission", 4, "SUB"),
    "blue_dec": ("blue", "win_by_decision", 5, "DEC"),
}


def clip_p(x):
    return np.clip(np.asarray(x, dtype=float), EPS, 1 - EPS)


def logit(x):
    p = clip_p(x)
    return np.log(p / (1 - p))


def american_from_decimal(dec: float) -> float:
    return 100.0 * (dec - 1.0) if dec >= 2.0 else -100.0 / (dec - 1.0)


def max_drawdown(profits: pd.Series) -> float:
    if len(profits) == 0:
        return 0.0
    equity = profits.astype(float).cumsum().to_numpy()
    equity = np.concatenate([[0.0], equity])
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def metrics(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return {
            "bets": 0, "wins": 0, "hit_rate": None, "profit": 0.0, "roi": None,
            "mean_odds": None, "median_odds": None, "avg_model_p": None,
            "avg_market_p": None, "avg_ev": None, "max_drawdown": 0.0,
        }
    g = d.sort_values(["date", "fight_id", "slug"]).copy()
    return {
        "bets": int(len(g)),
        "wins": int(g["won"].sum()),
        "hit_rate": float(g["won"].mean()),
        "profit": float(g["profit"].sum()),
        "roi": float(g["profit"].mean()),
        "mean_odds": float(g["american_odds"].mean()),
        "median_odds": float(g["american_odds"].median()),
        "avg_model_p": float(g["model_p"].mean()),
        "avg_market_p": float(g["market_p"].mean()),
        "avg_ev": float(g["ev"].mean()),
        "max_drawdown": max_drawdown(g["profit"]),
    }


def add_metric_row(rows: list[dict], family: str, label: str, d: pd.DataFrame, **extra) -> None:
    base = {"family": family, "label": label, **extra, **metrics(d)}
    rows.append(base)
    for yr in YEARS:
        rows.append({"family": family, "label": label, **extra, "year": yr, **metrics(d[d["year"] == yr])})


def normalized_moneyline(m: pd.DataFrame) -> dict[tuple[str, str], float]:
    ml = m[(m["bookmaker"] == "legacy_consensus") & (m["market_key"] == "moneyline")].copy()
    ml = ml[ml["outcome_side"].astype(str).isin(["red", "blue"])]
    ml["implied_probability"] = pd.to_numeric(ml["implied_probability"], errors="coerce")
    ml = ml.dropna(subset=["implied_probability"])
    ml = ml.sort_values(["fight_id", "outcome_side"]).drop_duplicates(["fight_id", "outcome_side"], keep=False)
    out: dict[tuple[str, str], float] = {}
    for fid, g in ml.groupby(ml["fight_id"].astype(str)):
        vals = {str(r.outcome_side): float(r.implied_probability) for r in g.itertuples(index=False)}
        if "red" not in vals or "blue" not in vals:
            continue
        s = vals["red"] + vals["blue"]
        if s <= 0:
            continue
        out[(str(fid), "red")] = vals["red"] / s
        out[(str(fid), "blue")] = vals["blue"] / s
    return out


def method_raw_prices(m: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    z = m[(m["bookmaker"] == "legacy_consensus") & m["outcome_side"].astype(str).isin(["red", "blue"])].copy()
    z["implied_probability"] = pd.to_numeric(z["implied_probability"], errors="coerce")
    z["american_odds"] = pd.to_numeric(z["american_odds"], errors="coerce")
    price: dict[tuple[str, str], tuple[float, float]] = {}
    for slug, (side, key, _, _) in META.items():
        q = z[(z["outcome_side"].astype(str) == side) & (z["market_key"] == key)].copy()
        q = q.dropna(subset=["implied_probability"]).drop_duplicates("fight_id", keep=False)
        for r in q.itertuples(index=False):
            raw = float(r.implied_probability)
            if not (0 < raw < 1):
                continue
            dec = 1.0 / raw
            amer = float(r.american_odds) if pd.notna(r.american_odds) else american_from_decimal(dec)
            price[(str(r.fight_id), slug)] = (dec, amer)
    return price


def build_candidates() -> pd.DataFrame:
    p = pd.read_csv(PRED)
    p["fight_id"] = p["fight_id"].astype(str)
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    p["year"] = p["date"].dt.year
    p = p[p["year"].isin(YEARS)].copy()
    if sorted(p["year"].unique().tolist()) != YEARS:
        raise ValueError(f"Expected OOF years {YEARS}, got {sorted(p['year'].unique().tolist())}")

    m = pd.read_parquet(MARKET)
    m["fight_id"] = m["fight_id"].astype(str)
    prices = method_raw_prices(m)
    ml = normalized_moneyline(m)

    rows: list[dict] = []
    for r in p.itertuples(index=False):
        red_win = float(r.v5_model_p_red)
        winner_side = "red" if red_win >= 0.5 else "blue"
        winner_p = red_win if winner_side == "red" else 1.0 - red_win

        side_slugs = [s for s in SLUGS if META[s][0] == winner_side]
        side_uncond = np.array([float(getattr(r, f"hier_{s}")) for s in side_slugs], dtype=float)
        # Conditional method probabilities are defined by the frozen hierarchy.
        cond = side_uncond / max(winner_p, EPS)
        cond = np.clip(cond, 0.0, 1.0)
        if cond.sum() > 0:
            cond = cond / cond.sum()
        order = np.argsort(-cond)
        top_slug = side_slugs[int(order[0])]
        top_cond = float(cond[order[0]])
        second_cond = float(cond[order[1]])
        gap = top_cond - second_cond
        entropy = float(-np.sum(np.where(cond > 0, cond * np.log(cond), 0.0)) / math.log(3.0))
        concentration = 1.0 - entropy

        ml_p = ml.get((str(r.fight_id), winner_side))
        ml_market_side = None
        if ml_p is not None:
            ml_market_side = winner_side if ml_p >= 0.5 else ("blue" if winner_side == "red" else "red")

        for slug, cp in zip(side_slugs, cond):
            side, _, target_idx, method = META[slug]
            market_p = float(getattr(r, f"market_{slug}"))
            model_p = float(getattr(r, f"hier_{slug}"))
            raw_price = prices.get((str(r.fight_id), slug))
            if raw_price is None or not (0 < market_p < 1) or not (0 < model_p < 1):
                continue
            dec, amer = raw_price
            won = int(int(r.target) == target_idx)
            profit = dec - 1.0 if won else -1.0
            ev = model_p * dec - 1.0
            prob_diff = model_p - market_p
            prob_ratio = model_p / market_p
            residual = float(logit(model_p) - logit(market_p))
            rows.append({
                "fight_id": str(r.fight_id), "date": pd.Timestamp(r.date), "year": int(r.year),
                "winner_side": winner_side, "method": method, "slug": slug,
                "ml_model_p": winner_p, "ml_market_p": ml_p,
                "ml_model_market_diff": (winner_p - ml_p) if ml_p is not None else np.nan,
                "ml_model_market_logit_residual": (float(logit(winner_p) - logit(ml_p)) if ml_p is not None else np.nan),
                "ml_market_agrees_winner": (bool(ml_p >= 0.5) if ml_p is not None else np.nan),
                "conditional_method_p": float(cp), "top_method": slug == top_slug,
                "top_conditional_p": top_cond, "second_conditional_p": second_cond,
                "top_second_gap": gap, "method_entropy": entropy, "method_concentration": concentration,
                "model_p": model_p, "market_p": market_p,
                "decimal_odds": dec, "american_odds": amer,
                "ev": ev, "prob_diff": prob_diff, "prob_ratio": prob_ratio, "logit_residual": residual,
                "won": won, "profit": profit,
            })

    d = pd.DataFrame(rows).sort_values(["date", "fight_id", "slug"]).reset_index(drop=True)
    if not set(d["year"].unique()).issubset(set(YEARS)):
        raise AssertionError("Candidate table contains data outside 2021-2024")
    return d


def add_bins(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["ml_conf_bin"] = pd.cut(out["ml_model_p"], [0.5, 0.6, 0.7, 0.8, 1.000001], right=False, labels=[".50-.599", ".60-.699", ".70-.799", ".80+"])
    out["cond_top_bin"] = pd.cut(out["top_conditional_p"], [0, 0.4, 0.5, 0.6, 1.000001], right=False, labels=["<.40", ".40-.499", ".50-.599", ".60+"])
    out["gap_bin"] = pd.cut(out["top_second_gap"], [-1e-9, 0.05, 0.15, 0.25, 1.000001], right=False, labels=["<.05", ".05-.149", ".15-.249", ".25+"])
    out["price_bin"] = pd.cut(out["american_odds"], [-np.inf, 0, 200, 400, 700, np.inf], right=False, labels=["negative", "+100-199", "+200-399", "+400-699", "+700+"])
    out["model_p_bin"] = pd.cut(out["model_p"], [0, .15, .25, .35, .50, 1.000001], right=False, labels=["<.15", ".15-.249", ".25-.349", ".35-.499", ".50+"])
    out["ev_bin"] = pd.cut(out["ev"], [-np.inf, 0, .10, .25, .50, np.inf], right=False, labels=["<=0", "0-.099", ".10-.249", ".25-.499", ".50+"])
    out["prob_diff_bin"] = pd.cut(out["prob_diff"], [-np.inf, 0, .03, .06, .10, np.inf], right=False, labels=["<=0", "0-.029", ".03-.059", ".06-.099", ".10+"])
    out["prob_ratio_bin"] = pd.cut(out["prob_ratio"], [-np.inf, 1, 1.15, 1.35, 1.60, np.inf], right=False, labels=["<1", "1-1.149", "1.15-1.349", "1.35-1.599", "1.60+"])
    out["residual_bin"] = pd.cut(out["logit_residual"], [-np.inf, .2, .3, .4, .5, .75, np.inf], right=False, labels=["<.20", ".20-.299", ".30-.399", ".40-.499", ".50-.749", ".75+"])
    return out


def signal_tables(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    specs = [
        ("ml_confidence", "ml_conf_bin"), ("top_conditional_probability", "cond_top_bin"),
        ("top_second_gap", "gap_bin"), ("price", "price_bin"), ("model_probability", "model_p_bin"),
        ("expected_value", "ev_bin"), ("probability_difference", "prob_diff_bin"),
        ("probability_ratio", "prob_ratio_bin"), ("logit_residual", "residual_bin"),
        ("method", "method"), ("top_method", "top_method"), ("ml_market_agreement", "ml_market_agrees_winner"),
    ]
    for fam, col in specs:
        for label, g in d.groupby(col, observed=True, dropna=False):
            add_metric_row(rows, fam, str(label), g)
            for method, gm in g.groupby("method"):
                add_metric_row(rows, fam + "_by_method", str(label), gm, method=method)
    return pd.DataFrame(rows)


def interaction_tables(d: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    # Coarse, interpretable two-signal interactions. These are diagnostics, not optimized thresholds.
    pairs = [
        ("ml_conf_bin", "cond_top_bin"), ("ml_conf_bin", "gap_bin"),
        ("cond_top_bin", "ev_bin"), ("gap_bin", "ev_bin"),
        ("method", "ev_bin"), ("method", "residual_bin"),
        ("method", "price_bin"), ("top_method", "ev_bin"),
        ("ml_market_agrees_winner", "ev_bin"), ("ml_conf_bin", "price_bin"),
    ]
    for a, b in pairs:
        for keys, g in d.groupby([a, b], observed=True, dropna=False):
            label = f"{a}={keys[0]} | {b}={keys[1]}"
            add_metric_row(rows, f"{a}__{b}", label, g)
    return pd.DataFrame(rows)


def one_per_fight(d: pd.DataFrame, score_col: str, eligible: pd.Series) -> pd.DataFrame:
    q = d[eligible].copy()
    if len(q) == 0:
        return q
    return q.sort_values(["fight_id", score_col], ascending=[True, False]).drop_duplicates("fight_id", keep="first")


def rule_sets(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Named structural hypotheses. Round cutoffs are intentionally sparse and interpretable.
    positive_ev = d["ev"] > 0
    top = d["top_method"]
    rules = {
        "benchmark_residual_030_multi": d[d["logit_residual"] >= .30],
        "benchmark_residual_030_one": one_per_fight(d, "logit_residual", d["logit_residual"] >= .30),
        "top_method_positive_ev": d[top & positive_ev],
        "dec_top_positive_ev": d[top & positive_ev & (d["method"] == "DEC")],
        "top_positive_ev_ml60": d[top & positive_ev & (d["ml_model_p"] >= .60)],
        "top_positive_ev_ml65": d[top & positive_ev & (d["ml_model_p"] >= .65)],
        "top_positive_ev_cond50": d[top & positive_ev & (d["top_conditional_p"] >= .50)],
        "top_positive_ev_gap15": d[top & positive_ev & (d["top_second_gap"] >= .15)],
        "top_positive_ev_ml60_gap15": d[top & positive_ev & (d["ml_model_p"] >= .60) & (d["top_second_gap"] >= .15)],
        "top_positive_ev_ml60_cond50": d[top & positive_ev & (d["ml_model_p"] >= .60) & (d["top_conditional_p"] >= .50)],
        "dec_top_positive_ev_ml60": d[top & positive_ev & (d["method"] == "DEC") & (d["ml_model_p"] >= .60)],
        "dec_top_positive_ev_gap15": d[top & positive_ev & (d["method"] == "DEC") & (d["top_second_gap"] >= .15)],
        "dec_top_positive_ev_ml60_gap15": d[top & positive_ev & (d["method"] == "DEC") & (d["ml_model_p"] >= .60) & (d["top_second_gap"] >= .15)],
        "dec_positive_ev_one_best_ev": one_per_fight(d, "ev", positive_ev & (d["method"] == "DEC")),
        "ko_positive_ev_one_best_ev": one_per_fight(d, "ev", positive_ev & (d["method"] == "KO")),
        "sub_positive_ev_one_best_ev": one_per_fight(d, "ev", positive_ev & (d["method"] == "SUB")),
    }
    return rules


def rule_table(rules: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, g in rules.items():
        add_metric_row(rows, "rule", name, g)
    return pd.DataFrame(rows)


def robustness_table(rules: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for name, g0 in rules.items():
        g0 = g0.sort_values(["date", "fight_id", "slug"]).copy()
        rows.append({"rule": name, "test": "base", **metrics(g0)})
        for omit in YEARS:
            rows.append({"rule": name, "test": f"leave_out_{omit}", **metrics(g0[g0["year"] != omit])})
        if len(g0):
            wins = g0[g0["won"] == 1]
            if len(wins):
                idx = wins["profit"].idxmax()
                rows.append({"rule": name, "test": "remove_largest_winner", **metrics(g0.drop(index=idx))})
        for cap in [500, 750, 1000]:
            rows.append({"rule": name, "test": f"odds_cap_+{cap}", **metrics(g0[g0["american_odds"] <= cap])})
        # Calibration diagnostic for the candidate segment.
        if len(g0):
            rows.append({
                "rule": name, "test": "calibration",
                "bets": int(len(g0)), "wins": int(g0["won"].sum()),
                "hit_rate": float(g0["won"].mean()),
                "avg_model_p": float(g0["model_p"].mean()),
                "calibration_error": float(g0["won"].mean() - g0["model_p"].mean()),
            })
    return pd.DataFrame(rows)


def summarize_shortlist(rule_df: pd.DataFrame, robust_df: pd.DataFrame) -> list[dict]:
    base = rule_df[rule_df["year"].isna()].copy()
    yearly = rule_df[rule_df["year"].notna()].copy()
    loo = robust_df[robust_df["test"].str.startswith("leave_out_")].copy()
    out = []
    for r in base.itertuples(index=False):
        name = r.label
        y = yearly[yearly["label"] == name]
        l = loo[loo["rule"] == name]
        positive_years = int((y["roi"].fillna(-999) > 0).sum())
        positive_loo = int((l["roi"].fillna(-999) > 0).sum())
        min_year_bets = int(y["bets"].min()) if len(y) else 0
        # Stability-first eligibility. ROI only breaks ties after sample/stability gates.
        eligible = int(r.bets) >= 80 and positive_years >= 3 and positive_loo >= 3 and min_year_bets >= 10
        if eligible:
            out.append({
                "rule": name, "bets": int(r.bets), "roi": float(r.roi),
                "positive_years": positive_years, "positive_leave_one_year_out": positive_loo,
                "min_year_bets": min_year_bets, "max_drawdown": float(r.max_drawdown),
            })
    out.sort(key=lambda x: (-x["positive_years"], -x["positive_leave_one_year_out"], -x["bets"], -x["roi"]))
    # Preserve structural diversity rather than returning near-duplicate threshold variants.
    chosen = []
    families_seen = set()
    for item in out:
        name = item["rule"]
        if name.startswith("dec_top") or name.startswith("dec_positive"):
            fam = "dec"
        elif name.startswith("benchmark"):
            fam = "residual"
        else:
            fam = "top_method"
        if fam in families_seen:
            continue
        families_seen.add(fam)
        chosen.append(item)
        if len(chosen) == 4:
            break
    return chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = add_bins(build_candidates())
    signals = signal_tables(d)
    interactions = interaction_tables(d)
    rules = rule_sets(d)
    rules_df = rule_table(rules)
    robust = robustness_table(rules)
    shortlist = summarize_shortlist(rules_df, robust)

    d.to_csv(CANDIDATES_CSV, index=False)
    signals.to_csv(SIGNAL_BINS_CSV, index=False)
    interactions.to_csv(INTERACTIONS_CSV, index=False)
    rules_df.to_csv(RULES_CSV, index=False)
    robust.to_csv(ROBUSTNESS_CSV, index=False)

    base_rules = rules_df[rules_df["year"].isna()].set_index("label")
    summary = {
        "scope": "chronological 2021-2024 OOF only",
        "candidate_rows": int(len(d)),
        "fights": int(d["fight_id"].nunique()),
        "years": sorted(d["year"].unique().tolist()),
        "benchmark_residual_030_multi": base_rules.loc["benchmark_residual_030_multi"].dropna().to_dict(),
        "benchmark_residual_030_one": base_rules.loc["benchmark_residual_030_one"].dropna().to_dict(),
        "top_method_positive_ev": base_rules.loc["top_method_positive_ev"].dropna().to_dict(),
        "dec_top_positive_ev": base_rules.loc["dec_top_positive_ev"].dropna().to_dict(),
        "stability_shortlist": shortlist,
        "selection_note": "Shortlist gating uses sample size and year/leave-one-year-out stability first; pooled ROI is not the primary selector.",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    for path in [CANDIDATES_CSV, SIGNAL_BINS_CSV, INTERACTIONS_CSV, RULES_CSV, ROBUSTNESS_CSV, SUMMARY_JSON]:
        print(path)


if __name__ == "__main__":
    main()
