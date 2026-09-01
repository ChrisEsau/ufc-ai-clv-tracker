from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/research/prop_mispricing")
MARKET = Path("data/market/historical_market_outcomes.parquet")
SUMMARY = OUT / "ko_method_market_price_integrity_2021_2025_summary.json"
YEARLY = OUT / "ko_method_market_price_integrity_2021_2025_yearly.csv"
SOURCES = OUT / "ko_method_market_price_integrity_2021_2025_sources.csv"
FIGHTS = OUT / "ko_method_market_price_integrity_2025_fights.csv"

START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2025-12-31")
METHOD_KEYS = ["win_by_ko_tko_dq", "win_by_submission", "win_by_decision"]
SLUG_MAP = {
    ("red", "win_by_ko_tko_dq"): "red_ko",
    ("red", "win_by_submission"): "red_sub",
    ("red", "win_by_decision"): "red_dec",
    ("blue", "win_by_ko_tko_dq"): "blue_ko",
    ("blue", "win_by_submission"): "blue_sub",
    ("blue", "win_by_decision"): "blue_dec",
}
SLUGS = ["red_ko", "red_sub", "red_dec", "blue_ko", "blue_sub", "blue_dec"]


def american_from_p(p: float) -> float:
    p = float(p)
    if p <= 0 or p >= 1:
        return float("nan")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def quantiles(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MARKET, filters=[("date", ">=", START), ("date", "<=", END)]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m[
        (m["bookmaker"] == "legacy_consensus")
        & m["market_key"].isin(METHOD_KEYS)
        & m["outcome_side"].astype(str).isin(["red", "blue"])
        & (m["result_status"] == "graded")
        & m["won"].notna()
    ].copy()
    m["implied_probability"] = pd.to_numeric(m["implied_probability"], errors="coerce")
    m["profit_per_100"] = pd.to_numeric(m["profit_per_100"], errors="coerce")
    m = m.dropna(subset=["date", "fight_id", "implied_probability"])
    m = m[np.isfinite(m["implied_probability"]) & (m["implied_probability"] > 0) & (m["implied_probability"] < 1)].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["side"] = m["outcome_side"].astype(str)
    m["slug"] = [SLUG_MAP.get((s, k)) for s, k in zip(m["side"], m["market_key"])]
    m = m[m["slug"].notna()].copy()
    m["year"] = m["date"].dt.year.astype(int)

    counts = m.groupby(["fight_id", "slug"]).size().unstack(fill_value=0)
    for s in SLUGS:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[SLUGS]
    complete_ids = counts.index[(counts == 1).all(axis=1)]
    c = m[m["fight_id"].isin(complete_ids)].copy()

    implied = c.pivot(index="fight_id", columns="slug", values="implied_probability").reindex(columns=SLUGS)
    overround = implied.sum(axis=1).rename("six_way_implied_sum")
    meta_cols = [x for x in ["fight_id", "date", "event_name", "source", "mapping_method"] if x in c.columns]
    meta = c.sort_values(["date", "fight_id"]).groupby("fight_id", as_index=False).first()[meta_cols]
    meta["fight_id"] = meta["fight_id"].astype(str)
    fights = meta.merge(overround, on="fight_id", how="inner")
    fights["year"] = pd.to_datetime(fights["date"]).dt.year.astype(int)
    for slug in SLUGS:
        fights[f"raw_p_{slug}"] = fights["fight_id"].map(implied[slug])
        fights[f"american_{slug}"] = fights[f"raw_p_{slug}"].map(american_from_p)

    yearly_rows = []
    for year in range(2021, 2026):
        g = fights[fights["year"].eq(year)].copy()
        q = quantiles(g["six_way_implied_sum"])
        yearly_rows.append({
            "year": year,
            "complete_six_way_fights": int(len(g)),
            "overround_mean": q.get("mean"),
            "overround_median": q.get("median"),
            "overround_p05": q.get("p05"),
            "overround_p95": q.get("p95"),
            "fraction_sum_below_0_80": float((g["six_way_implied_sum"] < 0.80).mean()) if len(g) else None,
            "fraction_sum_below_0_90": float((g["six_way_implied_sum"] < 0.90).mean()) if len(g) else None,
            "fraction_sum_below_1_00": float((g["six_way_implied_sum"] < 1.00).mean()) if len(g) else None,
            "fraction_sum_1_00_to_1_30": float(g["six_way_implied_sum"].between(1.00, 1.30, inclusive="both").mean()) if len(g) else None,
            "fraction_sum_above_1_30": float((g["six_way_implied_sum"] > 1.30).mean()) if len(g) else None,
            "red_ko_raw_p_median": float(g["raw_p_red_ko"].median()) if len(g) else None,
            "blue_ko_raw_p_median": float(g["raw_p_blue_ko"].median()) if len(g) else None,
        })
    yearly = pd.DataFrame(yearly_rows)
    yearly.to_csv(YEARLY, index=False)

    source_cols = [x for x in ["year", "source", "mapping_method"] if x in m.columns]
    if source_cols:
        sources = (
            m.groupby(source_cols, dropna=False)
            .agg(rows=("fight_id", "size"), fights=("fight_id", "nunique"))
            .reset_index()
            .sort_values(["year", "rows"], ascending=[True, False])
        )
    else:
        sources = pd.DataFrame()
    sources.to_csv(SOURCES, index=False)

    f2025 = fights[fights["year"].eq(2025)].sort_values(["date", "fight_id"]).reset_index(drop=True)
    f2025.to_csv(FIGHTS, index=False)

    known = {}
    for label, term in {
        "Sandhagen_vs_Figueiredo": "Sandhagen vs. Figueiredo",
        "Burns_vs_Morales": "Burns vs. Morales",
        "Topuria_vs_Oliveira": "Topuria vs. Oliveira",
    }.items():
        z = f2025[f2025["event_name"].astype(str).str.contains(term, case=False, regex=False, na=False)]
        if not z.empty:
            # Event can contain many fights; retain the most extreme underround rows for inspection.
            known[label] = z.sort_values("six_way_implied_sum").head(5).to_dict(orient="records")

    summary = {
        "experiment": "ko_method_market_price_integrity_2021_2025",
        "evaluation_end": "2025-12-31",
        "2026_read_or_evaluated": False,
        "bookmaker": "legacy_consensus",
        "market_keys": METHOD_KEYS,
        "complete_six_way_fights": int(len(fights)),
        "yearly": yearly.to_dict(orient="records"),
        "source_distribution": sources.to_dict(orient="records"),
        "2025_overround_quantiles": quantiles(f2025["six_way_implied_sum"]),
        "2025_known_event_rows": known,
        "integrity_flags": {
            "2025_majority_six_way_sum_below_1": bool((f2025["six_way_implied_sum"] < 1.0).mean() > 0.5) if len(f2025) else None,
            "2025_material_underround_below_0_8": bool((f2025["six_way_implied_sum"] < 0.8).mean() > 0.1) if len(f2025) else None,
        },
        "interpretation_policy": "Do not interpret 2025 KO ROI from this feed until source/mapping integrity is established.",
        "artifacts": [str(YEARLY), str(SOURCES), str(FIGHTS)],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
