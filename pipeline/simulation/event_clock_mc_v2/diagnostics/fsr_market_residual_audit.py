"""Research-only chronological audit of what market strength remains unexplained by FSR V3.

No simulator, FSR, or market data are modified. Uses two-way de-vigged historical
moneylines and strictly pre-fight FSR V3 snapshots. Models market favorite log-odds
from favorite-minus-underdog FSR deltas using chronological train/test evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.fsr_v3_adapter import REQUIRED_MATCHUP_COLUMNS

MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")

META_EXCLUDE = {
    "event_date", "fight_id", "fighter_id", "fighter_name", "name", "date",
    "r_id", "b_id", "winner_id", "red_win", "blue_win", "won", "result",
}


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1.0 - p))


def build_two_way_market(path: Path) -> pd.DataFrame:
    m = pd.read_parquet(path).copy()
    m = m[(m["market_key"] == "moneyline") & m["implied_probability"].notna()].copy()
    m["fight_id"] = m["fight_id"].astype(str)
    m["outcome_fighter_id"] = m["outcome_fighter_id"].astype(str)
    rows = []
    for fid, g in m.groupby("fight_id", sort=False):
        g = g.drop_duplicates("outcome_fighter_id")
        if len(g) != 2:
            continue
        probs = pd.to_numeric(g["implied_probability"], errors="coerce").to_numpy(float)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            continue
        fair = probs / probs.sum()
        gg = g.reset_index(drop=True)
        i = int(np.argmax(fair)); j = 1 - i
        rows.append({
            "fight_id": str(fid),
            "favorite_id": str(gg.loc[i, "outcome_fighter_id"]),
            "underdog_id": str(gg.loc[j, "outcome_fighter_id"]),
            "market_favorite_fair_p": float(fair[i]),
            "favorite_won": int(bool(gg.loc[i, "won"])) if pd.notna(gg.loc[i, "won"]) else np.nan,
        })
    return pd.DataFrame(rows)


def choose_trait_columns(fsr: pd.DataFrame) -> list[str]:
    # Start with all runtime FSR traits, then include other finite numeric pre-fight
    # snapshot fields that are clearly fighter-state variables rather than keys.
    runtime = [c for c in REQUIRED_MATCHUP_COLUMNS if c != "fighter_id" and c in fsr.columns]
    extras = []
    for c in fsr.columns:
        if c in runtime or c in META_EXCLUDE or c.startswith("_"):
            continue
        if not pd.api.types.is_numeric_dtype(fsr[c]):
            continue
        lc = c.lower()
        if any(tok in lc for tok in ("target", "winner", "label", "actual", "future", "postfight")):
            continue
        finite = pd.to_numeric(fsr[c], errors="coerce").notna().mean()
        if finite >= 0.80:
            extras.append(c)
    return sorted(dict.fromkeys(runtime + extras))


def build_matchups(market: pd.DataFrame, fsr: pd.DataFrame, master: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    fsr = fsr.copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    master = master.copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = next((c for c in ("date", "event_date", "fight_date") if c in master.columns), None)
    if date_col is None:
        raise RuntimeError("master has no recognized date column")
    dates = master[["fight_id", date_col]].drop_duplicates("fight_id").copy()
    dates["fight_date"] = pd.to_datetime(dates[date_col], errors="coerce").dt.normalize()

    look = fsr.set_index(["fight_id", "fighter_id"])
    rows = []
    for _, m in market.iterrows():
        fid = str(m["fight_id"]); fav = str(m["favorite_id"]); dog = str(m["underdog_id"])
        if (fid, fav) not in look.index or (fid, dog) not in look.index:
            continue
        fr = look.loc[(fid, fav)]; dr = look.loc[(fid, dog)]
        if isinstance(fr, pd.DataFrame) or isinstance(dr, pd.DataFrame):
            continue
        rec = dict(m)
        for c in traits:
            fv = pd.to_numeric(pd.Series([fr.get(c)]), errors="coerce").iloc[0]
            dv = pd.to_numeric(pd.Series([dr.get(c)]), errors="coerce").iloc[0]
            rec[f"delta__{c}"] = float(fv - dv) if pd.notna(fv) and pd.notna(dv) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows).merge(dates[["fight_id", "fight_date"]], on="fight_id", how="left")
    return out.sort_values(["fight_date", "fight_id"]).reset_index(drop=True)


def evaluate_temporal(frame: pd.DataFrame, features: list[str], train_frac: float = 0.70):
    usable = frame.dropna(subset=features + ["market_favorite_fair_p", "fight_date"]).copy()
    if len(usable) < 100:
        raise RuntimeError(f"too few complete market+FSR fights: {len(usable)}")
    cut = max(1, min(len(usable)-1, int(len(usable) * train_frac)))
    train = usable.iloc[:cut].copy(); test = usable.iloc[cut:].copy()
    y_train = safe_logit(train["market_favorite_fair_p"]); y_test = safe_logit(test["market_favorite_fair_p"])
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(train[features], y_train)
    pred_train = model.predict(train[features]); pred_test = model.predict(test[features])
    test["fsr_expected_market_logit"] = pred_test
    test["fsr_expected_market_p"] = 1.0 / (1.0 + np.exp(-pred_test))
    test["market_logit"] = y_test
    test["residual_logit"] = y_test - pred_test
    test["residual_pp"] = 100.0 * (test["market_favorite_fair_p"] - test["fsr_expected_market_p"])
    metrics = {
        "all_fights": len(usable), "train_fights": len(train), "test_fights": len(test),
        "train_start": train["fight_date"].min(), "train_end": train["fight_date"].max(),
        "test_start": test["fight_date"].min(), "test_end": test["fight_date"].max(),
        "train_r2_logit": r2_score(y_train, pred_train),
        "test_r2_logit": r2_score(y_test, pred_test),
        "train_rmse_logit": mean_squared_error(y_train, pred_train) ** 0.5,
        "test_rmse_logit": mean_squared_error(y_test, pred_test) ** 0.5,
        "mean_abs_residual_pp": float(test["residual_pp"].abs().mean()),
        "median_abs_residual_pp": float(test["residual_pp"].abs().median()),
        "corr_expected_vs_market_p": float(test["fsr_expected_market_p"].corr(test["market_favorite_fair_p"])),
    }
    return model, train, test, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    market = build_two_way_market(MARKET_PATH)
    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    master = pd.read_parquet(MASTER_PATH).copy()
    traits = choose_trait_columns(fsr)
    matchups = build_matchups(market, fsr, master, traits)
    features = [f"delta__{c}" for c in traits]

    model, train, test, metrics = evaluate_temporal(matchups, features)

    ridge = model.named_steps["ridge"]
    coef = pd.DataFrame({"feature": traits, "standardized_coefficient": ridge.coef_})
    coef["abs_coefficient"] = coef["standardized_coefficient"].abs()
    coef = coef.sort_values("abs_coefficient", ascending=False)

    # Market-strength buckets on untouched chronological test set.
    test["market_bucket"] = pd.cut(test["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.001], labels=["50-60","60-70","70-80","80-90","90+"], right=False)
    buckets = test.groupby("market_bucket", observed=True).agg(
        fights=("fight_id","size"), market_p=("market_favorite_fair_p","mean"),
        fsr_expected_p=("fsr_expected_market_p","mean"), residual_pp=("residual_pp","mean"),
        abs_residual_pp=("residual_pp", lambda s: s.abs().mean()), favorite_win_rate=("favorite_won","mean"),
    ).reset_index()

    biggest = test.reindex(test["residual_pp"].abs().sort_values(ascending=False).index).head(50)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out_dir/"temporal_summary.csv", index=False)
    coef.to_csv(args.out_dir/"fsr_market_coefficients.csv", index=False)
    test.to_csv(args.out_dir/"chronological_test_residuals.csv", index=False)
    biggest.to_csv(args.out_dir/"largest_market_fsr_residuals.csv", index=False)
    buckets.to_csv(args.out_dir/"market_strength_residual_buckets.csv", index=False)
    pd.DataFrame({"trait": traits}).to_csv(args.out_dir/"trait_columns_used.csv", index=False)

    print("FSR V3 -> MARKET STRENGTH CHRONOLOGICAL RESIDUAL AUDIT")
    print(f"two-way market fights={len(market)} | joined FSR fights={len(matchups)} | traits={len(traits)}")
    print("\nTEMPORAL SUMMARY")
    print(pd.DataFrame([metrics]).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nMARKET-STRENGTH TEST BUCKETS")
    print(buckets.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nTOP STANDARDIZED FSR COEFFICIENTS")
    print(coef.head(25).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print("\nLARGEST ABSOLUTE TEST RESIDUALS")
    cols = ["fight_id","fight_date","market_favorite_fair_p","fsr_expected_market_p","residual_pp","favorite_won"]
    print(biggest[cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

if __name__ == "__main__":
    main()
