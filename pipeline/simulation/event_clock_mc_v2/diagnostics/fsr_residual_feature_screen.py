"""Research-only screen for non-market feature families that explain FSR market residuals.

Uses the same historical two-way market cohort and chronological split as the FSR
market residual audit. Candidate features are strictly pre-fight and derived only
from historical UFC results, FSR trajectory, and optional master physical fields.
No simulator, FSR, market, or raw data are modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH
from pipeline.simulation.event_clock_mc_v2.diagnostics.fsr_market_residual_audit import (
    MASTER_PATH,
    build_matchups,
    build_two_way_market,
    choose_trait_columns,
    safe_logit,
)

MARKET_PATH = Path("data/market/historical_market_outcomes.parquet")
DATE_CANDIDATES = ("date", "event_date", "fight_date")


def norm_id(x) -> str:
    s = str(x).strip()
    return "" if s in {"", "nan", "None"} else s


def resolve_date_col(df: pd.DataFrame) -> str:
    for c in DATE_CANDIDATES:
        if c in df.columns:
            return c
    raise RuntimeError(f"master has no supported date column: {DATE_CANDIDATES}")


def build_history_features(master: pd.DataFrame) -> pd.DataFrame:
    """Sequential pre-fight fighter state. Elo is a screening feature, not a final model."""
    m = master.copy()
    m["fight_id"] = m["fight_id"].astype(str)
    date_col = resolve_date_col(m)
    m[date_col] = pd.to_datetime(m[date_col], errors="coerce").dt.normalize()
    m = m.dropna(subset=[date_col]).sort_values([date_col, "fight_id"]).reset_index(drop=True)

    rating = defaultdict(lambda: 1500.0)
    fights = defaultdict(int)
    wins = defaultdict(int)
    losses = defaultdict(int)
    last_date = {}
    outcomes = defaultdict(lambda: deque(maxlen=5))
    opp_elos = defaultdict(lambda: deque(maxlen=5))
    win_streak = defaultdict(int)
    loss_streak = defaultdict(int)
    rows = []

    for _, r in m.iterrows():
        fid = str(r["fight_id"]); dt = r[date_col]
        rid, bid, wid = norm_id(r.get("r_id")), norm_id(r.get("b_id")), norm_id(r.get("winner_id"))
        if not rid or not bid:
            continue
        re, be = rating[rid], rating[bid]
        for fighter, opp, elo, opp_elo in ((rid, bid, re, be), (bid, rid, be, re)):
            hist = list(outcomes[fighter])
            rows.append({
                "fight_id": fid,
                "fighter_id": fighter,
                "hist_elo": elo,
                "hist_prior_fights": fights[fighter],
                "hist_prior_win_rate": wins[fighter] / fights[fighter] if fights[fighter] else np.nan,
                "hist_last3_win_rate": float(np.mean(hist[-3:])) if hist else np.nan,
                "hist_last5_win_rate": float(np.mean(hist[-5:])) if hist else np.nan,
                "hist_win_streak": win_streak[fighter],
                "hist_loss_streak": loss_streak[fighter],
                "hist_days_since_last": (dt - last_date[fighter]).days if fighter in last_date else np.nan,
                "hist_recent_opp_elo": float(np.mean(opp_elos[fighter])) if opp_elos[fighter] else np.nan,
                "hist_current_opp_elo": opp_elo,
            })

        red_win = wid == rid
        blue_win = wid == bid
        if red_win or blue_win:
            expected_red = 1.0 / (1.0 + 10.0 ** ((be - re) / 400.0))
            score_red = 1.0 if red_win else 0.0
            k = 32.0
            rating[rid] = re + k * (score_red - expected_red)
            rating[bid] = be + k * ((1.0 - score_red) - (1.0 - expected_red))
            for fighter, opp, won, opp_elo in ((rid, bid, red_win, be), (bid, rid, blue_win, re)):
                fights[fighter] += 1
                wins[fighter] += int(won)
                losses[fighter] += int(not won)
                outcomes[fighter].append(int(won))
                opp_elos[fighter].append(float(opp_elo))
                if won:
                    win_streak[fighter] += 1; loss_streak[fighter] = 0
                else:
                    loss_streak[fighter] += 1; win_streak[fighter] = 0
                last_date[fighter] = dt
    return pd.DataFrame(rows)


def add_fsr_trajectory(fsr: pd.DataFrame, traits: list[str]) -> pd.DataFrame:
    x = fsr.copy()
    x["fight_id"] = x["fight_id"].astype(str); x["fighter_id"] = x["fighter_id"].astype(str)
    x["event_date"] = pd.to_datetime(x["event_date"], errors="coerce").dt.normalize()
    x = x.sort_values(["fighter_id", "event_date", "fight_id"])
    out = x[["fight_id", "fighter_id"]].copy()
    for c in traits:
        vals = pd.to_numeric(x[c], errors="coerce")
        out[f"traj__{c}"] = vals - vals.groupby(x["fighter_id"]).shift(1)
    return out


def optional_physical_features(master: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    pairs = []
    candidate_pairs = {
        "age": [("r_age", "b_age"), ("red_age", "blue_age")],
        "reach": [("r_reach", "b_reach"), ("red_reach", "blue_reach"), ("r_reach_cm", "b_reach_cm")],
        "height": [("r_height", "b_height"), ("red_height", "blue_height"), ("r_height_cm", "b_height_cm")],
    }
    for name, options in candidate_pairs.items():
        for rc, bc in options:
            if rc in master.columns and bc in master.columns:
                pairs.append((name, rc, bc)); break
    if not pairs:
        return pd.DataFrame(columns=["fight_id"]), []
    mm = master.copy(); mm["fight_id"] = mm["fight_id"].astype(str)
    needed = ["fight_id", "r_id", "b_id"] + [c for _, rc, bc in pairs for c in (rc, bc)]
    mm = mm[needed].drop_duplicates("fight_id").merge(market[["fight_id", "favorite_id", "underdog_id"]], on="fight_id", how="inner")
    rows = []
    for _, r in mm.iterrows():
        fav_red = norm_id(r["favorite_id"]) == norm_id(r["r_id"])
        rec = {"fight_id": str(r["fight_id"])}
        for name, rc, bc in pairs:
            rv = pd.to_numeric(pd.Series([r[rc]]), errors="coerce").iloc[0]
            bv = pd.to_numeric(pd.Series([r[bc]]), errors="coerce").iloc[0]
            rec[f"physical__{name}"] = (rv - bv) if fav_red else (bv - rv)
        rows.append(rec)
    cols = [f"physical__{name}" for name, _, _ in pairs]
    return pd.DataFrame(rows), cols


def orient_side_features(market: pd.DataFrame, side_frame: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    look = side_frame.set_index(["fight_id", "fighter_id"])
    rows = []
    for _, m in market.iterrows():
        fid, fav, dog = str(m["fight_id"]), str(m["favorite_id"]), str(m["underdog_id"])
        if (fid, fav) not in look.index or (fid, dog) not in look.index:
            continue
        fr, dr = look.loc[(fid, fav)], look.loc[(fid, dog)]
        rec = {"fight_id": fid}
        for c in cols:
            fv = pd.to_numeric(pd.Series([fr.get(c)]), errors="coerce").iloc[0]
            dv = pd.to_numeric(pd.Series([dr.get(c)]), errors="coerce").iloc[0]
            rec[f"{prefix}{c}"] = float(fv - dv) if pd.notna(fv) and pd.notna(dv) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_eval(frame: pd.DataFrame, features: list[str], cut_date: pd.Timestamp, label: str) -> tuple[dict, pd.DataFrame]:
    z = frame.dropna(subset=["fight_date", "market_favorite_fair_p"]).copy()
    train = z[z["fight_date"] < cut_date].copy(); test = z[z["fight_date"] >= cut_date].copy()
    ytr, yte = safe_logit(train["market_favorite_fair_p"]), safe_logit(test["market_favorite_fair_p"])
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10.0)),
    ])
    model.fit(train[features], ytr)
    ptr, pte = model.predict(train[features]), model.predict(test[features])
    pprob = 1.0 / (1.0 + np.exp(-pte))
    test[f"pred_p__{label}"] = pprob
    test[f"residual_pp__{label}"] = 100.0 * (test["market_favorite_fair_p"] - pprob)
    rec = {
        "model": label, "feature_count": len(features), "train_fights": len(train), "test_fights": len(test),
        "train_r2_logit": r2_score(ytr, ptr), "test_r2_logit": r2_score(yte, pte),
        "train_rmse_logit": mean_squared_error(ytr, ptr) ** 0.5,
        "test_rmse_logit": mean_squared_error(yte, pte) ** 0.5,
        "mean_abs_residual_pp": float(test[f"residual_pp__{label}"].abs().mean()),
        "corr_pred_vs_market_p": float(pd.Series(pprob, index=test.index).corr(test["market_favorite_fair_p"])),
    }
    return rec, test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    market = build_two_way_market(MARKET_PATH)
    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    master = pd.read_parquet(MASTER_PATH).copy()
    traits = choose_trait_columns(fsr)
    base = build_matchups(market, fsr, master, traits)
    fsr_features = [f"delta__{c}" for c in traits]

    hist = build_history_features(master)
    hist_cols = [c for c in hist.columns if c.startswith("hist_")]
    hist_delta = orient_side_features(market, hist, hist_cols, "delta__")
    quality = ["delta__hist_elo", "delta__hist_recent_opp_elo", "delta__hist_current_opp_elo"]
    form = ["delta__hist_prior_win_rate", "delta__hist_last3_win_rate", "delta__hist_last5_win_rate", "delta__hist_win_streak", "delta__hist_loss_streak"]
    experience = ["delta__hist_prior_fights", "delta__hist_days_since_last"]

    traj_side = add_fsr_trajectory(fsr, traits)
    traj_cols = [c for c in traj_side.columns if c.startswith("traj__")]
    traj_delta = orient_side_features(market, traj_side, traj_cols, "delta__")
    trajectory = [f"delta__{c}" for c in traj_cols]

    physical_df, physical = optional_physical_features(master, market)
    frame = base.merge(hist_delta, on="fight_id", how="left").merge(traj_delta, on="fight_id", how="left")
    if physical:
        frame = frame.merge(physical_df, on="fight_id", how="left")

    # Same chronological boundary as prior audit: first 70% of joined fights for training.
    frame = frame.sort_values(["fight_date", "fight_id"]).reset_index(drop=True)
    cut = max(1, min(len(frame)-1, int(len(frame) * 0.70)))
    cut_date = pd.Timestamp(frame.iloc[cut]["fight_date"])

    families = {
        "fsr_only": fsr_features,
        "fsr_plus_quality": fsr_features + quality,
        "fsr_plus_form": fsr_features + form,
        "fsr_plus_experience": fsr_features + experience,
        "fsr_plus_trajectory": fsr_features + trajectory,
    }
    if physical:
        families["fsr_plus_physical"] = fsr_features + physical
    all_candidate = sorted(dict.fromkeys(quality + form + experience + trajectory + physical))
    families["fsr_plus_all_candidates"] = fsr_features + all_candidate

    metrics, tests = [], {}
    for label, feats in families.items():
        rec, test = fit_eval(frame, feats, cut_date, label)
        metrics.append(rec); tests[label] = test
    metrics = pd.DataFrame(metrics)
    base_row = metrics.loc[metrics["model"] == "fsr_only"].iloc[0]
    metrics["delta_test_r2_vs_fsr"] = metrics["test_r2_logit"] - base_row["test_r2_logit"]
    metrics["delta_rmse_vs_fsr"] = metrics["test_rmse_logit"] - base_row["test_rmse_logit"]
    metrics = metrics.sort_values("test_rmse_logit")

    # High-favorite residual behavior for each family.
    bucket_rows = []
    for label, test in tests.items():
        test = test.copy()
        test["bucket"] = pd.cut(test["market_favorite_fair_p"], [0.5,0.6,0.7,0.8,0.9,1.001], labels=["50-60","60-70","70-80","80-90","90+"], right=False)
        for bucket, g in test.groupby("bucket", observed=True):
            bucket_rows.append({
                "model": label, "bucket": str(bucket), "fights": len(g),
                "market_p": g["market_favorite_fair_p"].mean(),
                "pred_p": g[f"pred_p__{label}"].mean(),
                "residual_pp": g[f"residual_pp__{label}"].mean(),
                "abs_residual_pp": g[f"residual_pp__{label}"].abs().mean(),
            })
    buckets = pd.DataFrame(bucket_rows)

    # Candidate-only correlations with the baseline FSR residual on untouched test rows.
    base_test = tests["fsr_only"].copy()
    candidate_corr = []
    for c in all_candidate:
        if c not in base_test.columns:
            continue
        z = base_test[[c, "residual_pp__fsr_only"]].dropna()
        if len(z) >= 30:
            candidate_corr.append({"feature": c, "n": len(z), "corr_with_fsr_residual_pp": z[c].corr(z["residual_pp__fsr_only"])})
    candidate_corr = pd.DataFrame(candidate_corr)
    if not candidate_corr.empty:
        candidate_corr["abs_corr"] = candidate_corr["corr_with_fsr_residual_pp"].abs()
        candidate_corr = candidate_corr.sort_values("abs_corr", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_dir/"family_model_comparison.csv", index=False)
    buckets.to_csv(args.out_dir/"family_market_strength_buckets.csv", index=False)
    candidate_corr.to_csv(args.out_dir/"candidate_residual_correlations.csv", index=False)
    pd.DataFrame({"feature": fsr_features}).to_csv(args.out_dir/"fsr_features.csv", index=False)
    pd.DataFrame({"feature": all_candidate}).to_csv(args.out_dir/"candidate_features.csv", index=False)

    print("FSR RESIDUAL FEATURE-FAMILY SCREEN")
    print(f"joined fights={len(frame)} | cut_date={cut_date.date()} | fsr_features={len(fsr_features)} | candidates={len(all_candidate)}")
    print(f"physical features detected={physical}")
    print("\nFAMILY MODEL COMPARISON (chronological OOS; lower RMSE / higher R2 better)")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print("\nMARKET-STRENGTH BUCKET RESIDUALS")
    print(buckets.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print("\nTOP CANDIDATE CORRELATIONS WITH BASELINE FSR RESIDUAL")
    if candidate_corr.empty:
        print("none")
    else:
        print(candidate_corr.head(30).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    main()
