"""Research-only audit: does pre-fight UFC form explain market-vs-MC separation?

Uses the same priced bantamweight fights as the dog-compression audit. All form
features are computed strictly from UFC fights before the target bout. No market
information enters any simulator or FSR calculation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

MASTER = Path("data/master/ufc_master.parquet")
DATE_CANDIDATES = ("event_date", "fight_date", "date")


def _norm_id(x):
    s = str(x).strip()
    return "" if s in {"", "nan", "None"} else s


def _winner_side(row):
    wid, rid, bid = _norm_id(row.get("winner_id")), _norm_id(row.get("r_id")), _norm_id(row.get("b_id"))
    if wid and wid == rid:
        return "red"
    if wid and wid == bid:
        return "blue"
    w = str(row.get("winner", "")).strip()
    if w and w == str(row.get("r_name", "")).strip():
        return "red"
    if w and w == str(row.get("b_name", "")).strip():
        return "blue"
    return None


def _is_finish(method):
    s = str(method).lower()
    return ("decision" not in s) and (s not in {"", "nan", "none"})


def _resolve_date_col(frame):
    for col in DATE_CANDIDATES:
        if col in frame.columns:
            return col
    raise RuntimeError(
        f"master has no supported fight-date column; tried {DATE_CANDIDATES}; "
        f"available columns={list(frame.columns)}"
    )


def fighter_history(master, fighter_id, before_date, exclude_fight_id, date_col):
    fid = _norm_id(fighter_id)
    if not fid:
        return pd.DataFrame()
    h = master[
        ((master["r_id"].astype(str) == fid) | (master["b_id"].astype(str) == fid))
        & (master[date_col] < before_date)
        & (master["fight_id"].astype(str) != str(exclude_fight_id))
    ].copy()
    return h.sort_values([date_col, "fight_id"])


def form_features(h, fighter_id):
    if h.empty:
        return {
            "prior_fights": 0, "win_streak": 0, "loss_streak": 0,
            "last3_win_rate": np.nan, "last5_win_rate": np.nan,
            "wins_last3": 0, "wins_last5": 0,
            "finish_win_streak": 0, "last_fight_win": np.nan,
        }
    fid = _norm_id(fighter_id)
    outcomes = []
    finish_wins = []
    for _, r in h.iterrows():
        side = _winner_side(r)
        my_side = "red" if _norm_id(r.get("r_id")) == fid else "blue"
        win = int(side == my_side) if side is not None else 0
        outcomes.append(win)
        finish_wins.append(int(win and _is_finish(r.get("method"))))

    win_streak = 0
    for x in reversed(outcomes):
        if x == 1:
            win_streak += 1
        else:
            break
    loss_streak = 0
    for x in reversed(outcomes):
        if x == 0:
            loss_streak += 1
        else:
            break
    finish_streak = 0
    for w, fw in zip(reversed(outcomes), reversed(finish_wins)):
        if w == 1 and fw == 1:
            finish_streak += 1
        else:
            break

    return {
        "prior_fights": len(outcomes),
        "win_streak": win_streak,
        "loss_streak": loss_streak,
        "last3_win_rate": float(np.mean(outcomes[-3:])),
        "last5_win_rate": float(np.mean(outcomes[-5:])),
        "wins_last3": int(sum(outcomes[-3:])),
        "wins_last5": int(sum(outcomes[-5:])),
        "finish_win_streak": finish_streak,
        "last_fight_win": int(outcomes[-1]),
    }


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1-p))


def loocv_rmse(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        m = LinearRegression().fit(X[mask], y[mask])
        preds[i] = m.predict(X[i:i+1])[0]
    return float(mean_squared_error(y, preds) ** 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced-fights-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    market = pd.read_csv(args.priced_fights_path).drop_duplicates("fight_id").copy()
    market["fight_id"] = market["fight_id"].astype(str)
    master = pd.read_parquet(MASTER).copy()
    master["fight_id"] = master["fight_id"].astype(str)
    date_col = _resolve_date_col(master)
    master[date_col] = pd.to_datetime(master[date_col], errors="coerce").dt.normalize()
    if master[date_col].isna().all():
        raise RuntimeError(f"resolved master date column {date_col!r} contains no parseable dates")

    target = master[master["fight_id"].isin(set(market["fight_id"]))].copy()
    if len(target) != len(market):
        raise RuntimeError(f"target mismatch master={len(target)} market={len(market)}")
    target = target.set_index("fight_id")

    rows = []
    for _, m in market.iterrows():
        fid = str(m["fight_id"])
        t = target.loc[fid]
        date = pd.Timestamp(t[date_col]).normalize()
        fav_side = str(m["favorite_side"])
        fav_id = t["r_id"] if fav_side == "red" else t["b_id"]
        dog_id = t["b_id"] if fav_side == "red" else t["r_id"]
        fav = form_features(fighter_history(master, fav_id, date, fid, date_col), fav_id)
        dog = form_features(fighter_history(master, dog_id, date, fid, date_col), dog_id)
        rec = {
            "fight_id": fid, "favorite": m["favorite"], "underdog": m["underdog"],
            "market_favorite_fair_p": float(m["market_favorite_fair_p"]),
            "mc_favorite_p": float(m["mc_favorite_p"]),
            "compression_pp": float(m["compression_pp"]),
            "favorite_won": int(m["favorite_won"]),
        }
        for k in fav:
            rec[f"fav_{k}"] = fav[k]
            rec[f"dog_{k}"] = dog[k]
            rec[f"delta_{k}"] = fav[k] - dog[k] if pd.notna(fav[k]) and pd.notna(dog[k]) else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["market_logit"] = safe_logit(out["market_favorite_fair_p"])
    out["mc_logit"] = safe_logit(out["mc_favorite_p"])
    out["market_minus_mc_logit"] = out["market_logit"] - out["mc_logit"]

    feature_cols = [
        "delta_win_streak", "delta_loss_streak", "delta_last3_win_rate",
        "delta_last5_win_rate", "delta_finish_win_streak", "delta_prior_fights",
    ]
    corr_rows = []
    for c in feature_cols:
        z = out[[c, "market_favorite_fair_p", "mc_favorite_p", "compression_pp", "market_minus_mc_logit", "favorite_won"]].dropna()
        corr_rows.append({
            "feature": c, "n": len(z),
            "corr_market_favorite_p": z[c].corr(z["market_favorite_fair_p"]),
            "corr_mc_favorite_p": z[c].corr(z["mc_favorite_p"]),
            "corr_compression_pp": z[c].corr(z["compression_pp"]),
            "corr_market_minus_mc_logit": z[c].corr(z["market_minus_mc_logit"]),
            "corr_actual_favorite_win": z[c].corr(z["favorite_won"]),
        })
    corrs = pd.DataFrame(corr_rows)

    models = []
    base = out[["mc_logit", "market_logit"]].dropna()
    models.append({"model": "mc_only", "features": "mc_logit", "n": len(base), "loocv_rmse_market_logit": loocv_rmse(base[["mc_logit"]], base["market_logit"])})
    for c in feature_cols:
        z = out[["mc_logit", c, "market_logit"]].dropna()
        models.append({"model": f"mc_plus_{c}", "features": f"mc_logit + {c}", "n": len(z), "loocv_rmse_market_logit": loocv_rmse(z[["mc_logit", c]], z["market_logit"])})
    z = out[["mc_logit", "delta_win_streak", "delta_last5_win_rate", "market_logit"]].dropna()
    models.append({"model": "mc_plus_streak_last5", "features": "mc_logit + delta_win_streak + delta_last5_win_rate", "n": len(z), "loocv_rmse_market_logit": loocv_rmse(z[["mc_logit", "delta_win_streak", "delta_last5_win_rate"]], z["market_logit"])})
    model_df = pd.DataFrame(models).sort_values("loocv_rmse_market_logit")

    out["streak_bucket"] = pd.cut(out["delta_win_streak"], [-99,-1,0,1,2,99], labels=["dog_ahead","equal","fav+1","fav+2","fav+3plus"], right=False)
    buckets = out.groupby("streak_bucket", observed=True).agg(
        fights=("fight_id","size"),
        mean_market_p=("market_favorite_fair_p","mean"),
        mean_mc_p=("mc_favorite_p","mean"),
        mean_compression_pp=("compression_pp","mean"),
        favorite_win_rate=("favorite_won","mean"),
    ).reset_index()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir/"fight_level_prefight_form.csv", index=False)
    corrs.to_csv(args.out_dir/"form_correlations.csv", index=False)
    model_df.to_csv(args.out_dir/"incremental_market_models.csv", index=False)
    buckets.to_csv(args.out_dir/"win_streak_buckets.csv", index=False)

    print("BANTAMWEIGHT PREFIGHT FORM VS MARKET AUDIT")
    print(f"fights={len(out)} | all form strictly prior to target fight | master date column={date_col}")
    print("\nFORM CORRELATIONS")
    print(corrs.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nINCREMENTAL MARKET MODELS (lower LOOCV RMSE is better)")
    print(model_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nWIN-STREAK BUCKETS")
    print(buckets.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()
