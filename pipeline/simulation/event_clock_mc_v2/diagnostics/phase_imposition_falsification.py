"""Measurement-only falsification of a prefight phase-imposition latent.

Question: do fighters consistently distort an opponent's normal phase usage, and
is that distortion useful out of sample beyond frozen FSR V3? No simulator or
FSR mechanics are modified.

The raw round-stats parquet is LONG format: one fighter/corner row per round.
This diagnostic aggregates that source to one red/blue fight row, then builds
strictly chronological prefight histories. For each fighter it measures how much
prior opponents' phase behavior changed versus those opponents' own prefight UFC
baselines.

Validation:
  1. next-fight stability of each fighter-level induced-distortion component;
  2. chronological 70/30 winner prediction: FSR-only vs FSR+imposition;
  3. mature priced bantamweight fights (>=3 prior UFC fights each): association
     with historical market-favorite minus frozen-MC residual.
"""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.fsr_v3.paths import FSR_V3_PREFIGHT_SNAPSHOTS_PATH

ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
MARKET_AUDIT_PATH = Path("/tmp/market_edge/bet_level_audit.csv")
OUT = Path("data/diagnostics/event_clock_mc_v2/phase_imposition_falsification")
MIN_PRIOR = 3
COMPONENTS = ("td_att", "ground_share", "clinch_share", "control_share")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _find(cols, aliases, required=True):
    norm = {_norm(c): c for c in cols}
    for a in aliases:
        if _norm(a) in norm:
            return norm[_norm(a)]
    for a in aliases:
        aa = _norm(a)
        hits = [c for c in cols if aa and aa in _norm(c)]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(f"none of aliases found: {aliases}; columns={list(cols)}")
    return None


def _num(s):
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def load_fight_aggregates() -> pd.DataFrame:
    """Convert long fighter-round rows to one red/blue fight row."""
    rd = pd.read_parquet(ROUND_PATH).copy()
    required = {
        "fight_id", "event_date", "corner", "fighter_id", "round",
        "td_attempted", "distance_attempted", "clinch_attempted",
        "ground_attempted", "ctrl_sec",
    }
    missing = sorted(required - set(rd.columns))
    if missing:
        raise RuntimeError(f"round stats missing required long-format columns: {missing}")

    rd["event_date"] = pd.to_datetime(rd["event_date"], errors="coerce").dt.normalize()
    rd["corner_norm"] = rd["corner"].astype(str).str.strip().str.lower()
    bad = ~rd["corner_norm"].isin(["red", "blue"])
    if bad.any():
        raise RuntimeError(f"unexpected corner values: {sorted(rd.loc[bad, 'corner'].astype(str).unique())}")

    for c in ["td_attempted", "distance_attempted", "clinch_attempted", "ground_attempted", "ctrl_sec"]:
        rd[c] = _num(rd[c])

    side = (
        rd.groupby(["fight_id", "event_date", "corner_norm", "fighter_id"], as_index=False)
        .agg(
            rounds=("round", "nunique"),
            td_attempted=("td_attempted", "sum"),
            distance_attempted=("distance_attempted", "sum"),
            clinch_attempted=("clinch_attempted", "sum"),
            ground_attempted=("ground_attempted", "sum"),
            ctrl_sec=("ctrl_sec", "sum"),
        )
    )
    # Normalize TD volume by observed rounds so fight length does not dominate.
    side["td_att"] = side["td_attempted"] / side["rounds"].clip(lower=1)
    phase_den = (
        side["distance_attempted"] + side["clinch_attempted"] + side["ground_attempted"]
    ).replace(0, np.nan)
    side["ground_share"] = (side["ground_attempted"] / phase_den).fillna(0.0)
    side["clinch_share"] = (side["clinch_attempted"] / phase_den).fillna(0.0)

    # Each fight should have exactly one aggregate row for each corner.
    counts = side.groupby("fight_id")["corner_norm"].nunique()
    valid_ids = counts[counts.eq(2)].index
    side = side[side["fight_id"].isin(valid_ids)].copy()

    red = side[side["corner_norm"].eq("red")].copy()
    blue = side[side["corner_norm"].eq("blue")].copy()
    red = red.rename(columns={
        "fighter_id": "r_id",
        **{c: f"r_{c}" for c in ["td_att", "ground_share", "clinch_share", "ctrl_sec"]},
    })
    blue = blue.rename(columns={
        "fighter_id": "b_id",
        **{c: f"b_{c}" for c in ["td_att", "ground_share", "clinch_share", "ctrl_sec"]},
    })
    keep_r = ["fight_id", "event_date", "r_id", "r_td_att", "r_ground_share", "r_clinch_share", "r_ctrl_sec"]
    keep_b = ["fight_id", "b_id", "b_td_att", "b_ground_share", "b_clinch_share", "b_ctrl_sec"]
    f = red[keep_r].merge(blue[keep_b], on="fight_id", how="inner", validate="one_to_one")

    ctrl_den = (f["r_ctrl_sec"] + f["b_ctrl_sec"]).replace(0, np.nan)
    f["r_control_share"] = (f["r_ctrl_sec"] / ctrl_den).fillna(0.5)
    f["b_control_share"] = (f["b_ctrl_sec"] / ctrl_den).fillna(0.5)
    f["fight_id"] = f["fight_id"].astype(str)
    f["r_id"] = f["r_id"].astype(str)
    f["b_id"] = f["b_id"].astype(str)
    return f.sort_values(["event_date", "fight_id"]).reset_index(drop=True)


def build_chronological_features(f: pd.DataFrame) -> pd.DataFrame:
    """Build strictly prefight own-history and induced-opponent-distortion ratings."""
    own: dict[str, dict[str, list[float]]] = {}
    induced: dict[str, dict[str, list[float]]] = {}
    rows = []

    def hist_mean(store, key, comp):
        vals = store.get(str(key), {}).get(comp, [])
        return float(np.mean(vals)) if vals else np.nan

    def count(store, key):
        z = store.get(str(key), {})
        return max((len(v) for v in z.values()), default=0)

    def add(store, key, comp, value):
        store.setdefault(str(key), {}).setdefault(comp, []).append(float(value))

    for _, r in f.iterrows():
        rid, bid = str(r.r_id), str(r.b_id)
        rec = {
            "fight_id": str(r.fight_id),
            "event_date": r.event_date,
            "r_id": rid,
            "b_id": bid,
            "r_prior": count(own, rid),
            "b_prior": count(own, bid),
        }

        # Snapshot all baselines BEFORE observing this fight.
        for side, fid, opp in [("r", rid, bid), ("b", bid, rid)]:
            for c in COMPONENTS:
                rec[f"{side}_own_prior_{c}"] = hist_mean(own, fid, c)
                rec[f"{side}_induced_{c}"] = hist_mean(induced, fid, c)
                rec[f"{side}_opp_prior_{c}"] = hist_mean(own, opp, c)
        rows.append(rec)

        # Add this fight's induced distortion versus each opponent's prefight norm.
        for side, fid, oside in [("r", rid, "b"), ("b", bid, "r")]:
            for c in COMPONENTS:
                base = rec[f"{side}_opp_prior_{c}"]
                if pd.notna(base):
                    add(induced, fid, c, float(r[f"{oside}_{c}"]) - float(base))

        # Only after induced residuals are computed, update each fighter's own history.
        for side, fid in [("r", rid), ("b", bid)]:
            for c in COMPONENTS:
                add(own, fid, c, float(r[f"{side}_{c}"]))

    return pd.DataFrame(rows)


def attach_outcome_and_fsr(feat: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH).copy()
    cols = master.columns
    mf = _find(cols, ["fight_id", "bout_id"])
    mr = _find(cols, ["r_id", "red_id", "r_fighter_id"])
    mb = _find(cols, ["b_id", "blue_id", "b_fighter_id"])
    win = _find(cols, ["winner_id", "winner_fighter_id", "winner"], False)
    if win is None:
        raise RuntimeError("winner field required in master")

    m = master[[mf, mr, mb, win]].drop_duplicates(mf).copy()
    m.columns = ["fight_id", "mr_id", "mb_id", "winner"]
    for c in ["fight_id", "mr_id", "mb_id", "winner"]:
        m[c] = m[c].astype(str)
    z = feat.merge(m, on="fight_id", how="left")
    z["red_won"] = (
        z["winner"].eq(z["r_id"]) | z["winner"].str.lower().eq("red")
    ).astype(int)

    fsr = pd.read_parquet(FSR_V3_PREFIGHT_SNAPSHOTS_PATH).copy()
    fsr["fight_id"] = fsr["fight_id"].astype(str)
    fsr["fighter_id"] = fsr["fighter_id"].astype(str)
    id_like = {"fight_id", "fighter_id", "event_date", "fighter_name", "name", "side"}
    nums = [c for c in fsr.columns if c not in id_like and pd.api.types.is_numeric_dtype(fsr[c])]
    red = fsr.rename(columns={"fighter_id": "r_id", **{c: f"r_fsr__{c}" for c in nums}})
    blue = fsr.rename(columns={"fighter_id": "b_id", **{c: f"b_fsr__{c}" for c in nums}})
    z = z.merge(red[["fight_id", "r_id"] + [f"r_fsr__{c}" for c in nums]], on=["fight_id", "r_id"], how="left")
    z = z.merge(blue[["fight_id", "b_id"] + [f"b_fsr__{c}" for c in nums]], on=["fight_id", "b_id"], how="left")
    for c in nums:
        z[f"fsr_delta__{c}"] = z[f"r_fsr__{c}"] - z[f"b_fsr__{c}"]
    for c in COMPONENTS:
        # Direction is intentionally not pre-labelled as good/bad. The OOS model
        # must learn whether positive/negative distortion is predictive.
        z[f"imp_delta__{c}"] = z[f"r_induced_{c}"] - z[f"b_induced_{c}"]
    return z


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


def stability_table(feat: pd.DataFrame) -> pd.DataFrame:
    """Next-fight stability on fighter-level prefight induced ratings, both corners."""
    long_rows = []
    for _, r in feat.iterrows():
        for side in ["r", "b"]:
            rec = {
                "fight_id": r["fight_id"],
                "event_date": r["event_date"],
                "fighter_id": r[f"{side}_id"],
                "prior": r[f"{side}_prior"],
            }
            for c in COMPONENTS:
                rec[c] = r[f"{side}_induced_{c}"]
            long_rows.append(rec)
    q = pd.DataFrame(long_rows).sort_values(["fighter_id", "event_date", "fight_id"])
    out = []
    for c in COMPONENTS:
        q[f"next__{c}"] = q.groupby("fighter_id")[c].shift(-1)
        mask = q[c].notna() & q[f"next__{c}"].notna() & q["prior"].ge(MIN_PRIOR)
        out.append({
            "feature": c,
            "n": int(mask.sum()),
            "next_fight_corr": float(q.loc[mask, c].corr(q.loc[mask, f"next__{c}"])) if int(mask.sum()) >= 5 else np.nan,
            "mean": float(q.loc[mask, c].mean()) if mask.any() else np.nan,
            "sd": float(q.loc[mask, c].std()) if mask.any() else np.nan,
        })
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = load_fight_aggregates()
    feat = build_chronological_features(f)
    z = attach_outcome_and_fsr(feat)
    mature = z[(z.r_prior >= MIN_PRIOR) & (z.b_prior >= MIN_PRIOR) & z.winner.notna()].sort_values(["event_date", "fight_id"]).copy()
    fsr_cols = [c for c in mature if c.startswith("fsr_delta__")]
    imp_cols = [c for c in mature if c.startswith("imp_delta__")]
    usable = mature[mature[fsr_cols].notna().any(axis=1)].copy()
    cut = max(1, int(0.70 * len(usable)))
    tr, te = usable.iloc[:cut], usable.iloc[cut:]

    rows = []
    for label, cols in [
        ("fsr_only", fsr_cols),
        ("fsr_plus_imposition", fsr_cols + imp_cols),
        ("imposition_only", imp_cols),
    ]:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=4000),
        )
        model.fit(tr[cols], tr.red_won)
        p = model.predict_proba(te[cols])[:, 1]
        rows.append({"model": label, **metrics(te.red_won.to_numpy(), p)})
    pred = pd.DataFrame(rows)
    stab = stability_table(feat)

    market_summary = pd.DataFrame()
    if MARKET_AUDIT_PATH.exists():
        b = pd.read_csv(MARKET_AUDIT_PATH)
        ml = b[b.market_key.eq("moneyline")].copy()
        market_rows = []
        for fid, g in ml.groupby(ml.fight_id.astype(str)):
            if len(g) < 2:
                continue
            g = g.sort_values("market_fair_probability", ascending=False)
            fav = g.iloc[0]
            market_rows.append({
                "fight_id": str(fid),
                "favorite_side": str(fav.outcome_side),
                "market_p": float(fav.market_fair_probability),
                "mc_p": float(fav.model_probability),
                "residual": float(fav.market_fair_probability - fav.model_probability),
                "red_prior_market": float(fav.red_prior_ufc_fights),
                "blue_prior_market": float(fav.blue_prior_ufc_fights),
            })
        mm = pd.DataFrame(market_rows)
        mm = mm[(mm.red_prior_market >= MIN_PRIOR) & (mm.blue_prior_market >= MIN_PRIOR)]
        mm = mm.merge(usable[["fight_id"] + imp_cols], on="fight_id", how="left")
        for c in imp_cols:
            mm[f"fav_{c}"] = np.where(mm.favorite_side.eq("red"), mm[c], -mm[c])
        xcols = [f"fav_{c}" for c in imp_cols]
        valid = mm.dropna(subset=["residual"]).copy()
        X = valid[xcols]
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
        pipe.fit(X, valid.residual)
        ph = pipe.predict(X)
        denom = float(np.sum((valid.residual - valid.residual.mean()) ** 2))
        r2 = np.nan if denom <= 0 else 1.0 - float(np.sum((valid.residual - ph) ** 2)) / denom
        market_summary = pd.DataFrame([{
            "mature_priced_fights": len(valid),
            "mean_market_p": valid.market_p.mean(),
            "mean_mc_p": valid.mc_p.mean(),
            "mean_residual_pp": 100 * valid.residual.mean(),
            "phase_fit_r2_insample": r2,
            "phase_fit_mae_pp": 100 * mean_absolute_error(valid.residual, ph),
            **{f"corr_residual__{x.replace('fav_imp_delta__', '')}": valid[x].corr(valid.residual) for x in xcols},
        }])
        valid.to_csv(OUT / "mature_market_rows.csv", index=False)

    pred.to_csv(OUT / "winner_prediction.csv", index=False)
    stab.to_csv(OUT / "phase_imposition_stability.csv", index=False)
    market_summary.to_csv(OUT / "market_residual_summary.csv", index=False)
    usable.to_csv(OUT / "mature_prefight_features.csv", index=False)

    print("PHASE IMPOSITION FALSIFICATION")
    print(f"fight aggregates={len(f)} mature usable={len(usable)} train={len(tr)} test={len(te)}")
    print("\nWINNER PREDICTION")
    print(pred.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nSTABILITY")
    print(stab.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if not market_summary.empty:
        print("\nMATURE MARKET RESIDUAL")
        print(market_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
